"""Slot fidelity — does P1 name the same property the same way twice?

RESULTS.md §11 introduced the `(subject, attribute)` slot and named its two residual
failure modes. Live testing across three parallel sessions promoted both from residual to
dominant, and neither the FactConsolidation bench nor the calibration harness can see
either one: the bench is one-attribute-per-subject by construction, and calibration
measures the read path.

    slot SPLIT      one property, two names. The contradiction never fires and a stale
                    fact stays live. Measured here as the `must-collide` miss rate.
    slot COLLISION  two properties, one name. A true fact is superseded. Measured here as
                    the `must-coexist` false-supersede rate.
    SUBJECT split   one entity, two names ("Lisa" and "the user's sister"). Facts scatter
                    across two subjects, so they never compete and the split costs nothing
                    a liveness check can see -- but recall then finds half an entity, and
                    it is what made the three live stores diverge most visibly. Measured
                    separately, because a coexist group can be fully live AND fully split.

The two move in OPPOSITE directions, which is the whole reason both are measured. Making
P1 reuse attributes harder fixes splits and causes collisions; making it coin freely does
the reverse. A change that improves one number and wrecks the other is not an improvement,
and a single "fragmentation" count would have hidden exactly that.

Run it three times, and read the runs separately rather than averaged. P1 is measurably
non-deterministic even at `temperature=0.0`, and the variance IS the finding — the three
live sessions that motivated this harness resolved the same Python-vs-Ruby contradiction
one, zero and one times respectively.

    uv run python -m memore.bench.slots --runs 3

Turns are reconstructed from those sessions. Each pair-bearing turn must yield exactly one
fact; a turn that yields none or two is reported as UNMEASURABLE rather than scored, so a
change to P1's salience never quietly inflates a rate.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import uuid
from dataclasses import dataclass, field

from ..config import EmbedConfig, StoreConfig, WritePathConfig
from ..consolidate import DeterministicConsolidator
from ..embed import OllamaEmbedder
from ..extract import OllamaExtractor
from ..keys import normalize_subject
from ..store.falkor import FalkorStore
from ..types import Message
from ..writepath import WritePath

# Turn text only. Indices below refer to positions in this list.
TURNS: list[str] = [
    # 0-2  identity: three properties of one subject, all true at once
    "My name is Bart and I'm 58 years old",
    "I'm a software engineer",
    "I specialise in memory systems for LLMs",
    # 3-4  the §11 residual case: two facts, one subject, same VALUE, different property
    "I was born in Den Haag",
    "I wrote the memory system in Den Haag",
    # 5-6  one subject, two independent properties
    "The memory system does a deterministic 100ms lookup",
    "The memory system is written in Python",
    # 7-8  COLLIDE: capital of the Netherlands
    "The capital of the Netherlands is Amsterdam",
    "Correction, the capital of the Netherlands is Den Haag",
    # 9-10 COLLIDE: language preference. The case all three live sessions disagreed on.
    "I like coding in Python",
    "Actually I've changed my mind, I prefer Ruby now",
    # 11   the "preference" trap: a DIFFERENT preference of the same subject. A
    #      document-frequency merge over attributes would fold this into 9/10 and
    #      supersede a true fact -- which is why §10's rule is not reused here.
    "I hate green sweets",
    # 12-13 COLLIDE: employer
    "I work at a company called Northgate",
    "I've switched jobs, I now work for Supreme Data Systems",
    # 14-15 COLLIDE: Lisa's location
    "My sister Lisa lives in Amsterdam",
    "Lisa has moved, she lives in Den Haag now",
    # 16-17 Lisa, two more independent properties
    "Lisa is 45 years old",
    "Lisa's birthday is on the 24th of August",
    # 18-20 one subject, three independent properties
    "My dog Pixel is a Rottweiler",
    "Pixel is 4 years old",
    "Pixel is female",
    # 21-22 COLLIDE on a LONG property name. Short attributes ("age", "employer") survive
    #       token-sorting unmangled; a five-word one does not, and the sorted key is what
    #       `subject_slots` shows back to P1 to reuse. The live store held
    #       `city code favourite python write` -- nobody reuses that verbatim.
    "Den Haag is my favourite city to write Python code",
    "Actually Lisbon is now my favourite city to write Python code",
    # 23-24 COLLIDE on another multi-word property, phrased differently the second time.
    "I always deploy to the test environment before going live",
    "I have changed that, I now deploy straight to production",
]

# (earlier turn, later turn): the later fact MUST supersede the earlier one.
MUST_COLLIDE: list[tuple[int, int]] = [(7, 8), (9, 10), (12, 13), (14, 15), (21, 22), (23, 24)]

# Every turn in a group MUST still be live: they are different properties.
MUST_COEXIST: list[list[int]] = [
    [0, 1, 2],        # name / age / occupation  (0 may yield 2 facts -- see note below)
    [3, 4],           # birthplace vs where the system was written -- both "Den Haag"
    [5, 6],           # lookup latency vs implementation language
    [10, 11],         # language preference vs sweets preference
    [16, 17],         # Lisa: age vs birthday
    [18, 19, 20],     # Pixel: breed / age / gender
    [22, 24],         # favourite coding city vs deploy habit -- both the user, both true
]


@dataclass
class TurnResult:
    index: int
    turn: str
    ordinals: list[int] = field(default_factory=list)
    # As P1 emitted them, for the vocabulary report.
    slots: list[tuple[str, str]] = field(default_factory=list)
    # After `normalize_subject`, which is what actually decides identity. The two differ:
    # "preferred programming language" and "language preferred programming" are distinct
    # strings and the SAME key, because normalization sorts tokens. Scoring the raw
    # strings would report splits that the store never suffered.
    keys: list[tuple[str, str]] = field(default_factory=list)
    cases: list[str] = field(default_factory=list)


@dataclass
class RunReport:
    run: int
    turns: list[TurnResult]
    live_by_ordinal: dict[int, bool]

    def _single(self, index: int) -> int | None:
        """The one ordinal a pair-bearing turn wrote, or None if it wrote 0 or 2+."""
        rows = self.turns[index].ordinals
        return rows[0] if len(rows) == 1 else None

    def collide_results(self) -> list[tuple[tuple[int, int], str, str]]:
        out = []
        for earlier, later in MUST_COLLIDE:
            a, b = self._single(earlier), self._single(later)
            if a is None or b is None:
                out.append(((earlier, later), "UNMEASURABLE", "turn wrote 0 or 2+ facts"))
                continue
            slot_a = self.turns[earlier].keys[0]
            slot_b = self.turns[later].keys[0]
            same_slot = slot_a == slot_b
            resolved = not self.live_by_ordinal.get(a, True) and self.live_by_ordinal.get(b, False)
            detail = f"{slot_a[0]}::{slot_a[1]}  vs  {slot_b[0]}::{slot_b[1]}"
            if resolved:
                out.append(((earlier, later), "RESOLVED", detail))
            else:
                out.append((
                    (earlier, later),
                    "SPLIT" if not same_slot else "MISSED",
                    detail,
                ))
        return out

    def subject_results(self) -> list[tuple[list[int], str, str]]:
        """Did every turn in a group land on ONE subject?

        Checked over the coexist groups and the collide pairs alike: both assert that the
        turns concern a single entity, and a split subject breaks that whether or not any
        fact was superseded.
        """
        out = []
        groups = MUST_COEXIST + [list(pair) for pair in MUST_COLLIDE]
        for group in groups:
            subjects = {k[0] for i in group for k in self.turns[i].keys}
            if not subjects:
                out.append((group, "UNMEASURABLE", "no facts written"))
            elif len(subjects) == 1:
                out.append((group, "OK", ""))
            else:
                out.append((group, "SUBJECT-SPLIT", " | ".join(sorted(subjects))))
        return out

    def coexist_results(self) -> list[tuple[list[int], str, str]]:
        out = []
        for group in MUST_COEXIST:
            ordinals = [o for i in group for o in self.turns[i].ordinals]
            if not ordinals:
                out.append((group, "UNMEASURABLE", "no facts written"))
                continue
            dead = [o for o in ordinals if not self.live_by_ordinal.get(o, False)]
            if dead:
                slots = ", ".join(
                    f"{k[0]}::{k[1]}" for i in group for k in self.turns[i].keys
                )
                out.append((group, "COLLIDED", f"{len(dead)} superseded — {slots}"))
            else:
                out.append((group, "OK", ""))
        return out


async def run_once(run: int, graph: str) -> RunReport:
    session = f"slotbench-{run}-{uuid.uuid4().hex[:6]}"
    embed_config = EmbedConfig.from_env()
    store = FalkorStore(StoreConfig.from_env(), dimension=embed_config.dimension)
    embedder = OllamaEmbedder(embed_config)
    await store.connect()
    write_path = WritePath(
        OllamaExtractor(WritePathConfig()),
        DeterministicConsolidator(store, embedder),
        WritePathConfig(),
        store=store,
    )
    try:
        results: list[TurnResult] = []
        history: list[Message] = []
        for index, turn in enumerate(TURNS):
            outcome = await write_path.run(session, turn, "", list(history))
            row = TurnResult(index=index, turn=turn)
            for item in outcome.outcomes:
                row.ordinals.append(item.ordinal)
                row.slots.append((item.candidate.subject_hint, item.candidate.attribute))
                row.keys.append((
                    normalize_subject(item.candidate.subject_hint),
                    normalize_subject(item.candidate.attribute),
                ))
                row.cases.append(item.case.value)
            results.append(row)
            history.append(Message(role="user", content=turn))
            history.append(Message(role="assistant", content="Understood."))

        facts = await store.facts_in_session(session)
        live = {f.ordinal: f.invalid_at is None for f in facts}
        await store.clear_session(session)
        return RunReport(run=run, turns=results, live_by_ordinal=live)
    finally:
        await embedder.aclose()
        await store.aclose()


def print_run(report: RunReport) -> tuple[int, int, int, int, int, int]:
    print(f"\n=== run {report.run} ===")
    collide = report.collide_results()
    resolved = sum(1 for _, verdict, _ in collide if verdict == "RESOLVED")
    measurable_c = sum(1 for _, verdict, _ in collide if verdict != "UNMEASURABLE")
    print(f"  must-collide  {resolved}/{measurable_c} resolved")
    for (a, b), verdict, detail in collide:
        flag = "ok  " if verdict == "RESOLVED" else "FAIL"
        print(f"    {flag} [{a}->{b}] {verdict:<13} {detail}")

    coexist = report.coexist_results()
    ok = sum(1 for _, verdict, _ in coexist if verdict == "OK")
    measurable_x = sum(1 for _, verdict, _ in coexist if verdict != "UNMEASURABLE")
    print(f"  must-coexist  {ok}/{measurable_x} intact")
    for group, verdict, detail in coexist:
        flag = "ok  " if verdict == "OK" else "FAIL"
        print(f"    {flag} {str(group):<14} {verdict:<13} {detail}")

    subjects = report.subject_results()
    whole = sum(1 for _, verdict, _ in subjects if verdict == "OK")
    measurable_s = sum(1 for _, verdict, _ in subjects if verdict != "UNMEASURABLE")
    print(f"  one-subject   {whole}/{measurable_s} coherent")
    for group, verdict, detail in subjects:
        if verdict != "OK":
            print(f"    FAIL {str(group):<14} {verdict:<13} {detail}")

    # Slot vocabulary per subject: the raw material of a split, whether or not it cost a
    # pair. Printed because a subject accumulating five near-synonymous slots is the
    # warning sign that precedes the failure. Keys, not raw strings -- the raw strings
    # over-report, since token sorting already collapses word-order variants.
    by_subject: dict[str, list[str]] = {}
    for row in report.turns:
        for subject, attribute in row.keys:
            by_subject.setdefault(subject, []).append(attribute)
    print("  slot keys coined")
    for subject, attributes in sorted(by_subject.items()):
        print(f"    {subject:<30} {', '.join(attributes)}")
    return resolved, measurable_c, ok, measurable_x, whole, measurable_s


async def main_async(runs: int) -> None:
    graph = os.environ.get("MEMORE_GRAPH", "memore")
    print(f"extractor={WritePathConfig().extractor_model}  graph={graph}  turns={len(TURNS)}")
    totals = [0, 0, 0, 0, 0, 0]
    for run in range(1, runs + 1):
        report = await run_once(run, graph)
        for i, value in enumerate(print_run(report)):
            totals[i] += value
    print(
        f"\n=== {runs} run(s) ===\n"
        f"  must-collide resolved   {totals[0]}/{totals[1]}   (slot split)\n"
        f"  must-coexist intact     {totals[2]}/{totals[3]}   (slot collision)\n"
        f"  one-subject coherent    {totals[4]}/{totals[5]}   (subject split)\n"
        "  read the runs separately -- variance across identical inputs is the finding."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3)
    asyncio.run(main_async(parser.parse_args().runs))


if __name__ == "__main__":
    main()
