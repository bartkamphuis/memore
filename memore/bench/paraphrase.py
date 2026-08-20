"""Metamorphic paraphrase harness (wrap-up-spec.md W2). RESULTS.md §26.

    MEMORE_GRAPH=memore_slots uv run python -m memore.bench.paraphrase --runs 1

## What this measures, and why it is not another slots.py

`slots.py` asks whether P1 gets a turn RIGHT. This asks whether P1 gets the same turn the
SAME WAY when the wording changes and the meaning does not. That is the property the whole
identity design rests on and the one thing never measured: subject identity is exact match
on `normalize_subject` (no threshold, no embedding, no LLM), so two paraphrases of one
assertion collide only if P1 names them identically. §15 fixed the co-reference failures it
could see in one script; this puts a number on what is left.

A metamorphic test needs no gold labels -- there is no "correct" subject string here, only
agreement -- which is exactly why it can be committed and re-run. The fixture below is the
whole instrument and it is pre-registered: it was written before the first run, and
`wrap-up-spec.md` W2 forbids tuning `_SYSTEM` to raise the number it produces. §20 already
closed those arms; this is not a licence to reopen them.

**A low number is the deliverable.** This test exists to measure a known weakness honestly,
not to be passed.

## The design decision that makes the number readable

Variant 0 and variant 1 of every turn are **byte-identical on purpose**. P1 is stochastic at
temperature 0 (§15, §18.5) and §22 then found nine runs byte-identical, so how much of any
disagreement is noise rather than paraphrase sensitivity is an open question, not an
assumption. The identical pair answers it in the same run, on the same axes:

    self_agreement         variant 0 vs variant 1. The floor. Disagreement here is
                           stochasticity and is NOT evidence about paraphrasing.
    paraphrase_agreement   variants 0, 2, 3, 4 -- the genuinely reworded ones.

Read the second against the first. Without the control the two effects are pooled, and
§22.4's caution is the standing example of what that costs.

## Held fixed, deliberately

Every variant is extracted with `recent=[]`, `assistant_response=""` and
`known_subjects=None`. The wording is then the ONLY variable. That is not how the system
runs -- in a live session the hint list is the dominant force on naming (§18.4, "the
magnet") -- and the number here is therefore about P1 in isolation, which is the component
under test. A hint list would improve agreement by construction (it shows the model what it
said last time) and would measure the hint list instead.

## Axes, and the rule for a variant that stores nothing

Four, reported separately and never pooled:

    cardinality   every variant of the turn emitted the same NUMBER of facts (and >= 1).
    subject       normalized `subject_hint` identical across variants.
    attribute     normalized `attribute` identical across variants.
    arity         `single_valued` identical across variants.

The field axes compare each variant's FIRST fact, and a variant that emitted nothing is
excluded from them -- with the turn reported as `NO-DATA` when fewer than two variants
survive, never as a pass. A silent turn trivially agrees with itself, which is the §17.4 /
§18.8 hole: an axis blind to the empty case scores highest on the input that destroys the
store. `cardinality` is what catches it, and it is why that axis is listed first.

Per axis two numbers are printed, because they answer different questions:

    unanimous     turns where every compared variant agrees. What a user experiences --
                  one dissenter splits the subject and the collision is lost.
    pairwise      agreeing variant pairs / compared pairs. Degrades smoothly, so a turn
                  with one dissenter reads differently from a turn with four answers.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import os
from dataclasses import dataclass, field

from ..config import WritePathConfig
from ..extract import OllamaExtractor
from ..keys import normalize_subject
from ..types import CandidateFact

# Twelve single-assertion turns, four variants each. Variants [0] and [1] are IDENTICAL --
# the stochasticity control described above; [2] and [3] are paraphrases that preserve the
# assertion and change the wording as far as ordinary speech does: word order, active vs
# passive, synonym, register, and the fronting of the value rather than the subject.
#
# The turns are deliberately UNAMBIGUOUS. A turn whose meaning is genuinely underdetermined
# ("Bart maintains memore", typed by Bart -- is the subject `Bart` or `the user`?) would
# fail this test for a reason that is not paraphrase sensitivity, and §15's rule 2 already
# owns that case. Failures here are meant to be attributable to wording alone.
TURNS: list[list[str]] = [
    [
        "I deploy to staging by default.",
        "I deploy to staging by default.",
        "By default I deploy to staging.",
        "Staging is where I deploy unless I say otherwise.",
    ],
    [
        "My sister Lisa lives in Rotterdam.",
        "My sister Lisa lives in Rotterdam.",
        "Lisa, my sister, is based in Rotterdam.",
        "Rotterdam is where my sister Lisa lives.",
    ],
    [
        "I'm allergic to penicillin.",
        "I'm allergic to penicillin.",
        "I have a penicillin allergy.",
        "Penicillin gives me an allergic reaction.",
    ],
    [
        "We use Postgres for the main database.",
        "We use Postgres for the main database.",
        "The main database is Postgres.",
        "Postgres is what we run as our primary database.",
    ],
    [
        "I have a cat called Miso.",
        "I have a cat called Miso.",
        "My cat's name is Miso.",
        "Miso is my cat.",
    ],
    [
        "The standup is at 9:30 every weekday.",
        "The standup is at 9:30 every weekday.",
        "Every weekday we have standup at 9:30.",
        "We do standup at half past nine, Monday to Friday.",
    ],
    [
        "I'm flying to Lisbon on 29 August 2026.",
        "I'm flying to Lisbon on 29 August 2026.",
        "On 29 August 2026 I fly to Lisbon.",
        "My flight to Lisbon is booked for 29 August 2026.",
    ],
    [
        "I speak Dutch and English.",
        "I speak Dutch and English.",
        "Dutch and English are the languages I speak.",
        "I can speak both English and Dutch.",
    ],
    [
        "I prefer tea over coffee in the morning.",
        "I prefer tea over coffee in the morning.",
        "In the morning I'd rather have tea than coffee.",
        "First thing, I go for tea rather than coffee.",
    ],
    [
        "The office moved to Utrecht last month.",
        "The office moved to Utrecht last month.",
        "Last month the office relocated to Utrecht.",
        "Our office is in Utrecht now -- we moved a month ago.",
    ],
    [
        "My laptop is a 14-inch MacBook Pro.",
        "My laptop is a 14-inch MacBook Pro.",
        "I work on a 14-inch MacBook Pro.",
        "The machine I use is a MacBook Pro, the 14-inch one.",
    ],
    [
        "The memore project is written in Python.",
        "The memore project is written in Python.",
        "memore is a Python project.",
        "We wrote memore in Python.",
    ],
]

# Indices into a turn's variant list. Kept as constants rather than slices so the control
# and the treatment cannot drift apart when the fixture grows.
CONTROL_PAIR = (0, 1)
PARAPHRASES = (0, 2, 3)

AXES = ("subject", "attribute", "arity")


@dataclass
class VariantResult:
    text: str
    facts: list[CandidateFact]

    @property
    def n(self) -> int:
        return len(self.facts)

    def field(self, axis: str) -> str | bool | None:
        """The compared value, or None when this variant emitted nothing.

        `attribute` is normalized with `normalize_subject` because that is exactly what
        `consolidate()` does to it before it becomes half of a slot key -- comparing the
        raw string would report a disagreement the system does not have.
        """
        if not self.facts:
            return None
        fact = self.facts[0]
        if axis == "subject":
            return normalize_subject(fact.subject_hint)
        if axis == "attribute":
            return normalize_subject(fact.attribute)
        return fact.single_valued


@dataclass
class TurnResult:
    index: int
    variants: list[VariantResult]

    def compared(self, which: tuple[int, ...]) -> list[VariantResult]:
        return [self.variants[i] for i in which]

    def cardinality_ok(self, which: tuple[int, ...]) -> bool:
        counts = {v.n for v in self.compared(which)}
        return len(counts) == 1 and counts != {0}

    def values(self, axis: str, which: tuple[int, ...]) -> list[str | bool]:
        return [v.field(axis) for v in self.compared(which) if v.field(axis) is not None]


@dataclass
class AxisScore:
    unanimous: int = 0
    turns: int = 0
    agreeing_pairs: int = 0
    pairs: int = 0
    no_data: int = 0

    def add(self, values: list) -> None:
        if len(values) < 2:
            # Fewer than two variants said anything: nothing to compare. Counted, never
            # passed -- a silent turn agrees with itself.
            self.no_data += 1
            return
        self.turns += 1
        if len(set(values)) == 1:
            self.unanimous += 1
        for a, b in itertools.combinations(values, 2):
            self.pairs += 1
            if a == b:
                self.agreeing_pairs += 1

    def line(self) -> str:
        u = f"{self.unanimous}/{self.turns}"
        p = f"{self.agreeing_pairs}/{self.pairs}"
        rate_u = self.unanimous / self.turns if self.turns else 0.0
        rate_p = self.agreeing_pairs / self.pairs if self.pairs else 0.0
        extra = f"   no-data {self.no_data}" if self.no_data else ""
        return f"unanimous {u} ({rate_u:.3f})   pairwise {p} ({rate_p:.3f}){extra}"


@dataclass
class RunReport:
    run: int
    turns: list[TurnResult] = field(default_factory=list)

    def score(self, which: tuple[int, ...]) -> dict[str, AxisScore]:
        scores = {axis: AxisScore() for axis in AXES}
        cardinality = AxisScore()
        for turn in self.turns:
            cardinality.turns += 1
            if turn.cardinality_ok(which):
                cardinality.unanimous += 1
            for a, b in itertools.combinations(turn.compared(which), 2):
                cardinality.pairs += 1
                if a.n == b.n and a.n >= 1:
                    cardinality.agreeing_pairs += 1
            for axis in AXES:
                scores[axis].add(turn.values(axis, which))
        return {"cardinality": cardinality, **scores}


async def run_once(run: int) -> RunReport:
    extractor = OllamaExtractor(WritePathConfig())
    report = RunReport(run=run)
    try:
        for index, variants in enumerate(TURNS):
            results = []
            for text in variants:
                # Empty context on every call: the wording is the only variable (see the
                # module docstring). `known_subjects=None` in particular -- the hint list
                # would show the model its own previous answer and measure the hint list.
                facts = await extractor.extract(text, "", [], None)
                results.append(VariantResult(text=text, facts=list(facts)))
            report.turns.append(TurnResult(index=index, variants=results))
    finally:
        await extractor.llm.aclose()
    return report


def print_run(report: RunReport) -> None:
    print(f"\n=== run {report.run} ===")
    for turn in report.turns:
        counts = [v.n for v in turn.variants]
        disagreements = []
        for axis in AXES:
            values = turn.values(axis, PARAPHRASES)
            if len(set(values)) > 1:
                disagreements.append(f"{axis}={sorted({str(v) for v in values})}")
        flag = "  " if not disagreements and turn.cardinality_ok(PARAPHRASES) else "!!"
        head = turn.variants[0].text
        print(f" {flag} [{turn.index:2d}] n={counts}  {head}")
        for line in disagreements:
            print(f"      {line}")
        if any(v.n == 0 for v in turn.variants):
            # Diagnostic, not an axis -- the §17.4 argument. It is `cardinality` that
            # scores this; the line exists so the reason is readable.
            silent = [i for i, v in enumerate(turn.variants) if v.n == 0]
            print(f"      wrote nothing: variants {silent}")

    for label, which in (("self (identical wording)", CONTROL_PAIR), ("paraphrase", PARAPHRASES)):
        print(f"\n  {label}, variants {which}:")
        for axis, score in report.score(which).items():
            print(f"    {axis:12s} {score.line()}")


async def main_async(runs: int) -> None:
    graph = os.environ.get("MEMORE_GRAPH", "(unset)")
    print(f"extractor={WritePathConfig().extractor_model}  graph={graph}  turns={len(TURNS)}")
    print("no store is written -- this harness calls P1 directly and compares its output")
    reports = [await run_once(run) for run in range(runs)]
    for report in reports:
        print_run(report)
    if runs > 1:
        # Runs are printed separately and never averaged, for the reason §11's command
        # block gives: P1 varies at temperature 0 and the variance IS the finding.
        print("\n=== per-run paraphrase unanimity (never averaged; see CLAUDE.md) ===")
        for report in reports:
            scores = report.score(PARAPHRASES)
            cells = "  ".join(f"{a} {s.unanimous}/{s.turns}" for a, s in scores.items())
            print(f"  run {report.run}: {cells}")


def main() -> None:
    parser = argparse.ArgumentParser(description="metamorphic paraphrase stability of P1")
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()
    asyncio.run(main_async(args.runs))


if __name__ == "__main__":
    main()
