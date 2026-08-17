"""Slot fidelity — does P1 name the same property the same way twice?

RESULTS.md §11 introduced the `(subject, attribute)` slot and named its two residual
failure modes. Live testing across three parallel sessions promoted both from residual to
dominant, and neither the FactConsolidation bench nor the calibration harness can see
either one: the bench is one-attribute-per-subject by construction, and calibration
measures the read path.

    slot SPLIT      one property, two names. The contradiction never fires and a stale
                    fact stays live. Measured here as the `must-collide` miss rate.
    slot COLLISION  two properties, one name. A true fact is superseded. Measured here as
                    the `must-coexist` false-supersede rate. Its worst form is the
                    CATEGORY attribute: P1 names a slot after the fact's TYPE
                    (`preference`, `interests`, `likes`) rather than the property it gives
                    a value for, and every simultaneously-true fact of that type then
                    lands in one slot and retires the last. The hint list amplifies it --
                    `subject_slots` shows the category back to P1 and the prompt asks for
                    exact reuse -- so a bad first naming is self-reinforcing for the rest
                    of the session. Turns 39-41. RESULTS.md §18.
    SUBJECT split   one entity, two names ("Lisa" and "the user's sister Lisa"). Facts
                    scatter across two subjects, so they never compete and the split costs
                    nothing a liveness check can see -- but recall then finds half an
                    entity, and it is what made the three live stores diverge most visibly.
                    Measured separately, because a coexist group can be fully live AND
                    fully split.
    SUBJECT merge   two entities, one name ("Lisa" and "Lisa's daughter Fien"). The
                    opposite error, and the worse one: merged subjects compete, so a true
                    fact about one is superseded by a fact about the other and is gone.

Both pairs move in OPPOSITE directions, which is the whole reason all four are measured.
Making P1 reuse attributes harder fixes slot splits and causes slot collisions; making it
reuse SUBJECTS harder fixes subject splits and causes subject merges. A change that
improves one number and wrecks its partner is not an improvement, and a single
"fragmentation" count would have hidden exactly that.

The two directions are not equally bad, and the harness does not pretend they are. A split
costs recall and is recoverable -- both facts are still in the store, correctly. A merge
destroys a fact permanently. That is the same asymmetry `AliasConfig` records, so a fix
that trades one over-merge for one recovered split is a REGRESSION, not a wash.

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
    # ---- appended for the co-reference axis. Deliberately APPENDED rather than
    # ---- interleaved: every index above keeps its meaning, so the collide/coexist
    # ---- numbers stay comparable with the runs recorded in RESULTS.md §11 and §14.
    #
    # 25-27 CO-REFER: an entity introduced with a descriptor, then named alone. This is
    #       the measured baseline failure, reproduced on a second entity so one flaky
    #       Lisa run cannot carry the axis on its own.
    "My colleague Tom reviews all of my pull requests",
    "Tom prefers small commits",
    "Tom is based in Rotterdam",
    # 28-29 CO-REFER the other way round: named first, described afterwards. The fix for
    #       one direction is not automatically the fix for the other -- "prefer the
    #       proper name" resolves 25-27 by shortening and 28-29 by refusing to lengthen.
    "I have a cat called Miso",
    "My cat Miso is 3 years old",
    # 30-31 CO-REFER, and simultaneously the over-merge trap below: Fien is her own
    #       entity, introduced through Lisa.
    "Lisa's daughter Fien is 12 years old",
    "Fien goes to school in Haarlem",
    # 32    OVER-MERGE TRAP: a PART of a subject already in the store. The test suite is
    #       not the memory system, and its runtime is not the system's lookup latency.
    "The memory system's test suite takes 4 minutes to run",
    # 33    OVER-MERGE TRAP: a second Tom, separated only by surname and relation. The
    #       hardest case here, and the one a merge rule keyed on a shared proper noun
    #       gets wrong by construction.
    "My neighbour Tom Bakker plays the trumpet",
    # 34    The SECOND fact about the test suite, and the reason turn 32's over-merge is
    #       a latent defect rather than a harmless one. On its own an over-merged subject
    #       costs nothing -- turn 32 took its own attribute and superseded nothing. This
    #       turn is the one that collects the bill: merged into `the memory system` it
    #       asks for `implementation language`, which turn 6 already holds, and a true
    #       fact dies. Paired with turn 6 on MUST_COEXIST so that death is a measured
    #       number rather than a paragraph of reasoning.
    "The test suite is written in pytest",
    # 35    Restores what [3, 4] was built for and no longer tests. §11's case is two
    #       properties of ONE subject sharing a VALUE, and it needs a subject that the
    #       "is it a property or a part" rule cannot move: both of these are plainly
    #       about the user, and neither has the "the X of Y" shape that sends a fact
    #       elsewhere.
    "I still live in Den Haag",
    # ---- appended for the assistant-reply axis (RESULTS.md §17). Appended, again, so
    # ---- every index above keeps the meaning the §11/§14/§15 runs measured.
    #
    # 36-37 QUESTION turns whose REPLY restates memory. This is the channel: the reader
    #       holds the whole conversation, P1 holds three messages, so a reply can assert
    #       anything from anywhere and P1 reads it as "what this turn asserts". Measured
    #       on the 2026-08-17 console run at turn 23, where a question stored a JOIN over
    #       two facts that no single turn or fact contained.
    "What is my favourite city to write Python code?",
    "Where does Lisa live?",
    # 38    The other direction, and the reason this axis cannot be passed by muting the
    #       reply channel entirely: a real user assertion, delivered WITH a reply that
    #       also restates old memory. The new fact must still land (paired with 24 on
    #       MUST_COLLIDE) while the restatement must not.
    "I've moved my deploys back to the staging environment",
    # ---- appended for the CATEGORY-ATTRIBUTE defect (RESULTS.md §18). Appended again, so
    # ---- every index above keeps the meaning §11/§14/§15/§17 measured. No replies: these
    # ---- turns measure how P1 NAMES a slot, and a reply would confound that with §17.
    #
    # 39-41 The defect, measured at 5-of-6 on a replay of the 2026-08-17 console run. P1
    #       names Bud's first preference with the bare CATEGORY `preference` -- a fact
    #       TYPE, not a property -- and the hint list then does the rest: `subject_slots`
    #       shows `Bud -> preference, ...` and the prompt tells P1 to reuse the exact
    #       property string for a listed property, so every later preference snaps to the
    #       same slot and retires the last. Three separate turns ON PURPOSE: inside one
    #       utterance the §16 same-batch guard already withholds CONTRADICTION, so a
    #       single-turn version of this would pass while the defect stood.
    #
    #       Four disjoint domains (beer / film / furniture / language, below) so that no
    #       two of these can legitimately compete. If they collapse into one slot it is
    #       the category magnet, not a real disagreement.
    "My colleague Bud likes beer",
    "Bud likes the first Matrix movie",
    "Bud likes red gaming chairs",
    # 42-43 The paired refuse-list for the axis above, written before the fix per
    #       §3/§10/§15 -- and specifically shaped against the fix that will suggest itself
    #       first. "Never supersede a preference" (a COLLECTION_TYPES keyword list in the
    #       consolidator) scores 3/3 on 39-41 and breaks THIS pair, because a favourite is
    #       a preference that genuinely holds one value at a time. A slot that can never
    #       resolve is the FactConsolidation task failing, which is the one thing this
    #       project may not trade away.
    "Bud's favourite programming language is Go",
    "Bud has changed his mind, his favourite programming language is Rust",
    # 44    §16's same-batch guard, which had no fixture here at all. ONE utterance, three
    #       facts, all true: before the guard "does not like milk in their tea" and "likes
    #       green tea" retired each other in both columns of the 2026-08-17 console run,
    #       because candidates from one extraction differ in ordinal only by their position
    #       in P1's output array. Scored as a one-index coexist group -- every fact this
    #       single turn writes must still be live.
    "I like milk in my coffee but not in my tea, and I drink a lot of green tea",
    # 45    Bud's seating, which §16's containment branch was written for -- and which is
    #       here as ONE turn, not two, because the two-turn version does not work and the
    #       reason is worth recording. A second turn saying only "Bud sits on the red
    #       chair" was tried and P1 emitted NOTHING for it in 3 runs of 3: a pure
    #       restatement asserts no new fact, and P1's salience gate is right to drop it.
    #       So the containment branch is not reachable from a restatement that merely
    #       repeats; it needs one that reads as news while saying less, which no scripted
    #       turn here produced. A group that cannot express the failure cannot guard
    #       against it, so the pair was removed rather than left scoring two easy passes.
    #       The branch stays guarded where it can be constructed directly -- the
    #       trace-derived tests at the end of tests/test_consolidation.py. RESULTS.md §18.
    "Bud sits on the red chair in the Whangarei office",
]

# The assistant's reply for a turn, when the turn has one. Absent means `""`, which is
# what turns 0-35 were measured with -- do not give them replies, or the collide/coexist
# numbers stop being comparable with the runs in §11, §14 and §15.
#
# These are written the way the console's reader actually answers: correct, helpful, and
# freely restating stored facts the user did not mention this turn.
REPLIES: dict[int, str] = {
    36: "Lisbon is your favourite city to write Python code. You deploy straight to "
        "production these days, and you're a software engineer specialising in memory "
        "systems for LLMs.",
    37: "Lisa lives in Den Haag — she moved there from Amsterdam. She's 45, and her "
        "birthday is on the 24th of August.",
    38: "Got it, staging is your deploy target again. Previously you deployed straight "
        "to production, and your favourite city to write Python code is Lisbon.",
}

# Turns that MUST yield no candidates at all: the user asserts nothing durable, and
# everything durable in earshot comes from the assistant's reply.
#
# The refuse-list of this axis, written before the fix, per §3/§10/§15. Note what it is
# NOT: a rule that drops every candidate on a question turn would score 2/2 here and walk
# straight back into §12a, where an explicit "remember that X" was silently refused. Turn
# 38 is the guard against that, and the four existing axes are the rest of it.
MUST_NOT_EXTRACT: list[int] = [36, 37]

# (earlier turn, later turn): the later fact MUST supersede the earlier one.
MUST_COLLIDE: list[tuple[int, int]] = [
    (7, 8), (9, 10), (12, 13), (14, 15), (21, 22), (23, 24),
    (24, 38),   # deploy target again -- turn 38 carries a reply, and must still land.
    (42, 43),   # Bud's favourite language. A scalar preference: it MUST still resolve, and
                # it is the pair that any "preferences never supersede" rule breaks.
]

# `(24, 38)` above fails 3/3 both before and after the §17 fix, exactly as `[3, 4]` fails
# below: P1 names turn 38's slot `deploy target` and turn 24's `deployment workflow`, which
# is ordinary attribute-naming variance and measures nothing about the reply channel. Left
# failing rather than relaxed, per §3 and §10 -- an assertion edited because a change broke
# it stops being evidence. So the collide total reads 18/21 where §11/§14/§15 recorded
# 18/18: the six original pairs are unchanged, and TWO known failures now sit in the
# denominator. RESULTS.md §17.5.

# Every turn in a group MUST still be live: they are different properties.
MUST_COEXIST: list[list[int]] = [
    [0, 1, 2],        # name / age / occupation  (0 may yield 2 facts -- see note below)
    [3, 4],           # birthplace vs where the system was written -- both "Den Haag"
    [5, 6],           # lookup latency vs implementation language
    [10, 11],         # language preference vs sweets preference
    [16, 17],         # Lisa: age vs birthday
    [18, 19, 20],     # Pixel: breed / age / gender
    [22, 24],         # favourite coding city vs deploy habit -- both the user, both true
    [26, 27],         # Tom: commit preference vs location
    [30, 31],         # Fien: age vs school
    [6, 34],          # the system's language vs the TEST SUITE's language. Both true,
                      # and both die into one slot if turn 34's subject over-merges into
                      # the memory system -- which is precisely the bill turn 32 defers.
    [3, 35],          # birth place vs current residence -- one subject, one value
                      # ("Den Haag"), two properties. §11's case, restored.
    # --- added with the category-attribute turns (RESULTS.md §18) ---
    [39, 40, 41, 43], # Bud: beer / film / furniture / language. Four disjoint domains, all
                      # true at once. 43 is included deliberately: it is the SURVIVOR of
                      # the (42, 43) collide, so if the category magnet swallows the
                      # favourite-language slot too, the later correction retires the three
                      # preferences as well and this group catches it. The measured defect
                      # fails this 5 runs out of 6.
    [44],             # one utterance, several true facts -- §16's same-batch guard.
]

# Every turn in a group names ONE entity, so every fact must land on ONE subject key.
#
# Written out rather than derived from the two lists above, which is what the previous
# version did. Derivation made the axis unreadable -- you could not see what was being
# asserted without mentally unioning two other lists -- and it could not express the two
# things this axis now needs: a group spanning a collide pair AND a coexist group (Lisa,
# below), and groups with no liveness assertion at all.
#
# The first thirteen entries ARE the derived set, unchanged, so their count is directly
# comparable with the 37/39 recorded in RESULTS.md §14.
MUST_COREFER: list[list[int]] = [
    [0, 1, 2], [3, 4], [5, 6], [10, 11], [16, 17], [18, 19, 20], [22, 24],
    [7, 8], [9, 10], [12, 13], [14, 15], [21, 22], [23, 24],
    # --- added with the co-reference turns ---
    [14, 15, 16, 17],  # Lisa across all four. Strictly stronger than [14,15] + [16,17]:
                       # both halves can be internally coherent and still disagree, which
                       # is exactly what run 2 of the baseline did (lisa | lisa sister user).
    [25, 26, 27],      # Tom
    [28, 29],          # Miso
    [30, 31],          # Fien
    [32, 34],          # the test suite -- both facts must land on it, not on the system
    [3, 35],           # the user, twice, same value
    # --- added with the category-attribute turns ---
    [39, 40, 41, 42, 43, 45],  # Bud, across all six. Turn 39 introduces him with a
                       # descriptor ("my colleague Bud") and every later turn names him
                       # alone, so this is rule 3 of §15 as well: the subject is `Bud`, not
                       # `the user's colleague Bud`. A split here would also make the
                       # coexist group above pass for the wrong reason -- facts scattered
                       # across two subject keys never compete, so nothing supersedes.
]

# `[3, 4]` above fails 3/3 under the shipped prompt and is KEPT failing on purpose.
# "I wrote the memory system in Den Haag" now files under `memory system` rather than
# `the user`, which is arguable either way -- and editing an assertion because a change
# broke it is what RESULTS.md §3 and §10 refuse. It is left as a standing failure rather
# than quietly relaxed, so read 3 known failures into the one-subject total. It also no
# longer exercises §11's same-value-different-property case, which is why `[3, 35]` was
# added rather than `[3, 4]` repaired. RESULTS.md §15.

# (a, b): these two turns are about DIFFERENT entities and MUST NOT share a subject key.
#
# The axis that did not exist before, and the reason it has to. Every fix for subject
# splitting pushes toward merging, and a merge is unrecoverable in the direction that
# started this whole thread: merged subjects compete, competing facts supersede, and a
# true fact is lost. `MUST_COREFER` alone would score a rule that merges everything at
# 100%. These pairs are the refuse-list, written before the fix, per the precedent of
# RESULTS.md §3 and §10 -- both of which hand-labelled the pairs a rule must decline
# before letting the rule's aggregate score decide anything.
MUST_DISTINGUISH: list[tuple[int, int]] = [
    (15, 30),   # Lisa vs her daughter Fien -- Fien's subject contains "Lisa"
    (26, 33),   # Tom the colleague vs Tom Bakker the neighbour -- same first name
    (6, 32),    # the memory system vs its test suite -- part-of, not identity
    (19, 29),   # Pixel the dog vs Miso the cat -- both are "the user's pet" if P1
                # generalises the descriptor instead of using the proper name
    (6, 34),    # the memory system vs its test suite, on the pair that actually costs a
                # fact. (6, 32) catches the same over-merge one turn earlier, where it is
                # still free; this one catches it after it has been paid for.
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
        """Did every turn in a group land on ONE subject?"""
        out = []
        for group in MUST_COREFER:
            subjects = {k[0] for i in group for k in self.turns[i].keys}
            if not subjects:
                out.append((group, "UNMEASURABLE", "no facts written"))
            elif len(subjects) == 1:
                out.append((group, "OK", ""))
            else:
                out.append((group, "SUBJECT-SPLIT", " | ".join(sorted(subjects))))
        return out

    def distinguish_results(self) -> list[tuple[tuple[int, int], str, str]]:
        """Did two turns about DIFFERENT entities stay on different subjects?

        Reported as a share of MEASURABLE pairs, like every other axis, and a pair where
        either turn wrote nothing is measurable at neither end. Note the asymmetry with
        the co-reference axis: a split is a recall loss, an over-merge destroys a fact.
        A change that trades one for one is not neutral.
        """
        out = []
        for a, b in MUST_DISTINGUISH:
            keys_a = {k[0] for k in self.turns[a].keys}
            keys_b = {k[0] for k in self.turns[b].keys}
            if not keys_a or not keys_b:
                out.append(((a, b), "UNMEASURABLE", "a turn wrote no facts"))
            elif keys_a & keys_b:
                out.append(((a, b), "OVER-MERGED", " | ".join(sorted(keys_a & keys_b))))
            else:
                out.append(((a, b), "OK", ""))
        return out

    def not_extracted_results(self) -> list[tuple[int, str, str]]:
        """Did a turn that asserts nothing stay silent, whatever its reply restated?

        Always measurable, unlike the other four axes: "wrote no facts" is the assertion
        here rather than the thing that makes a pair unscorable.

        Which is exactly why this number is meaningless alone. Anything that mutes P1 --
        a salience change, a stricter confidence floor, a question-suppressor -- scores
        2/2 here while destroying the store, and §12a records that failure happening for
        real. Read it next to turn 38 (a user assertion delivered WITH a leaky reply) and
        the four axes above, never on its own.
        """
        out = []
        for index in MUST_NOT_EXTRACT:
            row = self.turns[index]
            if not row.slots:
                out.append((index, "OK", ""))
            else:
                out.append((
                    index,
                    "LEAKED",
                    " | ".join(f"{s}::{a}" for s, a in row.slots),
                ))
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
            reply = REPLIES.get(index, "")
            outcome = await write_path.run(session, turn, reply, list(history))
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
            # "Understood." where the turn has no reply, so turns 0-35 see exactly the
            # history the earlier runs saw.
            history.append(Message(role="assistant", content=reply or "Understood."))

        facts = await store.facts_in_session(session)
        live = {f.ordinal: f.invalid_at is None for f in facts}
        await store.clear_session(session)
        return RunReport(run=run, turns=results, live_by_ordinal=live)
    finally:
        await embedder.aclose()
        await store.aclose()


def print_run(report: RunReport) -> tuple[int, ...]:
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

    distinct = report.distinguish_results()
    apart = sum(1 for _, verdict, _ in distinct if verdict == "OK")
    measurable_d = sum(1 for _, verdict, _ in distinct if verdict != "UNMEASURABLE")
    print(f"  distinct      {apart}/{measurable_d} kept apart")
    for pair, verdict, detail in distinct:
        if verdict != "OK":
            print(f"    FAIL {str(pair):<14} {verdict:<13} {detail}")

    silent = report.not_extracted_results()
    quiet = sum(1 for _, verdict, _ in silent if verdict == "OK")
    print(f"  no-reply-leak {quiet}/{len(silent)} silent")
    for index, verdict, detail in silent:
        if verdict != "OK":
            print(f"    FAIL [{index}]          {verdict:<13} {detail}")

    # DIAGNOSTIC, not an axis: turns that asserted something durable and stored nothing.
    # Deliberately unscored, because one observation does not make a rate and a
    # MUST_EXTRACT list would be over-fitted to it -- but it must be VISIBLE, because
    # every axis above is silent about it. A coexist group whose member wrote no fact
    # still reads OK (there is no dead ordinal to find), so a change that quietly mutes
    # P1 improves `no-reply-leak` and disturbs nothing else. Measured on the 2026-08-17
    # replay: "Bud likes red gaming chairs" was dropped in 6 runs out of 6 while the same
    # turn's "$560" landed every time. RESULTS.md §18.
    dropped = [
        row.index for row in report.turns
        if not row.slots and row.index not in MUST_NOT_EXTRACT
    ]
    print(f"  wrote nothing {dropped}   (diagnostic only -- not scored)")

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
    return (resolved, measurable_c, ok, measurable_x, whole, measurable_s,
            apart, measurable_d, quiet, len(silent))


async def main_async(runs: int) -> None:
    graph = os.environ.get("MEMORE_GRAPH", "memore")
    print(f"extractor={WritePathConfig().extractor_model}  graph={graph}  turns={len(TURNS)}")
    totals = [0] * 10
    for run in range(1, runs + 1):
        report = await run_once(run, graph)
        for i, value in enumerate(print_run(report)):
            totals[i] += value
    print(
        f"\n=== {runs} run(s) ===\n"
        f"  must-collide resolved   {totals[0]}/{totals[1]}   (slot split)\n"
        f"  must-coexist intact     {totals[2]}/{totals[3]}   (slot collision)\n"
        f"  one-subject coherent    {totals[4]}/{totals[5]}   (subject split)\n"
        f"  distinct kept apart     {totals[6]}/{totals[7]}   (subject OVER-merge)\n"
        f"  no reply leak           {totals[8]}/{totals[9]}   (assistant-reply channel)\n"
        "  read the runs separately -- variance across identical inputs is the finding.\n"
        "  the last two move against each other: any rule that merges harder improves\n"
        "  one and costs the other, so neither number means anything without the other."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3)
    asyncio.run(main_async(parser.parse_args().runs))


if __name__ == "__main__":
    main()
