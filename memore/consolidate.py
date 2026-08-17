"""The deterministic freshness primitive -- the novel core of this project.

recall-poc-spec.md §4 overrides recall-writepath-spec.md §2.3: the NEW / DUPLICATE /
CONTRADICTION / REFINEMENT decision is NOT delegated to a graph store's edge logic and
NOT delegated to an LLM's judgment. It is a deterministic function of (subject key,
freshness ordinal, value comparison).

Why that matters: on MemoryAgentBench's FactConsolidation task the field is measurably
bad at exactly this decision -- Zep/Graphiti 7%, HippoRAG-v2 54% single-hop -- even when
the tiebreak rule is stated outright in the prompt. The thesis is that treating
freshness as a primitive we own, rather than an emergent property of retrieval, fixes it.

The freshness ordinal is *arrival order within a session*. We never read an ordinal out
of the input; facts get the next counter value as they are ingested. For a corpus that
happens to arrive in chronological order, arrival order and chronology coincide, which
is the whole point -- no LLM is asked to reason about which fact is newer.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from .aliases import AliasConfig, SubjectVocabulary
from .embed import Embedder, cosine
from .keys import normalize_subject
from .store import ConsolidatingStore
from .store.falkor import normalize_subject as subject_key
from .types import (
    CandidateFact,
    ConsolidationCase,
    ConsolidationOutcome,
    StoredFact,
)


@dataclass(frozen=True)
class ConsolidationConfig:
    """Thresholds for the value comparison of recall-poc-spec.md §4.3.

    These govern *value* comparison only. Subject identity is an exact match on the
    normalized subject key -- no threshold, no fuzziness, because that is what makes the
    decision deterministic rather than a retrieval artifact.

    Embedding comparison defaults OFF, and the reason is an asymmetry measured on
    MemoryAgentBench sh_32k (RESULTS.md §6). The two error directions are not equally bad:

      false DUPLICATE     the update is DISCARDED. The store keeps the stale value and
                          answers wrongly, forever. Unrecoverable.
      false CONTRADICTION both facts are kept, the newer one is live, and the answer is
                          still right. Costs store size and an extra SUPERSEDED line.

    So the threshold must never be tuned for "accuracy" symmetrically. At 0.97, two real
    value changes were swallowed -- "flanker … rugby union" -> "… rugby" (cos 0.982) and
    "Catholic Church" -> "Catholicism" (cos 0.986) -- because a small value edit barely
    moves a sentence embedding. Meanwhile genuine paraphrases sit at 0.877, *below*
    real contradictions at 0.849-0.911, so no threshold separates the cases anyway.

    Exact normalized-string equality catches the duplicates that matter and cannot lose a
    fact. Turn this on only with a comparison that looks at the value, not the sentence.
    """

    duplicate_similarity: float = 0.97
    use_embedding_comparison: bool = False
    # Subject *identity* (`memore.aliases`), as opposed to the value comparison above.
    # Exact match on the normalized key stays the rule; this widens what counts as the same
    # key by absorbing generic relation words, which is the measured residual error after
    # order-insensitive sorting (RESULTS.md §3). It is still an exact match, still no
    # embedding and no LLM -- the only new input is the session's own token statistics.
    alias: AliasConfig = AliasConfig()


class Consolidator(Protocol):
    """The seam recall-writepath-spec.md §2.3 asks for, so the responsibility can move."""

    async def consolidate(
        self, session_id: str, candidates: list[CandidateFact]
    ) -> list[ConsolidationOutcome]: ...


def _normalize_value(text: str) -> str:
    return " ".join(
        "".join(c if c.isalnum() or c.isspace() else " " for c in text.lower()).split()
    )


def _competing(candidate: CandidateFact, live: list[StoredFact]) -> list[StoredFact]:
    """The live facts on this subject that the candidate is allowed to supersede.

    Subject identity says "these facts are about the same thing". It does NOT say "these
    facts cannot both be true", and treating it as if it did is the defect RESULTS.md §11
    measures: across three real conversational sessions, 18 supersedes fired and 1 was
    correct. "The user is a software engineer" was marked SUPERSEDED in all three,
    knocked out by "Bart specialises in memory systems for LLMs" -- same subject, no
    disagreement whatsoever.

    The attribute is the slot that holds exactly one value at a time, so only a fact in
    the same slot competes. `""` is unspecified and competes with everything, in EITHER
    direction:

      old fact, no attribute       any candidate supersedes it, as before this field
      candidate with no attribute  supersedes every live fact, as before this field

    That asymmetry is deliberate and is what makes the change inert where it has no
    evidence to act on -- a graph written before §11, or the bench harness, whose cached
    subject extraction carries no attribute (`bench/extract.py`). It also keeps the error
    direction the safe one: with no slot information we fall back to over-superseding,
    which leaves the right answer live and merely mislabels a stale-looking neighbour,
    rather than under-superseding and leaving a genuinely dead fact presented as current.
    """
    key = normalize_subject(candidate.attribute)
    if not key:
        return list(live)
    return [f for f in live if not f.attribute or f.attribute == key]


def _supersede_targets(incumbent: StoredFact, competing: list[StoredFact]) -> list[StoredFact]:
    """Which of the competing facts a contradiction with `incumbent` actually retires.

    Before the same-batch guard, a slot was supposed to hold exactly one live fact, so
    superseding the whole competing set was self-healing: any extra live fact in the slot
    was corruption, and the arriving fact outranked all of it.

    That guard deliberately breaks the one-live-fact rule. Two facts from one utterance now
    coexist in a slot on purpose, and superseding "everything in the slot" would collect
    them both on the next correction -- which is the original bug arriving one turn late:

        turn  4  "no milk in tea" + "likes green tea"   coexist, both live      (the fix)
        turn 30  "I hate green tea now"                 retires BOTH            (the bill)

    So a contradiction retires the incumbent, plus any live fact from a DIFFERENT batch --
    those are the ones the one-live-fact invariant was ever about. The incumbent's own
    batch siblings were never ordered against it, so disagreeing with the incumbent says
    nothing about them.

    Facts written before this field carried a batch (`source_episode_id == ""`) fall back
    to the old behaviour wholesale, exactly as `attribute == ""` does in `_competing`: with
    no batch information, over-superseding is the safe error, and old graphs and the bench
    stay inert rather than silently acquiring a rule their data cannot support.
    """
    if not incumbent.source_episode_id:
        return competing
    return [
        f
        for f in competing
        if f.id == incumbent.id or f.source_episode_id != incumbent.source_episode_id
    ]


class DeterministicConsolidator:
    """recall-poc-spec.md §4. No LLM, no graph inference, no judgment call."""

    def __init__(
        self,
        store: ConsolidatingStore,
        embedder: Embedder,
        config: ConsolidationConfig | None = None,
    ):
        self.store = store
        self.embedder = embedder
        self.config = config or ConsolidationConfig()
        # Session-scoped monotonic arrival counter. Seeded from the store on first use
        # so a restart does not reissue ordinals.
        self._counters: dict[str, int] = {}
        # Incumbent facts are re-compared on every arrival for the same subject; without
        # this the same text is re-embedded once per contradiction.
        self._vec_cache: dict[str, list[float]] = {}
        # Session subject vocabulary for alias resolution. Seeded from the store once, then
        # maintained in memory: rebuilding it per candidate would be a store round-trip per
        # fact, and rebuilding it per batch would make subject identity depend on where the
        # batch boundaries happen to fall.
        # Stays private so `test_no_llm_in_the_consolidation_decision` keeps its exact
        # collaborator assertion -- that tripwire is worth more than the convenience of a
        # public attribute. Read it through `vocabulary()`.
        self._vocabs: dict[str, SubjectVocabulary] = {}

    async def _vectors_for(self, texts: list[str]) -> dict[str, list[float]]:
        missing = [t for t in dict.fromkeys(texts) if t not in self._vec_cache]
        if missing:
            for text, vec in zip(missing, await self.embedder.embed(missing), strict=True):
                self._vec_cache[text] = vec
        return {t: self._vec_cache[t] for t in texts}

    async def prewarm(self, texts: list[str]) -> None:
        """Embed `texts` now so later `consolidate()` calls find them cached.

        Exists because a batch is a SEMANTIC unit, not a throughput one: everything in one
        `consolidate()` call is treated as one utterance whose candidates have no freshness
        order between them (RESULTS.md §16). A caller with facts that genuinely arrived
        separately -- the bench harnesses, where every fact is its own turn -- must hand
        them over one at a time, and would otherwise pay one embedder round-trip per fact.
        Prewarm restores the batching where it belongs, on the embedder.
        """
        if texts:
            await self._vectors_for(texts)

    async def _vocab_for(self, session_id: str) -> SubjectVocabulary:
        if session_id not in self._vocabs:
            keys = await self.store.live_subject_keys(session_id)
            self._vocabs[session_id] = SubjectVocabulary(self.config.alias, keys)
        return self._vocabs[session_id]

    def vocabulary(self, session_id: str) -> SubjectVocabulary | None:
        """The subject vocabulary built for a session, or None if it never consolidated.

        `vocabulary(session).merges` is the audit trail of every subject identity this
        consolidator widened -- the evidence the merge rule has to be judged on, since
        RESULTS.md §3 rejected the ungated version *despite* a better aggregate score.
        """
        return self._vocabs.get(session_id)

    async def _next_ordinal(self, session_id: str) -> int:
        if session_id not in self._counters:
            self._counters[session_id] = await self.store.max_ordinal(session_id)
        self._counters[session_id] += 1
        return self._counters[session_id]

    def _classify(
        self, candidate: CandidateFact, live: list[StoredFact], candidate_vec: list[float] | None,
        live_vecs: dict[str, list[float]] | None, batch_ids: frozenset[str] = frozenset(),
    ) -> tuple[ConsolidationCase, StoredFact | None, list[StoredFact]]:
        """Step 2 of §4: decide the case deterministically.

        Returns the case, the incumbent, and the *competing set* -- the live facts the
        candidate is allowed to supersede. That third value is the whole of the §11 fix:
        it is a subset of `live`, not all of it.

        `batch_ids` are the facts this same `consolidate()` call has already written. They
        are live, and they are in the slot, but the candidate may not out-rank them on
        FRESHNESS -- see the same-batch guard below.
        """
        candidate_value = _normalize_value(candidate.fact)
        candidate_slot = normalize_subject(candidate.attribute)

        # DUPLICATE is checked against every live fact on the subject, not just the
        # competing slot. The same sentence arriving under a different attribute is still
        # the same sentence, and a second copy is exactly the store bloat writepath §2.2
        # case 2 exists to prevent.
        for fact in live:
            if _normalize_value(fact.fact) == candidate_value:
                return ConsolidationCase.DUPLICATE, fact, []

        competing = _competing(candidate, live)
        if not competing:
            # Either the subject is empty, or every live fact on it occupies a DIFFERENT
            # slot and is still true. Coexist. Before §11 this branch did not exist and
            # the candidate superseded whatever was there.
            return ConsolidationCase.NEW, None, []

        # `competing` inherits `live`'s ordinal-descending order; the incumbent is the
        # freshest live fact IN THIS SLOT.
        incumbent = competing[0]

        if self.config.use_embedding_comparison and candidate_vec is not None and live_vecs:
            incumbent_vec = live_vecs.get(incumbent.id)
            if incumbent_vec is not None:
                similarity = cosine(candidate_vec, incumbent_vec)
                if similarity >= self.config.duplicate_similarity:
                    return ConsolidationCase.DUPLICATE, incumbent, []

        # More specific, not incompatible: the candidate says everything the incumbent
        # said and more ("works in Python" -> "works in Python, mainly async backend").
        incumbent_value = _normalize_value(incumbent.fact)
        if incumbent_value in candidate_value and len(candidate_value) > len(incumbent_value):
            return ConsolidationCase.REFINEMENT, incumbent, [incumbent]

        # The mirror image, and the one that was missing: the INCUMBENT already says
        # everything the candidate says. Without this branch a shorter restatement fell
        # through to CONTRADICTION below and retired the richer fact on recency alone --
        # measured in the 2026-08-17 console run, where
        #   "Bud sits in the Red chair in the Whangarei office"   (turn 11)
        # was superseded by
        #   "Bud sits in the red chair"                           (turn 16)
        # and the office left the live set. Freshness is the right tiebreak between two
        # claims that disagree; these two do not disagree, so it should never have been
        # consulted.
        #
        # COEXIST, not DUPLICATE, and that is measured rather than cautious. Containment
        # in this direction does NOT imply "adds nothing": the one containment pair in
        # 2310 facts of sh_32k is
        #   "flanker is associated with the sport of rugby union"  ->  "... of rugby"
        # which is a real value change, the same one RESULTS.md §6 records the embedding
        # threshold swallowing. Answering DUPLICATE there discards an update permanently
        # -- the unrecoverable direction `ConsolidationConfig` exists to warn about --
        # and no surface rule separates it from the Whangarei case, since both are strict
        # prefixes differing only in whether the trailing phrase opens with a preposition.
        # So neither fact is retired and neither is dropped: both stay live, the detail
        # survives, and the reader is handed both rather than a wrong one.
        #
        # Requires a real slot on BOTH sides, exactly as `_competing` treats `""` as
        # unspecified: with no slot there is no "same property" for a restatement to be a
        # restatement OF, and the bench (which supplies no attribute) stays inert.
        if (
            candidate_slot
            and incumbent.attribute
            and candidate_value in incumbent_value
            and len(candidate_value) < len(incumbent_value)
        ):
            return ConsolidationCase.NEW, None, []

        # Same subject, same slot, different value, neither a superset of the other --
        # but if the incumbent arrived in THIS batch there is no freshness relation to
        # decide it with. Both candidates came out of one extraction of one utterance;
        # their ordinals differ only by position in P1's output array, which carries no
        # information about which the user meant later. Ordering them anyway is what
        # marked a still-true fact SUPERSEDED in both columns of the 2026-08-17 run:
        #   "the user does not like milk in their tea"  (ord 11)
        #   "the user likes green tea"                  (ord 12) -> superseded ord 11
        # Both true, one utterance, no ordering. They coexist instead.
        #
        # Note where this guard sits: AFTER the two subsumption branches, deliberately.
        # REFINEMENT and the containment-DUPLICATE above are decided by what the strings
        # say, which needs no ordering and is just as valid between batch siblings.
        # Only CONTRADICTION rests on recency, so only CONTRADICTION is withheld.
        if incumbent.id in batch_ids:
            return ConsolidationCase.NEW, None, []

        # The money case: the higher freshness ordinal wins, and the candidate always has
        # it, because it is arriving now.
        return ConsolidationCase.CONTRADICTION, incumbent, _supersede_targets(incumbent, competing)

    async def consolidate(
        self, session_id: str, candidates: list[CandidateFact]
    ) -> list[ConsolidationOutcome]:
        outcomes: list[ConsolidationOutcome] = []
        if not candidates:
            return outcomes

        by_text = await self._vectors_for([c.fact for c in candidates])

        vocab = await self._vocab_for(session_id)

        # One batch = one extraction of one utterance. The id is what lets a later
        # contradiction tell "these two coexist because we could not order them" from
        # "this slot is corrupt", and the ids of the facts written here are what stop a
        # candidate superseding its own sibling. `source_episode_id` is the spec's field
        # for exactly this (`Episode`, writepath §3) and was being written empty.
        batch_id = uuid.uuid4().hex
        batch_ids: set[str] = set()

        for candidate in candidates:
            vector = by_text[candidate.fact]
            # Alias resolution runs BEFORE the lookup, and can only ever redirect it to a
            # key the session already holds -- it never invents one. If it declines, this
            # is exactly the exact-match behaviour it replaced.
            key = vocab.resolve(subject_key(candidate.subject_hint))
            live = await self.store.live_facts_for_subject(session_id, key)

            live_vecs = None
            if self.config.use_embedding_comparison and live:
                live_by_text = await self._vectors_for([f.fact for f in live])
                live_vecs = {f.id: live_by_text[f.fact] for f in live}

            case, incumbent, competing = self._classify(
                candidate, live, vector, live_vecs, frozenset(batch_ids)
            )
            now = datetime.now(UTC)

            if case is ConsolidationCase.DUPLICATE:
                # No second copy. This is what keeps the store from bloating
                # (writepath §2.2 case 2).
                outcomes.append(
                    ConsolidationOutcome(
                        candidate=candidate,
                        case=case,
                        ordinal=incumbent.ordinal if incumbent else 0,
                    )
                )
                continue

            superseded_id = None
            if case in (ConsolidationCase.CONTRADICTION, ConsolidationCase.REFINEMENT):
                # Supersede, do not delete: the prior fact stays and surfaces to recall
                # marked SUPERSEDED, so the model sees "was X, now Y" not a hole.
                assert incumbent is not None
                # `competing`, not `live` -- and that is the load-bearing word. Before §11
                # this read `live`, so a contradiction about the memory system's latency
                # also superseded live facts about its language and its birthplace. Once a
                # subject can legitimately hold several slots, `live` here would quietly
                # undo the entire fix.
                #
                # Which facts are in this list is now decided by `_supersede_targets`: a
                # slot may hold several live facts on purpose (same-batch siblings), so
                # "retire everything in the slot" is no longer the self-healing move it
                # was. Read that function before widening this back out.
                for target in competing:
                    await self.store.supersede(target.id, now)
                superseded_id = incumbent.id

            ordinal = await self._next_ordinal(session_id)
            stored = StoredFact(
                id=str(uuid.uuid4()),
                session_id=session_id,
                fact=candidate.fact,
                subject_key=key,
                subject_label=candidate.subject_hint.strip() or key,
                ordinal=ordinal,
                valid_at=candidate.valid_at or now,
                invalid_at=None,
                source_episode_id=batch_id,
                type=candidate.type,
                attribute=normalize_subject(candidate.attribute),
                # The raw phrasing, not the sorted key. This is what P1 gets shown back
                # to reuse; the key is what decides identity. RESULTS.md §14.
                attribute_label=candidate.attribute.strip(),
            )
            await self.store.add_fact(stored, vector)
            batch_ids.add(stored.id)
            vocab.add(key)
            outcomes.append(
                ConsolidationOutcome(
                    candidate=candidate,
                    case=case,
                    superseded_fact_id=superseded_id,
                    ordinal=ordinal,
                )
            )

        return outcomes
