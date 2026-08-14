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
        live_vecs: dict[str, list[float]] | None,
    ) -> tuple[ConsolidationCase, StoredFact | None]:
        """Step 2 of §4: decide the case deterministically."""
        if not live:
            return ConsolidationCase.NEW, None

        candidate_value = _normalize_value(candidate.fact)
        # `live` arrives ordinal-descending; the incumbent is the freshest live fact.
        incumbent = live[0]
        incumbent_value = _normalize_value(incumbent.fact)

        if candidate_value == incumbent_value:
            return ConsolidationCase.DUPLICATE, incumbent

        if self.config.use_embedding_comparison and candidate_vec is not None and live_vecs:
            incumbent_vec = live_vecs.get(incumbent.id)
            if incumbent_vec is not None:
                similarity = cosine(candidate_vec, incumbent_vec)
                if similarity >= self.config.duplicate_similarity:
                    return ConsolidationCase.DUPLICATE, incumbent

        # More specific, not incompatible: the candidate says everything the incumbent
        # said and more ("works in Python" -> "works in Python, mainly async backend").
        if incumbent_value in candidate_value and len(candidate_value) > len(incumbent_value):
            return ConsolidationCase.REFINEMENT, incumbent

        # Same subject, different value, neither a superset of the other. The higher
        # freshness ordinal wins -- and the candidate always has it, because it is
        # arriving now. This is the money case.
        return ConsolidationCase.CONTRADICTION, incumbent

    async def consolidate(
        self, session_id: str, candidates: list[CandidateFact]
    ) -> list[ConsolidationOutcome]:
        outcomes: list[ConsolidationOutcome] = []
        if not candidates:
            return outcomes

        by_text = await self._vectors_for([c.fact for c in candidates])

        vocab = await self._vocab_for(session_id)

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

            case, incumbent = self._classify(candidate, live, vector, live_vecs)
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
                # A subject should hold exactly one live fact, but if the invariant was
                # ever broken (concurrent writes, a store restored from elsewhere), the
                # arriving fact outranks *every* live fact on the subject, not just the
                # freshest. Superseding all of them makes the invariant self-healing
                # instead of leaving a stale fact live forever.
                targets = live if case is ConsolidationCase.CONTRADICTION else [incumbent]
                for target in targets:
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
                source_episode_id="",
                type=candidate.type,
            )
            await self.store.add_fact(stored, vector)
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
