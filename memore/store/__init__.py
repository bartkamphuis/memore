"""Store boundary. recall-stage-spec.md §13: no store-specific types leak past here."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from ..chain import ChainNode
from ..types import Episode, MemoryHit, StoredFact


@runtime_checkable
class MemoryStore(Protocol):
    """recall-stage-spec.md §3.2 -- the surface the gateway sees."""

    async def hybrid_search(
        self,
        query_text: str,
        query_vec: list[float],
        session_id: str,
        k: int,
    ) -> list[MemoryHit]: ...

    async def ingest(self, episode: Episode) -> None: ...


@runtime_checkable
class ChainWalkStore(MemoryStore, Protocol):
    """Optional capability: hand back the session's live facts for a multi-hop walk.

    Separate from `MemoryStore` rather than added to it, because `MemoryStore` is the
    verbatim surface `recall-stage-spec.md` §3.2 defines and multi-hop chaining is an
    extension the spec does not describe. A store that does not implement this still
    backs `recall()` -- expansion just stays off, checked at runtime and logged, never
    raised (§3.1).
    """

    async def live_chain_view(self, session_id: str) -> list[ChainNode]: ...


@runtime_checkable
class ConsolidatingStore(MemoryStore, Protocol):
    """The extra primitives the deterministic consolidator needs (recall-poc-spec.md §4).

    Kept separate from `MemoryStore` on purpose: the gateway only ever needs the two
    methods above, while consolidation needs to own freshness ordinals and supersession.
    A store that cannot support these can still back `recall()`.
    """

    async def live_facts_for_subject(self, session_id: str, subject_key: str) -> list[StoredFact]:
        """Facts on `subject_key` with no `invalid_at`, ordinal-descending.

        `subject_key` must already be normalized -- pass
        `memore.consolidate.subject_key(hint)`, never a raw `subject_hint`. The store
        matches it exactly; normalization policy belongs to the consolidator, because
        that policy is what defines subject identity.
        """
        ...

    async def max_ordinal(self, session_id: str, subject_key: str | None = None) -> int: ...

    async def live_subject_keys(self, session_id: str) -> list[str]:
        """Distinct normalized keys of the subjects currently live in this session.

        Seeds `memore.aliases.SubjectVocabulary`, which needs document frequency over
        subjects to tell a relation word from an entity. Keys, not labels, and distinct:
        the statistic counts SUBJECTS, so returning one row per fact would let a subject
        with many facts masquerade as a common token.

        Added here rather than as a separate capability protocol (the way `ChainWalkStore`
        was) on purpose: subject identity is the consolidator's own decision, so a store
        that cannot answer this cannot consolidate at all. `ConsolidatingStore` is already
        this project's extension rather than a verbatim spec type, so widening it is not
        the interface drift the discipline section warns about.
        """
        ...

    async def subject_labels(self, session_id: str) -> list[str]:
        """Readable names of the subjects held for this session, one per subject.

        Fed back into P1 so the extractor can reuse an existing subject instead of
        coining a synonym -- the measured cause of missed contradictions (RESULTS.md §3).
        Labels, not keys: the canonical key is sorted tokens and would only confuse the
        model being asked to reuse it.
        """
        ...

    async def add_fact(self, fact: StoredFact, embedding: list[float]) -> None: ...

    async def supersede(self, fact_id: str, invalid_at: datetime) -> None: ...

    async def count(self, session_id: str) -> int: ...


__all__ = ["ChainWalkStore", "ConsolidatingStore", "MemoryStore"]
