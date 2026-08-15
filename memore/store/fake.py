"""In-memory stores for tests (recall-stage-test-spec.md, Fixtures).

`FakeStore` is the workhorse of the gate/assembly suites: `hybrid_search` returns a
fixed, pre-scored list so the tests drive entirely off crafted hits. `InMemoryStore` is
a real (if naive) `ConsolidatingStore`, so the consolidation suite can assert on
supersession without FalkorDB running.
"""

from __future__ import annotations

import asyncio
from datetime import datetime

from ..types import Episode, MemoryHit, StoredFact


class FakeStore:
    """`hybrid_search` returns a fixed list. Records calls for assertions."""

    def __init__(self, hits: list[MemoryHit] | None = None):
        self.hits = hits or []
        self.calls: list[dict] = []
        self.ingested: list[Episode] = []

    async def hybrid_search(
        self, query_text: str, query_vec: list[float], session_id: str, k: int
    ) -> list[MemoryHit]:
        self.calls.append(
            {"query_text": query_text, "query_vec": query_vec, "session_id": session_id, "k": k}
        )
        return list(self.hits[:k])

    async def ingest(self, episode: Episode) -> None:
        self.ingested.append(episode)


class SlowStore(FakeStore):
    """Drives the timeout tests (Suite 2)."""

    def __init__(self, delay_ms: float, hits: list[MemoryHit] | None = None):
        super().__init__(hits)
        self.delay_ms = delay_ms

    async def hybrid_search(
        self, query_text: str, query_vec: list[float], session_id: str, k: int
    ) -> list[MemoryHit]:
        await asyncio.sleep(self.delay_ms / 1000.0)
        return await super().hybrid_search(query_text, query_vec, session_id, k)


class FailingStore(FakeStore):
    """Drives the failure-safety tests (Suite 2)."""

    def __init__(self, exc: Exception | None = None):
        super().__init__([])
        self.exc = exc or RuntimeError("store is down")

    async def hybrid_search(
        self, query_text: str, query_vec: list[float], session_id: str, k: int
    ) -> list[MemoryHit]:
        raise self.exc


class InMemoryStore:
    """A real `ConsolidatingStore` with dict-backed storage and cosine-only retrieval."""

    def __init__(self):
        self.facts: dict[str, StoredFact] = {}
        self.vectors: dict[str, list[float]] = {}

    async def hybrid_search(
        self, query_text: str, query_vec: list[float], session_id: str, k: int
    ) -> list[MemoryHit]:
        scored = []
        for fact in self.facts.values():
            if fact.session_id != session_id:
                continue
            vector = self.vectors.get(fact.id, [])
            similarity = (
                sum(a * b for a, b in zip(query_vec, vector, strict=False)) if vector else 0.0
            )
            scored.append(
                MemoryHit(
                    fact=fact.fact,
                    score=max(0.0, min(1.0, similarity)),
                    # Cosine-only retrieval here, so the two coincide.
                    similarity=max(0.0, min(1.0, similarity)),
                    valid_at=fact.valid_at,
                    invalid_at=fact.invalid_at,
                    source_episode_id=fact.source_episode_id,
                )
            )
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:k]

    async def ingest(self, episode: Episode) -> None:
        raise NotImplementedError("drive writes through the consolidator")

    async def live_facts_for_subject(self, session_id: str, subject_key: str) -> list[StoredFact]:
        matches = [
            f
            for f in self.facts.values()
            if f.session_id == session_id and f.subject_key == subject_key and f.invalid_at is None
        ]
        return sorted(matches, key=lambda f: f.ordinal, reverse=True)

    async def max_ordinal(self, session_id: str, subject_key: str | None = None) -> int:
        matches = [
            f.ordinal
            for f in self.facts.values()
            if f.session_id == session_id and (subject_key is None or f.subject_key == subject_key)
        ]
        return max(matches, default=0)

    async def live_subject_keys(self, session_id: str) -> list[str]:
        return sorted(
            {
                f.subject_key
                for f in self.facts.values()
                if f.session_id == session_id and f.invalid_at is None and f.subject_key
            }
        )

    async def subject_slots(self, session_id: str) -> list[str]:
        labels: dict[str, str] = {}
        slots: dict[str, set[str]] = {}
        for f in self.facts.values():
            if f.session_id != session_id:
                continue
            labels.setdefault(f.subject_key, f.subject_label or f.subject_key)
            if f.attribute:
                slots.setdefault(f.subject_key, set()).add(f.attribute)
        return sorted(
            f"{label} -> {', '.join(sorted(slots[key]))}" if slots.get(key) else label
            for key, label in labels.items()
        )

    async def add_fact(self, fact: StoredFact, embedding: list[float]) -> None:
        self.facts[fact.id] = fact
        self.vectors[fact.id] = list(embedding)

    async def supersede(self, fact_id: str, invalid_at: datetime) -> None:
        existing = self.facts[fact_id]
        self.facts[fact_id] = StoredFact(
            id=existing.id,
            session_id=existing.session_id,
            fact=existing.fact,
            subject_key=existing.subject_key,
            subject_label=existing.subject_label,
            ordinal=existing.ordinal,
            valid_at=existing.valid_at,
            invalid_at=invalid_at,
            source_episode_id=existing.source_episode_id,
            type=existing.type,
            attribute=existing.attribute,
        )

    async def count(self, session_id: str) -> int:
        return sum(1 for f in self.facts.values() if f.session_id == session_id)
