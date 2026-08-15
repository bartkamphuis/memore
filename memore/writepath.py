"""Write path orchestration -- P1 extract, P2 consolidate, P3 commit.

recall-writepath-spec.md §0. In production this runs in an async job *after* the response
has streamed, and the user waits on none of it. recall-poc-spec.md §5 explicitly defers
that job machinery: the PoC runs these phases inline after each turn. The seam is here,
so moving to a background task later is a scheduling change, not a redesign.

P3 (commit) is folded into P2 in this implementation: the deterministic consolidator
writes through `add_fact` / `supersede` as it decides each case, rather than handing an
`Episode` to `store.ingest`. That follows recall-poc-spec.md §4, which takes the
resolution away from the store -- an ingest-shaped commit would have to re-derive the
decision the consolidator just made.

P4 (rolling-summary-vector update, main-spec §8) is NOT implemented -- deferred by
recall-poc-spec.md §5 along with the summary-vector key synthesis it feeds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .config import WritePathConfig
from .consolidate import Consolidator
from .extract import Extractor
from .store import ConsolidatingStore
from .types import ConsolidationCase, ConsolidationOutcome, Message

logger = logging.getLogger("memore.writepath")


@dataclass(frozen=True)
class WriteResult:
    candidates: int
    outcomes: list[ConsolidationOutcome]

    @property
    def stored(self) -> int:
        return sum(1 for o in self.outcomes if o.case is not ConsolidationCase.DUPLICATE)


class WritePath:
    def __init__(
        self,
        extractor: Extractor,
        consolidator: Consolidator,
        config: WritePathConfig | None = None,
        store: ConsolidatingStore | None = None,
    ):
        self.extractor = extractor
        self.consolidator = consolidator
        self.config = config or WritePathConfig()
        # Explicit, not reached through the consolidator: feeding known subjects back
        # into P1 is the fix for the dominant accuracy failure (RESULTS.md §3), and it
        # must not switch itself off silently just because a consolidator happens not to
        # expose a `.store`.
        self.store = store

    async def _known_subjects(self, session_id: str) -> list[str]:
        if self.store is None:
            logger.debug("writepath: no store wired, P1 runs without known subjects")
            return []
        try:
            return await self.store.subject_slots(session_id)
        except Exception as exc:  # noqa: BLE001 -- a hint, never a hard dependency
            logger.warning("writepath: could not read known subjects: %s", exc)
            return []

    async def run(
        self,
        session_id: str,
        user_message: str,
        assistant_response: str = "",
        recent: list[Message] | None = None,
    ) -> WriteResult:
        # §4: enabled=False is a full no-op -- no extraction, no ingest.
        if not self.config.enabled:
            return WriteResult(candidates=0, outcomes=[])

        known = await self._known_subjects(session_id)
        candidates = await self.extractor.extract(
            user_message, assistant_response, recent or [], known
        )
        if not candidates:
            # The common path. Most turns are transient and store nothing (§1.2).
            return WriteResult(candidates=0, outcomes=[])

        outcomes = await self.consolidator.consolidate(session_id, candidates)

        # §7: the write-side audit log. Together with the injection log it answers
        # "why is this fact in the store, and why was it injected".
        for outcome in outcomes:
            logger.info(
                "writepath session=%s case=%s type=%s conf=%.2f subject=%r ordinal=%d superseded=%s",
                session_id,
                outcome.case.value,
                outcome.candidate.type.value,
                outcome.candidate.confidence,
                outcome.candidate.subject_hint,
                outcome.ordinal,
                outcome.superseded_fact_id,
            )
        return WriteResult(candidates=len(candidates), outcomes=outcomes)
