"""Core types, taken verbatim from the production specs.

These shapes are the integration contract: `recall-poc-spec.md` §7 requires that the
gateway could import `recall()` and the consolidator unchanged, which only holds if the
types here match `recall-stage-spec.md` §3 and `recall-writepath-spec.md` §1.3 exactly.
Do not "improve" a field name without changing the spec first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class FactType(str, Enum):
    """recall-writepath-spec.md §1.3."""

    PREFERENCE = "PREFERENCE"
    IDENTITY = "IDENTITY"
    STATE = "STATE"
    EVENT = "EVENT"


class ConsolidationCase(str, Enum):
    """The four cases of recall-writepath-spec.md §2.2."""

    NEW = "NEW"
    DUPLICATE = "DUPLICATE"
    CONTRADICTION = "CONTRADICTION"
    REFINEMENT = "REFINEMENT"


@dataclass(frozen=True)
class Message:
    role: str
    content: str


@dataclass(frozen=True)
class TurnContext:
    """recall-stage-spec.md §3.1."""

    session_id: str
    user_message: str
    rolling_summary_vec: list[float] | None = None
    recent_messages: list[Message] = field(default_factory=list)


@dataclass(frozen=True)
class MemoryHit:
    """recall-stage-spec.md §3.2. `score` is the fused hybrid score, normalized 0..1."""

    fact: str
    score: float
    valid_at: datetime | None
    invalid_at: datetime | None
    source_episode_id: str


@dataclass(frozen=True)
class RecallResult:
    """recall-stage-spec.md §3.1."""

    injected_block: str | None
    memories_used: list[MemoryHit]
    latency_ms: float
    gate_open: bool


@dataclass(frozen=True)
class CandidateFact:
    """recall-writepath-spec.md §1.3.

    `subject_hint` is the retrieval key naming what the fact is about ("deploy target").
    It carries more weight here than in the production spec: the deterministic
    consolidation primitive (recall-poc-spec.md §4) keys its freshness ordinals by
    normalized subject, so subject_hint is what decides whether two facts collide.
    """

    fact: str
    type: FactType
    confidence: float
    valid_at: datetime | None
    subject_hint: str


@dataclass(frozen=True)
class StoredFact:
    """A fact as it lives in the store, with the bookkeeping the store owns.

    `ordinal` is the monotonic per-subject freshness ordinal of recall-poc-spec.md §4 --
    the primitive this project exists to test. Higher always wins a contradiction.
    """

    id: str
    session_id: str
    fact: str
    subject_key: str
    # The canonical `subject_key` is sorted content words and reads poorly. This keeps
    # the first natural phrasing seen for a subject, for traces and for the hint list
    # shown back to P1.
    subject_label: str
    ordinal: int
    valid_at: datetime | None
    invalid_at: datetime | None
    source_episode_id: str
    type: FactType = FactType.STATE


@dataclass(frozen=True)
class Episode:
    """One turn's worth of candidate facts, keyed by turn id for idempotent commit
    (recall-writepath-spec.md §3)."""

    episode_id: str
    session_id: str
    facts: list[CandidateFact]
    occurred_at: datetime


@dataclass(frozen=True)
class ConsolidationOutcome:
    """What P2 decided for one candidate, for the audit log (writepath §7)."""

    candidate: CandidateFact
    case: ConsolidationCase
    superseded_fact_id: str | None = None
    ordinal: int = 0
