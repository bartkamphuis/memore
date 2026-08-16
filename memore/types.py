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
    """recall-stage-spec.md §3.2. `score` is the fused hybrid score, normalized 0..1.

    `similarity` is the SAME hit's un-fused cosine, and the two differ for a reason that
    turns out to matter (RESULTS.md §12). Fusion is `cos·(1 + w·bm25)/(1 + w)`, whose range
    is `[cos/(1+w), cos]` — so BM25 can only ever deduct, and a fact sharing no term with
    the query is docked ~23% before any threshold sees it. That is right for RANKING (a
    lexical match is better evidence) and wrong for GATING (an absolute relevance
    question), which is why `RecallConfig.gate_on` chooses between them.

    A store that cannot supply the un-fused value leaves it 0.0. That is deliberately not
    made to fall back to `score`: with `gate_on="cosine"` such a store shuts the gate
    visibly, rather than silently reverting to the behaviour being measured against.
    """

    fact: str
    score: float
    valid_at: datetime | None
    invalid_at: datetime | None
    source_episode_id: str
    similarity: float = 0.0


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

    `attribute` narrows that collision to one SLOT within the subject, and exists because
    subject alone conflates two different questions. A subject is a topic, and a topic
    accumulates many attributes that are all true at once ("the memory system" is written
    in Python *and* takes 70-90ms *and* was built in Den Haag). Without a slot, every one
    of those supersedes the last. Empty means unspecified, which collides with everything
    on the subject -- the pre-attribute behaviour, kept so old stores and the bench (which
    supplies no attribute) are unaffected. RESULTS.md §11.
    """

    fact: str
    type: FactType
    confidence: float
    valid_at: datetime | None
    subject_hint: str
    attribute: str = ""


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
    # Normalized slot within the subject -- see `CandidateFact.attribute`. Only a fact
    # sharing this slot can supersede this one. `""` means unspecified and collides with
    # every slot, which is what facts written before this field existed carry.
    #
    # Normalized through the same `normalize_subject` as `subject_key`, so it reads as
    # sorted tokens: "lookup latency" is stored as "latency lookup".
    attribute: str = ""
    # The first natural phrasing seen for this slot, exactly as `subject_label` is for
    # `subject_key` -- and for exactly the same reason, which this file previously argued
    # did not apply here. That argument held for MATCHING (sorting makes reuse
    # order-insensitive, so a rephrasing still lands in the same slot) and was wrong for
    # PROMPTING: `subject_slots` feeds these strings back to P1 with an instruction to
    # reuse them verbatim, and what it was feeding back was "list todo", "issues medical",
    # "city code favourite python write". Nobody reuses that, so P1 coined a fresh slot
    # instead -- which is the slot-split failure of RESULTS.md §11 arriving by way of the
    # hint list rather than the model's judgement.
    attribute_label: str = ""


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
