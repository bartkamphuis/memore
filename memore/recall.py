"""The recall stage -- components A-D (recall-stage-spec.md §1, §3.1).

Hard budget: 200ms P95 for A-D combined, of which the store lookup (B) dominates. A, C
and D must each stay in low single-digit milliseconds. There is no LLM call anywhere in
this file, and adding one violates the core design (§13).

`recall()` never raises. Any store error or timeout returns a closed result and logs --
memory is best-effort, and a recall failure must never fail the turn (§3.1).
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from .assemble import build_block
from .chain import walk
from .config import RecallConfig
from .embed import Embedder, blend
from .store import ChainWalkStore, MemoryStore
from .subjects import SubjectView, build_subject_view
from .types import MemoryHit, RecallResult, TurnContext

logger = logging.getLogger("memore.recall")


class Tokenizer(Protocol):
    def count(self, text: str) -> int: ...


class WordTokenizer:
    """Placeholder for the gateway's real tokenizer.

    recall-stage-test-spec.md Suite 5 requires the gate to count tokens with the
    gateway's existing tokenizer rather than a character heuristic. The PoC is not wired
    to the gateway, so this stands in behind the `Tokenizer` protocol -- swap the real
    one in at integration time; nothing else has to change.
    """

    def count(self, text: str) -> int:
        return len(text.split())


@dataclass(frozen=True)
class GateDecision:
    hits: list[MemoryHit]
    above_floor: int
    tokens: int
    overflow: bool


def _usable(hit: MemoryHit, now: datetime) -> bool:
    """Drop malformed and not-yet-valid hits (§6.3, Suite 2)."""
    if hit.score is None or math.isnan(hit.score) or math.isinf(hit.score):
        return False
    if not (0.0 <= hit.score <= 1.0):
        return False
    if hit.valid_at is not None and hit.valid_at > now:
        # Should not happen; guard against bad data rather than injecting the future.
        return False
    return True


def apply_gate(
    hits: list[MemoryHit],
    config: RecallConfig,
    tokenizer: Tokenizer,
    now: datetime,
    subjects: SubjectView | None = None,
    query: str = "",
) -> GateDecision:
    """Component C -- the relevance gate (§6). Deterministic, sub-millisecond, no LLM."""
    # 1. Floor test. The floor is inclusive.
    survivors = [h for h in hits if _usable(h, now) and h.score >= config.score_floor]
    # 1b. Subject admission (`memore.subjects`), when configured. Applied AFTER the floor
    # and as a veto only: it can never promote a hit the score rejected, so `score_floor`
    # keeps exactly the meaning it was calibrated with (RESULTS.md §5).
    if subjects is not None:
        kept = [
            h
            for h in survivors
            if subjects.admits(
                query, h.fact, config.subject_min_competitors, config.subject_df_max
            )
        ]
        if len(kept) != len(survivors):
            logger.info(
                "subject check refused %d of %d hits above floor", len(survivors) - len(kept), len(survivors)
            )
        survivors = kept
    if not survivors:
        return GateDecision(hits=[], above_floor=0, tokens=0, overflow=False)

    # 2. Budget fill. A hard cap, not a target: a turn that clears the floor with one
    # 20-token fact injects 20 tokens.
    survivors.sort(key=lambda h: h.score, reverse=True)
    chosen: list[MemoryHit] = []
    used = 0
    overflow = False
    for hit in survivors:
        cost = tokenizer.count(hit.fact)
        if not chosen and cost > config.inject_token_budget:
            # Pinned decision (test-spec Suite 5): a single relevant fact larger than the
            # whole budget is included rather than silently dropped, and warns.
            logger.warning(
                "recall budget overflow: single fact costs %d tokens, budget %d",
                cost,
                config.inject_token_budget,
            )
            chosen.append(hit)
            used = cost
            overflow = True
            break
        if used + cost > config.inject_token_budget:
            break
        chosen.append(hit)
        used += cost

    return GateDecision(hits=chosen, above_floor=len(survivors), tokens=used, overflow=overflow)


async def recall(
    turn: TurnContext,
    config: RecallConfig,
    store: MemoryStore,
    embedder: Embedder | None = None,
    tokenizer: Tokenizer | None = None,
) -> RecallResult:
    """The single function the gateway calls per turn (§3.1)."""
    started = time.perf_counter()

    def closed(reason: str | None = None) -> RecallResult:
        if reason:
            logger.info("recall gate closed: %s (session=%s)", reason, turn.session_id)
        return RecallResult(
            injected_block=None,
            memories_used=[],
            latency_ms=(time.perf_counter() - started) * 1000.0,
            gate_open=False,
        )

    # §11: the master kill switch is a full no-op. The store is not even consulted.
    if not config.enabled:
        return closed()

    tokenizer = tokenizer or WordTokenizer()

    try:
        # Component A -- key synthesis (§4). The only cost here is the local embedding.
        if embedder is None:
            return closed("no embedder configured")
        msg_vec = await embedder.embed_one(turn.user_message, query=True)
        query_vec = blend(msg_vec, turn.rolling_summary_vec, config.alpha)
    except Exception as exc:  # noqa: BLE001 -- embedder failure degrades to no-inject
        logger.warning("recall key synthesis failed: %s", exc)
        return closed("key synthesis failed")

    try:
        # Component B -- hybrid lookup (§5). `query_text` is the RAW user message: the
        # BM25 arm wants the literal terms, not the blended vector's notion of them.
        hits = await asyncio.wait_for(
            store.hybrid_search(
                query_text=turn.user_message,
                query_vec=query_vec,
                session_id=turn.session_id,
                k=config.k,
            ),
            timeout=config.lookup_timeout_ms / 1000.0,
        )
    except TimeoutError:
        logger.warning("recall lookup timed out after %dms", config.lookup_timeout_ms)
        return closed("lookup timeout")
    except Exception as exc:  # noqa: BLE001 -- a store failure must never fail the turn
        logger.warning("recall lookup failed: %s", exc)
        return closed("lookup failed")

    # Subject vocabulary for the admission rule, when configured. Failure to load it
    # degrades to no check rather than to a closed gate: this is a precision refinement,
    # and losing it must not cost recall (§3.1).
    subjects = None
    if config.subject_check:
        try:
            subjects = build_subject_view(await store.subject_view(turn.session_id))
        except Exception as exc:  # noqa: BLE001
            logger.warning("subject view unavailable, admission check skipped: %s", exc)

    # Component C -- the gate (§6).
    decision = apply_gate(
        hits, config, tokenizer, datetime.now(UTC), subjects=subjects, query=turn.user_message
    )
    latency_ms = (time.perf_counter() - started) * 1000.0

    # §6: this log is the eval substrate and the audit trail that replaces the
    # tool-call trace we gave up (§10). Do not drop it.
    logger.info(
        "recall session=%s gate=%s hits=%d above_floor=%d injected=%d tokens=%d latency_ms=%.1f",
        turn.session_id,
        "open" if decision.hits else "closed",
        len(hits),
        decision.above_floor,
        len(decision.hits),
        decision.tokens,
        latency_ms,
    )

    if not decision.hits:
        return closed()

    # Multi-hop chain expansion -- an extension to §5, off unless configured.
    chosen = decision.hits
    if config.expansion_hops > 0:
        try:
            chosen = await _expand(turn, config, store, decision, tokenizer)
        except Exception as exc:  # noqa: BLE001 -- same contract as the lookup: never fail a turn
            logger.warning("recall chain expansion failed: %s", exc)

    latency_ms = (time.perf_counter() - started) * 1000.0

    # Component D -- assembly (§7).
    return RecallResult(
        injected_block=build_block(chosen),
        memories_used=chosen,
        latency_ms=latency_ms,
        gate_open=True,
    )


async def _expand(
    turn: TurnContext,
    config: RecallConfig,
    store: MemoryStore,
    decision: GateDecision,
    tokenizer: Tokenizer,
) -> list[MemoryHit]:
    """Follow value->subject edges out of the gated hits (`memore.chain`).

    Deliberately placed after the gate. A multi-hop question's answer fact shares no
    entity with the question -- that is what makes it multi-hop -- so it cannot clear a
    similarity floor and must not be asked to. Judging relevance on the seeds and letting
    the chain ride along keeps `score_floor` exactly as calibrated (RESULTS.md §5) while
    still delivering the hop-2 fact the turn actually needs.

    Chain facts are appended, never reordered ahead of the gated hits, and stop at the
    same `inject_token_budget` -- a hard cap, not a target (§6.2).
    """
    if not isinstance(store, ChainWalkStore):
        logger.info("expansion_hops set but store has no live_chain_view; skipping")
        return decision.hits

    nodes = await store.live_chain_view(turn.session_id)
    if not nodes:
        return decision.hits

    by_fact = {node.fact: index for index, node in enumerate(nodes)}
    seeds = [by_fact[hit.fact] for hit in decision.hits if hit.fact in by_fact]
    reached = walk(nodes, seeds, config.expansion_hops, config.expansion_fanout)
    if not reached:
        return decision.hits

    out = list(decision.hits)
    used = decision.tokens
    already = {hit.fact for hit in decision.hits}
    for index in reached:
        node = nodes[index]
        if node.fact in already:
            continue
        cost = tokenizer.count(node.fact)
        if used + cost > config.inject_token_budget:
            break
        # Score 0.0 is honest: a chain fact was never ranked against the turn. It is
        # here because something relevant pointed at it, and the block records that by
        # position rather than by inventing a similarity it does not have.
        out.append(
            MemoryHit(
                fact=node.fact,
                score=0.0,
                valid_at=node.valid_at,
                invalid_at=node.invalid_at,
                source_episode_id="",
            )
        )
        used += cost
        already.add(node.fact)

    logger.info(
        "recall chain expansion session=%s seeds=%d reached=%d injected=%d tokens=%d",
        turn.session_id,
        len(seeds),
        len(reached),
        len(out) - len(decision.hits),
        used,
    )
    return out
