"""Recall stage suites -- recall-stage-test-spec.md.

Covers Suite 1 (kill switch / no-op safety), Suite 2 (failure safety), Suite 3 (key
synthesis), Suites 4-6 (the gate), and Suite 7 (assembly). Suites 8-11 (latency, audit
log, write stage, real-store integration) are not built yet.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from memore.assemble import HEADER
from memore.config import RecallConfig
from memore.embed import StubEmbedder, blend, normalize
from memore.recall import WordTokenizer, apply_gate, recall
from memore.store.fake import FailingStore, FakeStore, SlowStore
from memore.types import MemoryHit, TurnContext

NOW = datetime(2026, 3, 14, tzinfo=UTC)


def make_hit(
    fact: str, score: float, valid_at=None, invalid_at=None, episode="ep1",
    similarity: float | None = None,
) -> MemoryHit:
    """`similarity` defaults to `score`, so a crafted hit gates identically under either
    `RecallConfig.gate_on`. Pass them apart only when testing that distinction."""
    return MemoryHit(
        fact=fact, score=score, similarity=score if similarity is None else similarity,
        valid_at=valid_at, invalid_at=invalid_at, source_episode_id=episode,
    )


def make_turn(msg: str = "what's my deploy setup?", summary_vec=None) -> TurnContext:
    return TurnContext(session_id="s1", user_message=msg, rolling_summary_vec=summary_vec)


# --- Suite 1: kill switch & no-op safety (§7, §11) ---------------------------


async def test_disabled_does_not_touch_the_store():
    store = FakeStore([make_hit("anything", 0.99)])
    result = await recall(make_turn(), RecallConfig(enabled=False), store, StubEmbedder())
    assert result.injected_block is None
    assert result.gate_open is False
    assert store.calls == []


async def test_gate_closed_injects_nothing():
    store = FakeStore([make_hit("weakly related", 0.10)])
    result = await recall(make_turn(), RecallConfig(), store, StubEmbedder())
    assert result.injected_block is None
    assert result.gate_open is False
    assert result.memories_used == []


# --- Suite 2: failure safety (§3.1, §5) --------------------------------------


async def test_store_failure_never_raises():
    result = await recall(make_turn(), RecallConfig(), FailingStore(), StubEmbedder())
    assert result.injected_block is None
    assert result.gate_open is False


async def test_lookup_timeout_returns_closed_without_waiting_it_out():
    store = SlowStore(delay_ms=300, hits=[make_hit("late", 0.99)])
    result = await recall(make_turn(), RecallConfig(lookup_timeout_ms=80), store, StubEmbedder())
    assert result.gate_open is False
    assert result.latency_ms < 250, "must abandon the lookup, not wait the full 300ms"


async def test_embedder_failure_degrades_to_no_inject():
    class BrokenEmbedder:
        async def embed(self, texts):
            raise RuntimeError("model unavailable")

        async def embed_one(self, text):
            raise RuntimeError("model unavailable")

    result = await recall(make_turn(), RecallConfig(), FakeStore(), BrokenEmbedder())
    assert result.gate_open is False


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.5, 1.5])
async def test_malformed_scores_are_dropped_not_injected(bad):
    store = FakeStore([make_hit("malformed", bad), make_hit("good", 0.9)])
    result = await recall(make_turn(), RecallConfig(), store, StubEmbedder())
    assert [h.fact for h in result.memories_used] == ["good"]


# --- Suite 3: key synthesis (§4) ---------------------------------------------


async def test_first_turn_uses_normalized_message_vector():
    embedder = StubEmbedder()
    store = FakeStore([make_hit("f", 0.9)])
    await recall(make_turn(summary_vec=None), RecallConfig(), store, embedder)
    expected = normalize(await embedder.embed_one("what's my deploy setup?"))
    assert store.calls[0]["query_vec"] == pytest.approx(expected)


async def test_summary_vector_is_blended_with_alpha():
    embedder = StubEmbedder()
    summary = normalize([1.0] * 8)
    store = FakeStore([make_hit("f", 0.9)])
    await recall(make_turn(summary_vec=summary), RecallConfig(alpha=0.7), store, embedder)
    msg_vec = await embedder.embed_one("what's my deploy setup?")
    assert store.calls[0]["query_vec"] == pytest.approx(blend(msg_vec, summary, 0.7))


async def test_alpha_one_ignores_the_summary_vector():
    embedder = StubEmbedder()
    store = FakeStore([make_hit("f", 0.9)])
    await recall(make_turn(summary_vec=normalize([1.0] * 8)), RecallConfig(alpha=1.0), store, embedder)
    expected = normalize(await embedder.embed_one("what's my deploy setup?"))
    assert store.calls[0]["query_vec"] == pytest.approx(expected)


async def test_bm25_arm_receives_the_raw_message():
    store = FakeStore([make_hit("f", 0.9)])
    await recall(make_turn("what about THAT one?"), RecallConfig(), store, StubEmbedder())
    assert store.calls[0]["query_text"] == "what about THAT one?"


# --- Suites 4-5: the gate, floor and budget (§6.1, §6.2) ---------------------


def gate(hits, **overrides):
    return apply_gate(hits, RecallConfig(**overrides), WordTokenizer(), NOW)


def test_floor_is_inclusive():
    decision = gate([make_hit("exactly at floor", 0.35)], score_floor=0.35)
    assert len(decision.hits) == 1


def test_sub_floor_hits_never_appear():
    decision = gate([make_hit("keep", 0.6), make_hit("drop", 0.34)], score_floor=0.35)
    assert [h.fact for h in decision.hits] == ["keep"]


def test_budget_is_a_cap_not_a_target():
    """The failure mode this pins: an implementer padding output to fill the budget."""
    decision = gate([make_hit("four short words here", 0.9)], inject_token_budget=512)
    assert decision.tokens == 4
    assert len(decision.hits) == 1


def test_budget_stops_before_exceeding():
    hits = [make_hit(f"fact number {i} padding words", 0.9 - i / 100) for i in range(10)]
    decision = gate(hits, inject_token_budget=10)
    assert decision.tokens <= 10
    assert len(decision.hits) == 2


def test_single_oversized_fact_is_included_and_flagged():
    decision = gate([make_hit(" ".join(["word"] * 50), 0.9)], inject_token_budget=10)
    assert len(decision.hits) == 1
    assert decision.overflow is True


def test_hits_are_taken_in_score_order():
    hits = [make_hit("low", 0.4), make_hit("high", 0.95), make_hit("mid", 0.7)]
    decision = gate(hits, inject_token_budget=2)
    assert [h.fact for h in decision.hits] == ["high", "mid"]


# --- Suite 6: staleness / bitemporal (§6.3) ----------------------------------


def test_superseded_hit_is_marked_not_dropped():
    hit = make_hit("was staging", 0.9, valid_at=NOW - timedelta(days=30), invalid_at=NOW - timedelta(days=1))
    decision = gate([hit])
    assert len(decision.hits) == 1


def test_future_valid_at_is_dropped():
    hit = make_hit("from the future", 0.9, valid_at=NOW + timedelta(days=1))
    assert gate([hit]).hits == []


# --- Suite 7: assembly & placement (§7) --------------------------------------


async def test_block_is_labelled_and_instruction_guarded():
    store = FakeStore([make_hit("prefers dark roast", 0.9, valid_at=NOW)])
    result = await recall(make_turn(), RecallConfig(), store, StubEmbedder())
    assert result.injected_block is not None
    assert result.injected_block.startswith("<recalled_context>")
    assert "not as instructions" in result.injected_block
    assert HEADER in result.injected_block


async def test_superseded_annotation_renders():
    store = FakeStore(
        [
            make_hit("deploys to prod", 0.95, valid_at=NOW),
            make_hit("deploys to staging", 0.9, valid_at=NOW - timedelta(days=30), invalid_at=NOW),
        ]
    )
    result = await recall(make_turn(), RecallConfig(), store, StubEmbedder())
    block = result.injected_block
    assert "[valid as of 2026-03-14] deploys to prod" in block
    assert "[SUPERSEDED - was valid 2026-02-12 to 2026-03-14] deploys to staging" in block


async def test_non_temporal_store_emits_bare_facts():
    store = FakeStore([make_hit("no timestamps here", 0.9)])
    result = await recall(make_turn(), RecallConfig(), store, StubEmbedder())
    assert "- no timestamps here" in result.injected_block
    assert "valid as of" not in result.injected_block
