"""The frecency lane in `memore/demo/linger.py`, and the bar it has to clear.

Pre-registered, in the spirit of §10 and §15: this layer's failure mode is that it scores
perfectly by injecting everything forever, which destroys the property the gate exists for.
So the pass condition is TWO-SIDED and both sides are tested here --

  carry   the follow-up ("when?") still has the detail the previous turn surfaced, on a
          turn where the gate is shut and recall returned nothing;
  expire  a fact nobody refers to again falls out of the block on its own, and an empty
          store or a cleared session takes its carried copy with it.

The third condition is the one that would be a real defect rather than a weak demo: the
cache holds a fact's TEXT and its weight, never its rendered line, and the carried set is
re-read from the store every turn. A fact superseded on turn N+1 must come back labelled
SUPERSEDED, not replayed as though it were still true -- a harness resurrecting what the
store retired would invert the claim the store pane exists to make.

No FalkorDB and no Ollama: the runtime is faked, as in `test_demo_app.py`.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

pytest.importorskip("fastapi", reason="the demo lives behind the `demo` extra")

from fastapi.testclient import TestClient  # noqa: E402

from memore.demo import app as demo  # noqa: E402
from memore.demo.linger import LingerCache, LingerConfig  # noqa: E402
from memore.types import FactType, MemoryHit, RecallResult, StoredFact  # noqa: E402

TRIP = "the user is travelling to Lisbon on 2026-08-26"
CAT = "the user's cat is called Miso"


# --- the cache on its own -----------------------------------------------------------

def test_a_fact_decays_by_half_every_half_life_and_then_stops_existing():
    cache = LingerCache(LingerConfig(half_life_turns=2.0, floor=0.3))
    cache.begin_turn()
    cache.observe(TRIP, 0.8)
    assert cache.carried() == []                       # surfaced this turn, not carried
    cache.begin_turn()
    assert cache.carried()[0].weight == pytest.approx(0.8 * 0.5 ** 0.5, rel=1e-6)
    cache.begin_turn()
    assert cache.carried()[0].weight == pytest.approx(0.4)
    cache.begin_turn()
    cache.begin_turn()
    # 0.8 * 0.5**2 = 0.2, under the floor: gone, and gone from the cache rather than
    # merely filtered out of the answer.
    assert cache.carried() == []
    assert cache.carried() == []


def test_being_surfaced_again_lifts_the_strength_and_restarts_the_clock():
    cache = LingerCache(LingerConfig(half_life_turns=2.0, floor=0.3, repeat_bonus=0.15))
    cache.begin_turn()
    cache.observe(TRIP, 0.6)
    cache.begin_turn()
    cache.observe(TRIP, 0.6)
    cache.begin_turn()
    entry = cache.carried()[0]
    assert entry.seen == 2
    assert entry.age_turns == 1
    assert entry.strength == pytest.approx(0.69)       # 0.6 * 1.15
    assert entry.weight == pytest.approx(0.69 * 0.5 ** 0.5, rel=1e-6)


def test_repeat_bonus_can_never_push_a_carried_fact_past_a_fresh_one():
    """Strength is capped at 1.0. A fact mentioned five times must not outrank what the
    gate just scored on this turn's question -- the cache reorders nothing."""
    cache = LingerCache(LingerConfig(half_life_turns=99.0, floor=0.0))
    for _ in range(8):
        cache.begin_turn()
        cache.observe(TRIP, 0.9)
    cache.begin_turn()
    assert cache.carried()[0].strength <= 1.0


def test_the_carried_set_is_capped_and_heaviest_first():
    cache = LingerCache(LingerConfig(half_life_turns=99.0, floor=0.0, max_facts=2))
    cache.begin_turn()
    for i, score in enumerate([0.4, 0.9, 0.7, 0.5]):
        cache.observe(f"fact {i}", score)
    cache.begin_turn()
    carried = cache.carried()
    assert [c.fact for c in carried] == ["fact 1", "fact 2"]


def test_clear_drops_everything_including_the_turn_clock():
    cache = LingerCache(LingerConfig())
    cache.begin_turn()
    cache.observe(TRIP, 0.9)
    cache.clear()
    cache.begin_turn()
    assert cache.carried() == []


# --- the lane in the app ------------------------------------------------------------

def _fact(text: str, ordinal: int, *, superseded: bool = False,
          occurs_at: datetime | None = None) -> StoredFact:
    return StoredFact(
        id=f"f{ordinal}",
        session_id=demo.DEFAULT_SESSION,
        fact=text,
        subject_key="the user",
        subject_label="the user",
        ordinal=ordinal,
        valid_at=datetime(2026, 8, 21, tzinfo=UTC),
        invalid_at=datetime(2026, 8, 22, tzinfo=UTC) if superseded else None,
        source_episode_id="ep",
        type=FactType.STATE,
        attribute="travel",
        attribute_label="travel",
        occurs_at=occurs_at,
    )


class _Store:
    def __init__(self, facts):
        from memore.config import StoreConfig

        self.config = StoreConfig(graph_name="test_graph")
        self.facts = list(facts)

    async def connect(self): ...
    async def count(self, session_id): return len(self.facts)
    async def facts_in_session(self, session_id): return list(self.facts)
    async def sessions(self): return [("demo-web", len(self.facts), len(self.facts))]

    async def clear_session(self, session_id):
        self.facts = []


class _Chat:
    async def chat(self, messages, schema=None, max_tokens=None): return "ok"

    async def chat_stream(self, messages, max_tokens=None):
        # The prompt the model is actually handed. The point of the whole layer is what is
        # in here on a turn where the gate shut, so it is captured rather than discarded.
        SENT.append(messages)
        yield "ok"


class _WritePath:
    async def run(self, session_id, message, reply, history):
        from memore.writepath import WriteResult

        return WriteResult(candidates=0, outcomes=[])


SENT: list[list[dict]] = []


def _hit(text: str, similarity: float) -> MemoryHit:
    return MemoryHit(
        fact=text, score=similarity, similarity=similarity,
        valid_at=datetime(2026, 8, 21, tzinfo=UTC), invalid_at=None, source_episode_id="ep",
    )


@pytest.fixture
def demo_client(monkeypatch):
    """`answers` is a list of per-turn recall results, consumed in order."""

    def _build(facts, answers, config: LingerConfig | None = None):
        from memore.config import EmbedConfig, RecallConfig
        from memore.llm import LLMConfig

        SENT.clear()
        store = _Store(facts)
        runtime = demo.Runtime(
            store=store,
            embedder=None,
            write_path=_WritePath(),
            chat=_Chat(),
            recall_config=RecallConfig(),
            embed_config=EmbedConfig(),
            llm_config=LLMConfig(),
            linger=LingerCache(config or LingerConfig()),
        )
        runtime.preflight = [{"name": "FalkorDB", "ok": True, "detail": ""}]
        pending = list(answers)

        async def _recall(context, cfg, store_, embedder):
            hits = pending.pop(0) if pending else []
            return RecallResult(
                injected_block="<recalled_context>fresh</recalled_context>" if hits else None,
                memories_used=list(hits),
                latency_ms=9.9,
                gate_open=bool(hits),
            )

        monkeypatch.setattr(demo, "RUNTIME", runtime)
        monkeypatch.setattr(demo, "recall", _recall)
        return TestClient(demo.app, raise_server_exceptions=True), runtime, store

    return _build


def _events(http, message: str) -> dict[str, dict]:
    with http.stream("POST", "/api/turn", json={"message": message}) as response:
        out: dict[str, dict] = {}
        name = None
        for line in response.iter_lines():
            if line.startswith("event:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("data:") and name:
                out.setdefault(name, json.loads(line.split(":", 1)[1]))
        return out


def test_the_follow_up_question_still_has_the_detail(demo_client):
    """The case this exists for, in the user's own words.

    Turn 1 "Where am I travelling to?" -- the gate opens. Turn 2 "when?" -- two words that
    share no term and no neighbourhood with the stored sentence, so the gate shuts and
    `recall()` returns nothing. The date must still reach the model.
    """
    http, _, _ = demo_client([_fact(TRIP, 1)], [[_hit(TRIP, 0.689)], []])
    _events(http, "Where am I travelling to?")
    events = _events(http, "when?")

    assert events["recall"]["gate_open"] is False          # recall's answer, unchanged
    assert events["recall"]["block"] is None
    assert events["linger"]["rescued"] is True
    assert "2026-08-26" in events["linger"]["block"]
    memory = "\n".join(m["content"] for m in SENT[-1] if m["role"] == "system")
    assert "Lisbon" in memory and "2026-08-26" in memory


def test_a_fact_nobody_refers_to_again_falls_out_of_the_block(demo_client):
    """The other side of the bar. Without this the layer is 'inject everything forever'."""
    # The config is PINNED here rather than taken from the default: this test asserts an
    # arithmetic consequence (once-seen, gone by age 3) and a default that moves would
    # silently turn it into a test of nothing.
    http, _, _ = demo_client(
        [_fact(TRIP, 1)], [[_hit(TRIP, 0.689)], [], [], [], []],
        LingerConfig(half_life_turns=2.0, floor=0.3),
    )
    _events(http, "Where am I travelling to?")
    assert _events(http, "when?")["linger"]["carried"]
    for _ in range(3):
        events = _events(http, "unrelated chatter")
    assert events["linger"]["carried"] == []
    assert events["linger"]["block"] is None
    assert all("recalled_context" not in m["content"] for m in SENT[-1])


def test_a_carried_fact_superseded_meanwhile_comes_back_marked_not_replayed(demo_client):
    """The defect worth guarding: injecting a dead value under `[valid as of ...]`.

    The cache stores text and a weight, and the carried set is resolved against the store
    every turn, so `assemble.render_hit` labels this one for us. Nothing in the demo needs
    to know it happened.
    """
    store_facts = [_fact(TRIP, 1)]
    http, _, store = demo_client(store_facts, [[_hit(TRIP, 0.8)], []])
    _events(http, "Where am I travelling to?")
    store.facts = [_fact(TRIP, 1, superseded=True), _fact("the trip is cancelled", 2)]

    events = _events(http, "when?")
    assert events["linger"]["carried"][0]["superseded"] is True
    assert "[SUPERSEDED" in events["linger"]["block"]
    assert "valid as of" not in events["linger"]["block"]


def test_a_fact_the_store_no_longer_holds_is_not_carried(demo_client):
    """Reset, or a switch away and back. A fact that is not in the store is not background
    knowledge about this session, whatever the cache still weighs it at."""
    http, runtime, store = demo_client([_fact(TRIP, 1)], [[_hit(TRIP, 0.8)], []])
    _events(http, "Where am I travelling to?")
    store.facts = []

    events = _events(http, "when?")
    assert events["linger"]["carried"] == []
    assert events["linger"]["block"] is None


def test_recall_hits_come_first_and_the_carried_set_is_only_ever_appended(demo_client):
    """This layer may add to what the gate decided; it may not reorder or displace it."""
    http, _, _ = demo_client(
        [_fact(TRIP, 1), _fact(CAT, 2)],
        [[_hit(TRIP, 0.8)], [_hit(CAT, 0.9)]],
    )
    _events(http, "Where am I travelling to?")
    events = _events(http, "what is my cat called?")
    lines = [line for line in events["linger"]["block"].splitlines() if line.startswith("- ")]
    assert lines[0].endswith(CAT)          # this turn's hit, ranked against this question
    assert lines[1].endswith(TRIP)         # carried, appended after it
    assert events["linger"]["rescued"] is False   # the gate was open; nothing was rescued


def test_the_carried_set_respects_what_is_left_of_the_inject_budget(demo_client):
    """`apply_gate`'s budget fill is bypassed here, so the cap is re-applied explicitly --
    otherwise the block grows without bound across a long session."""
    long_fact = " ".join(["word"] * 600)
    http, runtime, _ = demo_client(
        [_fact(long_fact, 1)], [[_hit(long_fact, 0.9)], []]
    )
    assert runtime.recall_config.inject_token_budget < 600
    _events(http, "tell me")
    events = _events(http, "and?")
    assert events["linger"]["carried"] == []


def test_switching_session_clears_the_carried_context(demo_client):
    """Three pieces of session state move together, and this is the one that would leak
    silently: session A's facts appearing in session B's `<recalled_context>`."""
    http, runtime, _ = demo_client([_fact(TRIP, 1)], [[_hit(TRIP, 0.9)], []])
    _events(http, "Where am I travelling to?")

    body = http.post("/api/session", json={"session": "another"}).json()
    assert body["session"] == "another" and runtime.session == "another"
    assert runtime.history == []
    assert _events(http, "when?")["linger"]["carried"] == []


def test_reset_clears_the_carried_context_too(demo_client):
    http, runtime, _ = demo_client([_fact(TRIP, 1)], [[_hit(TRIP, 0.9)], []])
    _events(http, "Where am I travelling to?")
    http.post("/api/reset")
    assert _events(http, "when?")["linger"]["carried"] == []


def test_an_empty_session_name_is_refused(demo_client):
    http, runtime, _ = demo_client([], [[]])
    assert http.post("/api/session", json={"session": "  "}).status_code == 400
    assert runtime.session == demo.DEFAULT_SESSION


def test_the_session_list_always_contains_the_current_one(demo_client):
    """A session with no facts does not exist in the store -- a session IS its facts
    there -- and the page must still be able to show you the one you are talking into."""
    http, runtime, _ = demo_client([], [[]])
    http.post("/api/session", json={"session": "brand-new"})
    listed = http.get("/api/state").json()["sessions"]
    current = [row for row in listed if row["current"]]
    assert [row["session"] for row in current] == ["brand-new"]


def test_between_turns_the_state_read_projects_the_next_turn_not_this_one():
    """`/api/state` is read BETWEEN turns, and `carried()` would answer "nothing" there:
    the facts the last turn surfaced are still marked fresh. A panel claiming an empty
    context while the next turn is about to inject two facts is the panel lying."""
    cache = LingerCache(LingerConfig(half_life_turns=2.0, floor=0.3))
    cache.begin_turn()
    cache.observe(TRIP, 0.9)
    assert cache.carried() == []
    assert [c.fact for c in cache.upcoming()] == [TRIP]
    assert cache.upcoming()[0].age_turns == 1
    # Reading is not a turn: nothing was pruned and nothing decayed permanently.
    cache.begin_turn()
    assert [c.fact for c in cache.carried()] == [TRIP]


def test_a_projection_never_prunes_what_a_real_turn_would_drop():
    cache = LingerCache(LingerConfig(half_life_turns=2.0, floor=0.3))
    cache.begin_turn()
    cache.observe(TRIP, 0.35)
    for _ in range(4):
        cache.begin_turn()
    assert cache.upcoming() == []      # below the floor, but only projected
    assert cache.carried() == []       # this is the read that actually drops it


def test_a_carried_event_carries_its_DATE_which_is_what_when_asks_for(demo_client):
    """End to end, through the merged block, for the bug that started this.

    It was a recorded LIMIT first and the history is the point: P1 writes "the user is
    flying to Lisbon" and puts 2026-08-26 in `occurs_at` -- observed live, not
    hypothesised -- and `render_hit` used to emit a date only on the SUPERSEDED and PAST
    branches, so an upcoming event rendered `[valid as of <when it was learned>]`. A live
    demo turn then answered "when?" with **the learn date as the travel date**: carrying
    the fact answered "where" and not "when", and answered "when" wrongly.

    `assemble.render_hit` grew an UPCOMING branch. This is the only test that exercises it
    through the demo's own merged block rather than through `render_hit` directly, which
    is why it lives here rather than in `test_recall.py`.
    """
    text = "the user is flying to Lisbon"
    trip = datetime(2026, 8, 26, tzinfo=UTC)
    http, _, _ = demo_client([_fact(text, 1, occurs_at=trip)], [[_hit(text, 0.8)], []])
    _events(http, "Where am I travelling to?")

    block = _events(http, "when?")["linger"]["block"]
    assert "[UPCOMING - occurs 2026-08-26] the user is flying to Lisbon" in block
    # The learn date is gone from the line, not merely joined by the trip date: it is the
    # string the model was misreading.
    assert "valid as of" not in block
