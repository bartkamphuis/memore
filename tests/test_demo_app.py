"""The demo's API contract, and the failure it exists to make visible.

No FalkorDB, no Ollama: the runtime is faked, because what these pin is the **shape** the
page reads, and the one behaviour that is not cosmetic — a preflight failure has to reach
the browser as a readable line rather than as an empty store.

That last one is the whole reason `preflight()` exists. Three ordinary failures (FalkorDB
down, Ollama not serving the pinned models, a graph whose vector index was built at another
width) all produce *no facts* rather than an error, and a demo that renders an empty store
when the store is down teaches the opposite of what it is for.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

pytest.importorskip("fastapi", reason="the demo lives behind the `demo` extra")

from fastapi.testclient import TestClient  # noqa: E402

from memore.demo import app as demo  # noqa: E402
from memore.types import (  # noqa: E402
    CandidateFact,
    ConsolidationCase,
    ConsolidationOutcome,
    FactType,
    MemoryHit,
    RecallResult,
    StoredFact,
)


def _fact(fact: str, ordinal: int, *, superseded: bool = False) -> StoredFact:
    return StoredFact(
        id=f"f{ordinal}",
        session_id=demo.SESSION,
        fact=fact,
        subject_key="the user",
        subject_label="the user",
        ordinal=ordinal,
        valid_at=datetime(2026, 8, 21, tzinfo=UTC),
        invalid_at=datetime(2026, 8, 21, tzinfo=UTC) if superseded else None,
        source_episode_id="ep",
        type=FactType.PREFERENCE,
        attribute="deploy target",
        attribute_label="deploy target",
    )


class _Store:
    def __init__(self, facts: list[StoredFact] | None = None):
        from memore.config import StoreConfig

        self.config = StoreConfig(graph_name="test_graph")
        self.facts = facts or []
        self.cleared = False

    async def connect(self): ...
    async def count(self, session_id): return len(self.facts)
    async def facts_in_session(self, session_id): return list(self.facts)
    async def clear_session(self, session_id):
        self.cleared = True
        self.facts = []


class _Chat:
    def __init__(self, reply="ok"): self.reply, self.seen = reply, []
    async def chat(self, messages, schema=None, max_tokens=None):
        self.seen.append(messages)
        return self.reply


class _WritePath:
    def __init__(self, outcomes=()): self.outcomes, self.calls = list(outcomes), []
    async def run(self, session_id, user_message, assistant_response="", recent=None):
        from memore.writepath import WriteResult

        self.calls.append((user_message, assistant_response))
        return WriteResult(candidates=len(self.outcomes), outcomes=self.outcomes)


def _runtime(*, facts=None, hits=None, gate_open=True, outcomes=(), preflight_ok=True):
    from memore.config import EmbedConfig, RecallConfig
    from memore.llm import LLMConfig

    store = _Store(facts)
    runtime = demo.Runtime(
        store=store,
        embedder=None,
        write_path=_WritePath(outcomes),
        chat=_Chat(),
        recall_config=RecallConfig(),
        embed_config=EmbedConfig(),
        llm_config=LLMConfig(),
    )
    runtime.preflight = [
        {"name": "FalkorDB", "ok": preflight_ok,
         "detail": "graph 'test_graph'" if preflight_ok else "ConnectionError: refused"},
        {"name": "Ollama embedder", "ok": True, "detail": "mxbai"},
    ]

    async def _recall(context, config, store_, embedder):
        return RecallResult(
            injected_block="<recalled_context>…</recalled_context>" if gate_open else None,
            memories_used=list(hits or []),
            latency_ms=12.3,
            gate_open=gate_open,
        )

    return runtime, _recall


@pytest.fixture
def client(monkeypatch):
    def _build(**kwargs):
        runtime, fake_recall = _runtime(**kwargs)
        monkeypatch.setattr(demo, "RUNTIME", runtime)
        monkeypatch.setattr(demo, "recall", fake_recall)
        # lifespan would open a real store and a real Ollama connection.
        return TestClient(demo.app, raise_server_exceptions=True), runtime
    return _build


def test_a_preflight_failure_reaches_the_page_instead_of_an_empty_store(client):
    """The one non-cosmetic assertion. `ok:false` is what makes the banner render; without
    it a store that is DOWN and a store that is EMPTY are the same screen."""
    http, _ = client(preflight_ok=False)
    body = http.get("/api/state").json()
    assert body["ok"] is False
    assert body["store"] == []
    failed = [c for c in body["preflight"] if not c["ok"]]
    assert failed and "refused" in failed[0]["detail"]


def test_state_carries_what_you_need_to_debug_an_empty_store(client):
    """Session, graph and fact count, for the reason CLAUDE.md gives: recall is
    session-scoped, so an empty session and a broken lookup look identical."""
    http, _ = client(facts=[_fact("a", 1)])
    body = http.get("/api/state").json()
    assert body["session"] == demo.SESSION
    assert body["graph"] == "test_graph"
    assert body["gate"].startswith("cosine >=")
    assert body["store"][0]["subject"] == "the user"


def test_superseded_facts_are_returned_and_marked_not_hidden(client):
    """'Supersede, never delete' is the design claim, so the pane that would hide it is
    the pane that makes the claim unfalsifiable."""
    http, _ = client(facts=[_fact("staging", 1, superseded=True), _fact("production", 2)])
    group = http.get("/api/state").json()["store"][0]
    # Newest ordinal first, both present.
    assert [f["ordinal"] for f in group["facts"]] == [2, 1]
    assert [f["superseded"] for f in group["facts"]] == [False, True]


def test_a_turn_returns_recall_write_reply_and_the_new_store_state(client):
    outcome = ConsolidationOutcome(
        candidate=CandidateFact(
            fact="the user deploys to production",
            type=FactType.PREFERENCE,
            confidence=0.9,
            valid_at=None,
            subject_hint="the user",
            attribute="deploy target",
            single_valued=True,
        ),
        case=ConsolidationCase.CONTRADICTION,
        superseded_fact_id="f1",
        ordinal=2,
    )
    hit = MemoryHit(fact="the user deploys to staging", score=0.7, similarity=0.72,
                    valid_at=None, invalid_at=None, source_episode_id="ep")
    http, runtime = client(hits=[hit], outcomes=[outcome], facts=[_fact("x", 1)])

    body = http.post("/api/turn", json={"message": "where do I deploy?"}).json()
    assert body["recall"]["gate_open"] is True
    assert body["recall"]["hits"][0]["similarity"] == 0.72
    written = body["write"]["outcomes"][0]
    assert written["case"] == "CONTRADICTION"
    assert written["superseded_fact_id"] == "f1"
    assert written["single_valued"] is True
    assert body["reply"] == "ok"
    assert body["store"][0]["subject"] == "the user"


def test_the_assistant_reply_is_context_for_the_write_path_never_the_turn(client):
    """RESULTS.md §17: the reply sat inside the block P1 was told to extract from, and the
    model stored things the user never said. The demo must hand it over as the
    `assistant_response` argument, which is context, and never fold it into the message."""
    http, runtime = client()
    http.post("/api/turn", json={"message": "I deploy to staging"})
    user_message, assistant_response = runtime.write_path.calls[0]
    assert user_message == "I deploy to staging"
    assert assistant_response == "ok"


def test_recall_runs_before_the_model_call_and_is_injected_not_offered_as_a_tool(client):
    """The architecture in one assertion: the block reaches the model as a system message
    that is already there, rather than as a tool the model may decide to call."""
    http, runtime = client(gate_open=True)
    http.post("/api/turn", json={"message": "hello"})
    roles = [m["role"] for m in runtime.chat.seen[0]]
    system = [m["content"] for m in runtime.chat.seen[0] if m["role"] == "system"]
    assert roles[0] == "system"
    assert any("MEMORY:" in s for s in system)
    assert roles[-1] == "user"


def test_a_shut_gate_injects_nothing(client):
    http, runtime = client(gate_open=False)
    body = http.post("/api/turn", json={"message": "unrelated"}).json()
    assert body["recall"]["gate_open"] is False
    assert body["recall"]["block"] is None
    assert not any("MEMORY:" in m["content"] for m in runtime.chat.seen[0])


def test_reset_clears_the_session_and_the_history(client):
    http, runtime = client(facts=[_fact("a", 1)])
    http.post("/api/turn", json={"message": "hi"})
    assert runtime.history
    assert http.post("/api/reset").json() == {"store": [], "ok": True}
    assert runtime.store.cleared is True
    assert runtime.history == []


def test_an_empty_message_is_refused_rather_than_stored(client):
    http, runtime = client()
    assert http.post("/api/turn", json={"message": "   "}).status_code == 400
    assert runtime.write_path.calls == []
