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

import json
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

    async def chat_stream(self, messages, max_tokens=None):
        self.seen.append(messages)
        for character in self.reply:
            yield character


class _WritePath:
    def __init__(self, outcomes=()): self.outcomes, self.calls = list(outcomes), []
    async def run(self, session_id, user_message, assistant_response="", recent=None):
        from memore.writepath import WriteResult

        self.calls.append((user_message, assistant_response))
        return WriteResult(candidates=len(self.outcomes), outcomes=self.outcomes)


def _hit_obj() -> MemoryHit:
    return MemoryHit(fact="the user deploys to staging", score=0.7, similarity=0.72,
                     valid_at=None, invalid_at=None, source_episode_id="ep")


def _outcome() -> ConsolidationOutcome:
    return ConsolidationOutcome(
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


def _events(http, message: str) -> list[tuple[str, dict]]:
    """Read the SSE stream into `[(event, data)]`.

    Parses frames the same way the page does -- accumulate, split on a blank line -- and
    for the same reason: a `data:` line can arrive split across two reads, and appending
    straight through would corrupt that frame.
    """
    out: list[tuple[str, dict]] = []
    with http.stream("POST", "/api/turn", json={"message": message}) as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        buffer = ""
        for chunk in response.iter_text():
            buffer += chunk
            while "\n\n" in buffer:
                frame, buffer = buffer.split("\n\n", 1)
                event, data = "message", None
                for line in frame.split("\n"):
                    if line.startswith("event:"):
                        event = line[6:].strip()
                    elif line.startswith("data:"):
                        data = json.loads(line[5:].strip())
                if data is not None:
                    out.append((event, data))
    return out


def test_the_stages_arrive_as_separate_events_in_the_order_they_happen(client):
    """The ordering IS the contract, and it is the whole reason this endpoint streams.

    A single JSON body made recall (~80ms), the reply (seconds) and P1 extraction (seconds
    more) arrive together, so the page asserted a sequence the user never observed. In
    particular `write` must come after `reply_end`: awaiting the write path any earlier
    puts P1's own LLM call between the last token and the reply finishing.
    """
    http, _ = client(hits=[_hit_obj()], outcomes=[_outcome()])
    events = [name for name, _ in _events(http, "where do I deploy?")]
    assert [e for e in events if e != "delta"] == [
        "recall", "reply_start", "reply_end", "write", "store",
    ]
    assert events.index("recall") < events.index("delta") < events.index("write")


def test_every_event_carries_the_server_s_own_elapsed_ms(client):
    """The page must not compute timings from arrival times: the sequence is a claim about
    the system, not about the network."""
    http, _ = client(outcomes=[_outcome()])
    events = _events(http, "hello")
    assert all("ms" in data for _, data in events)
    stamps = [data["ms"] for _, data in events]
    assert stamps == sorted(stamps), stamps


def test_the_recall_event_carries_the_gate_the_hits_and_the_block(client):
    http, _ = client(hits=[_hit_obj()])
    recall_event = next(data for name, data in _events(http, "q") if name == "recall")
    assert recall_event["gate_open"] is True
    assert recall_event["hits"][0]["similarity"] == 0.72
    assert "recalled_context" in recall_event["block"]


def test_the_write_event_carries_the_case_the_ordinal_and_the_supersede(client):
    http, _ = client(outcomes=[_outcome()])
    write_event = next(data for name, data in _events(http, "q") if name == "write")
    written = write_event["outcomes"][0]
    assert written["case"] == "CONTRADICTION"
    assert written["superseded_fact_id"] == "f1"
    assert written["single_valued"] is True


def test_the_reply_is_assembled_from_the_deltas(client):
    http, _ = client()
    events = _events(http, "hi")
    deltas = "".join(data["text"] for name, data in events if name == "delta")
    reply = next(data["reply"] for name, data in events if name == "reply_end")
    assert deltas == reply == "ok"


def test_reset_clears_the_session_and_the_history(client):
    http, runtime = client(facts=[_fact("a", 1)])
    _events(http, "hi")
    assert runtime.history
    assert http.post("/api/reset").json() == {"store": [], "ok": True}
    assert runtime.store.cleared is True
    assert runtime.history == []


def test_an_empty_message_is_refused_rather_than_stored(client):
    http, runtime = client()
    assert http.post("/api/turn", json={"message": "   "}).status_code == 400
    assert runtime.write_path.calls == []


# --- the page itself, checked statically ------------------------------------------
#
# A JS rewrite dropped `boot()`'s definition while leaving its call site, so the header and
# the initial store silently never loaded -- caught by a browser, which the suite does not
# have. These two are the cheap static form of that check.

def _script() -> str:
    import re

    from memore.demo.page import PAGE

    return re.search(r"<script>(.*)</script>", PAGE, re.S).group(1)


def test_every_function_the_page_calls_at_top_level_is_defined():
    import re

    script = _script()
    defined = set(re.findall(r"(?:async\s+)?function\s+(\w+)", script))
    defined |= set(re.findall(r"const\s+(\w+)\s*=\s*(?:async\s*)?\(", script))
    called = set(re.findall(r"^(\w+)\(\);?$", script, re.M))
    assert called, "expected at least one top-level call (boot)"
    assert called <= defined, f"called but never defined: {sorted(called - defined)}"


def test_every_element_id_the_script_reaches_for_exists_in_the_markup():
    import re

    from memore.demo.page import PAGE

    wanted = set(re.findall(r"""\$\(["'](\w[\w-]*)["']\)""", _script()))
    present = set(re.findall(r"""id=["'](\w[\w-]*)["']""", PAGE))
    assert wanted <= present, f"referenced but absent: {sorted(wanted - present)}"
