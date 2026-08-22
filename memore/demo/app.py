"""The FastAPI app: three routes, one page, and a turn that streams.

## Why the turn streams, having first been written not to

The first version returned the whole trace in one JSON body, on the argument that an
assembled trace is more legible than one racing the reply. That argument was wrong here,
and it was wrong about the thing the demo exists to show. **The timings are the claim.**
Recall lands in ~70ms, before the model is called; the model takes seconds; P1 extraction
runs beside it and takes seconds of its own. A single response makes all three arrive
together and hides exactly the property being demonstrated -- the user waits, sees
"thinking", and then gets numbers asserting a sequence they never observed.

So a turn emits events when they actually happen:

    recall      the gate decision, the hits and the block RECALL produced   (~70ms)
    linger      the frecency lane's carried facts, and the block actually sent
    reply_start the model call has begun
    write_start the write lane has been launched, in parallel with it
    delta       content, as it arrives
    reply_end   the reply is complete
    write       P1's candidates and P2's case per candidate
    store       the store after the write

## The write lane runs CONCURRENTLY with the reply, and that is the point

The first streaming version awaited the write after `reply_end`, so P1's extraction --
seconds of `gemma4:26b` -- began only once the last token had landed. That is still not
what the architecture claims. "Off the response path" does not mean *after* the response
path; it means *not on it*. `OLLAMA_NUM_PARALLEL=3` and both lanes use the same model, so
the two requests share the loaded weights across slots with no reload.

So the write task is launched as soon as recall returns, beside the reply stream, and its
result is emitted whenever it lands. Events are multiplexed through a queue rather than
yielded in a fixed order, because a fixed order is exactly the thing being disproved.

**It is a trade and the numbers say so.** The same four turns, run both ways:

    write awaited after the reply    write 2578-4285ms   reply 1182-1468ms
    write launched beside it         write 1632-3098ms   reply 1616-2055ms
    mean change                      write -1089ms       reply  +452ms

The write finishes about a second sooner and the reply about half a second later: no
reload, but one GPU, so the lanes contend. An earlier single-turn observation had the write
landing mid-stream, and that was briefly written up as the rule -- on these four it landed
after the last token in *both* arms. What is always true, and what this demonstrates, is
that the write **starts** at ~76ms rather than at `reply_end`.

**Launched after recall, never before.** Recall must read the store as it was *before* this
turn's fact, and a write committing mid-lookup would let a turn recall itself.

**The tradeoff, stated because it is real.** P1 is given `assistant_response=""`: the reply
does not exist yet when the write starts. RESULTS.md §17 has the reply as *context* in the
P1 prompt -- deliberately demoted out of the block being extracted from, but present. The
cost here is narrow: `extract_window_turns` is 3, so P1 still sees the previous turns'
replies from history and misses only the current one. `memore.cli`'s terminal demo passes
`""` for the same reason. `llm_gateway` does the opposite and it is also right: it fires
`after_model_call(...)` with the reply, un-awaited, because a gateway has no reason to
start early. Two defensible choices; this one makes the parallelism visible.

Every event carries `ms`, measured from a single `t0` at request start, on the server.
The page does not compute it from arrival times: the sequence is a claim about the system,
not about the network.

**SSE framing over POST, read with a streaming `fetch`, not `EventSource`.** EventSource is
GET-only, and the turn's body is a message. Framing is standard `data: …\n\n` so the wire
is self-describing.

## Two events, because there are two decisions

`recall` is what `recall()` answered and nothing else. `linger` is the harness's own
frecency lane (`demo/linger.py`) deciding how long a fact recall already surfaced stays in
the block. They are reported separately on purpose: a turn showing `gate SHUT` **and** a
block going to the model is not a contradiction, it is the whole demonstration, and folding
the merged block back into the `recall` event would quietly make the page claim recall
returned something it did not.

## Sessions are switchable; graphs are not

The page's dropdown lists `store.sessions()` -- `(session_id, total, live)` -- and switching
one is a string assignment plus clearing history and the linger cache. Graphs are not
offered: the vector index is created at the embedder's width and `connect()` refuses a
mismatch, so a graph list is a list of things that may fail to open.

## Startup is where a demo of this system actually fails

Three failures produce a blank page rather than an error, and all three are ordinary:
FalkorDB not running, Ollama not serving the models `config.py` pins, and an existing graph
whose vector index was built at a different width (`FalkorStore.connect()` raises on that
deliberately -- see the invariant in CLAUDE.md). `preflight()` checks all three and the page
renders the result, because a demo that shows an empty store when the store is *down*
teaches the opposite of what it is for.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime

import httpx
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from ..assemble import build_block, is_past
from ..config import EmbedConfig, RecallConfig, StoreConfig, WritePathConfig
from ..consolidate import DeterministicConsolidator
from ..embed import OllamaEmbedder
from ..extract import OllamaExtractor
from ..llm import LLMConfig, OllamaClient
from ..recall import WordTokenizer, recall
from ..store.falkor import FalkorStore
from ..types import MemoryHit, Message, StoredFact, TurnContext
from ..writepath import WritePath
from .linger import LingerCache, LingerConfig
from .page import PAGE

logger = logging.getLogger("memore.demo")

# Its own graph and its own session, for the reason CLAUDE.md gives: recall is
# session-scoped, and sharing a graph with the bench or the terminal demo would mix
# corpora into what this page shows.
DEFAULT_GRAPH = "memore_demo"
DEFAULT_SESSION = "demo-web"

# Sessions are the unit the page's dropdown switches between, and they live inside the one
# graph. Not graphs: a graph's vector index is created at the embedder's width and
# `FalkorStore.connect()` refuses a mismatch (see the invariant in CLAUDE.md), so switching
# graphs at runtime means tearing down and rebuilding the store and can fail on a graph
# written by a different embedder. Sessions share the index, cost one string to switch, and
# are the boundary recall itself is scoped to.

SYSTEM = (
    "You are a concise, friendly assistant. Answer in at most three sentences. "
    "If the MEMORY block is present, use it as fact about this user and do not repeat it "
    "back verbatim unless asked."
)


@dataclass
class Runtime:
    """Everything the app holds open. One store, one embedder, one extractor."""

    store: FalkorStore
    embedder: OllamaEmbedder
    write_path: WritePath
    chat: OllamaClient
    recall_config: RecallConfig
    embed_config: EmbedConfig
    llm_config: LLMConfig
    # The session is runtime state, not a constant: the page switches it. Everything that
    # reads the store -- preflight, the store pane, the turn, reset -- must read it from
    # here, or a switch leaves one of them pointed at the previous session.
    session: str = DEFAULT_SESSION
    history: list[Message] = field(default_factory=list)
    preflight: list[dict] = field(default_factory=list)
    # Per session, and cleared on both reset and switch -- otherwise one session's facts
    # are carried into another's `<recalled_context>`.
    linger: LingerCache = field(default_factory=lambda: LingerCache(LingerConfig.from_env()))


RUNTIME: Runtime | None = None


async def preflight(runtime: Runtime) -> list[dict]:
    """Check the three things whose failure looks like an empty store.

    Each returns `(name, ok, detail)` and nothing raises: the page has to render even when
    everything is down, or the operator sees a browser error instead of the diagnosis.
    """
    checks: list[dict] = []

    try:
        await runtime.store.connect()
        held = await runtime.store.count(runtime.session)
        checks.append({
            "name": "FalkorDB",
            "ok": True,
            "detail": f"graph {runtime.store.config.graph_name!r}, session {runtime.session!r}, "
                      f"{held} fact(s)",
        })
    except Exception as exc:  # noqa: BLE001 -- the diagnosis IS the product here
        # The width mismatch is the interesting one and it is worth naming, because the
        # fix is a different graph rather than a restart.
        hint = ""
        if "dimension" in str(exc).lower() or "index" in str(exc).lower():
            hint = (" -- this graph's vector index was built for a different embedder. "
                    f"Use a fresh MEMORE_GRAPH, or drop it: "
                    f"docker exec memore-falkordb redis-cli GRAPH.DELETE "
                    f"{runtime.store.config.graph_name}")
        checks.append({
            "name": "FalkorDB",
            "ok": False,
            "detail": f"{type(exc).__name__}: {exc}{hint or ' -- is it up? docker compose up -d falkordb'}",
        })

    for label, model, probe in (
        ("embedder", runtime.embed_config.model, "embed"),
        ("chat / extractor", runtime.llm_config.model, "chat"),
    ):
        try:
            started = time.perf_counter()
            if probe == "embed":
                vector = await runtime.embedder.embed_one("preflight")
                extra = f"dim {len(vector)}"
            else:
                await runtime.chat.chat([{"role": "user", "content": "hi"}], max_tokens=1)
                extra = f"num_ctx {runtime.llm_config.num_ctx}"
            checks.append({
                "name": f"Ollama {label}",
                "ok": True,
                "detail": f"{model} ({extra}, {(time.perf_counter() - started) * 1000:.0f}ms)",
            })
        except Exception as exc:  # noqa: BLE001
            checks.append({
                "name": f"Ollama {label}",
                "ok": False,
                "detail": f"{model} unreachable -- {type(exc).__name__}: {exc}. "
                          f"`ollama pull {model}`, and check MEMORE_LLM_MODEL / "
                          f"MEMORE_EMBED_MODEL match how the host serves them.",
            })
    return checks


@asynccontextmanager
async def lifespan(app: FastAPI):
    global RUNTIME
    embed_config = EmbedConfig.from_env()
    store_config = StoreConfig.from_env()
    write_config = WritePathConfig()
    llm_config = LLMConfig(model=write_config.extractor_model)
    # The index width is the embedder's, never a store constant -- EmbedConfig owns it.
    store = FalkorStore(store_config, dimension=embed_config.dimension)
    embedder = OllamaEmbedder(embed_config)
    chat = OllamaClient(llm_config)
    RUNTIME = Runtime(
        store=store,
        embedder=embedder,
        write_path=WritePath(
            OllamaExtractor(write_config, OllamaClient(llm_config)),
            DeterministicConsolidator(store, embedder),
            write_config,
            store=store,
        ),
        chat=chat,
        recall_config=RecallConfig(),
        embed_config=embed_config,
        llm_config=llm_config,
    )
    RUNTIME.preflight = await preflight(RUNTIME)
    for check in RUNTIME.preflight:
        logger.log(logging.INFO if check["ok"] else logging.ERROR,
                   "preflight %s: %s", check["name"], check["detail"])
    yield
    await embedder.aclose()
    await chat.aclose()
    await store.aclose()


app = FastAPI(title="memore demo", lifespan=lifespan)


class Turn(BaseModel):
    message: str


class SessionSwitch(BaseModel):
    session: str


def _runtime() -> Runtime:
    assert RUNTIME is not None, "lifespan did not run"
    return RUNTIME


def _hit(hit, now) -> dict:
    return {
        "fact": hit.fact,
        "score": round(hit.score, 3),
        "similarity": round(hit.similarity, 3),
        "superseded": hit.invalid_at is not None,
        "past": is_past(hit, now),
    }


def _as_hit(fact: StoredFact) -> MemoryHit:
    """A stored row rendered as a hit, so `assemble.render_hit` labels it identically.

    `score`/`similarity` stay 0.0 here and the caller sets the decayed weight: a carried
    fact was ranked against a *previous* turn, so inventing a similarity for this one
    would be a number about a question nobody asked -- the same reason chain facts carry
    `score=0.0` (RESULTS.md §8).
    """
    return MemoryHit(
        fact=fact.fact,
        score=0.0,
        valid_at=fact.valid_at,
        invalid_at=fact.invalid_at,
        source_episode_id=fact.source_episode_id,
        occurs_at=fact.occurs_at,
        recurring=fact.recurring,
    )


async def _linger_hits(
    runtime: Runtime, budget_left: int
) -> tuple[list[MemoryHit], list[dict]]:
    """Resolve what the cache is carrying against the store AS IT IS NOW.

    The cache holds weights and fact text, never a rendered line, and this re-reads every
    carried fact from the store each turn. That is the whole safety argument: if turn N+1
    contradicts a fact carried from turn N, the store has it superseded and this renders
    it `[SUPERSEDED - was valid ...]` on the next injection rather than replaying the dead
    value under `[valid as of ...]`. A harness that resurrected what the store retired
    would invert the one claim the right-hand pane exists to make.

    Carried facts are also capped against what is LEFT of `inject_token_budget` after
    recall's own hits, because this path bypasses `apply_gate`'s budget fill and nothing
    else would bound the block across a long session.
    """
    carried = runtime.linger.carried()
    if not carried:
        return [], []

    rows = await runtime.store.facts_in_session(runtime.session)
    # Highest ordinal wins when one text appears twice: that row is the current state.
    current: dict[str, StoredFact] = {}
    for row in rows:
        held = current.get(row.fact)
        if held is None or row.ordinal > held.ordinal:
            current[row.fact] = row

    tokenizer = WordTokenizer()
    hits: list[MemoryHit] = []
    trace: list[dict] = []
    gone: list[str] = []
    now = datetime.now(UTC)
    for entry in carried:
        row = current.get(entry.fact)
        if row is None:
            # Cleared, or switched away and back. It is not in the store, so it is not
            # background knowledge about this session any more.
            gone.append(entry.fact)
            continue
        cost = tokenizer.count(row.fact)
        if cost > budget_left:
            continue
        budget_left -= cost
        hits.append(_as_hit(row))
        trace.append({
            "fact": row.fact,
            "weight": round(entry.weight, 3),
            "strength": round(entry.strength, 3),
            "age_turns": entry.age_turns,
            "seen": entry.seen,
            "superseded": row.invalid_at is not None,
            "past": is_past(hits[-1], now),
        })
    runtime.linger.forget(gone)
    return hits, trace


async def _store_view(runtime: Runtime) -> list[dict]:
    """The store, grouped by subject, newest ordinal first within each group.

    Superseded facts are included and marked -- hiding them would hide the design claim.
    """
    facts = await runtime.store.facts_in_session(runtime.session)
    groups: dict[str, dict] = {}
    for fact in facts:
        group = groups.setdefault(
            fact.subject_key, {"subject": fact.subject_label or fact.subject_key, "facts": []}
        )
        group["facts"].append({
            "fact": fact.fact,
            "attribute": fact.attribute_label or fact.attribute or "",
            "ordinal": fact.ordinal,
            "superseded": fact.invalid_at is not None,
            "type": fact.type.value,
            "occurs_at": fact.occurs_at.date().isoformat() if fact.occurs_at else None,
            "recurring": fact.recurring,
        })
    for group in groups.values():
        group["facts"].sort(key=lambda f: f["ordinal"], reverse=True)
    return sorted(groups.values(), key=lambda g: g["subject"].lower())


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return PAGE


async def _sessions(runtime: Runtime) -> list[dict]:
    """Every session in this graph, busiest first, with the current one always present.

    A session with no facts yet -- one just created from the dropdown -- does not exist in
    the store at all, because a session IS its facts there. It is added here so the page
    can show you the empty thing you are talking into rather than an unexplained blank.
    """
    try:
        rows = await runtime.store.sessions()
    except Exception as exc:  # noqa: BLE001 -- inspection must not break the page
        logger.warning("session list unavailable: %s", exc)
        rows = []
    out = [
        {"session": sid, "facts": total, "live": live, "current": sid == runtime.session}
        for sid, total, live in rows
    ]
    if not any(row["current"] for row in out):
        out.insert(0, {"session": runtime.session, "facts": 0, "live": 0, "current": True})
    return out


@app.get("/api/state")
async def state() -> JSONResponse:
    runtime = _runtime()
    config = runtime.recall_config
    linger = runtime.linger.config
    return JSONResponse({
        "session": runtime.session,
        "sessions": await _sessions(runtime) if runtime.preflight[0]["ok"] else [],
        "graph": runtime.store.config.graph_name,
        "embedder": runtime.embed_config.model,
        "model": runtime.llm_config.model,
        "gate": f"{config.gate_on} >= {config.score_floor}",
        "k": config.k,
        # What the cache is still holding, so a page RELOAD shows the standing state
        # rather than "nothing injected yet" on a server mid-conversation. Weights only;
        # the carried set is resolved against the store on the turn that uses it.
        "carried": [
            {"fact": c.fact, "weight": round(c.weight, 3), "age_turns": c.age_turns,
             "seen": c.seen, "strength": round(c.strength, 3),
             "superseded": False, "past": False}
            for c in runtime.linger.upcoming()
        ],
        "linger": {
            "enabled": linger.enabled,
            "half_life_turns": linger.half_life_turns,
            "floor": linger.floor,
            "max_facts": linger.max_facts,
        },
        "preflight": runtime.preflight,
        "ok": all(check["ok"] for check in runtime.preflight),
        "store": await _store_view(runtime) if runtime.preflight[0]["ok"] else [],
    })


def _sse(event: str, payload: dict, t0: float) -> str:
    """One SSE frame. `ms` is the server's own elapsed time, never the browser's."""
    body = dict(payload, ms=round((time.perf_counter() - t0) * 1000, 1))
    return f"event: {event}\ndata: {json.dumps(body)}\n\n"


async def _turn_events(runtime: Runtime, message: str) -> AsyncIterator[str]:
    """One turn. Recall first, then the reply and the write lane in parallel.

    The order is the architecture. Recall runs before the model call and is injected at
    prompt-assembly time rather than offered as a tool the model may decide to invoke. The
    write lane runs beside the reply, not after it, and never touches the response.
    """
    t0 = time.perf_counter()
    now = datetime.now(UTC)

    context = TurnContext(
        session_id=runtime.session, user_message=message, recent_messages=list(runtime.history)
    )
    result = await recall(context, runtime.recall_config, runtime.store, runtime.embedder)
    # Recall's answer, verbatim and unmodified -- its own `gate_open`, its own block, even
    # on turns where the harness goes on to inject anyway. The moment this event starts
    # reporting the merged block, the demo is asserting that recall produced something it
    # did not, and the trace stops being evidence.
    yield _sse("recall", {
        "gate_open": result.gate_open,
        "latency_ms": round(result.latency_ms, 1),
        "block": result.injected_block,
        "hits": [_hit(hit, now) for hit in result.memories_used],
    }, t0)

    # The frecency lane (`demo/linger.py`), harness-side. Recall has already run and is
    # untouched; this only decides how long what it found stays in the injected block.
    runtime.linger.begin_turn()
    tokenizer = WordTokenizer()
    for hit in result.memories_used:
        runtime.linger.observe(hit.fact, hit.similarity)
    spent = sum(tokenizer.count(hit.fact) for hit in result.memories_used)
    carried_hits, carried_trace = await _linger_hits(
        runtime, runtime.recall_config.inject_token_budget - spent
    )
    # Recall's hits first: they were ranked against THIS turn, and a carried fact may only
    # ever be appended to the gate's decision, never reorder or displace it.
    block = build_block(list(result.memories_used) + carried_hits, now)
    yield _sse("linger", {
        "carried": carried_trace,
        "block": block,
        # True exactly when the harness injected on a turn recall would have injected
        # nothing -- the case this layer exists for, and the one to watch for regressions.
        "rescued": bool(carried_hits) and not result.gate_open,
        "half_life_turns": runtime.linger.config.half_life_turns,
        "floor": runtime.linger.config.floor,
    }, t0)

    messages = [{"role": "system", "content": SYSTEM}]
    if block:
        messages.append({"role": "system", "content": f"MEMORY:\n{block}"})
    messages += [{"role": m.role, "content": m.content} for m in runtime.history[-6:]]
    messages.append({"role": "user", "content": message})

    # Snapshot: the write lane must see the history as it was before this turn, and it runs
    # while `runtime.history` is being appended to below.
    history = list(runtime.history)
    events: asyncio.Queue[tuple[str, dict] | None] = asyncio.Queue()

    async def pump_reply() -> None:
        await events.put(("reply_start", {}))
        chunks: list[str] = []
        try:
            async for delta in runtime.chat.chat_stream(messages, max_tokens=300):
                chunks.append(delta)
                await events.put(("delta", {"text": delta}))
        except httpx.HTTPError as exc:
            logger.exception("chat failed")
            chunks.append(f"[the model is unreachable: {type(exc).__name__}]")
            await events.put(("delta", {"text": chunks[-1]}))
        reply = "".join(chunks)
        runtime.history.append(Message(role="user", content=message))
        runtime.history.append(Message(role="assistant", content=reply))
        await events.put(("reply_end", {"reply": reply}))

    async def pump_write() -> None:
        await events.put(("write_start", {}))
        try:
            # `assistant_response=""` -- it does not exist yet. See the module docstring;
            # this is the one thing concurrency costs and it is not free.
            write = await runtime.write_path.run(runtime.session, message, "", history)
        except Exception as exc:  # noqa: BLE001 -- a failed write must not kill the stream
            logger.exception("write path failed")
            await events.put(("write_error", {"error": f"{type(exc).__name__}: {exc}"}))
            return
        await events.put(("write", {
            "candidates": write.candidates,
            "outcomes": [
                {
                    "case": outcome.case.value,
                    "fact": outcome.candidate.fact,
                    "type": outcome.candidate.type.value,
                    "confidence": round(outcome.candidate.confidence, 2),
                    "subject": outcome.candidate.subject_hint,
                    "attribute": outcome.candidate.attribute,
                    "single_valued": outcome.candidate.single_valued,
                    "ordinal": outcome.ordinal,
                    "superseded_fact_id": outcome.superseded_fact_id,
                }
                for outcome in write.outcomes
            ],
        }))
        await events.put(("store", {"store": await _store_view(runtime)}))

    async def drain() -> None:
        await asyncio.gather(pump_reply(), pump_write())
        await events.put(None)   # the sentinel, once BOTH lanes are done

    runner = asyncio.create_task(drain())
    try:
        while (item := await events.get()) is not None:
            yield _sse(item[0], item[1], t0)
    finally:
        # A browser that navigates away closes the response mid-stream. Cancel rather than
        # leak the two lanes, and let the write finish if it already has.
        if not runner.done():
            runner.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await runner


# `response_model=None`: the union return type is two Response classes, and FastAPI would
# otherwise try to build a Pydantic response model out of them.
@app.post("/api/turn", response_model=None)
async def turn(body: Turn) -> StreamingResponse | JSONResponse:
    runtime = _runtime()
    message = body.message.strip()
    if not message:
        return JSONResponse({"error": "empty message"}, status_code=400)
    return StreamingResponse(
        _turn_events(runtime, message),
        media_type="text/event-stream",
        # Without this an intervening proxy will buffer the stream and hand the browser
        # everything at once -- which is the bug this endpoint was rewritten to fix.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/session")
async def switch(body: SessionSwitch) -> JSONResponse:
    """Point the app at another session in the same graph. Creates by naming.

    Three pieces of state are session-scoped and all three must move together: the store
    reads (`runtime.session`), the conversation (`history`), and the linger cache. Leaving
    any behind leaks one session into another -- the cache most visibly, since it would put
    the previous session's facts straight into the next one's `<recalled_context>`.
    """
    runtime = _runtime()
    name = body.session.strip()
    if not name:
        return JSONResponse({"error": "empty session name"}, status_code=400)
    runtime.session = name
    runtime.history.clear()
    runtime.linger.clear()
    return JSONResponse({
        "ok": True,
        "session": name,
        "sessions": await _sessions(runtime),
        "store": await _store_view(runtime),
    })


@app.post("/api/reset")
async def reset() -> JSONResponse:
    runtime = _runtime()
    await runtime.store.clear_session(runtime.session)
    runtime.history.clear()
    # The cache resolves carried facts against the store every turn and would drop these
    # on its own; clearing here means the block is empty on the very next turn rather than
    # one turn later.
    runtime.linger.clear()
    return JSONResponse({
        "store": [], "ok": True, "sessions": await _sessions(_runtime())
    })
