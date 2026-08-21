"""The FastAPI app: three routes, one page, and a turn that streams.

## Why the turn streams, having first been written not to

The first version returned the whole trace in one JSON body, on the argument that an
assembled trace is more legible than one racing the reply. That argument was wrong here,
and it was wrong about the thing the demo exists to show. **The timings are the claim.**
Recall lands in ~80ms, before the model is called; the model takes seconds; P1 extraction
takes seconds more, after the reply. A single response makes all three arrive together and
hides exactly the property being demonstrated -- the user waits, sees "thinking", and then
gets numbers that assert a sequence they never observed.

So a turn emits events when they actually happen:

    recall      the gate decision, the hits and the injected block   (~70ms)
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
the two requests share the loaded weights across slots with no reload: measured, two
concurrent calls take 1221ms against 2213ms serial, a 45% overlap.

So the write task is launched as soon as recall returns and the reply stream starts, and
its result is emitted whenever it lands -- often while the reply is still streaming. Events
are multiplexed through a queue rather than yielded in a fixed order, because a fixed order
is exactly the thing being disproved.

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

from ..assemble import is_past
from ..config import EmbedConfig, RecallConfig, StoreConfig, WritePathConfig
from ..consolidate import DeterministicConsolidator
from ..embed import OllamaEmbedder
from ..extract import OllamaExtractor
from ..llm import LLMConfig, OllamaClient
from ..recall import recall
from ..store.falkor import FalkorStore
from ..types import Message, TurnContext
from ..writepath import WritePath
from .page import PAGE

logger = logging.getLogger("memore.demo")

# Its own graph and its own session, for the reason CLAUDE.md gives: recall is
# session-scoped, and sharing a graph with the bench or the terminal demo would mix
# corpora into what this page shows.
DEFAULT_GRAPH = "memore_demo"
SESSION = "demo-web"

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
    history: list[Message] = field(default_factory=list)
    preflight: list[dict] = field(default_factory=list)


RUNTIME: Runtime | None = None


async def preflight(runtime: Runtime) -> list[dict]:
    """Check the three things whose failure looks like an empty store.

    Each returns `(name, ok, detail)` and nothing raises: the page has to render even when
    everything is down, or the operator sees a browser error instead of the diagnosis.
    """
    checks: list[dict] = []

    try:
        await runtime.store.connect()
        held = await runtime.store.count(SESSION)
        checks.append({
            "name": "FalkorDB",
            "ok": True,
            "detail": f"graph {runtime.store.config.graph_name!r}, session {SESSION!r}, "
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


async def _store_view(runtime: Runtime) -> list[dict]:
    """The store, grouped by subject, newest ordinal first within each group.

    Superseded facts are included and marked -- hiding them would hide the design claim.
    """
    facts = await runtime.store.facts_in_session(SESSION)
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


@app.get("/api/state")
async def state() -> JSONResponse:
    runtime = _runtime()
    config = runtime.recall_config
    return JSONResponse({
        "session": SESSION,
        "graph": runtime.store.config.graph_name,
        "embedder": runtime.embed_config.model,
        "model": runtime.llm_config.model,
        "gate": f"{config.gate_on} >= {config.score_floor}",
        "k": config.k,
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
        session_id=SESSION, user_message=message, recent_messages=list(runtime.history)
    )
    result = await recall(context, runtime.recall_config, runtime.store, runtime.embedder)
    yield _sse("recall", {
        "gate_open": result.gate_open,
        "latency_ms": round(result.latency_ms, 1),
        "block": result.injected_block,
        "hits": [_hit(hit, now) for hit in result.memories_used],
    }, t0)

    messages = [{"role": "system", "content": SYSTEM}]
    if result.injected_block:
        messages.append({"role": "system", "content": f"MEMORY:\n{result.injected_block}"})
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
            write = await runtime.write_path.run(SESSION, message, "", history)
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


@app.post("/api/reset")
async def reset() -> JSONResponse:
    runtime = _runtime()
    await runtime.store.clear_session(SESSION)
    runtime.history.clear()
    return JSONResponse({"store": [], "ok": True})
