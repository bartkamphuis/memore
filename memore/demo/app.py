"""The FastAPI app: three routes, one page, no streaming.

Deliberately plain. A turn completes atomically and returns the whole trace in one JSON
body, because the trace is more legible when it arrives assembled than when it races the
reply token by token -- and because SSE would add a moving part that teaches nothing about
memore.

## Startup is where a demo of this system actually fails

Three failures produce a blank page rather than an error, and all three are ordinary:
FalkorDB not running, Ollama not serving the models `config.py` pins, and an existing graph
whose vector index was built at a different width (`FalkorStore.connect()` raises on that
deliberately -- see the invariant in CLAUDE.md). `preflight()` checks all three and the page
renders the result, because a demo that shows an empty store when the store is *down*
teaches the opposite of what it is for.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import httpx
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
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


@app.post("/api/turn")
async def turn(body: Turn) -> JSONResponse:
    """One turn: recall -> reply -> write. The order is the whole architecture.

    Recall runs BEFORE the model call and its result is injected at prompt-assembly time,
    rather than being offered as a tool the model may decide to call. The write path runs
    after, and is handed the assistant's reply as CONTEXT only -- never as text to extract
    from (RESULTS.md §17).
    """
    from datetime import UTC, datetime

    runtime = _runtime()
    message = body.message.strip()
    if not message:
        return JSONResponse({"error": "empty message"}, status_code=400)

    trace: dict[str, Any] = {}
    now = datetime.now(UTC)

    context = TurnContext(
        session_id=SESSION, user_message=message, recent_messages=list(runtime.history)
    )
    result = await recall(context, runtime.recall_config, runtime.store, runtime.embedder)
    trace["recall"] = {
        "gate_open": result.gate_open,
        "latency_ms": round(result.latency_ms, 1),
        "block": result.injected_block,
        "hits": [_hit(hit, now) for hit in result.memories_used],
    }

    messages = [{"role": "system", "content": SYSTEM}]
    if result.injected_block:
        messages.append({"role": "system", "content": f"MEMORY:\n{result.injected_block}"})
    messages += [{"role": m.role, "content": m.content} for m in runtime.history[-6:]]
    messages.append({"role": "user", "content": message})
    try:
        reply = await runtime.chat.chat(messages, max_tokens=300)
    except httpx.HTTPError as exc:
        reply = f"[the model is unreachable: {type(exc).__name__}]"
        logger.exception("chat failed")

    write = await runtime.write_path.run(SESSION, message, reply, list(runtime.history))
    trace["write"] = {
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
    }

    runtime.history.append(Message(role="user", content=message))
    runtime.history.append(Message(role="assistant", content=reply))
    trace["reply"] = reply
    trace["store"] = await _store_view(runtime)
    return JSONResponse(trace)


@app.post("/api/reset")
async def reset() -> JSONResponse:
    runtime = _runtime()
    await runtime.store.clear_session(SESSION)
    runtime.history.clear()
    return JSONResponse({"store": [], "ok": True})
