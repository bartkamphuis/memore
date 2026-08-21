"""Local LLM access via Ollama.

Where an LLM is allowed: P1 extraction (write path, off the response path) and the bench
reader. Where it is forbidden: recall components A-D, and the consolidation decision
(recall-poc-spec.md §7). Nothing in this module is imported by `memore.recall` or
`memore.consolidate`, and that is intentional -- keep it that way.
"""

from __future__ import annotations

import json
import logging
import logging.config
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

import httpx

from .config import DEFAULT_KEEP_ALIVE, DEFAULT_LLM_MODEL, DEFAULT_LLM_NUM_CTX

LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s [in %(filename)s:%(lineno)d]'
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'default': {
            'format': LOG_FORMAT,
            'datefmt': '%Y-%m-%d %H:%M:%S',
        },
    },
    'handlers': {
        'file': {
            'class': 'logging.handlers.TimedRotatingFileHandler',
            'level': 'DEBUG',
            'formatter': 'default',
            'filename': Path('logs/app.log'),
            'when': 'midnight',
            'interval': 1,
            'backupCount': 30,
            'encoding': 'utf8',
        },
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'default',
            'stream': 'ext://sys.stdout',
        },
    },
    'root': {
        'level': 'DEBUG',
        'handlers': ['file', 'console'],
    },
    'loggers': {
        'httpcore': {
            'level': 'WARNING',
        },
        # httpx logs one INFO line per request. The embedder issues one per batch, so at
        # bench scale this buries the result the run exists to print.
        'httpx': {
            'level': 'WARNING',
        },
    },
}

# dictConfig opens the file handler at import time and will NOT create its directory, so a
# fresh clone would fail to import this module -- and everything that touches P1 imports it.
Path('logs').mkdir(exist_ok=True)
logging.config.dictConfig(LOGGING_CONFIG)

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class LLMConfig:
    """See `config.DEFAULT_LLM_MODEL` for why the model and `num_ctx` travel together."""

    base_url: str = "http://localhost:11434"
    model: str = DEFAULT_LLM_MODEL
    temperature: float = 0.0
    num_ctx: int = DEFAULT_LLM_NUM_CTX
    keep_alive: int | str = DEFAULT_KEEP_ALIVE


class OllamaClient:
    def __init__(self, config: LLMConfig | None = None, client: httpx.AsyncClient | None = None):
        self.config = config or LLMConfig()
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(1800.0))
        self._owns_client = client is None

    def _payload(
        self,
        messages: list[dict[str, str]],
        schema: dict | None,
        max_tokens: int | None,
        *,
        stream: bool,
    ) -> dict:
        """The single place that knows what an Ollama chat request looks like here.

        Extracted rather than copied into the streaming path, and `num_ctx` and
        `keep_alive` are why. Ollama reloads a model when a request asks for options it was
        not loaded with, and `keep_alive` is **per-request** -- so a second call site that
        forgot it would silently replace an operator's `Forever` pin with the 5-minute
        default and make every later process start pay an 18GB reload. See CLAUDE.md,
        "Match the served models".
        """
        options: dict = {"temperature": self.config.temperature, "num_ctx": self.config.num_ctx}
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        payload: dict = {
            "model": self.config.model,
            "messages": messages,
            "stream": stream,
            "options": options,
            "think": False,
            "keep_alive": self.config.keep_alive,
        }
        if schema is not None:
            payload["format"] = schema
        return payload

    async def chat(
        self,
        messages: list[dict[str, str]],
        schema: dict | None = None,
        max_tokens: int | None = None,
    ) -> str:
        payload = self._payload(messages, schema, max_tokens, stream=False)
        logger.debug(payload)
        resp = await self._client.post(f"{self.config.base_url}/api/chat", json=payload)
        resp.raise_for_status()
        logger.debug(resp.json())
        return resp.json()["message"]["content"]

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Yield content deltas as they arrive. Same payload builder as `chat`.

        Exists for `memore.demo`, which has to show that recall lands in ~80ms and the
        model takes seconds -- a claim a blocking call cannot make visible. Nothing on the
        write path or the bench uses it: P1 needs a whole JSON document before it can be
        parsed, so streaming there would buy nothing.

        Ollama's final frame carries `done: true` and no content; it is consumed and not
        yielded, so a caller never sees an empty delta at the end.
        """
        payload = self._payload(messages, None, max_tokens, stream=True)
        async with self._client.stream(
            "POST", f"{self.config.base_url}/api/chat", json=payload
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                frame = json.loads(line)
                if frame.get("done"):
                    break
                delta = frame.get("message", {}).get("content", "")
                if delta:
                    yield delta

    async def chat_json(self, messages: list[dict[str, str]], schema: dict) -> dict | list:
        raw = await self.chat(messages, schema=schema)
        return json.loads(raw)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
