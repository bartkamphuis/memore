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

    async def chat(
        self,
        messages: list[dict[str, str]],
        schema: dict | None = None,
        max_tokens: int | None = None,
    ) -> str:
        options: dict = {"temperature": self.config.temperature, "num_ctx": self.config.num_ctx}
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        payload: dict = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "options": options,
            "think": False,
            "keep_alive": self.config.keep_alive,
        }
        if schema is not None:
            payload["format"] = schema

        logger.debug(payload)
        resp = await self._client.post(f"{self.config.base_url}/api/chat", json=payload)
        resp.raise_for_status()
        logger.debug(resp.json())
        return resp.json()["message"]["content"]

    async def chat_json(self, messages: list[dict[str, str]], schema: dict) -> dict | list:
        raw = await self.chat(messages, schema=schema)
        return json.loads(raw)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
