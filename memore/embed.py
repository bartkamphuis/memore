"""Embedding -- component A's only cost (recall-stage-spec.md §4).

The embedder is a *caller* concern, not a store concern: `hybrid_search` takes a
`query_vec` because §3.2 forbids the store from embedding on its own.
"""

from __future__ import annotations

import math
from typing import Protocol

import httpx

from .config import DEFAULT_KEEP_ALIVE, EmbedConfig


class Embedder(Protocol):
    async def embed(self, texts: list[str], *, query: bool = False) -> list[list[float]]: ...


def normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return list(vec)
    return [x / norm for x in vec]


def blend(msg_vec: list[float], summary_vec: list[float] | None, alpha: float) -> list[float]:
    """recall-stage-spec.md §4: query_vec = normalize(a*msg_vec + (1-a)*summary_vec)."""
    if summary_vec is None:
        return normalize(msg_vec)
    return normalize([alpha * m + (1.0 - alpha) * s for m, s in zip(msg_vec, summary_vec, strict=True)])


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


class OllamaEmbedder:
    """Local embedding model, kept warm. Must stay local -- see §4's latency ceiling and
    the no-cloud-LLM invariant in recall-poc-spec.md §7."""

    def __init__(self, config: EmbedConfig | None = None, client: httpx.AsyncClient | None = None):
        self.config = config or EmbedConfig.from_env()
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(600.0))
        self._owns_client = client is None

    async def embed(self, texts: list[str], *, query: bool = False) -> list[list[float]]:
        """`query=True` marks a retrieval query rather than a stored fact.

        Asymmetric models encode the two differently; for symmetric ones both prefixes
        are empty and this is a no-op.
        """
        if not texts:
            return []
        prefix = self.config.query_prefix if query else self.config.document_prefix
        payload = [prefix + t for t in texts] if prefix else texts
        resp = await self._client.post(
            f"{self.config.ollama_url}/api/embed",
            # No `options`: the embedder is served at its own context width and sending a
            # different one would reload it. `keep_alive` preserves the operator's pin --
            # it is per-request, so omitting it would silently impose the 5-minute default
            # on a model deliberately kept resident (config.DEFAULT_KEEP_ALIVE).
            json={
                "model": self.config.model,
                "input": payload,
                "keep_alive": DEFAULT_KEEP_ALIVE,
            },
        )
        resp.raise_for_status()
        return [normalize(v) for v in resp.json()["embeddings"]]

    async def embed_one(self, text: str, *, query: bool = False) -> list[float]:
        return (await self.embed([text], query=query))[0]

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


class StubEmbedder:
    """Deterministic fake for tests -- a fixed vector per string, no model required
    (recall-stage-test-spec.md fixtures: `local_embedder_stub`)."""

    def __init__(self, dimension: int = 8):
        self.dimension = dimension
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str], *, query: bool = False) -> list[list[float]]:
        self.calls.append(list(texts))
        out = []
        for text in texts:
            seed = sum((i + 1) * ord(c) for i, c in enumerate(text))
            out.append(normalize([float((seed >> (i * 3)) % 97) + 1.0 for i in range(self.dimension)]))
        return out

    async def embed_one(self, text: str, *, query: bool = False) -> list[float]:
        return (await self.embed([text], query=query))[0]
