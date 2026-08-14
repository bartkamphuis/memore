"""Spike arm (a): graphiti-core resolves contradictions itself.

This is the approach `recall-writepath-spec.md` §2.3 specifies and `recall-poc-spec.md`
§4 overrides. The spike exists to decide between them with a number, so the delegated
path has to actually be built and run, not argued about.

Graphiti's ingest pipeline is LLM-driven: `add_episode` extracts entities and edges,
then invalidates edges it judges contradictory. That is the component under test, so it
runs as designed -- pointed at the local Ollama through its OpenAI-compatible endpoint,
because there are no cloud keys here and recall-poc-spec.md §7 forbids a cloud model in
this path anyway.

Two properties of this arm are worth stating plainly, because they bound what the
comparison can claim:

  * Retrieval differs. Graphiti's ingest IS its edge-invalidation, so it cannot be run
    against our FalkorStore. Arm (a) therefore uses graphiti's own hybrid search. A
    delta between arms is attributable to "the whole delegated design vs. the whole
    owned design", not to the resolution rule in isolation.
  * It is slow. Several LLM calls per episode against a local 12B model. The run is
    timeboxed; whatever completes is reported with the fact that it was truncated.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass

from ..assemble import build_block
from ..types import MemoryHit
from . import data as bench_data
from .scoring import normalize_answer, score


@dataclass
class _Cfg:
    episode_batch: int = 10
    per_episode_timeout_s: float = 300.0


async def _build_graphiti(model: str):
    """Wire graphiti-core to FalkorDB + local Ollama."""
    from graphiti_core import Graphiti
    from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
    from graphiti_core.driver.falkordb_driver import FalkorDriver
    from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
    from graphiti_core.llm_client.config import LLMConfig
    from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

    ollama = os.getenv("MEMORE_OLLAMA_URL", "http://localhost:11434")
    base_url = f"{ollama}/v1"
    llm_config = LLMConfig(
        api_key="ollama",
        base_url=base_url,
        model=model,
        small_model=model,
    )
    driver = FalkorDriver(
        host=os.getenv("MEMORE_FALKOR_HOST", "localhost"),
        port=int(os.getenv("MEMORE_FALKOR_PORT", "6379")),
        database="graphiti_arm",
    )
    embedder = OpenAIEmbedder(
        config=OpenAIEmbedderConfig(
            api_key="ollama",
            base_url=base_url,
            embedding_model=os.getenv("MEMORE_EMBED_MODEL", "embeddinggemma:latest"),
            embedding_dim=int(os.getenv("MEMORE_EMBED_DIM", "768")),
        )
    )
    # Every client must be passed explicitly. Graphiti defaults each unset one to an
    # OpenAI client built with no api_key, which raises "Missing credentials" at
    # construction -- including the cross-encoder, which is easy to forget.
    # OpenAIGenericClient (not OpenAIClient) is the one documented for OpenAI-compatible
    # endpoints such as Ollama; it uses json_schema constrained decoding.
    graphiti = Graphiti(
        graph_driver=driver,
        llm_client=OpenAIGenericClient(config=llm_config),
        embedder=embedder,
        cross_encoder=OpenAIRerankerClient(config=llm_config),
    )
    await graphiti.build_indices_and_constraints()
    return graphiti


async def run(
    sample: bench_data.BenchSample,
    *,
    k: int,
    reader,
    candidates,
    limit: int | None,
    ingest_timeout_s: float,
    model: str,
):
    from .run import ArmResult  # local import: run.py imports this module

    result = ArmResult(arm="graphiti", source=sample.source, n_questions=0)
    cfg = _Cfg()

    try:
        graphiti = await _build_graphiti(model)
    except Exception as exc:  # noqa: BLE001
        result.notes.append(f"arm unavailable: {type(exc).__name__}: {exc}")
        return result

    from datetime import UTC, datetime

    from graphiti_core.nodes import EpisodeType

    started = time.perf_counter()
    ingested = 0
    truncated = False
    # Feed the SAME candidate facts, in the same order, as the deterministic arm.
    for i in range(0, len(candidates), cfg.episode_batch):
        if time.perf_counter() - started > ingest_timeout_s:
            truncated = True
            break
        batch = candidates[i : i + cfg.episode_batch]
        body = "\n".join(c.fact for c in batch)
        try:
            # Per-episode timeout, not just an overall one: a single add_episode can
            # grind indefinitely (several LLM calls, each with tenacity retries on schema
            # validation). Without this the outer process timeout kills the run and the
            # arm reports nothing at all -- and "could not ingest one episode in N
            # seconds" is itself the finding Step 0 needs recorded.
            await asyncio.wait_for(
                graphiti.add_episode(
                    name=f"{sample.source}-{i}",
                    episode_body=body,
                    source=EpisodeType.text,
                    source_description="knowledge pool",
                    reference_time=datetime.now(UTC),
                    group_id=f"bench-{sample.source}",
                ),
                timeout=cfg.per_episode_timeout_s,
            )
            ingested += len(batch)
        except TimeoutError:
            result.notes.append(
                f"episode at fact {i} ({len(batch)} facts) exceeded "
                f"{cfg.per_episode_timeout_s:.0f}s; arm abandoned"
            )
            truncated = True
            break
        except Exception as exc:  # noqa: BLE001
            result.notes.append(f"ingest error at {i}: {type(exc).__name__}: {exc}")
            break
    result.ingest_seconds = time.perf_counter() - started
    result.facts_ingested = ingested
    if truncated:
        result.notes.append(
            f"ingest timeboxed at {ingest_timeout_s:.0f}s after {ingested}/{len(candidates)} facts"
        )

    pairs = list(zip(sample.questions, sample.answers, strict=True))
    if limit:
        pairs = pairs[:limit]
    result.n_questions = len(pairs)

    started = time.perf_counter()
    acc = em = ret = 0.0
    for index, (question, gold) in enumerate(pairs):
        try:
            edges = await graphiti.search(question, group_ids=[f"bench-{sample.source}"], num_results=k)
        except Exception as exc:  # noqa: BLE001
            result.notes.append(f"search error: {type(exc).__name__}: {exc}")
            break
        hits = [
            MemoryHit(
                fact=e.fact,
                score=1.0 - (rank / max(1, len(edges))),
                valid_at=e.valid_at,
                invalid_at=e.invalid_at,
                source_episode_id=str(getattr(e, "episodes", [""])[0] if getattr(e, "episodes", None) else ""),
            )
            for rank, e in enumerate(edges)
        ]
        live = [h for h in hits if h.invalid_at is None]
        if live and any(normalize_answer(g) in normalize_answer(live[0].fact) for g in gold):
            ret += 1.0
        if reader is not None:
            answer = await reader.answer(question, build_block(hits))
            metrics = score(answer, gold)
            acc += metrics["substring_exact_match"]
            em += metrics["exact_match"]
            if index < 5:
                result.samples.append(
                    {
                        "question": question,
                        "gold": gold,
                        "answer": answer,
                        "top_live": live[0].fact if live else None,
                    }
                )
    result.query_seconds = time.perf_counter() - started
    n = max(1, result.n_questions)
    result.accuracy = acc / n
    result.exact_match = em / n
    result.retrieval_hit = ret / n
    result.notes.append(
        "retrieval differs from the deterministic arm (graphiti's own hybrid search); "
        "the delta compares whole designs, not the resolution rule in isolation"
    )
    try:
        await graphiti.close()
    except Exception:  # noqa: BLE001
        pass
    return result
