"""Run the consolidation oracle against a store that has already been ingested.

    uv run python -m memore.bench.oracle_run --source factconsolidation_sh_6k

Ingests the corpus through the deterministic consolidator, then scores the decision
directly -- no retrieval, no reader, no LLM (the cached subject extraction aside).
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path

from ..aliases import AliasConfig
from ..config import EmbedConfig, StoreConfig
from ..consolidate import ConsolidationConfig, DeterministicConsolidator
from ..embed import OllamaEmbedder
from ..store.falkor import FalkorStore
from ..types import ConsolidationCase
from . import data as bench_data
from .extract import CACHED_EXTRACTOR_MODEL, KnowledgePoolExtractor
from .oracle import evaluate


async def main_async(args: argparse.Namespace) -> None:
    sample = bench_data.load(args.source)
    extractor = KnowledgePoolExtractor(model=args.extractor_model, concurrency=args.concurrency)
    subjects = await extractor.subjects_for(sample.source, sample.facts)
    candidates = await extractor.candidates_for(sample.source, sample.facts)
    await extractor.aclose()

    embed_config = EmbedConfig.from_env()
    store = FalkorStore(StoreConfig.from_env(), dimension=embed_config.dimension)
    embedder = OllamaEmbedder(embed_config)
    session = f"oracle-{sample.source}"
    await store.connect()
    await store.clear_session(session)

    alias = AliasConfig(
        enabled=not args.no_alias,
        df_ratio=args.alias_df_ratio,
        min_subjects=args.alias_min_subjects,
    )
    consolidator = DeterministicConsolidator(
        store, embedder, ConsolidationConfig(alias=alias)
    )
    counts = {c.value: 0 for c in ConsolidationCase}
    # Chunked for the EMBEDDER, consolidated one fact at a time. Every FactConsolidation
    # fact is its own turn with a real arrival order, and `consolidate()` reads a batch as
    # one utterance whose candidates cannot supersede each other (RESULTS.md §16) -- so
    # handing it 50 at a time silently asserted a simultaneity the corpus does not have.
    # It cost 16 supersedes at 32k and 18 at 6k before this was split.
    for i in range(0, len(candidates), 50):
        chunk = candidates[i : i + 50]
        await consolidator.prewarm([c.fact for c in chunk])
        for candidate in chunk:
            for outcome in await consolidator.consolidate(session, [candidate]):
                counts[outcome.case.value] += 1

    live = await store.live_fact_texts(session)
    # The SAME alias config the ingest ran with. Scoring groups the store never built
    # measures a system that was never run (see `oracle.build_groups`).
    result = evaluate(sample.facts, subjects, sample.questions, sample.answers, live, alias)
    vocabulary = consolidator.vocabulary(session)
    merges = vocabulary.merges if vocabulary else []

    print(f"[{sample.source}] {len(sample.facts)} facts -> {result.n_groups} subject groups "
          f"({result.n_multi_fact_groups} with >1 fact)")
    print(f"  consolidation cases: {counts}")
    print(f"  live facts in store: {len(live)}")
    print()
    print(f"  ORACLE consolidation accuracy: {result.accuracy:.3f} "
          f"({result.consolidation_correct}/{result.n_questions})")
    print(f"    matched to a subject group : {result.matched}")
    print(f"    unmatched questions        : {result.unmatched}")
    print(f"    gold fact wrongly superseded (over-merge): {result.gold_fact_superseded}")
    print(f"    groups left with >1 live fact            : {result.stale_live}")

    # Printed in full, never sampled. An aggregate score cannot distinguish a merge that
    # fixed an under-merge from one that destroyed a correct fact, which is precisely why
    # RESULTS.md §3 refused to accept the ungated rule on its score.
    print(f"\n  subject aliases merged: {len(merges)}"
          f" (df_ratio={alias.df_ratio}, min_subjects={alias.min_subjects}, "
          f"enabled={alias.enabled})")
    for arriving, target in merges:
        print(f"   - {arriving!r}\n     -> {target!r}")

    if result.failures:
        print(f"\n  first {min(args.show, len(result.failures))} failures:")
        for failure in result.failures[: args.show]:
            print(f"   - [{failure['kind']}] {failure['question']}")
            print(f"       gold={failure['gold']}")
            if "newest_fact" in failure:
                print(f"       subject={failure['matched_subject']!r} (score {failure['match_score']}, "
                      f"{failure['group_size']} facts)")
                print(f"       newest={failure['newest_fact']!r} live={failure['newest_is_live']}")
                for text in failure["live_in_group"][:3]:
                    print(f"       live  = {text!r}")

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"cases": counts, "alias": asdict(alias), "merges": merges, **asdict(result)},
                indent=2,
            )
        )
        print(f"\nwrote {path}")

    await embedder.aclose()
    await store.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(description="Consolidation oracle")
    parser.add_argument("--source", default="factconsolidation_sh_6k")
    parser.add_argument("--extractor-model", default=CACHED_EXTRACTOR_MODEL)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--show", type=int, default=8)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--no-alias", action="store_true", help="disable DF-gated subject aliasing"
    )
    parser.add_argument("--alias-df-ratio", type=float, default=AliasConfig.df_ratio)
    parser.add_argument("--alias-min-subjects", type=int, default=AliasConfig.min_subjects)
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
