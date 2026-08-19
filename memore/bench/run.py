"""Step-0 spike runner (recall-poc-spec.md §3).

Measures FactConsolidation under two arms that differ in ONE thing -- who resolves a
contradiction:

  deterministic  our freshness-ordinal primitive (recall-poc-spec.md §4);
                 FalkorDB is storage + hybrid retrieval only.
  graphiti       graphiti-core's own edge-invalidation at ingest time
                 (recall-writepath-spec.md §2.3, the approach the PoC spec overrides).

Both arms consume the identical cached `CandidateFact` list and the identical reader, so
extraction quality and reader quality cancel out. What does NOT cancel out is retrieval:
graphiti's ingest *is* its edge-invalidation, so it cannot be run against our store.
That confound is real and is reported alongside the numbers rather than buried.

Metrics per arm:
  accuracy          MemoryAgentBench's own `substring_exact_match` over the reader's
                    answer -- the number comparable to the field's ~54% single-hop.
  retrieval_hit     does the top-ranked LIVE recalled fact contain the gold answer.
                    No reader involved, so it isolates store+consolidation from reader
                    error. The right question for single-hop, where one fact answers
                    outright.
  retrieval_any     does ANY live injected fact carry it. The right question for
                    multi-hop, where the answer arrives as the tail of a chain walk and
                    is deliberately ranked below the seeds that found it -- top-1 is
                    0.000 there by construction (RESULTS.md §8).
  *_clean           both, restricted to questions whose gold string is carried by exactly
                    one fact in the corpus. For the rest it appears in up to 5+ facts
                    ("Italy") and a retrieval metric is close to trivially satisfiable.

Three flags change what is measured rather than how: `--via-recall` routes through the
real recall stage instead of raw top-k, `--expansion-hops` turns on the chain walk, and
`--no-context` answers with no recalled block at all -- the parametric-knowledge floor,
without which a retrieval number cannot be interpreted.

## `--via-recall` and what it does and does not cover (RESULTS.md §21.3, §23)

§21.3 names the largest untested surface in the shipped read path: no headline number
routes through `recall()`, so `score_floor`, the lookup timeout and the failure-safety
path are in none of them. `--via-recall` is the mechanism; running it is the item.

Under `--via-recall` the raw `hybrid_search` above is **still performed, and is the
control**, not waste. It is the same retrieval `recall()` itself performs -- `embed_one`
normalizes and `blend(v, None, alpha)` is idempotent, so the two query vectors are
identical, and `RecallConfig.k` is `--k`. That makes it the right comparator for the one
number this item exists to produce:

    gate_shut_but_retrievable   the gate returned nothing on a question whose gold WAS
                                in the store's live top-k. The gate's cost in recall,
                                measured against what the same lookup would have found.

**Pre-registered expectation, written before the first run.** FactConsolidation questions
are short, third-person and lexically near-identical to their answer facts ("What is the
capital of Italy?" against "The capital of Italy is Rome."), so the gate should open on
nearly all of single-hop and `gate_shut_but_retrievable` should be small. If it is, that
is NOT a validation of `score_floor`: it is §11's argument applied to the gate -- a
corpus that cannot express the failure the floor exists to prevent cannot be evidence
about the floor. Only a materially non-zero cost would be a finding about 0.57 itself.

Multi-hop is the opposite and is already measured: §8's arm B ran `recall()` gate-only on
mh_6k and `retrieval_any` fell 0.400 -> 0.050, for the structural reason the chain-walk
invariant states -- a multi-hop answer shares no entity with the question, so it cannot
clear a similarity floor and must not be asked to.

Three surfaces, and this covers one. `score_floor` is genuinely exercised. The lookup
timeout is only *touched*: Falkor answers in 2.6-4.3ms against a 180ms timeout, so zero
timeouts is the expected result and is not evidence that the timeout path works. Failure
safety is not exercised at all -- it stays covered by `FailingStore`/`SlowStore` in the
unit suite. The counters below exist to make that distinction checkable rather than
assumed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..assemble import build_block
from ..config import (
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_NUM_CTX,
    EmbedConfig,
    RecallConfig,
    StoreConfig,
)
from ..consolidate import DeterministicConsolidator
from ..embed import OllamaEmbedder
from ..recall import recall
from ..store.falkor import FalkorStore
from ..types import ConsolidationCase, MemoryHit, TurnContext
from . import data as bench_data
from .extract import CACHED_EXTRACTOR_MODEL, KnowledgePoolExtractor
from .reader import Reader
from .scoring import normalize_answer, score


@dataclass
class ArmResult:
    arm: str
    source: str
    n_questions: int
    accuracy: float = 0.0
    exact_match: float = 0.0
    retrieval_hit: float = 0.0
    # Multi-hop needs a different retrieval metric than single-hop. `retrieval_hit` asks
    # whether the TOP live fact carries the gold, which is the right question when one
    # fact answers the question outright. A chained answer arrives as the tail of a
    # walk, deliberately ranked below the seeds that found it, so top-1 is 0.000 by
    # construction. What matters is whether the reader was given the answer at all.
    retrieval_any: float = 0.0
    # Same two metrics restricted to the unambiguous questions.
    n_clean: int = 0
    retrieval_any_clean: float = 0.0
    exact_match_clean: float = 0.0
    facts_ingested: int = 0
    facts_stored: int = 0
    cases: dict[str, int] = field(default_factory=dict)
    ingest_seconds: float = 0.0
    query_seconds: float = 0.0
    # --via-recall only (RESULTS.md §23). All default to the inert value so a raw top-k
    # run serializes exactly as it did before this existed.
    #
    # `gate_open_rate` is how often the shipped gate let anything through.
    # `gate_shut_but_retrievable` is the one that matters: gate shut on a question whose
    # gold was in the control lookup's LIVE hits. `_live_first` on both sides on purpose
    # -- a gate that shuts on a question answerable only by a SUPERSEDED fact cost
    # nothing, and counting it would credit the gate with a loss it did not cause.
    gate_open_rate: float = 0.0
    gate_shut_but_retrievable: float = 0.0
    # A-D latency as `recall()` itself measures it, which is the first such figure taken
    # on the bench corpora rather than on calibration fixtures. Reported apart from
    # `query_seconds`, which now also carries the control lookup and the reader.
    recall_p95_ms: float = 0.0
    # Zero is the EXPECTED result for both, not a passing grade -- see the module
    # docstring. They exist so that "the timeout never fired" is a checked statement
    # rather than an assumption.
    lookup_timeouts: int = 0
    lookup_failures: int = 0
    notes: list[str] = field(default_factory=list)
    samples: list[dict] = field(default_factory=list)


def _live_first(hits: list[MemoryHit]) -> list[MemoryHit]:
    return [h for h in hits if h.invalid_at is None]


class _DegradationCounter(logging.Handler):
    """Counts the two ways `recall()` degrades to a closed gate without raising.

    `recall()` never raises by contract (§3.1), so a store that timed out and a store
    that answered are indistinguishable from the return value -- both give
    `gate_open=False`. Without this, a run in which the store fell over on every question
    would report a gate that simply never opened, which is the same silent-failure shape
    `FalkorStore._assert_vector_dimension` exists to catch.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.timeouts = 0
        self.failures = 0

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if "lookup timed out" in message:
            self.timeouts += 1
        elif "lookup failed" in message:
            self.failures += 1


async def run_deterministic(
    sample: bench_data.BenchSample,
    *,
    k: int,
    reader: Reader | None,
    embedder: OllamaEmbedder,
    store: FalkorStore,
    candidates,
    limit: int | None,
    recall_config: RecallConfig | None = None,
    no_context: bool = False,
) -> ArmResult:
    session = f"bench-{sample.source}"
    result = ArmResult(arm="deterministic", source=sample.source, n_questions=0)

    await store.connect()
    await store.clear_session(session)
    consolidator = DeterministicConsolidator(store, embedder)

    started = time.perf_counter()
    # Facts arrive in serial order; the ordinal is arrival order, never read from input.
    # The chunk batches the EMBEDDER only -- consolidation is one fact at a time, because
    # a `consolidate()` batch means "one utterance" and withholds contradictions inside
    # itself (RESULTS.md §16). Every fact here is its own turn.
    chunk = 50
    counts: dict[str, int] = {c.value: 0 for c in ConsolidationCase}
    for i in range(0, len(candidates), chunk):
        window = candidates[i : i + chunk]
        await consolidator.prewarm([c.fact for c in window])
        for candidate in window:
            for outcome in await consolidator.consolidate(session, [candidate]):
                counts[outcome.case.value] += 1
    result.ingest_seconds = time.perf_counter() - started
    result.cases = counts
    result.facts_ingested = len(candidates)
    result.facts_stored = await store.count(session)

    pairs = list(zip(sample.questions, sample.answers, strict=True))
    if limit:
        pairs = pairs[:limit]
    result.n_questions = len(pairs)

    # A question is "clean" when exactly one fact in the corpus carries its gold string.
    # For the rest the gold appears in up to 5+ facts ("Italy"), so a retrieval metric is
    # close to trivially satisfiable and the aggregate flatters itself. Reported split.
    corpus = [f.text for f in sample.facts]
    clean_flags = [
        sum(1 for t in corpus if normalize_answer(g[0]) in normalize_answer(t)) == 1
        for g in [a for a in sample.answers]
    ]
    result.n_clean = sum(1 for i, _ in enumerate(pairs) if clean_flags[i])

    started = time.perf_counter()
    acc = em = ret = ret_any = 0.0
    em_clean = ret_any_clean = 0.0
    gate_open = gate_shut_retrievable = 0
    recall_latencies: list[float] = []
    degradations = _DegradationCounter()
    recall_log = logging.getLogger("memore.recall")
    if recall_config is not None:
        recall_log.addHandler(degradations)
    for index, (question, gold) in enumerate(pairs):
        clean = clean_flags[index]
        if no_context:
            # Parametric-knowledge control: the reader answers with no recalled block at
            # all. Without this floor, a retrieval number is uninterpretable -- the
            # reader knows real-world facts, and this corpus deliberately corrupts them,
            # so some questions are answerable (and some mis-answerable) from weights.
            hits = []
        else:
            query_vec = await embedder.embed_one(question, query=True)
            hits = await store.hybrid_search(question, query_vec, session, k)
        if not no_context and recall_config is not None:
            # Route through recall(): the shipped path, gate first and chain expansion
            # after it, under one budget. Kept opt-in because the default path above is
            # raw top-k with no gate, which is what every existing number in RESULTS.md
            # was measured with -- switching it silently would break comparability.
            # With expansion_hops=0 this isolates the gate's contribution from the
            # walk's, which is the only way `0.40 -> 0.91` means anything.
            #
            # The raw hits computed just above are NOT discarded: they are the control
            # this arm is measured against (see the module docstring). Same query vector,
            # same k, same store -- so the only difference between them and what the gate
            # returns is the gate.
            control_live = _live_first(hits)
            recalled = await recall(
                TurnContext(session_id=session, user_message=question),
                recall_config,
                store,
                embedder,
            )
            hits = recalled.memories_used
            recall_latencies.append(recalled.latency_ms)
            if recalled.gate_open:
                gate_open += 1
            elif any(
                normalize_answer(g) in normalize_answer(h.fact)
                for h in control_live
                for g in gold
            ):
                # The gate shut on a question the same lookup COULD have answered. This
                # is the gate's cost in recall, and the whole reason §21.3 is an item.
                gate_shut_retrievable += 1
        live = _live_first(hits)
        if live and any(normalize_answer(g) in normalize_answer(live[0].fact) for g in gold):
            ret += 1.0
        # LIVE only. The corpus holds both "The capital of Italy is Rome." and its
        # superseded "...is Duluth.", so counting any hit would hand free credit to the
        # stale value -- the one thing this benchmark exists to punish.
        if any(normalize_answer(g) in normalize_answer(h.fact) for h in live for g in gold):
            ret_any += 1.0
            if clean:
                ret_any_clean += 1.0
        block = build_block(hits)
        if reader is not None:
            answer = await reader.answer(question, block)
            metrics = score(answer, gold)
            acc += metrics["substring_exact_match"]
            em += metrics["exact_match"]
            if clean:
                em_clean += metrics["exact_match"]
            # Keep a head sample plus every imperfect answer: a headline SubEM near 1.0
            # is only trustworthy if the cases where exact match disagrees are readable.
            if index < 5 or metrics["exact_match"] < 1.0:
                result.samples.append(
                    {
                        "question": question,
                        "gold": gold,
                        "answer": answer,
                        "top_live": live[0].fact if live else None,
                        "exact_match": metrics["exact_match"],
                        "substring_exact_match": metrics["substring_exact_match"],
                    }
                )
    result.query_seconds = time.perf_counter() - started
    n = max(1, result.n_questions)
    result.accuracy = acc / n
    result.exact_match = em / n
    result.retrieval_hit = ret / n
    result.retrieval_any = ret_any / n
    c = max(1, result.n_clean)
    result.retrieval_any_clean = ret_any_clean / c
    result.exact_match_clean = em_clean / c
    if recall_config is not None:
        recall_log.removeHandler(degradations)
        result.gate_open_rate = gate_open / n
        result.gate_shut_but_retrievable = gate_shut_retrievable / n
        result.lookup_timeouts = degradations.timeouts
        result.lookup_failures = degradations.failures
        if recall_latencies:
            ordered = sorted(recall_latencies)
            result.recall_p95_ms = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
    return result


async def run_graphiti(
    sample: bench_data.BenchSample,
    *,
    k: int,
    reader: Reader | None,
    candidates,
    limit: int | None,
    ingest_timeout_s: float,
    model: str,
) -> ArmResult:
    """Arm (a): let graphiti-core's edge-invalidation resolve contradictions."""
    from .graphiti_arm import run as run_arm

    return await run_arm(
        sample,
        k=k,
        reader=reader,
        candidates=candidates,
        limit=limit,
        ingest_timeout_s=ingest_timeout_s,
        model=model,
    )


async def main_async(args: argparse.Namespace) -> None:
    sample = bench_data.load(args.source)
    extractor = KnowledgePoolExtractor(model=args.extractor_model, concurrency=args.concurrency)
    print(f"[{sample.source}] {len(sample.facts)} facts, {len(sample.questions)} questions")

    started = time.perf_counter()
    candidates = await extractor.candidates_for(sample.source, sample.facts)
    distinct = len({c.subject_hint for c in candidates})
    print(
        f"  extraction: {len(candidates)} candidates, {distinct} distinct subjects "
        f"({time.perf_counter() - started:.1f}s)"
    )
    await extractor.aclose()

    if args.fact_limit:
        candidates = candidates[: args.fact_limit]

    reader = None if args.no_reader else Reader(model=args.reader_model, num_ctx=args.num_ctx)
    embed_config = EmbedConfig.from_env()
    embedder = OllamaEmbedder(embed_config)
    store = FalkorStore(StoreConfig.from_env(), dimension=embed_config.dimension)

    results: list[ArmResult] = []
    if args.arm in ("deterministic", "both"):
        results.append(
            await run_deterministic(
                sample,
                k=args.k,
                reader=reader,
                embedder=embedder,
                store=store,
                candidates=candidates,
                limit=args.limit,
                recall_config=(
                    RecallConfig(
                        k=args.k,
                        expansion_hops=args.expansion_hops,
                        expansion_fanout=args.expansion_fanout,
                        inject_token_budget=args.inject_token_budget,
                        subject_check=not args.no_subject_check,
                    )
                    if (args.via_recall or args.expansion_hops)
                    else None
                ),
                no_context=args.no_context,
            )
        )
    if args.arm in ("graphiti", "both"):
        results.append(
            await run_graphiti(
                sample,
                k=args.k,
                reader=reader,
                candidates=candidates,
                limit=args.limit,
                ingest_timeout_s=args.graphiti_timeout,
                model=args.reader_model,
            )
        )

    for result in results:
        print(f"\n=== arm: {result.arm} / {result.source} ===")
        print(f"  facts ingested {result.facts_ingested} -> stored {result.facts_stored}")
        print(f"  cases          {result.cases}")
        print(f"  accuracy (SubEM) {result.accuracy:.3f}   exact_match {result.exact_match:.3f}")
        print(f"  retrieval_hit    {result.retrieval_hit:.3f}  (top live fact contains gold)")
        print(f"  retrieval_any    {result.retrieval_any:.3f}  (gold in any LIVE injected fact)")
        print(
            f"  clean subset     n={result.n_clean}  retrieval_any {result.retrieval_any_clean:.3f}"
            f"   exact_match {result.exact_match_clean:.3f}"
        )
        print(f"  ingest {result.ingest_seconds:.1f}s   query {result.query_seconds:.1f}s")
        if result.arm == "deterministic" and (args.via_recall or args.expansion_hops):
            # RESULTS.md §23. `gate shut w/ answer` is the number this arm exists for;
            # the rest is context for reading it. Timeouts/failures print even at 0
            # deliberately -- a zero that is never shown cannot be distinguished from a
            # counter nobody wired up.
            print(
                f"  gate opened      {result.gate_open_rate:.3f}"
                f"   gate shut w/ answer in store {result.gate_shut_but_retrievable:.3f}"
            )
            print(
                f"  recall() A-D p95 {result.recall_p95_ms:.1f}ms"
                f"   lookup timeouts {result.lookup_timeouts}"
                f"   lookup failures {result.lookup_failures}"
            )
        for note in result.notes:
            print(f"  note: {note}")

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([asdict(r) for r in results], indent=2))
        print(f"\nwrote {path}")

    if reader is not None:
        await reader.aclose()
    await embedder.aclose()
    await store.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(description="MemoryAgentBench FactConsolidation spike")
    parser.add_argument("--source", default="factconsolidation_sh_6k")
    parser.add_argument("--arm", default="deterministic", choices=["deterministic", "graphiti", "both"])
    parser.add_argument("--k", type=int, default=12)
    parser.add_argument("--limit", type=int, default=None, help="cap questions")
    parser.add_argument("--fact-limit", type=int, default=None, help="cap ingested facts")
    parser.add_argument("--no-reader", action="store_true", help="retrieval-only, no LLM answer step")
    parser.add_argument("--reader-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--extractor-model", default=CACHED_EXTRACTOR_MODEL)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--num-ctx", type=int, default=DEFAULT_LLM_NUM_CTX)
    parser.add_argument("--graphiti-timeout", type=float, default=1800.0)
    parser.add_argument(
        "--expansion-hops", type=int, default=0, help="multi-hop chain expansion depth (0 = off)"
    )
    parser.add_argument("--expansion-fanout", type=int, default=4)
    parser.add_argument(
        "--no-subject-check",
        action="store_true",
        help="disable the §9 subject admission rule inside recall() (--via-recall only). "
        "Separates what an OPEN gate discarded from what the gate shut out, and is the "
        "only way to price the check: it is O(session) and B6 proposes retiring it.",
    )
    parser.add_argument(
        "--via-recall",
        action="store_true",
        help="route retrieval through recall() (gate + budget) instead of raw top-k",
    )
    parser.add_argument(
        "--no-context",
        action="store_true",
        help="parametric control: reader answers with no recalled block (the floor)",
    )
    parser.add_argument("--inject-token-budget", type=int, default=RecallConfig.inject_token_budget)
    parser.add_argument("--out", default=None)
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
