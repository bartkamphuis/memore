"""Terminal harness -- the PoC's deliverable (recall-poc-spec.md §2, §6).

Feed turns in as text; the memory behaviour prints out. Every turn runs the read path
first (what would have been injected into the prompt), then the write path (what the turn
taught the store). The target trace is recall-poc-spec.md §6.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from .config import EmbedConfig, RecallConfig, StoreConfig, WritePathConfig
from .consolidate import DeterministicConsolidator
from .embed import OllamaEmbedder
from .extract import OllamaExtractor
from .llm import LLMConfig, OllamaClient
from .recall import recall
from .store.falkor import FalkorStore
from .types import Message, TurnContext
from .writepath import WritePath

BANNER = """memo_re demo -- deterministic input-driven recall
  type a turn and press enter.  /quit to exit, /reset to clear the session.
"""


async def run_demo(session: str, config: RecallConfig, write_config: WritePathConfig) -> None:
    embed_config = EmbedConfig.from_env()
    # The index width is the embedder's, not a store constant -- see EmbedConfig.
    store = FalkorStore(StoreConfig.from_env(), dimension=embed_config.dimension)
    embedder = OllamaEmbedder(embed_config)
    await store.connect()

    extractor = OllamaExtractor(
        write_config,
        OllamaClient(LLMConfig(model=write_config.extractor_model)),
    )
    consolidator = DeterministicConsolidator(store, embedder)
    write_path = WritePath(extractor, consolidator, write_config, store=store)
    history: list[Message] = []

    print(BANNER)
    # Which session and graph this turn writes to. Printed because recall is
    # session-scoped (§13) and an empty session is indistinguishable from a broken
    # lookup from the outside -- the first thing you need when nothing comes back.
    held = await store.count(session)
    print(
        f"  session={session!r}  graph={store.config.graph_name!r}  "
        f"model={embed_config.model}  holding {held} fact(s)\n"
    )
    loop = asyncio.get_running_loop()
    while True:
        try:
            line = (await loop.run_in_executor(None, sys.stdin.readline)).strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line == "/quit":
            break
        if line == "/reset":
            await store.clear_session(session)
            history.clear()
            print("  [session cleared]")
            continue

        turn = TurnContext(session_id=session, user_message=line, recent_messages=list(history))

        result = await recall(turn, config, store, embedder)
        if result.gate_open:
            print(
                f"  [recall: gate OPEN  {len(result.memories_used)} facts  "
                f"{result.latency_ms:.0f}ms]"
            )
            print("  -> injected:")
            for block_line in (result.injected_block or "").split("\n"):
                print(f"    {block_line}")
        else:
            print(f"  [recall: gate CLOSED  nothing injected  {result.latency_ms:.0f}ms]")

        write = await write_path.run(session, line, "", history)
        if not write.outcomes:
            print("  [P1 extract: []  (transient -- salience gate returned nothing)]")
            print("  [no write]")
        else:
            for outcome in write.outcomes:
                candidate = outcome.candidate
                print(
                    f"  [P1 extract: {candidate.type.value} {candidate.fact!r} "
                    f"conf={candidate.confidence:.2f} subject={candidate.subject_hint!r}]"
                )
                detail = f"ordinal {outcome.ordinal}"
                if outcome.superseded_fact_id:
                    detail = f"higher ordinal wins -> supersede prior, write ordinal {outcome.ordinal}"
                print(f"  [P2 consolidate: {outcome.case.value}  ({detail})]")

        history.append(Message(role="user", content=line))

    await embedder.aclose()
    await store.aclose()


async def run_inspect(session: str | None, query: str | None, config: RecallConfig) -> None:
    """Show what the store actually holds -- see `FalkorStore.sessions`."""
    embed_config = EmbedConfig.from_env()
    store = FalkorStore(StoreConfig.from_env(), dimension=embed_config.dimension)
    await store.connect()
    print(f"graph={store.config.graph_name!r}  model={embed_config.model}  dim={embed_config.dimension}")

    if session is None:
        rows = await store.sessions()
        if not rows:
            print("\n  (no facts in this graph -- nothing has been written yet)")
        else:
            print(f"\n  {'session':<42} {'facts':>7} {'live':>7}")
            for session_id, total, live in rows:
                print(f"  {session_id!r:<42} {total:>7} {live:>7}")
            print("\n  memore inspect --session <name>        list its facts")
            print("  memore inspect --session <name> --query '...'   trace a recall")
        await store.aclose()
        return

    facts = await store.facts_in_session(session)
    live = [f for f in facts if f.invalid_at is None]
    print(f"session={session!r}  {len(facts)} fact(s), {len(live)} live\n")
    if not facts:
        print("  (empty -- recall from this session will always find nothing)")

    # Grouped by subject, because the subject key IS the identity contract: one live
    # fact per subject is the invariant, and a group showing two is the bug.
    by_subject: dict[str, list] = {}
    for fact in facts:
        by_subject.setdefault(fact.subject_key, []).append(fact)
    for subject_key, group in sorted(by_subject.items(), key=lambda kv: kv[1][0].ordinal):
        label = group[0].subject_label or subject_key
        flag = "" if sum(1 for f in group if f.invalid_at is None) == 1 else "  <-- not exactly 1 live"
        print(f"  [{label}]{flag}")
        for fact in group:
            mark = "SUPERSEDED" if fact.invalid_at is not None else "live      "
            print(f"    #{fact.ordinal:<4} {mark}  {fact.fact}")

    if query:
        embedder = OllamaEmbedder(embed_config)
        result = await recall(
            TurnContext(session_id=session, user_message=query), config, store, embedder
        )
        print(f"\n  query: {query!r}")
        print(
            f"  gate {'OPEN' if result.gate_open else 'CLOSED'}  "
            f"{len(result.memories_used)} fact(s)  {result.latency_ms:.0f}ms  "
            f"(floor {config.score_floor}, hops {config.expansion_hops})"
        )
        for hit in result.memories_used:
            # score 0.0 means the walk reached it; it was never ranked against the turn.
            origin = "chain" if hit.score == 0.0 else f"{hit.score:.3f}"
            print(f"    {origin:>7}  {hit.fact}")
        if not result.gate_open:
            print("    nothing cleared the floor -- lower it with --score-floor to see near misses")
        await embedder.aclose()
    await store.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(prog="memore", description="memo_re terminal harness")
    sub = parser.add_subparsers(dest="command")

    demo = sub.add_parser("demo", help="interactive recall/write trace")
    demo.add_argument("--session", default="demo", help="recall is session-scoped (§13)")
    demo.add_argument("--score-floor", type=float, default=RecallConfig.score_floor)
    demo.add_argument("--k", type=int, default=RecallConfig.k)
    demo.add_argument("--expansion-hops", type=int, default=RecallConfig.expansion_hops)
    demo.add_argument("--extractor-model", default=WritePathConfig.extractor_model)

    inspect = sub.add_parser("inspect", help="show what the store holds; no session lists all")
    inspect.add_argument("--session", default=None)
    inspect.add_argument("--query", default=None, help="also trace a recall against it")
    inspect.add_argument("--score-floor", type=float, default=RecallConfig.score_floor)
    inspect.add_argument("--k", type=int, default=RecallConfig.k)
    inspect.add_argument("--expansion-hops", type=int, default=RecallConfig.expansion_hops)

    args = parser.parse_args()
    config = RecallConfig(
        score_floor=getattr(args, "score_floor", RecallConfig.score_floor),
        k=getattr(args, "k", RecallConfig.k),
        expansion_hops=getattr(args, "expansion_hops", RecallConfig.expansion_hops),
    )
    if args.command == "inspect":
        asyncio.run(run_inspect(args.session, args.query, config))
    elif args.command in (None, "demo"):
        write_config = WritePathConfig(
            extractor_model=getattr(args, "extractor_model", WritePathConfig.extractor_model)
        )
        asyncio.run(run_demo(getattr(args, "session", "demo"), config, write_config))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
