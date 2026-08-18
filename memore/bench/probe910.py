"""What does P1 actually emit for turns 9/10/11, attribute AND single_valued together?

The harness prints attributes and cases but never `single_valued`, so the MISSED mode
(`likes` vs `likes`, same slot, no supersede) can only be inferred. This shows it.

Replays turns 0..11 with the real history and hint list, exactly as slots.py would, and
stops there -- the later turns cannot reach turn 10's extraction at extract_window_turns=3.
"""
import asyncio
import sys
import uuid

from memore.bench.slots import REPLIES, TURNS
from memore.config import EmbedConfig, StoreConfig, WritePathConfig
from memore.consolidate import DeterministicConsolidator
from memore.embed import OllamaEmbedder
from memore.extract import OllamaExtractor
from memore.store.falkor import FalkorStore
from memore.types import Message
from memore.writepath import WritePath

WATCH = (9, 10, 11)


async def one(run: int) -> None:
    session = f"probe910-{run}-{uuid.uuid4().hex[:6]}"
    embed_config = EmbedConfig.from_env()
    store = FalkorStore(StoreConfig.from_env(), dimension=embed_config.dimension)
    embedder = OllamaEmbedder(embed_config)
    await store.connect()
    wp = WritePath(
        OllamaExtractor(WritePathConfig()),
        DeterministicConsolidator(store, embedder),
        WritePathConfig(),
        store=store,
    )
    history: list[Message] = []
    try:
        for index, turn in enumerate(TURNS[: max(WATCH) + 1]):
            reply = REPLIES.get(index, "")
            outcome = await wp.run(session, turn, reply, list(history))
            if index in WATCH:
                for item in outcome.outcomes:
                    c = item.candidate
                    print(
                        f"  run{run} [{index:2d}] {item.case.value:<14} "
                        f"attr={c.attribute!r:<34} sv={str(c.single_valued):<5} "
                        f"ord={item.ordinal} sup={'yes' if item.superseded_fact_id else 'no '}  {c.fact}"
                    )
            history.append(Message(role="user", content=turn))
            history.append(Message(role="assistant", content=reply or "Understood."))
    finally:
        await store.clear_session(session)
        await store.aclose()


async def main() -> None:
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    for r in range(runs):
        await one(r)
        print()


asyncio.run(main())
