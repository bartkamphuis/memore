"""Show P1's `attribute` and `single_valued` for chosen turns, side by side.

`slots.py` prints attributes, cases and ordinals but never `single_valued`, so a
`must-collide` miss cannot be told apart from its two causes: a SPLIT (two slot names) or
a MISSED (one slot name, `single_valued=false`, so no contradiction can fire). This prints
both fields on one line, which is what made §20's mechanism visible. That pairing is the
reusable part -- point `WATCH` at whatever turns you are investigating.

It replays from turn 0 through `max(WATCH)` with the real history and hint list, exactly as
`slots.py` would, and stops there. Stopping is safe for any window that ends at least
`extract_window_turns` (3) before a later turn you care about: nothing downstream can reach
an earlier turn's extraction, and the hint list at turn N depends only on turns 0..N-1.

Defaults to turns 9/10/11, the case §20 diagnoses. **§20 is settled** -- read it before
re-running that one, rather than re-deriving the same answer.

    MEMORE_GRAPH=memore_probe uv run python -m memore.bench.probe910 3
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
