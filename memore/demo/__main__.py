"""`python -m memore.demo`, and the `memore-demo` console script.

Its own port (8900) and its own graph default, so it cannot collide with the gateway
console or with anything the bench is doing.
"""

from __future__ import annotations

import argparse
import logging
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="memore demo: chat left, store right")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8900)
    parser.add_argument(
        "--graph",
        default=None,
        help="MEMORE_GRAPH to use. Defaults to `memore_demo` -- its own, because recall is "
        "session-scoped and a shared graph mixes bench and demo corpora into this page.",
    )
    # The linger lane's control arm (`memore/demo/linger.py`). `--no-linger` is what the
    # carry claim has to be measured against: the prompt already carries `history[-6:]`,
    # so a follow-up answered correctly with the lane ON proves nothing on its own.
    parser.add_argument("--no-linger", action="store_true",
                        help="turn the frecency lane off -- the control arm")
    parser.add_argument("--linger-half-life", type=float, default=None,
                        help="turns until a carried fact's weight halves (default 4)")
    parser.add_argument("--linger-floor", type=float, default=None,
                        help="weight below which a carried fact is dropped (default 0.20)")
    args = parser.parse_args()

    if args.no_linger:
        os.environ["MEMORE_DEMO_LINGER"] = "0"
    if args.linger_half_life is not None:
        os.environ["MEMORE_DEMO_LINGER_HALF_LIFE"] = str(args.linger_half_life)
    if args.linger_floor is not None:
        os.environ["MEMORE_DEMO_LINGER_FLOOR"] = str(args.linger_floor)

    from .app import DEFAULT_GRAPH

    os.environ.setdefault("MEMORE_GRAPH", args.graph or DEFAULT_GRAPH)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    import uvicorn

    print(f"\n  memore demo -> http://{args.host}:{args.port}   graph={os.environ['MEMORE_GRAPH']}\n")
    uvicorn.run("memore.demo.app:app", host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
