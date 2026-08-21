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
    args = parser.parse_args()

    from .app import DEFAULT_GRAPH

    os.environ.setdefault("MEMORE_GRAPH", args.graph or DEFAULT_GRAPH)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    import uvicorn

    print(f"\n  memore demo -> http://{args.host}:{args.port}   graph={os.environ['MEMORE_GRAPH']}\n")
    uvicorn.run("memore.demo.app:app", host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
