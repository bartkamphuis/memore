"""A small FastAPI app that makes memore's two paths visible in a browser.

    uv sync --extra demo
    docker compose up -d falkordb
    uv run memore-demo            # http://127.0.0.1:8900

Not part of the library and not wired into the bench, the calibration harness or
`slots.py`. It is a **viewer**: it runs the same `recall()` and `WritePath` the gateway
imports, and shows what they did.

## Why a page rather than a third CLI

`memore demo` already prints this trace, and a terminal is the wrong shape for it. The
interesting object is not the turn — it is **the store, changing**. A scrolling log shows
you a supersede as two lines eight screens apart; a pane shows you the old value greying
out next to the new one on the turn it happens. That is the claim this project makes
("supersede, never delete") and it is the one thing no other memory system puts on screen.

So: chat left, store right, and the trace of what the two paths decided in between.

## Two limits it has on purpose

**One conversation per process.** History and session are process-global, so two browser
tabs share one store and one thread. A single-user demo is what this is; multi-session
would need a session id in the URL and a store per id, and would add a moving part that
teaches nothing about memore.

**The full history goes to `recall()`** via `TurnContext.recent_messages`, unbounded, while
the extractor windows at `WritePathConfig.extract_window_turns` (3). That asymmetry is the
library's, not the demo's -- it is just visible here because the demo never truncates.
"""
