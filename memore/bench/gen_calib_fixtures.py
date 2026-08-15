"""Regenerate `calib_fixtures` fact text by running the REAL extractor over learn-turns.

Why this exists. RESULTS.md §12 found the calibration fixtures had drifted from what P1
actually stores: `CHAT_FACTS` held terse fragments ("deploys to staging by default")
where the live write path stores third-person sentences ("The user deploys to staging by
default"), and the fixture's own positives shared word stems with its facts. The BM25
deduction that dominates real gate behaviour therefore almost never fired during the run
that chose `score_floor = 0.48`, so the floor was calibrated against a distribution that
does not occur in production.

Hand-authoring the facts again would drift again. So the facts are *derived*: author the
user turn, run `OllamaExtractor`, keep what it emits. Re-run this after any change to the
extractor prompt or model, and re-run `bench.calibrate` after that -- the fixtures, the
embedder and the floor are one chain, not three settings.

    uv run python -m memore.bench.gen_calib_fixtures            # print, verify only
    uv run python -m memore.bench.gen_calib_fixtures --write    # rewrite calib_fixtures.py

Two invariants this script enforces rather than assumes, because violating either breaks
the calibration silently rather than loudly:

  1. EXACTLY ONE fact per turn. `CHAT_POSITIVES` maps `(query, index)` into the fact list,
     and `Fixture.queries` carries `on_subject_facts` as a frozenset of the fact STRING.
     A turn yielding 0 or 2 facts shifts every later index, which does not raise -- it
     just makes `on_subject`/`useful_tpr` measure nothing. When a turn misbehaves, fix the
     TURN below until it yields one fact; never patch the mapping.
  2. The crowded fixture's ENTITY must survive. `CROWDED_CHAT_HARD_NEGATIVES` are
     near-misses by construction ("where does the mobile app deploy?" against a store
     holding the web app, the api and the docs site). If extraction drops or replaces the
     entity, the hard negatives stop being near and that fixture stops measuring
     wrong-subject recall at all. Each turn therefore declares the spellings that still
     count as naming its entity, and generation fails if none appears.

Regeneration is NOT bit-reproducible. `temperature` is already 0.0, so the variation is
decoder-level rather than sampling, but it is real: one run emitted "the prod cluster is
in us-east-1" and the next "the production cluster is located in us-east-1". That is why
the generated literals are CHECKED IN rather than derived at import time -- the fixture
has to be a fixed object for a floor to be calibrated against it. Regenerating is a
deliberate act, and the calibration run afterwards is what verifies the result: if the
crowded fixture's hard negatives have gone soft, `fpr_hard` collapses at low floors and
says so.
"""

from __future__ import annotations

import argparse
import asyncio
import re
from pathlib import Path

from ..config import WritePathConfig
from ..extract import OllamaExtractor

# (learn-turn, spellings that count as the entity surviving). The turns are written
# in first person, as a user actually types them, and tuned until each yields exactly one
# fact -- see invariant 1. Order defines the index `CHAT_POSITIVES` refers to, so append
# rather than reorder.
CHAT_TURNS: list[tuple[str, tuple[str, ...]]] = [
    ("I deploy to staging by default", ("staging",)),
    ("I prefer Python for backend work", ("Python",)),
    ("I'm based in Amsterdam", ("Amsterdam",)),
    ("I use Neovim as my editor", ("Neovim",)),
    ("I run my tests with pytest", ("pytest",)),
    ("I have a dog named Pixel", ("Pixel",)),
    ("I work at a company called Northgate", ("Northgate",)),
    ("I prefer dark mode in the terminal", ("dark mode",)),
    ("I'm allergic to peanuts", ("peanut",)),
    ("I drink oat milk in my coffee", ("oat milk",)),
    ("I manage a team of six engineers", ("six",)),
    # Was "I fly out to Lisbon on the 14th" while these were hand-authored. P1 refuses it,
    # correctly -- a dated one-off expires, and §1.2 says drop those. That the old fixture
    # contained a fact the write path would never produce is exactly the drift this script
    # exists to stop, so the turn is now a durable travel fact rather than an itinerary.
    ("I go to Lisbon every quarter for work", ("Lisbon",)),
]

CROWDED_TURNS: list[tuple[str, tuple[str, ...]]] = [
    ("The web app deploys to staging", ("web app",)),
    ("The api deploys to prod", ("api",)),
    ("The docs site deploys to netlify", ("docs site",)),
    ("The work laptop runs Linux", ("work laptop",)),
    ("The home desktop runs Windows", ("home desktop",)),
    ("Pixel is a border collie", ("Pixel",)),
    ("Mochi is a siamese cat", ("Mochi",)),
    ("Sarah leads the platform team", ("Sarah",)),
    ("Tom leads the data team", ("Tom",)),
    ("The platform team has six engineers", ("platform team",)),
    ("The data team has four engineers", ("data team",)),
    ("The staging cluster is in eu-west-1", ("staging cluster",)),
    ("The prod cluster is in us-east-1", ("prod cluster", "production cluster")),
]


async def derive(turns: list[tuple[str, tuple[str, ...]]], label: str) -> list[tuple[str, str, str]]:
    """`(fact, subject_hint, attribute)` per turn, or raise with what went wrong."""
    extractor = OllamaExtractor(WritePathConfig())
    rows: list[tuple[str, str, str]] = []
    problems: list[str] = []
    # Ollama here serves 3 parallel slots, and each turn is extracted standalone (no
    # `recent`), so they are independent and the fixture does not depend on the order the
    # turns happen to be listed in. `gather` preserves order, which the index mapping in
    # `CHAT_POSITIVES` depends on. Safe here precisely because nothing timed is measured;
    # the calibration query loop stays serial for exactly the opposite reason.
    slots = asyncio.Semaphore(3)

    async def one(turn: str):
        async with slots:
            return await extractor.extract(turn, "", [], None)

    results = await asyncio.gather(*(one(turn) for turn, _ in turns))
    for (turn, entity), out in zip(turns, results, strict=True):
        if len(out) != 1:
            problems.append(f"  {turn!r} -> {len(out)} facts (need exactly 1): "
                            f"{[c.fact for c in out]}")
            continue
        candidate = out[0]
        if not any(spelling.lower() in candidate.fact.lower() for spelling in entity):
            problems.append(f"  {turn!r} -> {candidate.fact!r} names none of {entity}")
            continue
        rows.append((candidate.fact, candidate.subject_hint, candidate.attribute))
        print(f"  {turn!r}\n      -> {candidate.fact!r} [{candidate.subject_hint} :: "
              f"{candidate.attribute}]")
    if problems:
        raise SystemExit(
            f"\n{label}: {len(problems)} turn(s) unusable -- fix the TURN, not the mapping "
            f"(see invariant 1 in the module docstring):\n" + "\n".join(problems)
        )
    return rows


def render(name: str, rows: list[tuple[str, str, str]]) -> str:
    body = "\n".join(f"    ({fact!r}, {subject!r}, {attribute!r})," for fact, subject, attribute in rows)
    return f"{name}: list[tuple[str, str, str]] = [\n{body}\n]"


def splice(source: str, name: str, block: str) -> str:
    """Replace one `NAME: list[...] = [ ... ]` literal, leaving every comment around it."""
    # `[^=\n]*` for the annotation, not `[^\]]*`: `list[tuple[str, str, str]]` contains
    # brackets, and a bracket-excluding class stops inside it and matches nothing.
    pattern = re.compile(rf"^{re.escape(name)}\s*:[^=\n]*=\s*\[\n.*?^\]", re.S | re.M)
    if not pattern.search(source):
        raise SystemExit(f"could not find the {name} literal to replace")
    return pattern.sub(lambda _: block, source, count=1)


async def main_async(write: bool) -> None:
    print(f"extractor: {WritePathConfig().extractor_model}\n\nCHAT_FACTS:")
    chat = await derive(CHAT_TURNS, "CHAT_FACTS")
    print("\nCROWDED_CHAT_FACTS:")
    crowded = await derive(CROWDED_TURNS, "CROWDED_CHAT_FACTS")

    if not write:
        print("\n(dry run -- pass --write to update calib_fixtures.py)")
        return
    path = Path(__file__).with_name("calib_fixtures.py")
    source = path.read_text()
    source = splice(source, "CHAT_FACTS", render("CHAT_FACTS", chat))
    source = splice(source, "CROWDED_CHAT_FACTS", render("CROWDED_CHAT_FACTS", crowded))
    path.write_text(source)
    print(f"\nwrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="rewrite calib_fixtures.py in place")
    asyncio.run(main_async(parser.parse_args().write))


if __name__ == "__main__":
    main()
