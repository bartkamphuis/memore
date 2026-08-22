"""A frecency cache for facts recall has already surfaced -- HARNESS-SIDE, not memore.

## What this compensates for, stated plainly

The gate is a per-turn similarity decision. Ask "Where am I travelling to?" and the stored
fact clears the floor; follow up with "when?" and it cannot -- a two-word question shares
no term and almost no embedding neighbourhood with "the user is travelling to Lisbon on
2026-08-26", so the gate shuts and the detail the previous turn surfaced is gone. Nothing
is wrong in the store; the *conversation* carried the reference and the lookup key did not.

This module is the demo harness holding what recall already found for a few more turns, so
`<recalled_context>` lingers instead of vanishing between a question and its follow-up. It
is deliberately NOT a memore feature:

- It changes no scoring, no floor and no store, and imports nothing from `memore.recall`.
  `recall()` is called exactly as before and its answer is reported verbatim -- see
  `demo/app.py`, where the `recall` event still carries recall's own `gate_open` and
  recall's own block even on turns where the harness injects anyway.
- The alternative that looks equivalent and is not: routing this through
  `TurnContext.rolling_summary_vec`. That changes recall's *key*, not what gets injected,
  so it would not recover the lost detail -- and rolling-summary key synthesis is deferred
  scope (`recall-poc-spec.md` §5).

## Frecency, and why the clock is turns

`strength` is the similarity the fact scored when recall last surfaced it, lifted slightly
each time it is surfaced again -- the *frequency* half. It then decays by half every
`half_life_turns` -- the *recency* half. Below `floor` the entry is dropped and the detail
is gone again, which is the property that keeps this from degenerating into "inject
everything forever": a fact that stops being referred to leaves on its own.

The clock is **turns, never wall-clock**. A coffee break in the middle of a conversation
must not expire the thing being discussed, and two rapid-fire turns are two steps away
from the reference, not zero.

Text is the key because `MemoryHit` carries no fact id (`assemble.py` never touches one).
The caller resolves that text back to the live store row every turn -- this cache stores
weights, never the fact's rendered text, precisely so a superseded value cannot be
replayed after the store has retired it.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class LingerConfig:
    """Off by default nowhere -- the demo turns it on; it has no effect on any other path.

    `max_facts` is a cap on the *carried* set only. Facts recall surfaced this turn are
    never counted against it: this layer may only ever add to what the gate decided.
    """

    half_life_turns: float = 4.0
    floor: float = 0.20
    max_facts: int = 4
    repeat_bonus: float = 0.15
    enabled: bool = True

    @classmethod
    def from_env(cls) -> LingerConfig:
        """`--no-linger` and the two knobs are the CONTROL, not tuning conveniences.

        The layer's whole claim is that it holds a reference the prompt would otherwise
        lose, and the prompt already carries `history[-6:]` -- three turns of both roles.
        A window shorter than that answers the follow-up correctly for a reason that has
        nothing to do with this cache, so an on/off arm is the only way to tell the two
        apart. It is a flag rather than a fixture because the difference only appears
        against the real extractor and the real gate.
        """
        return cls(
            half_life_turns=float(os.getenv("MEMORE_DEMO_LINGER_HALF_LIFE", "4.0")),
            floor=float(os.getenv("MEMORE_DEMO_LINGER_FLOOR", "0.20")),
            max_facts=int(os.getenv("MEMORE_DEMO_LINGER_MAX", "4")),
            enabled=os.getenv("MEMORE_DEMO_LINGER", "1") not in ("0", "false", "no"),
        )


@dataclass
class _Entry:
    strength: float
    last_turn: int
    seen: int


@dataclass(frozen=True)
class Carried:
    """One fact still being carried, with the arithmetic that kept it visible."""

    fact: str
    weight: float
    strength: float
    age_turns: int
    seen: int


@dataclass
class LingerCache:
    """Per session. `clear()` on reset AND on session switch, or A's facts reach B.

    Turn numbering is the cache's own: `begin_turn()` once per turn, before `observe()`.
    """

    config: LingerConfig = field(default_factory=LingerConfig)
    _entries: dict[str, _Entry] = field(default_factory=dict)
    _turn: int = 0

    def begin_turn(self) -> int:
        self._turn += 1
        return self._turn

    def observe(self, fact: str, similarity: float) -> None:
        """Record that recall surfaced this fact on the current turn."""
        entry = self._entries.get(fact)
        seen = entry.seen + 1 if entry is not None else 1
        # Repeated surfacing lifts strength but can never exceed 1.0: a fact referred to
        # five times must not outrank a fact the gate just scored 0.9 on.
        strength = min(1.0, max(similarity, 0.0) * (1.0 + self.config.repeat_bonus * (seen - 1)))
        if entry is not None:
            strength = max(strength, entry.strength)
        self._entries[fact] = _Entry(strength=strength, last_turn=self._turn, seen=seen)

    def carried(self) -> list[Carried]:
        """What is still above the floor, heaviest first, minus what recall just surfaced.

        A fact observed on the CURRENT turn is not carried -- it is a hit, the gate
        admitted it, and it is already in the block on its own merits. That is the one
        rule that keeps this from double-counting, and it is here rather than in the
        caller so there is a single definition of "carried".

        Entries that have decayed below `floor` are dropped here rather than merely
        filtered -- an entry the harness will never inject again is not state worth
        keeping, and pruning on read keeps the cache the size of the live reference set.
        """
        return self._at(self._turn, prune=True)

    def upcoming(self) -> list[Carried]:
        """What the NEXT turn would carry -- for showing standing state between turns.

        Not the same question as `carried()`, and the difference is a whole turn: after a
        turn ends, the facts that turn surfaced have `last_turn == _turn` and are excluded
        as fresh, so asking `carried()` between turns answers "nothing" about a cache that
        is about to hand over three facts. Reading is not a turn, so this prunes nothing
        and decays nothing permanently: it projects, and the next real turn decides.
        """
        return self._at(self._turn + 1, prune=False)

    def _at(self, turn: int, *, prune: bool) -> list[Carried]:
        if not self.config.enabled:
            return []
        out: list[Carried] = []
        for fact, entry in list(self._entries.items()):
            age = max(0, turn - entry.last_turn)
            weight = entry.strength * 0.5 ** (age / self.config.half_life_turns)
            if weight < self.config.floor:
                if prune:
                    del self._entries[fact]
                continue
            if entry.last_turn == turn:
                continue
            out.append(
                Carried(
                    fact=fact, weight=weight, strength=entry.strength,
                    age_turns=age, seen=entry.seen,
                )
            )
        out.sort(key=lambda c: c.weight, reverse=True)
        return out[: self.config.max_facts]

    def forget(self, facts: Iterable[str]) -> None:
        """Drop entries the store no longer holds -- a cleared or switched session."""
        for fact in facts:
            self._entries.pop(fact, None)

    def clear(self) -> None:
        self._entries.clear()
        self._turn = 0
