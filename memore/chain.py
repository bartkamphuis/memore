"""Deterministic multi-hop chaining over stored facts.

The problem, measured. A multi-hop question names only its *first* entity: "In which
location did the spouse of Igor of Kiev pass away?" is answered by "Olga of Kiev died in
the city of Rodez.", which shares no entity with the question at all -- "Olga of Kiev"
only exists after hop 1 resolves. One query vector therefore cannot reach it, and
RESULTS.md §5 already established that cosine is specifically blind to entity identity.
Measured on `factconsolidation_mh_6k`: the top-ranked live hit carried the gold answer on
**0 of 100** questions.

The fix is not a smarter score, it is a second lookup keyed on what the first one found
-- a graph walk. This module is that walk, and it is deterministic end to end: no LLM (§13
forbids one in A-D), no embeddings, no threshold. An edge exists when one fact's VALUE
names another fact's SUBJECT, decided by exact token containment over the same normalized
key that decides subject identity for consolidation.

Two properties make this cheap rather than explosive, both measured on the 6k corpus
before any of it was built:

  reachability   96/100 gold facts reachable within 3 hops (97% of the 63 questions with
                 an unambiguous answer-carrying fact).
  branching      mean 1.0 neighbours per node, p95 3, max 4.

The branching factor is the surprise, and it is a consequence of freshness. Walking
**live facts only** collapses 455 facts to 303 before the walk starts, so a value like
"Italy" points at the handful of subjects currently asserting something about Italy
rather than at every historical assertion. Consolidation is what keeps the graph sparse,
which is the same primitive this project exists to test, doing a second job.

No values are stored. A fact already carries its text and its `subject_key`, and the
value is the part of the text the subject does not cover -- so this is derived at read
time and needs no schema change, no migration, and no P1 change.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from .keys import normalize_subject

# Relation connectors in the corpus templates ("X is a citizen of Y", "X is married to
# Y", "The capital of X is Y"). The value is always the trailing span after one of these.
# Every split point is tried rather than guessing the right one: a spurious candidate
# costs one containment test that fails, while a missed one costs a broken chain.
_CONNECTORS = (" is ", " to ", " by ", " of ", " in ", " for ", " with ", " from ")

_TRAILING_PUNCT = re.compile(r"[.,;:]+$")


@dataclass(frozen=True)
class ChainNode:
    """One live fact, as the walk sees it. Deliberately not a store type -- the walk is
    pure and testable without a database."""

    fact: str
    subject_key: str
    valid_at: datetime | None = None
    invalid_at: datetime | None = None


def candidate_values(fact: str, subject_key: str) -> list[str]:
    """Every trailing span of `fact` that could be its value.

    Generous by design: "Igor of Kiev is married to Olga of Kiev." yields both "Olga of
    Kiev" (after " to ") and "Kiev" (after the second " of "). Spans whose tokens the
    subject already covers are dropped -- they point back at the fact's own subject and
    would make every fact its own neighbour.
    """
    body = _TRAILING_PUNCT.sub("", fact.strip())
    subject_tokens = set(subject_key.split())
    out: list[str] = []
    seen: set[str] = set()
    low = body.lower()
    for connector in _CONNECTORS:
        start = 0
        while (index := low.find(connector, start)) != -1:
            tail = body[index + len(connector) :].strip()
            start = index + 1
            if not tail:
                continue
            key = normalize_subject(tail)
            if not key or key in seen:
                continue
            # Covered by the subject already -> not a value, just part of what the fact
            # is about.
            if set(key.split()) <= subject_tokens:
                continue
            seen.add(key)
            out.append(key)
    return out


def build_adjacency(nodes: list[ChainNode], fanout: int) -> dict[int, list[int]]:
    """index -> indices of nodes this one's value names.

    An edge is exact token containment: every token of the value key must appear in the
    target's subject key. That is the same all-or-nothing rule consolidation uses to
    decide subject identity, deliberately -- a fuzzy edge here would reintroduce exactly
    the wrong-subject failure the gate already cannot catch (RESULTS.md §5).
    """
    by_token: dict[str, list[int]] = {}
    for index, node in enumerate(nodes):
        for token in set(node.subject_key.split()):
            by_token.setdefault(token, []).append(index)

    adjacency: dict[int, list[int]] = {}
    for index, node in enumerate(nodes):
        found: set[int] = set()
        for value in candidate_values(node.fact, node.subject_key):
            tokens = set(value.split())
            if not tokens:
                continue
            candidates: set[int] = set()
            for token in tokens:
                candidates.update(by_token.get(token, ()))
            for other in candidates:
                if other != index and tokens <= set(nodes[other].subject_key.split()):
                    found.add(other)
        # Deterministic order, then bounded. Sorted by index == arrival order, so the
        # cap is stable across runs rather than dependent on set iteration.
        adjacency[index] = sorted(found)[:fanout]
    return adjacency


def walk(
    nodes: list[ChainNode],
    seeds: list[int],
    hops: int,
    fanout: int,
) -> list[int]:
    """Breadth-first from `seeds`, returning newly reached indices in hop order.

    Seeds themselves are excluded -- they are already in the recalled block. Hop order is
    preserved so the caller can fill a token budget nearest-first: a hop-1 fact is more
    likely to matter than a hop-3 one.
    """
    if hops <= 0 or not seeds:
        return []
    adjacency = build_adjacency(nodes, fanout)
    seen = set(seeds)
    frontier = list(seeds)
    reached: list[int] = []
    for _ in range(hops):
        nxt: list[int] = []
        for index in frontier:
            for other in adjacency.get(index, ()):
                if other not in seen:
                    seen.add(other)
                    nxt.append(other)
                    reached.append(other)
        if not nxt:
            break
        frontier = nxt
    return reached
