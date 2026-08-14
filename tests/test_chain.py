"""Multi-hop chain walk (`memore.chain`).

Pure functions over fact strings -- no store, no embedder, no LLM. The walk is the thing
that turns "the answer fact shares no entity with the question" from unanswerable into
a graph traversal, so what these pin is the *edge rule*: exact token containment of one
fact's value in another's subject key.

The worked example is the real one from `factconsolidation_mh_6k`, including its
supersession, because the freshness interaction is the point: the live chain and the
stale chain lead to different answers.
"""

from __future__ import annotations

from memore.chain import ChainNode, build_adjacency, candidate_values, walk
from memore.keys import normalize_subject


def node(fact: str, subject: str) -> ChainNode:
    return ChainNode(fact=fact, subject_key=normalize_subject(subject))


# The live view of the corpus chain: Our Mutual Friend -> Darwin -> Amala Paul -> Belgium.
# "Charles Dickens" and "India" are the superseded values and are absent by construction:
# `live_chain_view` never returns them.
CHAIN = [
    node("The author of Our Mutual Friend is Charles Darwin.", "author of Our Mutual Friend"),
    node("Charles Darwin is married to Amala Paul.", "spouse of Charles Darwin"),
    node("Amala Paul is a citizen of Belgium.", "country of citizenship of Amala Paul"),
    node("Thomas Kyd was born in the city of London.", "birthplace of Thomas Kyd"),
]


def test_value_is_the_span_the_subject_does_not_cover():
    values = candidate_values(
        "Igor of Kiev is married to Olga of Kiev.", normalize_subject("spouse of Igor of Kiev")
    )
    # "kiev olga" is the usable one; spans the subject already covers are dropped so a
    # fact never becomes its own neighbour.
    assert "kiev olga" in values
    assert "igor kiev" not in values


def test_walk_reaches_the_answer_three_hops_out():
    """The question names "Our Mutual Friend"; the gold answer lives on "Amala Paul"."""
    reached = walk(CHAIN, seeds=[0], hops=3, fanout=4)
    assert [CHAIN[i].fact for i in reached] == [
        "Charles Darwin is married to Amala Paul.",
        "Amala Paul is a citizen of Belgium.",
    ]


def test_walk_respects_the_hop_bound():
    assert walk(CHAIN, seeds=[0], hops=1, fanout=4) == [1]
    assert walk(CHAIN, seeds=[0], hops=0, fanout=4) == []


def test_walk_excludes_its_own_seeds_and_never_revisits():
    reached = walk(CHAIN, seeds=[0, 1], hops=3, fanout=4)
    assert 0 not in reached and 1 not in reached
    assert len(reached) == len(set(reached))


def test_unrelated_facts_are_not_neighbours():
    """The edge rule is containment, not overlap -- 'Kyd' shares nothing with the chain."""
    adjacency = build_adjacency(CHAIN, fanout=4)
    assert adjacency[3] == []
    assert 3 not in walk(CHAIN, seeds=[0], hops=3, fanout=4)


def test_fanout_bounds_the_frontier():
    """A popular value must not drag its whole neighbourhood in.

    "Italy" is the real hub in the corpus: the value of one fact and the subject of many.
    """
    hub = [
        node("Charlie Hebdo was created in the country of Italy.", "country Charlie Hebdo created in")
    ] + [node(f"The capital of Italy is city {i}.", f"topic {i} of Italy") for i in range(10)]
    assert len(build_adjacency(hub, fanout=10)[0]) == 10
    assert len(build_adjacency(hub, fanout=3)[0]) == 3


def test_walk_is_deterministic_across_runs():
    first = walk(CHAIN, seeds=[0], hops=3, fanout=4)
    for _ in range(5):
        assert walk(CHAIN, seeds=[0], hops=3, fanout=4) == first
