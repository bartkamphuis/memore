"""Subject admission (`memore.subjects`) -- the wrong-subject defence.

Pure functions over fact strings and subject keys; no store, no embedder, no LLM.

What these pin is the COMPETITOR precondition, because that is the part that makes the
rule safe rather than merely strict. A rule that just demanded token overlap would block
44% of conversational positives (measured), since a chat subject is an abstract label
matched by paraphrase. Every test below that asserts something is admitted is guarding
that property.
"""

from __future__ import annotations

from memore.chain import ChainNode
from memore.keys import normalize_subject
from memore.subjects import build_subject_view

MIN_COMPETITORS = 2
DF_MAX = 2


def node(fact: str, subject: str, superseded: bool = False) -> ChainNode:
    from datetime import UTC, datetime

    return ChainNode(
        fact=fact,
        subject_key=normalize_subject(subject),
        invalid_at=datetime(2020, 1, 1, tzinfo=UTC) if superseded else None,
    )


# A crowded relation: the same question asked of many entities. This is the shape that
# breaks a similarity floor.
CROWDED = [
    node("goaltender is associated with the sport of pesäpallo.", "sport associated with goaltender"),
    node("Steve Sax is associated with the sport of baseball.", "sport associated with Steve Sax"),
    node("power forward is associated with the sport of rugby.", "sport associated with power forward"),
    node("Blair Walsh is associated with the sport of rugby union.", "sport associated with Blair Walsh"),
]

# A personal store: one fact per subject, nothing competing.
CHAT = [
    node("is based in Amsterdam", "location"),
    node("deploys to staging by default", "deploy target"),
    node("has a dog named Pixel", "dog name"),
]


def test_query_naming_the_entity_is_admitted():
    view = build_subject_view(CROWDED)
    assert view.admits(
        "Which sport is goaltender associated with?",
        "goaltender is associated with the sport of pesäpallo.",
        MIN_COMPETITORS,
        DF_MAX,
    )


def test_query_naming_a_different_entity_is_refused():
    """The whole point: right relation, wrong entity, high similarity."""
    view = build_subject_view(CROWDED)
    assert not view.admits(
        "Which sport is Hines Ward associated with?",
        "goaltender is associated with the sport of pesäpallo.",
        MIN_COMPETITORS,
        DF_MAX,
    )


def test_conversational_paraphrase_survives_with_no_shared_token():
    """The failure mode of the naive rule. "what city am I in?" shares nothing with
    subject "location", and must still be admitted -- `location` has no competitors."""
    view = build_subject_view(CHAT)
    assert view.admits("what city am I in?", "is based in Amsterdam", MIN_COMPETITORS, DF_MAX)
    assert view.admits("what's my deploy setup?", "deploys to staging by default", MIN_COMPETITORS, DF_MAX)
    assert view.admits("tell me about my dog", "has a dog named Pixel", MIN_COMPETITORS, DF_MAX)


def test_an_uncontested_subject_is_never_policed():
    """One competing subject is below the threshold, so the check stays out of the way."""
    two = [
        node("is based in Amsterdam", "location of the user"),
        node("the office is in Berlin", "location of the office"),
    ]
    view = build_subject_view(two)
    # They share "location", so each has exactly 1 competitor -- under the minimum.
    assert view.competitors[normalize_subject("location of the user")] == 1
    assert view.admits("where am I?", "is based in Amsterdam", MIN_COMPETITORS, DF_MAX)


def test_superseded_facts_are_admissible_but_do_not_inflate_the_vocabulary():
    """A superseded hit can still be injected (§6.3), so it needs a subject mapping --
    but its subject is already counted under its live value and must not count twice."""
    facts = [
        node("The capital of Italy is Rome.", "capital of Italy", superseded=True),
        node("The capital of Italy is Duluth.", "capital of Italy"),
        node("The capital of France is Paris.", "capital of France"),
        node("The capital of Spain is Madrid.", "capital of Spain"),
    ]
    view = build_subject_view(facts)
    assert view.df["capital"] == 3  # three live subjects, not four facts
    # The superseded fact is still resolvable, and still admitted when named.
    assert view.admits("what is the capital of Italy?", "The capital of Italy is Rome.", MIN_COMPETITORS, DF_MAX)
    assert not view.admits("what is the capital of France?", "The capital of Italy is Rome.", MIN_COMPETITORS, DF_MAX)


def test_unknown_fact_is_admitted_rather_than_guessed_at():
    view = build_subject_view(CROWDED)
    assert view.admits("anything", "a fact this view has never seen", MIN_COMPETITORS, DF_MAX)


def test_subject_of_only_relation_words_is_admitted():
    """Nothing distinguishes it, so there is no entity for the query to have named."""
    facts = [node("sport one", "sport"), node("sport two", "sport associated"), node("s3", "sport with")]
    view = build_subject_view(facts)
    assert view.admits("which sport?", "sport one", MIN_COMPETITORS, DF_MAX)
