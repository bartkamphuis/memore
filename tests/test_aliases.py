"""Subject aliasing -- the DF-gated merge of RESULTS.md §3.

Every fixture here is built from the ratios actually measured on the two benchmark
corpora, because a unit test for a corpus-statistics rule is worthless otherwise: with a
vocabulary where every token has df=1, both the merges and the refusals pass for the wrong
reason and the test proves nothing about the rule.

Measured, and reproduced by `subjects()` below:

    affiliated  9/303  = 0.0297  and  60/1559 = 0.0385   relation word, must merge
    location    3/303  = 0.0099  and  13/1559 = 0.0083   relation word, near the boundary
    kingdom            2/303-ish = 0.0066  11/1559 = 0.0071   ENTITY, must not merge
    shingon     1/303  = 0.0033                          ENTITY, must not merge
"""

from __future__ import annotations

from memore.aliases import AliasConfig, SubjectVocabulary
from memore.consolidate import ConsolidationConfig, DeterministicConsolidator, subject_key
from memore.embed import StubEmbedder
from memore.store.fake import InMemoryStore
from memore.types import CandidateFact, ConsolidationCase, FactType

SESSION = "s1"


def subjects(*keys: str, token_counts: dict[str, int] | None = None, filler: int = 300):
    """A vocabulary of `filler` distinct subjects, plus tokens at controlled frequencies.

    `filler` defaults above `AliasConfig.min_subjects` so the rule is actually active --
    the whole point of the guard is that it is inert below that, which is pinned
    separately by `test_inert_in_a_conversational_store`.
    """
    vocabulary = SubjectVocabulary()
    for i in range(filler):
        vocabulary.add(f"filler{i} topic{i}")
    for token, count in (token_counts or {}).items():
        for i in range(count):
            vocabulary.add(f"{token} carrier{i} thing{i}")
    for key in keys:
        vocabulary.add(key)
    return vocabulary


def test_generic_relation_word_merges_the_split_subject():
    """§3's named residual: one subject, two namings, differing by a relation word.

    `religion of Karen Armstrong` and `religion Karen Armstrong is affiliated with` are the
    same subject. Before this rule they never collided, so a contradiction between them was
    never detected and both facts stayed live.
    """
    vocabulary = subjects(subject_key("religion of Karen Armstrong"), token_counts={"affiliated": 9})
    arriving = subject_key("religion Karen Armstrong is affiliated with")

    assert vocabulary.resolve(arriving) == subject_key("religion of Karen Armstrong")


def test_narrower_entity_is_refused_even_though_it_is_a_subset():
    """The over-merge §3 refused to ship, now refused for a stated reason.

    `shingon` appears in a single subject, so it reads as an entity, not a relation word.
    Merging would supersede a correct fact about Buddhism with one about Shingon Buddhism
    and lose it permanently -- the unrecoverable direction.
    """
    vocabulary = subjects(subject_key("founder of Buddhism"), token_counts={"shingon": 1})
    arriving = subject_key("founder of Shingon Buddhism")

    assert vocabulary.resolve(arriving) == arriving


def test_boundary_entity_that_a_naive_threshold_would_have_merged():
    """`Kingdom of England` is not `England`, and it sits closest to the cutoff.

    Measured at 11/1559 = 0.0071 it is the highest-frequency token that must NOT merge --
    England is in Europe, the Kingdom of England is a separate subject with its own answer
    chain. It is the reason the default sits at 0.015 rather than in the narrow band just
    above 0.0071: this refusal has a 2x margin instead of 1.17x.
    """
    vocabulary = subjects(subject_key("continent England is located in"), token_counts={"kingdom": 2})
    arriving = subject_key("continent Kingdom of England is located in")

    assert vocabulary.resolve(arriving) == arriving


def test_rare_relation_word_is_a_known_miss_not_a_merge():
    """A limitation, pinned deliberately so nobody tunes the threshold to "fix" it.

    `employed` and `married` are genuine relation words appearing in exactly one subject
    each at 32k. This rule refuses them, and must: any threshold low enough to catch a
    df=1 relation word also catches `shingon`, `gaelic` and `nazi` at the same frequency.
    Reading surface statistics cannot separate them -- that needs subject identity, the
    same wall RESULTS.md §9 hits from the other side.
    """
    vocabulary = subjects(subject_key("employer of Martin Luther"), token_counts={"employed": 1})
    arriving = subject_key("employer Martin Luther is employed by")

    assert vocabulary.resolve(arriving) == arriving


def test_inert_in_a_conversational_store():
    """Small stores are not policed, because document frequency needs a corpus.

    In a 12-subject personal session `location` in two subjects reads as ratio 0.17 --
    eleven times the threshold -- so an unguarded relative rule would fire hardest exactly
    where its evidence is weakest. The measured corpora are 303 and 1559 subjects; nothing
    smaller has been validated, so nothing smaller is merged.
    """
    vocabulary = subjects(
        "location home", "location work", filler=10, token_counts={}
    )
    arriving = subject_key("home location city")

    assert len(vocabulary) < AliasConfig.min_subjects
    assert vocabulary.resolve(arriving) == arriving


def test_ambiguity_refuses_rather_than_choosing():
    """A key between two existing subjects is evidence of nothing.

    Picking the "closest" would be a similarity judgement, which is exactly what this
    design does not make anywhere in the consolidation decision.
    """
    vocabulary = subjects(
        "capital france", "capital france modern", token_counts={"current": 12, "modern": 12}
    )
    # A strict superset of BOTH existing subjects, generic-apart from each. Nothing in the
    # statistics says which of the two it continues.
    arriving = "capital current france modern"

    assert vocabulary.resolve(arriving) == arriving
    assert vocabulary.merges == []


def test_merges_are_recorded_for_inspection():
    """RESULTS.md §3 rejected the ungated rule despite a better score, because aggregates
    hide over-merges. The log is the evidence, so it is part of the object."""
    vocabulary = subjects(subject_key("religion of Karen Armstrong"), token_counts={"affiliated": 9})
    vocabulary.resolve(subject_key("religion Karen Armstrong is affiliated with"))

    assert vocabulary.merges == [
        (subject_key("religion Karen Armstrong is affiliated with"),
         subject_key("religion of Karen Armstrong"))
    ]


def test_disabled_is_a_full_no_op():
    vocabulary = SubjectVocabulary(AliasConfig(enabled=False))
    for i in range(300):
        vocabulary.add(f"filler{i} topic{i}")
    for i in range(9):
        vocabulary.add(f"affiliated carrier{i} thing{i}")
    vocabulary.add(subject_key("religion of Karen Armstrong"))
    arriving = subject_key("religion Karen Armstrong is affiliated with")

    assert vocabulary.resolve(arriving) == arriving
    assert vocabulary.merges == []


def test_narrower_key_arriving_after_the_wider_one_also_merges():
    """Both directions, or the result would depend on which naming P1 happened to coin
    first -- and the same two facts would consolidate or not by luck of arrival order."""
    wider = subject_key("religion Karen Armstrong is affiliated with")
    vocabulary = subjects(wider, token_counts={"affiliated": 9})

    assert vocabulary.resolve(subject_key("religion of Karen Armstrong")) == wider


async def test_split_subject_now_consolidates_as_a_contradiction():
    """End to end: the under-merge §3 measured, resolved by the write path.

    `min_subjects` is lowered here only to keep the fixture small -- the default's
    inertness at conversational scale is pinned by `test_inert_in_a_conversational_store`.
    """
    store = InMemoryStore()
    alias = AliasConfig(min_subjects=1)
    con = DeterministicConsolidator(store, StubEmbedder(), ConsolidationConfig(alias=alias))

    def candidate(fact: str, subject: str) -> CandidateFact:
        return CandidateFact(
            fact=fact, type=FactType.STATE, confidence=1.0, valid_at=None, subject_hint=subject
        )

    # Enough same-relation subjects for "affiliated" to read as a relation word.
    for i in range(9):
        await con.consolidate(SESSION, [candidate(f"P{i} is a Quaker.", f"religion P{i} is affiliated with")])
    await con.consolidate(SESSION, [candidate("Karen Armstrong is a Catholic.", "religion of Karen Armstrong")])

    outcomes = await con.consolidate(
        SESSION,
        [candidate("Karen Armstrong is a Quaker.", "religion Karen Armstrong is affiliated with")],
    )

    assert outcomes[0].case is ConsolidationCase.CONTRADICTION
    live = await store.live_facts_for_subject(SESSION, subject_key("religion of Karen Armstrong"))
    assert [f.fact for f in live] == ["Karen Armstrong is a Quaker."]
