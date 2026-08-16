"""Consolidation suite -- recall-writepath-spec.md §8, adapted to recall-poc-spec.md §4.

The CONTRADICTION test is the critical one: it is the case the field scores 7-54% on and
the reason this project exists.
"""

from __future__ import annotations

import pytest

from memore.consolidate import ConsolidationConfig, DeterministicConsolidator, subject_key
from memore.embed import StubEmbedder
from memore.store.fake import InMemoryStore
from memore.types import CandidateFact, ConsolidationCase, FactType

SESSION = "s1"


def candidate(
    fact: str, subject: str, confidence: float = 0.9, attribute: str = ""
) -> CandidateFact:
    """`attribute` defaults to "" -- unspecified, colliding with every slot.

    That default is why the tests written before RESULTS.md §11 still pin exactly what
    they pinned: an extractor that names no slot gets the pre-§11 behaviour.
    """
    return CandidateFact(
        fact=fact,
        type=FactType.PREFERENCE,
        confidence=confidence,
        valid_at=None,
        subject_hint=subject,
        attribute=attribute,
    )


@pytest.fixture
def consolidator() -> tuple[DeterministicConsolidator, InMemoryStore]:
    store = InMemoryStore()
    # Embedding comparison off by default here: the StubEmbedder's vectors are
    # deterministic but meaningless, so a similarity threshold would be noise. The
    # string path is what these tests pin.
    con = DeterministicConsolidator(
        store, StubEmbedder(), ConsolidationConfig(use_embedding_comparison=False)
    )
    return con, store


async def test_new_on_empty_store(consolidator):
    con, store = consolidator
    outcomes = await con.consolidate(SESSION, [candidate("deploys to staging by default", "deploy target")])
    assert [o.case for o in outcomes] == [ConsolidationCase.NEW]
    assert await store.count(SESSION) == 1


async def test_duplicate_writes_no_second_copy(consolidator):
    con, store = consolidator
    await con.consolidate(SESSION, [candidate("deploys to staging by default", "deploy target")])
    outcomes = await con.consolidate(SESSION, [candidate("deploys to staging by default", "deploy target")])
    assert outcomes[0].case is ConsolidationCase.DUPLICATE
    assert await store.count(SESSION) == 1


async def test_contradiction_supersedes_without_deleting(consolidator):
    """The money case (writepath §2.2 case 3)."""
    con, store = consolidator
    await con.consolidate(SESSION, [candidate("deploys to staging by default", "deploy target")])
    outcomes = await con.consolidate(SESSION, [candidate("deploys to prod by default", "deploy target")])

    assert outcomes[0].case is ConsolidationCase.CONTRADICTION
    assert outcomes[0].superseded_fact_id is not None

    # Both facts still exist: the old one is marked, not removed.
    assert await store.count(SESSION) == 2
    old = store.facts[outcomes[0].superseded_fact_id]
    assert old.invalid_at is not None
    assert old.fact == "deploys to staging by default"

    live = await store.live_facts_for_subject(SESSION, subject_key("deploy target"))
    assert [f.fact for f in live] == ["deploys to prod by default"]
    assert live[0].valid_at is not None


async def test_higher_ordinal_always_wins(consolidator):
    """The resolution is arrival order, not a judgment call -- so the last write wins
    even when the earlier value is the one that is true in the real world."""
    con, store = consolidator
    for value in ["Berlin", "Bonn", "Paris"]:
        await con.consolidate(SESSION, [candidate(f"The capital of Germany is {value}", "capital of Germany")])
    live = await store.live_facts_for_subject(SESSION, subject_key("capital of Germany"))
    assert len(live) == 1
    assert live[0].fact == "The capital of Germany is Paris"
    assert live[0].ordinal == 3


async def test_refinement_updates_rather_than_duplicating(consolidator):
    con, store = consolidator
    await con.consolidate(SESSION, [candidate("works in Python", "primary language")])
    outcomes = await con.consolidate(
        SESSION, [candidate("works in Python, mainly async backend", "primary language")]
    )
    assert outcomes[0].case is ConsolidationCase.REFINEMENT
    live = await store.live_facts_for_subject(SESSION, subject_key("primary language"))
    assert [f.fact for f in live] == ["works in Python, mainly async backend"]


async def test_distinct_subjects_do_not_collide(consolidator):
    con, store = consolidator
    await con.consolidate(SESSION, [candidate("The capital of Germany is Berlin", "capital of Germany")])
    outcomes = await con.consolidate(
        SESSION, [candidate("The capital of France is Paris", "capital of France")]
    )
    assert outcomes[0].case is ConsolidationCase.NEW
    assert await store.count(SESSION) == 2


async def test_subject_key_normalization_collides_variants(consolidator):
    """Subject identity is an exact match on the *normalized* key, so casing,
    punctuation and leading articles must not fork a subject."""
    con, store = consolidator
    await con.consolidate(SESSION, [candidate("deploys to staging", "The Deploy-Target")])
    outcomes = await con.consolidate(SESSION, [candidate("deploys to prod", "deploy target")])
    assert outcomes[0].case is ConsolidationCase.CONTRADICTION


async def test_sessions_are_isolated(consolidator):
    con, store = consolidator
    await con.consolidate("a", [candidate("deploys to staging", "deploy target")])
    outcomes = await con.consolidate("b", [candidate("deploys to prod", "deploy target")])
    assert outcomes[0].case is ConsolidationCase.NEW
    assert await store.count("a") == 1
    assert await store.count("b") == 1


async def test_empty_candidate_list_is_a_no_op(consolidator):
    """Most turns extract nothing; the common path must be cheap and silent."""
    con, store = consolidator
    assert await con.consolidate(SESSION, []) == []
    assert await store.count(SESSION) == 0


async def test_no_llm_in_the_consolidation_decision(consolidator):
    """recall-poc-spec.md §7: no cloud LLM anywhere in the consolidation decision.

    Structural assertion: the consolidator's only collaborators are the store and the
    embedder. If someone wires an LLM client in, this fails.
    """
    con, _ = consolidator
    collaborators = {k: type(v).__name__ for k, v in vars(con).items() if not k.startswith("_")}
    assert set(collaborators) == {"store", "embedder", "config"}
    assert "llm" not in " ".join(collaborators.values()).lower()


async def test_subject_key_is_order_insensitive(consolidator):
    """The canonicalization that took sh_32k from 0.87 to 0.94.

    P1's dominant failure was naming one subject two ways, and the two namings are
    almost always the same content words reordered. The key sorts tokens and drops
    function words so those collide deterministically -- no threshold, no embedding.
    """
    con, store = consolidator
    await con.consolidate(
        SESSION, [candidate("associated with football", "sport associated with Tunisia team")]
    )
    outcomes = await con.consolidate(
        SESSION, [candidate("associated with basketball", "sport Tunisia team is associated with")]
    )
    assert outcomes[0].case is ConsolidationCase.CONTRADICTION
    live = await store.live_facts_for_subject(
        SESSION, subject_key("sport associated with Tunisia team")
    )
    assert [f.fact for f in live] == ["associated with basketball"]


def test_subject_key_examples():
    assert subject_key("city where Diego Rivera worked") == subject_key("city Diego Rivera worked in")
    assert subject_key("The Deploy-Target") == subject_key("deploy target")
    assert subject_key("country where baseball was created") == subject_key(
        "country baseball was created in"
    )
    # Distinct subjects must still be distinct: different content words, not just order.
    assert subject_key("capital of Germany") != subject_key("capital of France")


async def test_subject_label_keeps_the_readable_name(consolidator):
    """The canonical key reads as sorted tokens, so the natural phrasing is preserved
    separately -- it is what gets shown back to P1 and printed in traces."""
    con, store = consolidator
    await con.consolidate(SESSION, [candidate("deploys to staging", "deploy target")])
    assert await store.subject_slots(SESSION) == ["deploy target"]
    stored = next(iter(store.facts.values()))
    assert stored.subject_key == "deploy target"  # already canonical
    assert stored.subject_label == "deploy target"


def test_narrower_entity_is_not_the_same_subject():
    """Guards against a tempting-but-wrong "fix".

    Subset-merging subject keys (treat A as B when tokens(A) ⊂ tokens(B)) scores better
    in aggregate -- sh_32k 94 -> 96 -- and is still wrong. Most of its merges add a
    relation word ("religion of X" vs "religion X is affiliated with"), but some add an
    entity word and NARROW the subject: `buddhism founder` ⊂ `buddhism founder shingon`.
    Buddhism and Shingon Buddhism have different founders and different answer chains;
    merging them supersedes a correct fact and loses it permanently.

    Over-merge is the unrecoverable direction (see ConsolidationConfig), so the extra two
    points are not worth it.

    The gated version this docstring used to ask for now exists -- `memore.aliases` merges
    on set containment ONLY when the extra tokens are generic relation words by document
    frequency, and refuses this exact pair (`tests/test_aliases.py`). This assertion still
    holds and still belongs here: `subject_key` alone must never merge them. Aliasing is a
    separate layer with its own evidence, not a loosening of normalization.
    """
    assert subject_key("founder of Buddhism") != subject_key("founder of Shingon Buddhism")
    assert subject_key("headquarters of Google") != subject_key("headquarters of Google Cloud")


# ---------------------------------------------------------------------------
# Attribute slots (RESULTS.md §11)
# ---------------------------------------------------------------------------
#
# These are built from three real gateway sessions, not from MemoryAgentBench, and that
# is the point. FactConsolidation is constructed as repeated updates to ONE attribute per
# subject, so "a subject holds exactly one live fact" is true there by construction and
# the defect below is structurally invisible to the bench -- sh_6k, sh_32k and mh_6k
# cannot move whether this works or not. They stay a regression guard; this is the test.
#
# Measured on the three sessions before the fix: 18 supersedes fired, 1 was correct.


async def test_compatible_attributes_of_one_subject_all_stay_live(consolidator):
    """The §11 defect, verbatim from session 1's `current memory system` subject.

    Six facts, every one of them true at the same time, arriving under one subject
    because that is the topic P1 named. Before the fix, five were marked SUPERSEDED --
    including "written in Python", which the model was then asked about two turns later
    and had to answer from a fact labelled stale.
    """
    con, store = consolidator
    subject = "the memory system"
    facts = [
        ("the memory system is written in Python", "implementation language"),
        ("the memory system extraction is asynchronous", "extraction timing"),
        ("the memory system extraction returns structured data", "extraction output"),
        ("the memory system uses SUPERSEDED tags", "supersession scheme"),
        ("the memory system was written in Den Haag", "origin location"),
        ("the memory system lookup takes 70-90ms", "lookup latency"),
    ]
    for fact, attribute in facts:
        await con.consolidate(SESSION, [candidate(fact, subject, attribute=attribute)])

    live = await store.live_facts_for_subject(SESSION, subject_key(subject))
    assert len(live) == len(facts)
    assert {f.fact for f in live} == {f for f, _ in facts}


async def test_a_new_slot_supersedes_nothing(consolidator):
    """"The user is a software engineer" was marked SUPERSEDED in all three sessions,
    knocked out by "Bart specialises in memory systems for LLMs" -- same subject, no
    disagreement at all. Both are still true; both must stay live."""
    con, store = consolidator
    await con.consolidate(SESSION, [candidate("the user is a software engineer", "the user",
                                              attribute="profession")])
    outcomes = await con.consolidate(
        SESSION,
        [candidate("Bart specialises in memory systems for LLMs", "the user",
                   attribute="specialisation")],
    )
    assert [o.case for o in outcomes] == [ConsolidationCase.NEW]
    assert [o.superseded_fact_id for o in outcomes] == [None]
    live = await store.live_facts_for_subject(SESSION, subject_key("the user"))
    assert len(live) == 2


async def test_same_slot_still_contradicts(consolidator):
    """The half that must NOT regress. Amsterdam -> Den Haag was the one correct
    supersede in three sessions of traces; slotting must not cost it."""
    con, store = consolidator
    await con.consolidate(SESSION, [candidate("The capital of the Netherlands is Amsterdam",
                                              "the Netherlands", attribute="capital city")])
    outcomes = await con.consolidate(
        SESSION,
        [candidate("The capital of the Netherlands is Den Haag", "the Netherlands",
                   attribute="capital city")],
    )
    assert [o.case for o in outcomes] == [ConsolidationCase.CONTRADICTION]
    live = await store.live_facts_for_subject(SESSION, subject_key("the Netherlands"))
    assert [f.fact for f in live] == ["The capital of the Netherlands is Den Haag"]
    # Supersede, never delete.
    assert len(store.facts) == 2


async def test_contradiction_supersedes_only_its_own_slot(consolidator):
    """The landmine: `targets = live` would have made this fix a no-op.

    A contradiction about the deploy target must not take the unrelated live facts on
    the same subject down with it.
    """
    con, store = consolidator
    subject = "the user"
    await con.consolidate(SESSION, [candidate("deploys to staging by default", subject,
                                              attribute="deploy target")])
    await con.consolidate(SESSION, [candidate("the user is 58", subject, attribute="age")])
    await con.consolidate(SESSION, [candidate("the user likes Python", subject,
                                              attribute="language preference")])
    outcomes = await con.consolidate(
        SESSION,
        [candidate("the default deployment target is now production", subject,
                   attribute="deploy target")],
    )
    assert [o.case for o in outcomes] == [ConsolidationCase.CONTRADICTION]
    live = {f.fact for f in await store.live_facts_for_subject(SESSION, subject_key(subject))}
    assert live == {
        "the default deployment target is now production",
        "the user is 58",
        "the user likes Python",
    }


async def test_attribute_key_is_order_insensitive_like_the_subject(consolidator):
    """Slots go through the same normalization as subjects, so P1 rephrasing a property
    does not silently open a second slot and lose the contradiction."""
    con, store = consolidator
    await con.consolidate(SESSION, [candidate("deploys to staging", "the user",
                                              attribute="the deploy target")])
    outcomes = await con.consolidate(
        SESSION, [candidate("deploys to prod", "the user", attribute="Deploy-Target")]
    )
    assert [o.case for o in outcomes] == [ConsolidationCase.CONTRADICTION]


async def test_unslotted_candidate_still_supersedes_everything(consolidator):
    """Backward compatibility, and the safe error direction.

    With no slot information there is nothing to reason with, so we fall back to
    over-superseding: the right answer stays live and a neighbour is merely mislabelled.
    Under-superseding would leave a genuinely dead fact presented as current, which is
    the failure this project exists to prevent.
    """
    con, store = consolidator
    await con.consolidate(SESSION, [candidate("deploys to staging", "the user",
                                              attribute="deploy target")])
    await con.consolidate(SESSION, [candidate("the user is 58", "the user", attribute="age")])
    outcomes = await con.consolidate(SESSION, [candidate("deploys to prod", "the user")])
    assert [o.case for o in outcomes] == [ConsolidationCase.CONTRADICTION]
    live = await store.live_facts_for_subject(SESSION, subject_key("the user"))
    assert [f.fact for f in live] == ["deploys to prod"]


async def test_duplicate_is_caught_across_slots(consolidator):
    """DUPLICATE is checked against every live fact on the subject, not just the
    competing slot: the same sentence under a different property is still a second copy,
    and store bloat is what the DUPLICATE case exists to prevent."""
    con, store = consolidator
    await con.consolidate(SESSION, [candidate("the user is 58", "the user", attribute="age")])
    outcomes = await con.consolidate(
        SESSION, [candidate("the user is 58", "the user", attribute="age in years")]
    )
    assert [o.case for o in outcomes] == [ConsolidationCase.DUPLICATE]
    assert len(store.facts) == 1


async def test_subject_slots_shows_properties_back_to_p1(consolidator):
    """P1 can only reuse a property string it has been shown -- and reusing it is what
    keeps a contradiction colliding."""
    con, store = consolidator
    await con.consolidate(SESSION, [candidate("deploys to staging", "deploy setup",
                                              attribute="deploy target")])
    await con.consolidate(SESSION, [candidate("uses GitHub Actions", "deploy setup",
                                              attribute="ci provider")])
    assert await store.subject_slots(SESSION) == ["deploy setup -> ci provider, deploy target"]


async def test_attribute_label_keeps_the_natural_phrasing(consolidator):
    """The key is sorted tokens; the label is what P1 gets shown back to reuse.

    RESULTS.md §14: `subject_slots` was feeding the extractor the normalized key --
    "list todo", "latency lookup", "city code favourite python write" -- with an
    instruction to reuse it verbatim. Nobody reuses that, so P1 coined a fresh slot and
    the contradiction never fired. Exactly why `subject_label` exists for subjects.
    """
    con, store = consolidator
    await con.consolidate(
        SESSION,
        [candidate("the user has 'Get dogfood' on their todo list", "the user",
                   attribute="todo list")],
    )
    stored = next(iter(store.facts.values()))
    assert stored.attribute == "list todo"          # sorted key decides identity
    assert stored.attribute_label == "todo list"    # natural phrasing is what P1 sees
    assert await store.subject_slots(SESSION) == ["the user -> todo list"]


async def test_attribute_label_does_not_participate_in_matching(consolidator):
    """The label is display-and-prompt only.

    If it ever fed the competing-set check, identity would become order-sensitive again
    and §11's tests would not catch it -- they pass attributes explicitly, so they never
    exercise two phrasings of one slot.
    """
    con, store = consolidator
    await con.consolidate(SESSION, [candidate("deploys to staging", "the user",
                                              attribute="deploy target")])
    outcomes = await con.consolidate(
        SESSION, [candidate("deploys to prod", "the user", attribute="target deploy")]
    )
    assert [o.case for o in outcomes] == [ConsolidationCase.CONTRADICTION]
    live = await store.live_facts_for_subject(SESSION, subject_key("the user"))
    assert [f.fact for f in live] == ["deploys to prod"]
