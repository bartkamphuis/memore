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
    fact: str,
    subject: str,
    confidence: float = 0.9,
    attribute: str = "",
    single_valued: bool = True,
) -> CandidateFact:
    """`attribute` defaults to "" -- unspecified, colliding with every slot.

    That default is why the tests written before RESULTS.md §11 still pin exactly what
    they pinned: an extractor that names no slot gets the pre-§11 behaviour. `single_valued`
    defaults True for the same reason and RESULTS.md §18's: an extractor that answers
    nothing gets the pre-§18 behaviour.
    """
    return CandidateFact(
        fact=fact,
        type=FactType.PREFERENCE,
        confidence=confidence,
        valid_at=None,
        subject_hint=subject,
        attribute=attribute,
        single_valued=single_valued,
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


# ---------------------------------------------------------------------------
# Same-batch collisions and subsumption
# ---------------------------------------------------------------------------
#
# Trace-derived, like the §11 block above, and from the same kind of source: the
# two-column gateway console run of 2026-08-17, 24 turns per column against identical
# input. The FactConsolidation bench cannot express any of this either -- it feeds one
# fact per turn, so a batch is always of size one and a same-batch collision cannot
# occur. The candidate texts below are copied from that run's write log verbatim.
#
# The two defects are opposite in shape and both cost a live fact:
#
#   same-batch supersede   one utterance, two independently-true facts, one slot. The
#                          second retired the first on an ordinal that only recorded its
#                          position in P1's output array. Fired in BOTH columns.
#   subsumption inversion  a shorter restatement of a stored fact retired the longer
#                          original on recency, dropping the detail it did not repeat.


async def test_same_batch_siblings_do_not_supersede_each_other(consolidator):
    """Turn 4, verbatim, both columns: "I like milk in my coffee, but not in my tea.
    I like Green tea".

    P1 emitted both tea facts under one `tea preference` slot. Both are true. They
    arrived in one `consolidate()` call, so nothing orders them -- ordinal 11 vs 12 is
    array position, not chronology. Before the guard the second retired the first and
    the reader was handed a still-true fact labelled SUPERSEDED.
    """
    con, store = consolidator
    outcomes = await con.consolidate(
        SESSION,
        [
            candidate("the user likes milk in their coffee", "the user",
                      attribute="coffee preference"),
            candidate("the user does not like milk in their tea", "the user",
                      attribute="tea preference"),
            candidate("the user likes green tea", "the user", attribute="tea preference"),
        ],
    )
    assert [o.case for o in outcomes] == [ConsolidationCase.NEW] * 3
    assert all(o.superseded_fact_id is None for o in outcomes)

    live = await store.live_facts_for_subject(SESSION, subject_key("the user"))
    assert {f.fact for f in live} == {
        "the user likes milk in their coffee",
        "the user does not like milk in their tea",
        "the user likes green tea",
    }


async def test_the_guard_is_per_batch_not_per_slot(consolidator):
    """The money case must survive the fix.

    Same subject, same slot, different values, but two SEPARATE calls -- which is a real
    freshness ordering and must still resolve. A guard written against the slot rather
    than the batch would disarm the primitive this project exists to test.
    """
    con, store = consolidator
    await con.consolidate(
        SESSION, [candidate("the user likes green tea", "the user", attribute="tea preference")]
    )
    outcomes = await con.consolidate(
        SESSION, [candidate("the user hates green tea", "the user", attribute="tea preference")]
    )
    assert [o.case for o in outcomes] == [ConsolidationCase.CONTRADICTION]
    live = await store.live_facts_for_subject(SESSION, subject_key("the user"))
    assert [f.fact for f in live] == ["the user hates green tea"]


async def test_same_batch_refinement_is_still_allowed(consolidator):
    """Subsumption needs no ordering, so the guard must not block it.

    This is why the guard sits AFTER the two containment branches rather than before
    them: REFINEMENT is justified by what the strings say, CONTRADICTION by which
    arrived later. Only the second is unavailable inside a batch.
    """
    con, store = consolidator
    outcomes = await con.consolidate(
        SESSION,
        [
            candidate("Lisa leaves Amsterdam at 3:00 PM", "Lisa", attribute="departure"),
            candidate("Lisa leaves Amsterdam at 3:00 PM on Tuesday, August 18th", "Lisa",
                      attribute="departure"),
        ],
    )
    assert [o.case for o in outcomes] == [ConsolidationCase.NEW, ConsolidationCase.REFINEMENT]
    live = await store.live_facts_for_subject(SESSION, subject_key("Lisa"))
    assert [f.fact for f in live] == ["Lisa leaves Amsterdam at 3:00 PM on Tuesday, August 18th"]


async def test_same_batch_duplicate_still_collapses(consolidator):
    """The guard withholds superseding, not the duplicate scan.

    Two identical candidates in one extraction must still store one copy, or the guard
    trades a mislabelled fact for store bloat.
    """
    con, store = consolidator
    outcomes = await con.consolidate(
        SESSION,
        [
            candidate("Lisa likes red tulips", "Lisa", attribute="preference"),
            candidate("Lisa likes red tulips", "Lisa", attribute="preference"),
        ],
    )
    assert [o.case for o in outcomes] == [ConsolidationCase.NEW, ConsolidationCase.DUPLICATE]
    assert await store.count(SESSION) == 1


async def test_a_restatement_that_says_less_does_not_supersede(consolidator):
    """Turns 11 and 16, column 2, verbatim.

    The arriving fact is a strict substring of the stored one: it disagrees about
    nothing. Before the containment branch it fell through to CONTRADICTION and won on
    recency, and "in the Whangarei office" left the live set. Both stay live now -- see
    the next test for why the shorter one is not discarded instead.
    """
    con, store = consolidator
    await con.consolidate(
        SESSION,
        [candidate("Bud sits in the Red chair in the Whangarei office", "Bud",
                   attribute="seating")],
    )
    outcomes = await con.consolidate(
        SESSION, [candidate("Bud sits in the red chair", "Bud", attribute="seating")]
    )
    assert [o.case for o in outcomes] == [ConsolidationCase.NEW]
    assert outcomes[0].superseded_fact_id is None
    live = await store.live_facts_for_subject(SESSION, subject_key("Bud"))
    assert "Bud sits in the Red chair in the Whangarei office" in {f.fact for f in live}


async def test_a_narrowing_update_is_not_discarded_as_a_restatement(consolidator):
    """Why the containment branch coexists instead of answering DUPLICATE.

    The only containment pair in sh_32k's 2310 facts is a real value change, and it has
    exactly the same surface shape as the Whangarei restatement above. Answering
    DUPLICATE would drop the update permanently -- the unrecoverable direction
    `ConsolidationConfig` documents -- so both facts stay live and the reader gets both.
    """
    con, store = consolidator
    await con.consolidate(
        SESSION,
        [candidate("flanker is associated with the sport of rugby union", "flanker",
                   attribute="sport")],
    )
    outcomes = await con.consolidate(
        SESSION,
        [candidate("flanker is associated with the sport of rugby", "flanker",
                   attribute="sport")],
    )
    assert [o.case for o in outcomes] == [ConsolidationCase.NEW]
    live = {f.fact for f in await store.live_facts_for_subject(SESSION, subject_key("flanker"))}
    assert "flanker is associated with the sport of rugby" in live


async def test_the_containment_branch_is_inert_without_a_slot(consolidator):
    """No attribute means no "same property", so the branch must not fire.

    This is what keeps the bench and pre-§11 graphs on exactly the behaviour they were
    measured with: `bench/extract.py` supplies no attribute, so every FactConsolidation
    run takes the CONTRADICTION path it always took.
    """
    con, store = consolidator
    await con.consolidate(
        SESSION, [candidate("flanker is associated with the sport of rugby union", "flanker")]
    )
    outcomes = await con.consolidate(
        SESSION, [candidate("flanker is associated with the sport of rugby", "flanker")]
    )
    assert [o.case for o in outcomes] == [ConsolidationCase.CONTRADICTION]
    live = await store.live_facts_for_subject(SESSION, subject_key("flanker"))
    assert [f.fact for f in live] == ["flanker is associated with the sport of rugby"]


async def test_a_changed_value_is_not_mistaken_for_a_restatement(consolidator):
    """The containment branch must not swallow an update.

    Containment in that direction is safe precisely because a changed value is not a
    substring of the fact it changes -- this pins that, since a false DUPLICATE
    discards the update permanently (see `ConsolidationConfig`).
    """
    con, store = consolidator
    await con.consolidate(
        SESSION, [candidate("Bud sits in the red chair", "Bud", attribute="seating")]
    )
    outcomes = await con.consolidate(
        SESSION, [candidate("Bud sits in the blue chair", "Bud", attribute="seating")]
    )
    assert [o.case for o in outcomes] == [ConsolidationCase.CONTRADICTION]
    live = await store.live_facts_for_subject(SESSION, subject_key("Bud"))
    assert [f.fact for f in live] == ["Bud sits in the blue chair"]


async def test_a_later_contradiction_spares_the_incumbents_batch_siblings(consolidator):
    """The bill for letting two facts coexist in one slot, and why it is not paid.

    Turn 4 leaves `tea preference` holding two live facts on purpose. A correction two
    turns later contradicts one of them. Superseding "everything in the slot" -- the
    pre-fix self-healing rule -- would collect the milk fact as collateral, which is the
    original defect arriving one turn late instead of being fixed.
    """
    con, store = consolidator
    await con.consolidate(
        SESSION,
        [
            candidate("the user does not like milk in their tea", "the user",
                      attribute="tea preference"),
            candidate("the user likes green tea", "the user", attribute="tea preference"),
        ],
    )
    outcomes = await con.consolidate(
        SESSION,
        [candidate("the user hates green tea", "the user", attribute="tea preference")],
    )
    assert [o.case for o in outcomes] == [ConsolidationCase.CONTRADICTION]

    live = await store.live_facts_for_subject(SESSION, subject_key("the user"))
    assert {f.fact for f in live} == {
        "the user does not like milk in their tea",
        "the user hates green tea",
    }


async def test_facts_written_without_a_batch_keep_the_old_supersede_behaviour(consolidator):
    """Inertness on a graph written before `source_episode_id` carried a batch.

    Same argument as `attribute == ""` in `_competing`: with no batch information the
    safe error is to over-supersede, so a pre-fix store behaves exactly as it did rather
    than silently acquiring a rule its data cannot support.
    """
    from datetime import UTC, datetime

    from memore.types import StoredFact

    con, store = consolidator
    subject = subject_key("the user")
    for i, text in enumerate(("old fact one", "old fact two"), start=1):
        await store.add_fact(
            StoredFact(
                id=f"legacy-{i}",
                session_id=SESSION,
                fact=text,
                subject_key=subject,
                subject_label="the user",
                ordinal=i,
                valid_at=datetime.now(UTC),
                invalid_at=None,
                source_episode_id="",
                attribute=subject_key("tea preference"),
            ),
            [0.0] * 8,
        )

    outcomes = await con.consolidate(
        SESSION, [candidate("a brand new claim", "the user", attribute="tea preference")]
    )
    assert [o.case for o in outcomes] == [ConsolidationCase.CONTRADICTION]
    live = await store.live_facts_for_subject(SESSION, subject)
    assert [f.fact for f in live] == ["a brand new claim"]


async def test_a_multi_valued_slot_accumulates_instead_of_superseding(consolidator):
    """RESULTS.md §18: the defect, and the field that answers it.

    Four preferences of one subject, arriving on four SEPARATE turns so the §16 same-batch
    guard cannot be what saves them, all landing in the one slot P1 actually names them
    with (`preference` -- a category, not a property). Before `single_valued` each retired
    the last and three true facts were presented as SUPERSEDED.
    """
    con, store = consolidator
    for text in (
        "Bud likes Lisa",
        "Bud likes beer",
        "Bud likes the first Matrix movie",
        "Bud likes red gaming chairs",
    ):
        outcomes = await con.consolidate(
            SESSION, [candidate(text, "Bud", attribute="preference", single_valued=False)]
        )
        assert [o.case for o in outcomes] == [ConsolidationCase.NEW]

    live = await store.live_facts_for_subject(SESSION, subject_key("Bud"))
    assert {f.fact for f in live} == {
        "Bud likes Lisa",
        "Bud likes beer",
        "Bud likes the first Matrix movie",
        "Bud likes red gaming chairs",
    }


async def test_a_single_valued_slot_still_resolves(consolidator):
    """The refuse-list, as a test rather than a harness pair.

    A favourite IS a preference and it holds one value at a time. Any fix that keys on the
    fact's TYPE or on words like "likes"/"prefers" -- the `COLLECTION_TYPES` shape §18.6
    refuses -- passes the test above and fails this one, leaving a slot that can never
    resolve. That is the FactConsolidation task failing.
    """
    con, store = consolidator
    slot = "favourite programming language"
    await con.consolidate(
        SESSION,
        [candidate("Bud's favourite programming language is Go", "Bud", attribute=slot)],
    )
    outcomes = await con.consolidate(
        SESSION,
        [candidate("Bud's favourite programming language is Rust", "Bud", attribute=slot)],
    )
    assert [o.case for o in outcomes] == [ConsolidationCase.CONTRADICTION]

    live = await store.live_facts_for_subject(SESSION, subject_key("Bud"))
    assert [f.fact for f in live] == ["Bud's favourite programming language is Rust"]


async def test_a_multi_valued_slot_still_rejects_an_exact_duplicate(consolidator):
    """Withholding CONTRADICTION must not also withhold DUPLICATE.

    The check sits below the exact-duplicate scan deliberately: a collection that
    accumulates copies of the same sentence is the store bloat writepath §2.2 case 2
    exists to prevent, and nothing about a slot holding several values makes a repeat of
    one of them news.
    """
    con, store = consolidator
    for _ in range(2):
        await con.consolidate(
            SESSION,
            [candidate("Bud likes beer", "Bud", attribute="preference", single_valued=False)],
        )
    outcomes = await con.consolidate(
        SESSION,
        [candidate("Bud likes beer", "Bud", attribute="preference", single_valued=False)],
    )
    assert [o.case for o in outcomes] == [ConsolidationCase.DUPLICATE]
    live = await store.live_facts_for_subject(SESSION, subject_key("Bud"))
    assert [f.fact for f in live] == ["Bud likes beer"]


async def test_an_extractor_that_never_answers_gets_the_pre_18_behaviour(consolidator):
    """Inertness, the same argument `attribute == ""` makes.

    The bench's cached subject extraction (`bench/extract.py`) constructs CandidateFacts
    without this field, so every §3 oracle number must be reproducible unchanged. The
    default is True, and True is exactly the old code path.
    """
    con, store = consolidator
    await con.consolidate(SESSION, [candidate("the user likes tea", "the user")])
    outcomes = await con.consolidate(SESSION, [candidate("the user likes coffee", "the user")])
    assert [o.case for o in outcomes] == [ConsolidationCase.CONTRADICTION]
    live = await store.live_facts_for_subject(SESSION, subject_key("the user"))
    assert [f.fact for f in live] == ["the user likes coffee"]
