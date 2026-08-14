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


def candidate(fact: str, subject: str, confidence: float = 0.9) -> CandidateFact:
    return CandidateFact(
        fact=fact, type=FactType.PREFERENCE, confidence=confidence, valid_at=None, subject_hint=subject
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
    assert await store.subject_labels(SESSION) == ["deploy target"]
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
