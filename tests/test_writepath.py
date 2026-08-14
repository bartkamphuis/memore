"""Write path suite -- recall-writepath-spec.md §8.

Uses a scripted extractor so P2/P3 are testable without a model, per §8's instruction.
"""

from __future__ import annotations

import pytest

from memore.config import WritePathConfig
from memore.consolidate import ConsolidationConfig, DeterministicConsolidator
from memore.embed import StubEmbedder
from memore.extract import ScriptedExtractor
from memore.store.fake import InMemoryStore
from memore.types import CandidateFact, ConsolidationCase, FactType
from memore.writepath import WritePath

SESSION = "s1"


def candidate(fact: str, subject: str, confidence: float = 0.9) -> CandidateFact:
    return CandidateFact(
        fact=fact, type=FactType.PREFERENCE, confidence=confidence, valid_at=None, subject_hint=subject
    )


def build(script, **config_kwargs):
    store = InMemoryStore()
    consolidator = DeterministicConsolidator(
        store, StubEmbedder(), ConsolidationConfig(use_embedding_comparison=False)
    )
    path = WritePath(
        ScriptedExtractor(script), consolidator, WritePathConfig(**config_kwargs), store=store
    )
    return path, store


async def test_transient_turn_stores_nothing():
    """The headline test (§8): the common case writes nothing."""
    path, store = build([[]])
    result = await path.run(SESSION, "run the tests again")
    assert result.candidates == 0
    assert result.outcomes == []
    assert await store.count(SESSION) == 0


async def test_durable_turn_stores_a_fact():
    path, store = build([[candidate("deploys to staging by default", "deploy target")]])
    result = await path.run(SESSION, "I deploy to staging by default")
    assert [o.case for o in result.outcomes] == [ConsolidationCase.NEW]
    assert await store.count(SESSION) == 1


async def test_contradiction_across_turns_supersedes():
    path, store = build(
        [
            [candidate("deploys to staging by default", "deploy target")],
            [candidate("deploys to prod by default", "deploy target")],
        ]
    )
    await path.run(SESSION, "I deploy to staging by default")
    result = await path.run(SESSION, "actually I've moved everything to prod now")
    assert result.outcomes[0].case is ConsolidationCase.CONTRADICTION
    assert result.outcomes[0].superseded_fact_id is not None
    assert await store.count(SESSION) == 2


async def test_disabled_is_a_full_no_op():
    """§4: enabled=False -> no extraction, no ingest."""
    extractor_script = [[candidate("deploys to staging", "deploy target")]]
    path, store = build(extractor_script, enabled=False)
    result = await path.run(SESSION, "I deploy to staging by default")
    assert result.candidates == 0
    assert await store.count(SESSION) == 0
    assert path.extractor.calls == 0, "the extractor must not even be called"


async def test_known_subjects_are_fed_back_to_p1():
    """The fix for the dominant accuracy failure (RESULTS.md §3): P1 must see the
    subjects already in the store so it reuses a key instead of coining a synonym."""
    seen: list[list[str]] = []

    class RecordingExtractor(ScriptedExtractor):
        async def extract(self, user_message, assistant_response, recent, known_subjects=None):
            seen.append(list(known_subjects or []))
            return await super().extract(user_message, assistant_response, recent, known_subjects)

    store = InMemoryStore()
    consolidator = DeterministicConsolidator(
        store, StubEmbedder(), ConsolidationConfig(use_embedding_comparison=False)
    )
    extractor = RecordingExtractor(
        [
            [candidate("deploys to staging by default", "deploy target")],
            [candidate("deploys to prod by default", "deploy target")],
        ]
    )
    path = WritePath(extractor, consolidator, WritePathConfig(), store=store)

    await path.run(SESSION, "turn one")
    await path.run(SESSION, "turn two")

    assert seen[0] == [], "nothing in the store on the first turn"
    assert seen[1] == ["deploy target"], "the second turn must see the existing subject"


async def test_missing_store_does_not_crash_but_is_visible(caplog):
    """Subject feedback is a hint, not a hard dependency -- but it must not vanish
    silently, because losing it costs accuracy."""
    import logging

    store = InMemoryStore()
    consolidator = DeterministicConsolidator(
        store, StubEmbedder(), ConsolidationConfig(use_embedding_comparison=False)
    )
    path = WritePath(ScriptedExtractor([[]]), consolidator, WritePathConfig(), store=None)
    with caplog.at_level(logging.DEBUG, logger="memore.writepath"):
        assert await path._known_subjects(SESSION) == []
    assert any("no store wired" in r.message for r in caplog.records)


@pytest.mark.parametrize("confidence", [0.1, 0.59])
async def test_low_confidence_candidates_are_dropped_in_p1(confidence):
    """§1.3's floor lives in the extractor, so a scripted extractor bypasses it. This
    asserts the real extractor's contract via its filtering logic."""
    from memore.extract import OllamaExtractor

    class FakeLLM:
        async def chat_json(self, messages, schema):
            return {
                "facts": [
                    {
                        "fact": "deploys to staging",
                        "type": "PREFERENCE",
                        "confidence": confidence,
                        "subject_hint": "deploy target",
                    }
                ]
            }

    extractor = OllamaExtractor(WritePathConfig(min_extract_confidence=0.6), FakeLLM())
    assert await extractor.extract("msg", "", []) == []
