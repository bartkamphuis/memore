"""Suite 11 -- store adapter, against real FalkorDB (recall-stage-test-spec.md).

The only suite that touches an external service. Gated behind the `integration` marker
so unit runs stay fast:

    uv run pytest -m integration        # needs `docker compose up -d falkordb`
    uv run pytest -m "not integration"  # default-fast
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from memore.config import StoreConfig
from memore.consolidate import ConsolidationConfig, DeterministicConsolidator, subject_key
from memore.embed import StubEmbedder
from memore.store.falkor import FalkorStore
from memore.subjects import build_subject_view
from memore.types import CandidateFact, ConsolidationCase, FactType

pytestmark = pytest.mark.integration

DIM = 8


def _ollama_reachable() -> bool:
    """Whether the configured Ollama is answering.

    Every test here runs on `StubEmbedder` and needs only FalkorDB -- except
    `test_irrelevant_query_stays_below_the_gate_floor`, whose whole claim is about real
    semantic distance and so cannot be made with stub vectors. CI has FalkorDB (a service
    container) and no Ollama: the models are served on Marvin's GPUs, not on a runner.

    Probed rather than env-flagged so a local run never skips silently while Ollama is up.
    A reachable-but-broken Ollama still FAILS the test, which is the point -- this only
    distinguishes "not installed here" from "wrong answer".
    """
    import httpx

    from memore.config import EmbedConfig

    try:
        return httpx.get(f"{EmbedConfig.from_env().ollama_url}/api/tags", timeout=2.0).is_success
    except Exception:
        return False


needs_ollama = pytest.mark.skipif(
    not _ollama_reachable(), reason="needs a live Ollama for real embeddings; CI has none"
)


async def drop_graph(falkor: FalkorStore) -> None:
    """Delete the throwaway graph itself, not just the facts in it.

    `MATCH (f:Fact) DELETE f` empties a graph but leaves the *key*, and an empty graph
    is not free: FalkorDB keeps per-graph schema and index structures for it. Every
    fixture instance in this suite mints a fresh `test_<hex>` name, so each run leaked
    one shell per test -- 740 zero-node graphs had piled up before anyone ran
    `GRAPH.LIST`. Call this in teardown, before `aclose()` closes the connection.

    Best-effort like `aclose()`: a failure here must not mask the assertion that
    actually failed.
    """
    try:
        await falkor._graph.delete()
    except Exception:  # noqa: BLE001 -- teardown is best-effort
        pass


@pytest.fixture
async def store():
    # StubEmbedder emits 8-dim vectors; use a throwaway graph so the dimension of the
    # real 768-d graph is untouched and tests never collide with bench data.
    config = StoreConfig(graph_name=f"test_{uuid.uuid4().hex[:8]}")
    falkor = FalkorStore(config, dimension=DIM)
    await falkor.connect()
    yield falkor
    await falkor._q("MATCH (f:Fact) DELETE f")
    await drop_graph(falkor)
    await falkor.aclose()


def candidate(fact: str, subject: str) -> CandidateFact:
    return CandidateFact(
        fact=fact, type=FactType.STATE, confidence=0.9, valid_at=None, subject_hint=subject
    )


async def test_ingest_search_roundtrip(store):
    embedder = StubEmbedder(DIM)
    con = DeterministicConsolidator(store, embedder)
    await con.consolidate("s1", [candidate("deploys to staging by default", "deploy target")])

    query_vec = await embedder.embed_one("deploys to staging by default")
    hits = await store.hybrid_search("deploy staging", query_vec, "s1", 5)
    assert [h.fact for h in hits] == ["deploys to staging by default"]
    assert 0.0 <= hits[0].score <= 1.0
    assert hits[0].valid_at is not None


async def test_scores_are_normalized_zero_to_one(store):
    embedder = StubEmbedder(DIM)
    con = DeterministicConsolidator(store, embedder)
    await con.consolidate(
        "s1",
        [candidate(f"fact number {i} about topic {i}", f"topic {i}") for i in range(8)],
    )
    query_vec = await embedder.embed_one("fact number 3 about topic 3")
    hits = await store.hybrid_search("fact number 3", query_vec, "s1", 8)
    assert hits
    assert all(0.0 <= h.score <= 1.0 for h in hits)
    assert hits == sorted(hits, key=lambda h: h.score, reverse=True)


async def test_session_scoping(store):
    """§13: no cross-session recall in v1."""
    embedder = StubEmbedder(DIM)
    con = DeterministicConsolidator(store, embedder)
    await con.consolidate("session-a", [candidate("a-only secret", "secret")])
    await con.consolidate("session-b", [candidate("b-only secret", "secret")])

    query_vec = await embedder.embed_one("a-only secret")
    hits = await store.hybrid_search("secret", query_vec, "session-b", 10)
    assert [h.fact for h in hits] == ["b-only secret"]


async def test_session_scoping_survives_a_crowded_index(store):
    """The regression this pins: FalkorDB's vector index is global and its query
    procedure takes no filter, so session scoping happens after the ANN fetch. With a
    fixed over-fetch, a session holding one fact silently returns nothing once other
    sessions crowd the index."""
    embedder = StubEmbedder(DIM)
    con = DeterministicConsolidator(store, embedder)
    await con.consolidate(
        "noisy", [candidate(f"unrelated filler fact {i}", f"filler {i}") for i in range(300)]
    )
    await con.consolidate("tiny", [candidate("the one fact in this session", "the subject")])

    query_vec = await embedder.embed_one("the one fact in this session")
    hits = await store.hybrid_search("the one fact in this session", query_vec, "tiny", 12)
    assert [h.fact for h in hits] == ["the one fact in this session"]


async def test_live_chain_view_returns_only_live_facts(store):
    """Live-only is what keeps the multi-hop walk affordable, not a nicety.

    Superseded facts multiply the out-degree of exactly the hub values that would
    otherwise explode the frontier, so a chain view that leaked them would change the
    walk's cost characteristics, not just its answers (`memore.chain`).
    """
    embedder = StubEmbedder(DIM)
    con = DeterministicConsolidator(store, embedder)
    await con.consolidate("s1", [candidate("The capital of Italy is Rome.", "capital of Italy")])
    await con.consolidate("s1", [candidate("The capital of Italy is Duluth.", "capital of Italy")])
    await con.consolidate("other", [candidate("A fact in another session.", "elsewhere")])

    view = await store.live_chain_view("s1")
    assert [n.fact for n in view] == ["The capital of Italy is Duluth."]
    assert all(n.invalid_at is None for n in view)
    # Subject keys come back normalized, which is what the edge rule compares against.
    assert view[0].subject_key == subject_key("capital of Italy")


async def test_expansion_reaches_a_fact_the_gate_alone_cannot(store):
    """End to end: a chained answer that no similarity score would have surfaced.

    Pinned because this is the whole multi-hop claim. The gate is calibrated on
    similarity to the turn, and a hop-2 fact shares no entity with the turn -- measured
    on `factconsolidation_mh_6k`, the gate alone put the gold answer in the block on 5%
    of questions and the walk took that to 91% (RESULTS.md §8).
    """
    from memore.config import RecallConfig
    from memore.recall import recall
    from memore.types import TurnContext

    embedder = StubEmbedder(DIM)
    con = DeterministicConsolidator(store, embedder)
    await con.consolidate(
        "s1",
        [
            candidate("Igor of Kiev is married to Olga of Kiev.", "spouse of Igor of Kiev"),
            candidate("Olga of Kiev died in the city of Rodez.", "place of death of Olga of Kiev"),
        ],
    )

    # floor 0 isolates the walk from the gate, and k=1 forces retrieval to return only
    # the hop-1 fact -- so reaching Rodez is the walk's doing and cannot be retrieval
    # having found it anyway.
    turn = TurnContext(session_id="s1", user_message="Igor of Kiev is married to Olga of Kiev.")
    base = await recall(turn, RecallConfig(score_floor=0.0, k=1, expansion_hops=0), store, embedder)
    walked = await recall(turn, RecallConfig(score_floor=0.0, k=1, expansion_hops=2), store, embedder)

    assert [h.fact for h in base.memories_used] == ["Igor of Kiev is married to Olga of Kiev."]
    assert "Olga of Kiev died in the city of Rodez." in [h.fact for h in walked.memories_used]
    assert len(walked.memories_used) > len(base.memories_used)
    # Chain facts carry no similarity score -- they were never ranked against the turn.
    chained = [h for h in walked.memories_used if h.score == 0.0]
    assert chained and all(h.invalid_at is None for h in chained)


async def test_widening_is_not_bounded_by_the_graph_size(store):
    """The second half of the crowded-index problem, which hid behind the fix for the
    first: HNSW returns FEWER nodes than asked for, so "I asked for every node in the
    graph" is not "I saw every node in the graph".

    Measured on the calibration graph: 467 facts, ask the index for 467, get 182 back, of
    which zero belonged to the 12-fact session under test -- `recall()` then returned
    nothing and the gate read as correctly shut. Bounding the widening by `total` (which
    is what "stop once fetch covers the graph" means) reintroduces exactly the silent
    empty result the widening exists to prevent.
    """
    embedder = StubEmbedder(DIM)
    con = DeterministicConsolidator(store, embedder)
    await con.consolidate(
        "noisy", [candidate(f"unrelated filler fact {i}", f"filler {i}") for i in range(600)]
    )
    session_facts = [f"session fact number {i}" for i in range(12)]
    await con.consolidate("small", [candidate(f, f"subject {f}") for f in session_facts])

    query_vec = await embedder.embed_one("session fact number 0")
    hits = await store.hybrid_search("session fact number 0", query_vec, "small", 12)
    # Every fact the session holds, not merely "something".
    assert sorted(h.fact for h in hits) == sorted(session_facts)


async def test_supersede_persists_bitemporal_fields(store):
    embedder = StubEmbedder(DIM)
    # StubEmbedder vectors are deterministic but semantically meaningless, so any
    # similarity threshold over them is noise -- two contradictory strings can land
    # above 0.97 and be called DUPLICATE. Real embeddings put contradictions at
    # 0.85-0.91 (measured); the string path is what this test pins.
    con = DeterministicConsolidator(
        store, embedder, ConsolidationConfig(use_embedding_comparison=False)
    )
    await con.consolidate("s1", [candidate("deploys to staging", "deploy target")])
    outcomes = await con.consolidate("s1", [candidate("deploys to prod", "deploy target")])
    assert outcomes[0].case is ConsolidationCase.CONTRADICTION

    live = await store.live_facts_for_subject("s1", subject_key("deploy target"))
    assert [f.fact for f in live] == ["deploys to prod"]
    assert await store.count("s1") == 2

    query_vec = await embedder.embed_one("deploys to staging")
    hits = await store.hybrid_search("deploy", query_vec, "s1", 5)
    superseded = [h for h in hits if h.invalid_at is not None]
    assert [h.fact for h in superseded] == ["deploys to staging"]


async def test_commit_is_idempotent_per_fact(store):
    """writepath §3: re-running a turn's job must not create duplicate facts."""
    from memore.types import StoredFact

    embedder = StubEmbedder(DIM)
    vector = await embedder.embed_one("stable fact")
    fact = StoredFact(
        id="fixed-id",
        session_id="s1",
        fact="stable fact",
        subject_key="stable",
        subject_label="stable subject",
        ordinal=1,
        valid_at=None,
        invalid_at=None,
        source_episode_id="turn-7",
    )
    await store.add_fact(fact, vector)
    await store.add_fact(fact, vector)
    assert await store.count("s1") == 1


async def test_embedder_dimension_mismatch_fails_at_connect():
    """A store whose index was built for another embedder must refuse to open.

    This one degrades into silence rather than an error: the wrong-width write succeeds,
    only the query raises, and `recall()` swallows every store exception by design (§3.1)
    -- so the gate is shut forever and the only symptom is one WARNING per turn. It is a
    single step away, because changing MEMORE_EMBED_MODEL does not change the graph name.
    """
    config = StoreConfig(graph_name=f"test_{uuid.uuid4().hex[:8]}")
    first = FalkorStore(config, dimension=DIM)
    await first.connect()
    try:
        mismatched = FalkorStore(config, dimension=DIM * 2)
        with pytest.raises(RuntimeError, match="dimension vector index"):
            await mismatched.connect()
        await mismatched.aclose()
        # Same width still opens: the guard rejects mismatches, not reconnections.
        again = FalkorStore(config, dimension=DIM)
        await again.connect()
        await again.aclose()
    finally:
        # All three stores share one graph name, so drop it once via the handle that is
        # known to have connected -- `mismatched` is expected to have raised.
        await first._q("MATCH (f:Fact) DELETE f")
        await drop_graph(first)
        await first.aclose()


@needs_ollama
async def test_irrelevant_query_stays_below_the_gate_floor():
    """The precision regression this pins (§6 is the differentiator: inject only when
    the store actually has something relevant).

    With additive fusion `w_v*cos + w_t*bm25_norm`, BM25's max-normalization gave the
    best of two terrible lexical matches a full 1.0, lifting an unrelated fact to 0.419
    against a 0.35 floor. Multiplicative fusion anchors the score on cosine, so an
    irrelevant query cannot be boosted over the floor by keyword noise alone.
    """
    from memore.config import EmbedConfig, RecallConfig
    from memore.embed import OllamaEmbedder

    # Its own graph: the shared fixture indexes 8-dim stub vectors, and this test needs
    # the real embedder to say anything about actual relevance. from_env() (not the bare
    # default) so the configured prefixes and dimension are the ones we ship -- and the
    # index width has to follow the embedder, not a constant, or the store silently holds
    # vectors it cannot search.
    embed_config = EmbedConfig.from_env()
    embedder = OllamaEmbedder(embed_config)
    real = FalkorStore(
        StoreConfig(graph_name=f"test_{uuid.uuid4().hex[:8]}"), dimension=embed_config.dimension
    )
    await real.connect()
    con = DeterministicConsolidator(real, embedder)
    await con.consolidate("s1", [candidate("The user deploys to prod by default.", "deploy target")])

    floor = RecallConfig().score_floor
    relevant = await real.hybrid_search(
        "remind me where I deploy?",
        await embedder.embed_one("remind me where I deploy?", query=True),
        "s1",
        5,
    )
    irrelevant = await real.hybrid_search(
        "what is the weather in Paris?",
        await embedder.embed_one("what is the weather in Paris?", query=True),
        "s1",
        5,
    )
    await embedder.aclose()
    try:
        assert relevant and relevant[0].score >= floor
        assert all(h.score < floor for h in irrelevant)
    finally:
        await real._q("MATCH (f:Fact) DELETE f")
        await drop_graph(real)
        await real.aclose()


# --- Temporal expiry round-trip (RESULTS.md §19) -----------------------------


async def test_occurs_at_and_recurring_survive_the_store(store):
    """The fields are useless unless they come back. Both search arms hydrate them, so
    this asserts through `hybrid_search` rather than off the StoredFact."""
    from datetime import UTC, datetime

    embedder = StubEmbedder(DIM)
    con = DeterministicConsolidator(store, embedder)
    trip = CandidateFact(
        fact="the user is flying to Porto",
        type=FactType.STATE, confidence=0.9, valid_at=None, subject_hint="porto trip",
        attribute="flight date",
        occurs_at=datetime(2026, 5, 12, tzinfo=UTC),
    )
    gym = CandidateFact(
        fact="the user's gym membership renews monthly",
        type=FactType.STATE, confidence=0.9, valid_at=None, subject_hint="gym membership",
        attribute="renewal", recurring=True,
    )
    await con.consolidate("s1", [trip, gym])

    hits = {h.fact: h for h in await store.hybrid_search(
        "porto gym", await embedder.embed_one("porto gym"), "s1", 10
    )}
    assert hits["the user is flying to Porto"].occurs_at.date().isoformat() == "2026-05-12"
    assert hits["the user is flying to Porto"].recurring is False
    # A recurring event carries no date, and must not acquire one on the way through.
    assert hits["the user's gym membership renews monthly"].occurs_at is None
    assert hits["the user's gym membership renews monthly"].recurring is True


async def test_a_fact_with_no_date_reads_back_inert(store):
    """Every fact written before §19 has no `occurs_at` property at all. The read path
    must give None/False for a missing property, not raise and not invent a date."""
    embedder = StubEmbedder(DIM)
    con = DeterministicConsolidator(store, embedder)
    await con.consolidate("s1", [candidate("the user's car is a 2019 Subaru", "car")])

    hits = await store.hybrid_search(
        "car", await embedder.embed_one("car"), "s1", 5
    )
    assert [h.occurs_at for h in hits] == [None]
    assert [h.recurring for h in hits] == [False]


async def test_supersede_preserves_the_temporal_fields(store):
    """Supersede rewrites the node; the date of the event it describes is not what
    changed, so it must still be there afterwards."""
    from datetime import UTC, datetime

    embedder = StubEmbedder(DIM)
    con = DeterministicConsolidator(
        store, embedder, ConsolidationConfig(use_embedding_comparison=False)
    )
    def dated(fact: str, day: int) -> CandidateFact:
        return CandidateFact(
            fact=fact, type=FactType.STATE, confidence=0.9, valid_at=None,
            subject_hint="porto trip", attribute="flight date",
            occurs_at=datetime(2026, 5, day, tzinfo=UTC),
        )

    await con.consolidate("s1", [dated("flying to Porto on the 12th", 12)])
    await con.consolidate("s1", [dated("flying to Porto on the 19th", 19)])

    hits = {h.fact: h for h in await store.hybrid_search(
        "porto", await embedder.embed_one("porto"), "s1", 10
    )}
    superseded = hits["flying to Porto on the 12th"]
    assert superseded.invalid_at is not None, "precondition: the first fact was retired"
    assert superseded.occurs_at.date().isoformat() == "2026-05-12"


async def test_slot_arity_round_trips_and_clear_session_takes_it(store):
    """identity-and-gate-spec.md A1, against the real graph.

    Three things the in-memory store cannot prove: `ON CREATE SET` really refuses the
    implicit overwrite in Cypher, the bare `SET` really performs the deliberate one, and
    `clear_session` really takes the `:Slot` nodes with it. That last one is why it is
    here at all -- nothing else lists slots (`sessions()` counts Facts), so orphans in a
    shared graph would accumulate completely invisibly.
    """
    session = f"slots-{uuid.uuid4().hex[:8]}"
    key, slot = subject_key("the user"), subject_key("creation location")

    await store.ensure_slot_schema(session, key, slot, True)
    assert await store.slot_schemas(session) == [(key, slot, True)]

    # Create-only: a second fact in the slot cannot revise what the first declared.
    await store.ensure_slot_schema(session, key, slot, False)
    assert await store.slot_schemas(session) == [(key, slot, True)]

    # The one deliberate correction path.
    await store.set_slot_schema(session, key, slot, False)
    assert await store.slot_schemas(session) == [(key, slot, False)]

    # `""` is unspecified, not a slot -- nothing to be the arity of, nothing recorded.
    await store.ensure_slot_schema(session, key, "", True)
    assert await store.slot_schemas(session) == [(key, slot, False)]

    await store.clear_session(session)
    assert await store.slot_schemas(session) == []


async def test_a_correction_survives_a_fresh_consolidator(store):
    """A1's third acceptance clause, against the real graph rather than a dict.

    `DeterministicConsolidator.set_slot_schema` writes through to the store AND mutates the
    dict `_slots_for` caches. If those two ever disagree the correction appears to work and
    then evaporates on the next process start -- which is §16.4's shape exactly: a
    store-level behaviour whose unit tests pass while the real path silently loses it. So
    the assertion is made by a SECOND consolidator, which has no cache to be right by
    accident and must read the corrected value back out of the graph.
    """
    session = f"correct-{uuid.uuid4().hex[:8]}"
    key, slot = subject_key("the user"), subject_key("creation location")

    def slotted(fact: str) -> CandidateFact:
        return CandidateFact(
            fact=fact, type=FactType.STATE, confidence=0.9, valid_at=None,
            subject_hint="the user", attribute="creation location",
        )

    con = DeterministicConsolidator(store, StubEmbedder(DIM))
    await con.consolidate(session, [slotted("the user wrote the memory system in Den Haag")])
    outcomes = await con.consolidate(session, [slotted("the user was born in Den Haag")])
    assert [o.case for o in outcomes] == [ConsolidationCase.CONTRADICTION]

    await con.set_slot_schema(session, key, slot, single_valued=False)

    fresh = DeterministicConsolidator(store, StubEmbedder(DIM))
    outcomes = await fresh.consolidate(session, [slotted("the user learned to sail in Den Haag")])
    assert [o.case for o in outcomes] == [ConsolidationCase.NEW]

    live = {f.fact for f in await store.live_facts_for_subject(session, key)}
    assert live == {
        "the user was born in Den Haag",
        "the user learned to sail in Den Haag",
    }
    # No rewrite: the fact retired before the correction is still retired.
    assert "the user wrote the memory system in Den Haag" not in live


async def test_the_cached_subject_view_survives_writes_exactly(store):
    """C6 (RESULTS.md §24). The warm view must equal a cold fetch, after writes.

    The cache is MAINTAINED across writes rather than dropped, because a conversational
    turn is read-then-write and a drop-on-write cache is cold on every turn (§24.3
    measured 1 warm read in 20). That buys the hit rate and takes on the risk: an
    incrementally-updated row that does not match what the query would have returned makes
    a warm view silently different from a cold one, which moves the gate's precision
    without failing anything.

    So this compares against the real thing: do the writes, then drop the cache and
    re-fetch, and require the two to be identical field for field.

    `supersede` is the trap it is written around. It is handed a fact id and no session,
    so the write most likely to corrupt this cache is the one that could least easily
    reach it -- and the field it changes, `invalid_at`, is exactly the one
    `build_subject_view` reads to decide which subjects are live.
    """
    session = f"sv-{uuid.uuid4().hex[:8]}"
    con = DeterministicConsolidator(store, StubEmbedder(DIM))

    await con.consolidate(session, [candidate("the sport is rugby", "sport of Tunisia")])
    warm = await store.subject_view(session)          # populates the cache
    await con.consolidate(session, [candidate("the sport is hurling", "sport of Ireland")])
    live = await store.live_facts_for_subject(session, subject_key("sport of Tunisia"))
    await store.supersede(live[0].id, datetime.now(UTC))

    warm = await store.subject_view(session)
    store._subject_view_cache.pop(session, None)
    cold = await store.subject_view(session)

    # fact, subject_key and LIVENESS, exactly. Those three are what `build_subject_view`
    # reads and therefore all that can reach the gate.
    assert sorted((n.fact, n.subject_key, n.invalid_at is None) for n in warm) == sorted(
        (n.fact, n.subject_key, n.invalid_at is None) for n in cold
    )
    # The datetime VALUES cannot be asserted equal, and the reason is a property of the
    # store rather than of the cache: `_ts`/`_dt` persist a datetime as a float epoch and
    # the wire format truncates it, so a value that has been to the graph and back differs
    # from the one handed in -- measured at 5 microseconds here. Reproducing that rounding
    # in Python is not possible reliably, and reading each row back after writing it would
    # reintroduce the round trip this cache exists to remove.
    #
    # This is stated rather than hidden because it is a real limit: nothing reads these
    # beyond `is None` today, and the first thing that does will see warm and cold differ
    # in the low microseconds. `valid_at` is not read by the fold at all.
    assert all(
        (w.invalid_at is None) == (c.invalid_at is None)
        for w, c in zip(sorted(warm, key=lambda n: n.fact), sorted(cold, key=lambda n: n.fact),
                        strict=True)
    )
    # And the fold agrees, which is what actually reaches the gate: the superseded
    # subject must drop out of the LIVE statistics under both.
    warm_view, cold_view = build_subject_view(warm), build_subject_view(cold)
    assert warm_view.competitors == cold_view.competitors
    assert warm_view.df == cold_view.df
    assert subject_key("sport of Tunisia") not in warm_view.competitors
    assert subject_key("sport of Ireland") in warm_view.competitors


async def test_clear_session_drops_the_cached_subject_view(store):
    """The one write that drops rather than maintains -- there is nothing left to keep."""
    session = f"sv-{uuid.uuid4().hex[:8]}"
    con = DeterministicConsolidator(store, StubEmbedder(DIM))
    await con.consolidate(session, [candidate("the sport is rugby", "sport of Tunisia")])
    assert len(await store.subject_view(session)) == 1
    await store.clear_session(session)
    assert await store.subject_view(session) == []


async def test_the_cached_subject_view_decides_identically_to_the_uncached_one(store):
    """C6's other acceptance clause: this changes latency and nothing else.

    Built twice over the same session -- once cold, once from cache -- the two views must
    produce the same `admits()` answer for every (query, fact) pair the session can form.
    A cache that is merely *fast* and subtly different would move the gate's precision
    without moving any number this repo currently prints.
    """
    session = f"sv-{uuid.uuid4().hex[:8]}"
    con = DeterministicConsolidator(store, StubEmbedder(DIM))
    for subject, fact in (
        ("sport of Tunisia", "the sport of Tunisia is rugby"),
        ("sport of Ireland", "the sport of Ireland is hurling"),
        ("sport of Wales", "the sport of Wales is rugby"),
        ("deploy target", "deploys to staging"),
    ):
        await con.consolidate(session, [candidate(fact, subject)])

    store._subject_view_cache.pop(session, None)
    cold = build_subject_view(await store.subject_view(session))
    warm = build_subject_view(await store.subject_view(session))

    assert cold.df == warm.df
    assert cold.competitors == warm.competitors
    assert cold.subject_of == warm.subject_of
    queries = ["which sport does Tunisia play", "what sport", "where do i deploy", "sport of Wales"]
    for query in queries:
        for fact in cold.subject_of:
            assert cold.admits(query, fact, 2, 2) == warm.admits(query, fact, 2, 2)
