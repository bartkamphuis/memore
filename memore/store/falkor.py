"""FalkorDB-backed store: storage + hybrid retrieval only.

recall-poc-spec.md §4 is explicit that the store does NOT make the consolidation
decision -- that lives in `memore.consolidate`. This module owns persistence, the two
retrieval arms, and the score normalization the gate depends on.

Score normalization contract (recall-stage-spec.md §5 requires this documented):

  vector arm   FalkorDB's vector index yields *cosine distance* (0.0 == identical).
               We convert to similarity as `1 - distance`, clamped to [0, 1]. This is
               an absolute measure -- it means the same thing across queries, which is
               what makes the gate's `score_floor` meaningful.
  text arm     The full-text index yields an unbounded BM25-style relevance score. It is
               only comparable *within* one result set, so it is max-normalized against
               the top hit of that same query. That makes it a keyword signal, NOT an
               absolute relevance level -- in a store holding two facts, the better of
               two terrible matches still normalizes to 1.0.
  fused        `cos_sim * (1 + text_weight * bm25_norm) / (1 + text_weight)`.

The fusion is multiplicative, not additive, and that is the whole point. Adding a
set-relative term to an absolute one destroys the gate: with `0.7*cos + 0.3*bm25_norm`,
an unrelated query ("what is the weather in Paris?") scored 0.419 against a deploy fact
purely because it was the best of two bad lexical matches -- clearing a 0.35 floor that
exists precisely to keep irrelevant memories out of the prompt (spec §6). Anchoring on
cosine keeps the score absolute and comparable across queries, so `score_floor` means
something; BM25 then boosts a keyword match and demotes a hit with no lexical overlap.

The tradeoff, stated plainly: the text arm can re-rank but cannot introduce a hit the
vector arm missed, since a text-only hit has no cosine to scale. The vector arm's
widening over-fetch is what makes that acceptable.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import replace
from datetime import UTC, datetime

from falkordb.asyncio import FalkorDB

from ..chain import ChainNode
from ..config import StoreConfig
from ..keys import FT_SPECIAL, normalize_subject  # noqa: F401 -- re-exported, see below
from ..types import Episode, FactType, MemoryHit, StoredFact

# Subject-key normalization lives in `memore.keys` (a domain rule, not a storage one).
# Re-exported here because the consolidator and the tests import it from this module.
_FT_SPECIAL = FT_SPECIAL

# FalkorDB's vector index is global per (label, property) and its query procedure takes
# no filter, so session scoping can only happen *after* the ANN fetch. A fixed over-fetch
# degrades silently as the graph grows: with 900 facts across sessions, a session holding
# 1 fact has to win a global top-48 before the filter ever sees it, and recall quietly
# returns nothing. So the fetch widens until it has enough in-session hits or gives up.
_SESSION_OVERFETCH = 4
_MAX_VECTOR_FETCH = 4096

# How many sessions' subject-view rows to keep cached (C6, RESULTS.md §24). Bounded
# because a gateway process serves many sessions and the rows are a copy of the session's
# whole fact text -- unbounded, this is a slow leak rather than a cache. 32 is well past
# what any one process holds warm at PoC scale and is a few MB at bench sizes.
_SUBJECT_VIEW_CACHE_SESSIONS = 32


def _ts(value: datetime | None) -> float | None:
    return value.timestamp() if value is not None else None


def _dt(value: float | None) -> datetime | None:
    return datetime.fromtimestamp(value, tz=UTC) if value is not None else None


def sanitize_fulltext(query: str) -> str:
    """Turn a raw user turn into a full-text *union* query.

    The index's query language intersects space-separated terms, so passing a natural
    question through verbatim demands that every word appear in the fact -- which almost
    never holds, and the text arm silently returns nothing on every turn. That is not a
    degraded BM25 signal, it is no BM25 signal: the hybrid quietly collapses to
    vector-only. Joining terms with `|` asks for a union, which is what a BM25 arm is
    supposed to do; the index still ranks by how many terms hit and how rare they are.

    Single characters are dropped -- "what's" tokenises to "what" + "s", and a bare "s"
    matches noise.
    """
    terms = [t for t in _FT_SPECIAL.sub(" ", query).split() if len(t) > 1]
    return "|".join(terms)


class FalkorStore:
    """Implements `ConsolidatingStore`."""

    def __init__(self, config: StoreConfig | None = None, dimension: int = 768):
        self.config = config or StoreConfig.from_env()
        self.dimension = dimension
        self._db: FalkorDB | None = None
        self._graph = None
        self._ready = False
        # session -> subject_view rows. C6: the fetch is O(session) and ran on EVERY read,
        # which was the whole of the 200ms budget breach at 32k (RESULTS.md §23.3).
        # Invalidated by this instance's own writes -- see `_invalidate_subject_view`.
        self._subject_view_cache: OrderedDict[str, list[ChainNode]] = OrderedDict()

    async def connect(self) -> None:
        if self._graph is None:
            self._db = FalkorDB(host=self.config.falkor_host, port=self.config.falkor_port)
            self._graph = self._db.select_graph(self.config.graph_name)
        if not self._ready:
            await self._ensure_indexes()
            self._ready = True

    async def _q(self, query: str, params: dict | None = None):
        assert self._graph is not None, "call connect() first"
        return await self._graph.query(query, params or {})

    async def _assert_vector_dimension(self) -> None:
        """Fail loudly if the graph's vector index was built for a different embedder.

        Nothing else in the system notices this. Creating a 1024-dim index over a graph
        that already has a 768-dim one raises "already indexed", which `_ensure_indexes`
        correctly swallows; `add_fact` then writes wrong-width vectors without complaint,
        and only the *query* raises. `recall()` catches every store exception by design
        (§3.1: a store failure must never fail the turn), so the whole thing degrades to
        a gate that is silently shut forever -- memory quietly dead, one WARNING a turn.

        Reachable in one step: `MEMORE_EMBED_MODEL` is the documented way to change
        embedder, and the graph name does not change with it. So the mismatch is checked
        at connect(), where raising is safe because it is startup rather than a turn.
        """
        result = await self._q("CALL db.indexes()")
        try:
            options = result.header.index([1, "options"])
        except ValueError:  # pragma: no cover -- older FalkorDB without the column
            return
        for row in result.result_set:
            existing = (row[options] or {}).get("embedding", {}).get("dimension")
            if existing is not None and int(existing) != self.dimension:
                raise RuntimeError(
                    f"graph {self.config.graph_name!r} has a {existing}-dimension vector "
                    f"index but this store is configured for {self.dimension}. The stored "
                    "vectors are another embedder's and cannot be searched with this one: "
                    "use a different MEMORE_GRAPH, or rebuild this one."
                )

    async def _ensure_indexes(self) -> None:
        # Index creation is idempotent-by-exception here: FalkorDB errors if the index
        # already exists, and there is no CREATE IF NOT EXISTS for all three kinds.
        statements = [
            f"CREATE VECTOR INDEX FOR (f:Fact) ON (f.embedding) "
            f"OPTIONS {{dimension: {self.dimension}, similarityFunction: 'cosine'}}",
            "CALL db.idx.fulltext.createNodeIndex('Fact', 'fact')",
            "CREATE INDEX FOR (f:Fact) ON (f.session_id)",
            "CREATE INDEX FOR (f:Fact) ON (f.subject_key)",
            "CREATE INDEX FOR (f:Fact) ON (f.id)",
            # RESULTS.md §22 / identity-and-gate-spec A1. Slot arity lives on its own
            # node, not on the Fact, so it survives the facts it governs and can be
            # corrected without rewriting any of them.
            "CREATE INDEX FOR (s:Slot) ON (s.session_id)",
        ]
        for statement in statements:
            try:
                await self._q(statement)
            except Exception as exc:  # noqa: BLE001 -- "already indexed" is the happy path
                if "already" not in str(exc).lower():
                    raise
        # After creation, not before: on a fresh graph there is no index to compare to.
        await self._assert_vector_dimension()

    # ---- MemoryStore -----------------------------------------------------------

    async def hybrid_search(
        self,
        query_text: str,
        query_vec: list[float],
        session_id: str,
        k: int,
    ) -> list[MemoryHit]:
        await self.connect()
        vector_task = self._vector_arm(query_vec, session_id, k)
        text_task = self._text_arm(query_text, session_id, k)
        vector_hits, text_hits = await asyncio.gather(vector_task, text_task)

        best_text = max((s for _, s in text_hits.values()), default=0.0)
        weight = self.config.text_weight
        fused: dict[str, tuple[dict, float, float]] = {}
        for fact_id, (row, sim) in vector_hits.items():
            raw = text_hits.get(fact_id, (None, 0.0))[1]
            bm25_norm = raw / best_text if best_text > 0 else 0.0
            # The un-fused cosine travels alongside the fused score. Ranking uses the
            # fused one; the gate may use either (RecallConfig.gate_on, RESULTS.md §12).
            fused[fact_id] = (row, sim * (1.0 + weight * bm25_norm) / (1.0 + weight), sim)

        ranked = sorted(fused.values(), key=lambda triple: triple[1], reverse=True)[:k]
        return [
            MemoryHit(
                fact=row["fact"],
                score=min(1.0, max(0.0, score)),
                similarity=min(1.0, max(0.0, sim)),
                valid_at=_dt(row.get("valid_at")),
                invalid_at=_dt(row.get("invalid_at")),
                source_episode_id=row.get("source_episode_id") or "",
                # Carried for the PAST label only -- nothing above reads them, so ranking
                # and the gate are unchanged by their presence (RESULTS.md §19.1).
                occurs_at=_dt(row.get("occurs_at")),
                recurring=bool(row.get("recurring")),
            )
            for row, score, sim in ranked
        ]

    async def _vector_arm(self, query_vec: list[float], session_id: str, k: int) -> dict:
        """ANN fetch, widened until the session's own facts are actually in hand.

        Two separate traps here, and the second one hid behind the fix for the first.

        1. The index is global per (label, property) and the query procedure takes no
           filter, so `session_id` can only be applied *after* the fetch. A fixed
           over-fetch means a session holding a few facts silently returns nothing once
           other sessions crowd the index.

        2. The obvious bound for the widening -- stop once `fetch` covers every node in
           the graph -- is wrong, because **HNSW returns fewer nodes than asked for**.
           Measured on a 467-fact graph: asking for 467 returned 182, of which 0 were the
           12-fact session's; asking for 2000 returned all 12. So "I asked for the whole
           graph" is not "I saw the whole graph", and bounding by `total` reintroduced
           exactly the silent-empty-result failure the widening was added to fix.

        So the loop is bounded by `_MAX_VECTOR_FETCH`, not by the graph size, and it
        stops on the *session's* fact count rather than on `k` -- a session holding 3
        facts must not widen forever chasing a 12th hit that does not exist.
        """
        fetch = k * _SESSION_OVERFETCH
        # How many in-session hits are worth waiting for. Starts at k and is refined to
        # the session's real size only if the first pass falls short, so the common case
        # costs no extra query.
        target = k
        counted = False
        out: dict = {}
        while True:
            result = await self._q(
                "CALL db.idx.vector.queryNodes('Fact', 'embedding', $k, vecf32($vec)) "
                "YIELD node, score WITH node, score WHERE node.session_id = $sid "
                "RETURN node.id AS id, node.fact AS fact, node.valid_at AS valid_at, "
                "node.invalid_at AS invalid_at, node.source_episode_id AS ep, "
                "node.occurs_at AS occurs_at, node.recurring AS recurring, score",
                {"k": fetch, "vec": query_vec, "sid": session_id},
            )
            out = {}
            for (
                fact_id, fact, valid_at, invalid_at, episode, occurs_at, recurring, distance
            ) in result.result_set:
                similarity = max(0.0, min(1.0, 1.0 - float(distance)))
                out[fact_id] = (
                    {
                        "fact": fact,
                        "valid_at": valid_at,
                        "invalid_at": invalid_at,
                        "source_episode_id": episode,
                        "occurs_at": occurs_at,
                        "recurring": recurring,
                    },
                    similarity,
                )
            if len(out) >= target or fetch >= _MAX_VECTOR_FETCH:
                return out
            if not counted:
                # The session may simply hold fewer than k facts, in which case the first
                # pass already has everything there is. One indexed count settles it.
                target = min(k, await self.count(session_id))
                counted = True
                if len(out) >= target:
                    return out
            fetch = min(fetch * 8, _MAX_VECTOR_FETCH)

    async def _text_arm(self, query_text: str, session_id: str, k: int) -> dict:
        cleaned = sanitize_fulltext(query_text)
        if not cleaned:
            return {}
        try:
            result = await self._q(
                "CALL db.idx.fulltext.queryNodes('Fact', $q) YIELD node, score "
                "WITH node, score WHERE node.session_id = $sid "
                "RETURN node.id AS id, node.fact AS fact, node.valid_at AS valid_at, "
                "node.invalid_at AS invalid_at, node.source_episode_id AS ep, "
                "node.occurs_at AS occurs_at, node.recurring AS recurring, score "
                "LIMIT $lim",
                {"q": cleaned, "sid": session_id, "lim": k * _SESSION_OVERFETCH},
            )
        except Exception:  # noqa: BLE001 -- a malformed full-text query degrades to vector-only
            return {}
        return {
            fact_id: (
                {
                    "fact": fact,
                    "valid_at": valid_at,
                    "invalid_at": invalid_at,
                    "source_episode_id": episode,
                    "occurs_at": occurs_at,
                    "recurring": recurring,
                },
                float(score),
            )
            for fact_id, fact, valid_at, invalid_at, episode, occurs_at, recurring, score
            in result.result_set
        }

    async def ingest(self, episode: Episode) -> None:
        """Present for protocol conformance. The PoC's write path drives the store
        through the consolidator (`add_fact` / `supersede`) instead, because
        recall-poc-spec.md §4 moves the NEW/DUP/CONTRADICT/REFINE decision out of the
        store. Raising here keeps that from being bypassed by accident.
        """
        raise NotImplementedError(
            "FalkorStore is storage-only; drive writes through memore.consolidate "
            "(recall-poc-spec.md §4)."
        )

    # ---- ConsolidatingStore ----------------------------------------------------

    async def live_facts_for_subject(self, session_id: str, subject_key: str) -> list[StoredFact]:
        await self.connect()
        result = await self._q(
            "MATCH (f:Fact) WHERE f.session_id = $sid AND f.subject_key = $key "
            "AND f.invalid_at IS NULL RETURN f.id, f.fact, f.subject_key, f.ordinal, "
            "f.valid_at, f.invalid_at, f.source_episode_id, f.type, f.subject_label, f.attribute, f.attribute_label, "
            "f.occurs_at, f.recurring "
            "ORDER BY f.ordinal DESC",
            {"sid": session_id, "key": subject_key},
        )
        return [
            StoredFact(
                id=row[0],
                session_id=session_id,
                fact=row[1],
                subject_key=row[2],
                subject_label=row[8] or row[2],
                ordinal=int(row[3]),
                valid_at=_dt(row[4]),
                invalid_at=_dt(row[5]),
                source_episode_id=row[6] or "",
                type=FactType(row[7]) if row[7] else FactType.STATE,
                attribute=row[9] or "",
                attribute_label=row[10] or row[9] or "",
                # Absent on every fact written before §19; `_dt(None)` is None and
                # `bool(None)` is False, which are exactly the inert values.
                occurs_at=_dt(row[11]),
                recurring=bool(row[12]),
            )
            for row in result.result_set
        ]

    async def live_chain_view(self, session_id: str) -> list[ChainNode]:
        """Every LIVE fact in the session, as the multi-hop walk needs it.

        Live-only is not a filter, it is the reason the walk is affordable: superseded
        facts multiply the out-degree of exactly the popular values ("Italy", "Soviet
        Union") that would otherwise explode the frontier. Dropping them collapses 455
        facts to 303 on the 6k corpus and holds mean branching at 1.0 (see `memore.chain`).

        One query per expanding turn, returning the session -- acceptable at PoC scale
        (~300 rows, single-digit ms) and honestly not the production shape: with long
        sessions this wants materialized edges written at `add_fact` time and a
        variable-length Cypher path, so the cost tracks the chain rather than the session.
        """
        await self.connect()
        result = await self._q(
            "MATCH (f:Fact) WHERE f.session_id = $sid AND f.invalid_at IS NULL "
            "RETURN f.fact, f.subject_key, f.valid_at, f.invalid_at, f.occurs_at, f.recurring "
            "ORDER BY f.ordinal",
            {"sid": session_id},
        )
        return [
            ChainNode(
                fact=row[0],
                subject_key=row[1] or "",
                valid_at=_dt(row[2]),
                invalid_at=_dt(row[3]),
                occurs_at=_dt(row[4]),
                recurring=bool(row[5]),
            )
            for row in result.result_set
        ]

    def _invalidate_subject_view(self, session_id: str) -> None:
        """Drop the cached rows for a session entirely. Only `clear_session` needs this.

        The two ordinary writes UPDATE the cache instead -- see `_cache_added_fact` and
        `_cache_superseded_fact` and, more importantly, the reason below.
        """
        self._subject_view_cache.pop(session_id, None)

    # ---- keeping the cache warm across writes ----------------------------------
    #
    # Dropping the entry on every write is correct and was the first version, and it is
    # very nearly useless. A conversational turn is READ then WRITE: recall() runs, the
    # response streams, then the write path stores what the turn asserted. So a
    # drop-on-write cache is invalidated by turn N and cold again for turn N+1, every
    # time. Measured at ~330 facts: **warm at read time 1 turn in 20**, against 19 in 20
    # in a read-only loop. The bench, which ingests everything and then asks 100
    # questions, is the one workload where dropping looks fine -- the same
    # fixture-cannot-express-the-failure shape as §13 and §23.3.
    #
    # So the writes maintain the rows rather than discarding them. Both are exact: a fact
    # is appended exactly as the query would have returned it, and a supersede flips the
    # one field it changed. Anything a write cannot express exactly must go back to
    # dropping -- a cache that is subtly wrong moves the gate's precision silently, which
    # is worse than one that is merely cold.

    def _cache_added_fact(self, fact: StoredFact) -> None:
        rows = self._subject_view_cache.get(fact.session_id)
        if rows is None:
            return
        # Field-for-field what `subject_view`'s query returns -- fact, subject_key,
        # valid_at, invalid_at, and NOT occurs_at/recurring, which that query does not
        # select. A mismatch here would make a warm view differ from a cold one, which is
        # what `test_the_cached_subject_view_survives_writes_exactly` pins.
        #
        # `_dt(_ts(...))` rather than the datetime itself, and that is not pedantry: the
        # store persists a datetime as a float epoch, so a value that has been to the
        # graph and back has LESS precision than the one handed in. Caching the original
        # made the warm row disagree with the cold one by 5 microseconds -- found by the
        # test, not by reasoning. Nothing reads these today beyond `is None`, so it
        # changed no behaviour; it would have been waiting for the first thing that did.
        rows.append(
            ChainNode(
                fact=fact.fact,
                subject_key=fact.subject_key,
                valid_at=_dt(_ts(fact.valid_at)),
                invalid_at=_dt(_ts(fact.invalid_at)),
            )
        )
        # A duplicate row is harmless and is not filtered: `build_subject_view` keys
        # `subject_of` by fact text and folds `live_keys` into a set, so appending the
        # same fact twice (the MERGE-on-id retry path) folds to the same view.

    def _cache_superseded_fact(self, session_id: str, fact_text: str, invalid_at: datetime) -> None:
        rows = self._subject_view_cache.get(session_id)
        if rows is None:
            return
        # Round-tripped through the store's own float encoding, for the reason in
        # `_cache_added_fact`.
        stored = _dt(_ts(invalid_at))
        for index, node in enumerate(rows):
            if node.fact == fact_text:
                rows[index] = replace(node, invalid_at=stored)

    async def subject_view(self, session_id: str) -> list[ChainNode]:
        """Every fact in the session with its subject key, live and superseded.

        Feeds `memore.subjects`. Superseded facts are included because a superseded hit
        can still be injected (§6.3 marks it rather than dropping it) and so has to be
        admissible on the same terms; the vocabulary statistics themselves are computed
        over live subjects only, in `build_subject_view`.

        Deliberately a separate query from `live_chain_view` rather than one shared
        fetch. Both features are off by default and independent, so paying ~14ms twice
        only happens when both are on; folding them together would couple two things
        that have no reason to be coupled. Worth revisiting if both become defaults.

        **Cached per session, invalidated by this instance's writes (C6, RESULTS.md §24).**
        This query is O(session) and `recall()` ran it on every read, which was the entire
        200ms budget breach at 32k: 21ms at 455 facts, 106ms at 2310, exactly linear
        (§23.3). Nothing else in A-D grows with session size.

        Two things this cache does NOT do, both deliberate:

          it does not cache the FOLD.  `build_subject_view` is 1.4ms at 455 facts and
                                      12.2ms at 2310 -- an order of magnitude under the
                                      fetch. Caching it would mean a domain fold living
                                      behind the store boundary, which is exactly what
                                      `normalize_subject` was lifted out of this module to
                                      avoid. Revisit only if the fold starts to matter.
          it does not see other       The counter is per store INSTANCE. Another process
          processes' writes.          writing leaves this one's view stale until it writes
                                      itself. That is acceptable and not a correctness
                                      bug: the admission rule is a precision refinement,
                                      and §3.1 already requires that losing it costs
                                      recall nothing. It would NOT be acceptable for
                                      anything that decides freshness.
        """
        cached = self._subject_view_cache.get(session_id)
        if cached is not None:
            self._subject_view_cache.move_to_end(session_id)
            return cached
        await self.connect()
        result = await self._q(
            "MATCH (f:Fact) WHERE f.session_id = $sid "
            "RETURN f.fact, f.subject_key, f.valid_at, f.invalid_at ORDER BY f.ordinal",
            {"sid": session_id},
        )
        rows = [
            ChainNode(
                fact=row[0],
                subject_key=row[1] or "",
                valid_at=_dt(row[2]),
                invalid_at=_dt(row[3]),
            )
            for row in result.result_set
        ]
        self._subject_view_cache[session_id] = rows
        self._subject_view_cache.move_to_end(session_id)
        while len(self._subject_view_cache) > _SUBJECT_VIEW_CACHE_SESSIONS:
            self._subject_view_cache.popitem(last=False)
        return rows

    async def max_ordinal(self, session_id: str, subject_key: str | None = None) -> int:
        """Highest ordinal issued in this session, or within one subject if given."""
        await self.connect()
        if subject_key is None:
            result = await self._q(
                "MATCH (f:Fact) WHERE f.session_id = $sid RETURN max(f.ordinal)",
                {"sid": session_id},
            )
        else:
            result = await self._q(
                "MATCH (f:Fact) WHERE f.session_id = $sid AND f.subject_key = $key "
                "RETURN max(f.ordinal)",
                {"sid": session_id, "key": subject_key},
            )
        value = result.result_set[0][0] if result.result_set else None
        return int(value) if value is not None else 0

    async def add_fact(self, fact: StoredFact, embedding: list[float]) -> None:
        await self.connect()
        props = {
            "id": fact.id,
            "session_id": fact.session_id,
            "fact": fact.fact,
            "subject_key": fact.subject_key,
            "subject_label": fact.subject_label,
            "ordinal": fact.ordinal,
            "source_episode_id": fact.source_episode_id,
            "type": fact.type.value,
            "attribute": fact.attribute,
            "attribute_label": fact.attribute_label,
            # Temporal expiry (RESULTS.md §19). `recurring` goes in `props` because a
            # bool is always present; `occurs_at` follows the `valid_at` pattern below,
            # being nullable.
            "recurring": fact.recurring,
        }
        valid_at = _ts(fact.valid_at)
        invalid_at = _ts(fact.invalid_at)
        occurs_at = _ts(fact.occurs_at)
        # MERGE on id makes commit idempotent per turn (writepath §3).
        await self._q(
            "MERGE (f:Fact {id: $id}) SET f += $props, f.embedding = vecf32($vec)"
            + (", f.valid_at = $valid_at" if valid_at is not None else "")
            + (", f.invalid_at = $invalid_at" if invalid_at is not None else "")
            + (", f.occurs_at = $occurs_at" if occurs_at is not None else ""),
            {
                "id": fact.id,
                "props": props,
                "vec": embedding,
                "valid_at": valid_at,
                "invalid_at": invalid_at,
                "occurs_at": occurs_at,
            },
        )
        self._cache_added_fact(fact)

    async def supersede(self, fact_id: str, invalid_at: datetime) -> None:
        """Mark superseded -- never delete (recall-writepath-spec.md §2.2 case 3)."""
        await self.connect()
        # RETURN the session and the text so the subject-view cache can be MAINTAINED:
        # this signature takes a fact id and nothing else, so without asking, the one
        # write most likely to corrupt that cache is the one that cannot reach it. Costs
        # nothing -- the node is already matched. The text is the cache's key for a row,
        # matching `build_subject_view`, which keys `subject_of` by fact text.
        result = await self._q(
            "MATCH (f:Fact {id: $id}) SET f.invalid_at = $ts RETURN f.session_id, f.fact",
            {"id": fact_id, "ts": _ts(invalid_at)},
        )
        for row in result.result_set:
            self._cache_superseded_fact(row[0], row[1], invalid_at)

    async def subject_slots(self, session_id: str) -> list[str]:
        await self.connect()
        result = await self._q(
            "MATCH (f:Fact) WHERE f.session_id = $sid "
            "RETURN f.subject_key, min(f.subject_label), "
            "collect(coalesce(f.attribute_label, f.attribute))",
            {"sid": session_id},
        )
        lines = []
        for key, label, attributes in (
            (row[0], row[1], row[2] or []) for row in result.result_set
        ):
            if not (key or label):
                continue
            slots = sorted({a for a in attributes if a})
            lines.append(f"{label or key} -> {', '.join(slots)}" if slots else (label or key))
        return lines

    async def slot_schemas(self, session_id: str) -> list[tuple[str, str, bool]]:
        await self.connect()
        result = await self._q(
            "MATCH (s:Slot) WHERE s.session_id = $sid "
            "RETURN s.subject_key, s.attribute, s.single_valued",
            {"sid": session_id},
        )
        return [(row[0], row[1], bool(row[2])) for row in result.result_set if row[1]]

    async def ensure_slot_schema(
        self, session_id: str, subject_key: str, attribute: str, single_valued: bool
    ) -> None:
        if not attribute:
            return
        await self.connect()
        # ON CREATE SET, not SET: the "do not overwrite implicitly" rule of A1 is enforced
        # in the query, so no caller can get it wrong by passing the flag the wrong way.
        await self._q(
            "MERGE (s:Slot {session_id: $sid, subject_key: $key, attribute: $attr}) "
            "ON CREATE SET s.single_valued = $sv",
            {"sid": session_id, "key": subject_key, "attr": attribute, "sv": single_valued},
        )

    async def set_slot_schema(
        self, session_id: str, subject_key: str, attribute: str, single_valued: bool
    ) -> None:
        if not attribute:
            return
        await self.connect()
        await self._q(
            "MERGE (s:Slot {session_id: $sid, subject_key: $key, attribute: $attr}) "
            "SET s.single_valued = $sv",
            {"sid": session_id, "key": subject_key, "attr": attribute, "sv": single_valued},
        )

    async def live_subject_keys(self, session_id: str) -> list[str]:
        await self.connect()
        result = await self._q(
            "MATCH (f:Fact) WHERE f.session_id = $sid AND f.invalid_at IS NULL "
            "RETURN DISTINCT f.subject_key",
            {"sid": session_id},
        )
        return [row[0] for row in result.result_set if row[0]]

    async def live_fact_texts(self, session_id: str) -> set[str]:
        """Every fact in the session with no `invalid_at`. Used by the bench oracle."""
        await self.connect()
        result = await self._q(
            "MATCH (f:Fact) WHERE f.session_id = $sid AND f.invalid_at IS NULL RETURN f.fact",
            {"sid": session_id},
        )
        return {row[0] for row in result.result_set}

    async def count(self, session_id: str) -> int:
        await self.connect()
        result = await self._q(
            "MATCH (f:Fact) WHERE f.session_id = $sid RETURN count(f)", {"sid": session_id}
        )
        return int(result.result_set[0][0]) if result.result_set else 0

    # ---- inspection ------------------------------------------------------------
    # Deliberately NOT on the `MemoryStore` protocol: the gateway never needs these, and
    # §3.2's surface stays the two methods it defines. This is the read-only half of the
    # queryable audit log that `recall-poc-spec.md` §5 defers -- enough to answer "what
    # does the store actually hold", which is the question you have when recall returns
    # nothing and you cannot tell an empty session from a broken lookup.

    async def sessions(self) -> list[tuple[str, int, int]]:
        """(session_id, total facts, live facts), busiest first."""
        await self.connect()
        result = await self._q(
            "MATCH (f:Fact) RETURN f.session_id, count(f), "
            "sum(CASE WHEN f.invalid_at IS NULL THEN 1 ELSE 0 END) "
            "ORDER BY count(f) DESC"
        )
        return [(row[0], int(row[1]), int(row[2] or 0)) for row in result.result_set]

    async def facts_in_session(self, session_id: str) -> list[StoredFact]:
        """Every fact, live and superseded, in arrival order.

        Superseded ones are included on purpose: "supersede, never delete" is the design
        claim, and an inspector that hid them would make the one thing worth checking
        invisible.
        """
        await self.connect()
        result = await self._q(
            "MATCH (f:Fact) WHERE f.session_id = $sid "
            "RETURN f.id, f.fact, f.subject_key, f.ordinal, f.valid_at, f.invalid_at, "
            "f.source_episode_id, f.type, f.subject_label, f.attribute, f.attribute_label, "
            "f.occurs_at, f.recurring ORDER BY f.ordinal",
            {"sid": session_id},
        )
        return [
            StoredFact(
                id=row[0],
                session_id=session_id,
                fact=row[1],
                subject_key=row[2],
                subject_label=row[8] or row[2],
                ordinal=int(row[3]),
                valid_at=_dt(row[4]),
                invalid_at=_dt(row[5]),
                source_episode_id=row[6] or "",
                type=FactType(row[7]) if row[7] else FactType.STATE,
                attribute=row[9] or "",
                attribute_label=row[10] or row[9] or "",
                # Absent on every fact written before §19; `_dt(None)` is None and
                # `bool(None)` is False, which are exactly the inert values.
                occurs_at=_dt(row[11]),
                recurring=bool(row[12]),
            )
            for row in result.result_set
        ]

    async def clear_session(self, session_id: str) -> None:
        await self.connect()
        await self._q("MATCH (f:Fact) WHERE f.session_id = $sid DELETE f", {"sid": session_id})
        # The slot schema is session-scoped state like the facts, and nothing else lists
        # it -- `sessions()` counts Facts, so orphaned Slot nodes would accumulate in a
        # shared graph completely invisibly. A harness that clears between runs (the
        # `--runs 9` scripts do) must not leave the next run's arity decided by the last.
        await self._q("MATCH (s:Slot) WHERE s.session_id = $sid DELETE s", {"sid": session_id})
        self._invalidate_subject_view(session_id)

    async def aclose(self) -> None:
        if self._db is not None:
            try:
                await self._db.connection.aclose()
            except Exception:  # noqa: BLE001 -- close is best-effort
                pass
