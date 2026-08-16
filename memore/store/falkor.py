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
                "node.invalid_at AS invalid_at, node.source_episode_id AS ep, score",
                {"k": fetch, "vec": query_vec, "sid": session_id},
            )
            out = {}
            for fact_id, fact, valid_at, invalid_at, episode, distance in result.result_set:
                similarity = max(0.0, min(1.0, 1.0 - float(distance)))
                out[fact_id] = (
                    {
                        "fact": fact,
                        "valid_at": valid_at,
                        "invalid_at": invalid_at,
                        "source_episode_id": episode,
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
                "node.invalid_at AS invalid_at, node.source_episode_id AS ep, score "
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
                },
                float(score),
            )
            for fact_id, fact, valid_at, invalid_at, episode, score in result.result_set
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
            "f.valid_at, f.invalid_at, f.source_episode_id, f.type, f.subject_label, f.attribute, f.attribute_label "
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
            "RETURN f.fact, f.subject_key, f.valid_at, f.invalid_at ORDER BY f.ordinal",
            {"sid": session_id},
        )
        return [
            ChainNode(
                fact=row[0],
                subject_key=row[1] or "",
                valid_at=_dt(row[2]),
                invalid_at=_dt(row[3]),
            )
            for row in result.result_set
        ]

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
        """
        await self.connect()
        result = await self._q(
            "MATCH (f:Fact) WHERE f.session_id = $sid "
            "RETURN f.fact, f.subject_key, f.valid_at, f.invalid_at ORDER BY f.ordinal",
            {"sid": session_id},
        )
        return [
            ChainNode(
                fact=row[0],
                subject_key=row[1] or "",
                valid_at=_dt(row[2]),
                invalid_at=_dt(row[3]),
            )
            for row in result.result_set
        ]

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
        }
        valid_at = _ts(fact.valid_at)
        invalid_at = _ts(fact.invalid_at)
        # MERGE on id makes commit idempotent per turn (writepath §3).
        await self._q(
            "MERGE (f:Fact {id: $id}) SET f += $props, f.embedding = vecf32($vec)"
            + (", f.valid_at = $valid_at" if valid_at is not None else "")
            + (", f.invalid_at = $invalid_at" if invalid_at is not None else ""),
            {
                "id": fact.id,
                "props": props,
                "vec": embedding,
                "valid_at": valid_at,
                "invalid_at": invalid_at,
            },
        )

    async def supersede(self, fact_id: str, invalid_at: datetime) -> None:
        """Mark superseded -- never delete (recall-writepath-spec.md §2.2 case 3)."""
        await self.connect()
        await self._q(
            "MATCH (f:Fact {id: $id}) SET f.invalid_at = $ts",
            {"id": fact_id, "ts": _ts(invalid_at)},
        )

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
            "f.source_episode_id, f.type, f.subject_label, f.attribute, f.attribute_label ORDER BY f.ordinal",
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
            )
            for row in result.result_set
        ]

    async def clear_session(self, session_id: str) -> None:
        await self.connect()
        await self._q("MATCH (f:Fact) WHERE f.session_id = $sid DELETE f", {"sid": session_id})

    async def aclose(self) -> None:
        if self._db is not None:
            try:
                await self._db.connection.aclose()
            except Exception:  # noqa: BLE001 -- close is best-effort
                pass
