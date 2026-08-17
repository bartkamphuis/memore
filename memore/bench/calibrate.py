"""`score_floor` calibration across embedding models (RESULTS.md §5's next step).

Why this exists. `RecallConfig.score_floor` is the gate, and the gate is the design's
stated differentiator (`recall-stage-spec.md` §6): it is the only thing standing between
an irrelevant turn and an injected memory. The floor is also *embedder-dependent* -- the
faster models compress cosine similarities upward, so an unrelated query that scores 0.22
against `embeddinggemma` scores ~0.35 against `nomic-embed-text`, landing exactly on the
default floor. That coupling is what has kept the default on a model costing ~300ms per
call inside a 200ms budget (§14). Recalibrating over the two ad-hoc queries of §5 would
have been worse than the latency; this module is the real query set that replaces them.

What is measured. For every query the harness runs the **actual `recall()` path** with
the floor dropped to 0.0, which yields two things at once: the top-1 fused score the gate
would have tested, and `RecallResult.latency_ms`, which is components A-D end to end --
the exact quantity §14 budgets. Nothing is re-implemented here, so a floor chosen from
these numbers is a floor measured on the code that will use it.

The query set, and its two regimes:

  bench   MemoryAgentBench FactConsolidation sh_6k. Its ~300 subjects are split in half
          by a stable hash into two folds, each ingested into its own session. A subject
          lands wholly in one fold, so its update chain -- the thing consolidation exists
          to resolve -- is never cut in half. Each of the 100 questions is then a
          POSITIVE against the fold holding its subject and a HARD NEGATIVE against the
          other one: same corpus, same phrasing, same relation templates, subject simply
          absent. Two folds means every question is used as both, which is where 100
          questions turn into 100 positives and 100 hard negatives at no extra labelling.

  chat    An authored 12-fact conversational session with 24 positives (`calib_fixtures`),
          the regime the gateway actually runs in. Authored, and labelled as such.

Off-domain negatives run against every session in both regimes.

The deliberate limitation: a hard negative here is *lexically* close but semantically
absent, which is the case the gate is for. What this set does not contain is a query
whose answer the store holds under a different phrasing -- that would be a false negative
for the gate, and measuring it needs relevance judgements this corpus does not carry.

Usage:
    uv run python -m memore.bench.calibrate --configs all --out data/results/calibration.json
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..config import EmbedConfig, RecallConfig, StoreConfig
from ..consolidate import DeterministicConsolidator
from ..embed import OllamaEmbedder
from ..recall import recall
from ..store.falkor import FalkorStore, normalize_subject
from ..types import CandidateFact, FactType, TurnContext
from . import data as bench_data
from .calib_fixtures import (
    CHAT_FACTS,
    CHAT_POSITIVES,
    CROWDED_CHAT_FACTS,
    CROWDED_CHAT_HARD_NEGATIVES,
    CROWDED_CHAT_POSITIVES,
    OFF_DOMAIN_NEGATIVES,
)
from .extract import CACHED_EXTRACTOR_MODEL, KnowledgePoolExtractor
from .scoring import normalize_answer


# The gate is measured with the floor removed, so `recall()` returns every hit it found
# and the top-1 score is the number the floor would have been compared against.
#
# With `subject_check` on, a refused hit is simply absent, so `top1_score` becomes the
# score of the top ADMITTED hit -- which is exactly the quantity the floor would see in
# production. That is why the rule is measured by re-running this harness rather than
# replayed offline: the stored observations only carry top-1, and the rule can promote a
# lower hit into that position.
def _open_gate(subject_check: bool) -> RecallConfig:
    return RecallConfig(score_floor=0.0, subject_check=subject_check)

POSITIVE = "positive"
HARD_NEGATIVE = "hard_negative"
OFF_DOMAIN = "off_domain_negative"


@dataclass(frozen=True)
class EmbedderVariant:
    """One calibration subject: a model, and whether its asymmetric prefixes are used.

    Each variant gets its own FalkorDB graph. FalkorDB's vector index is global per
    (label, property), so two models' vectors in one graph share an ANN index and the
    session filter runs after the fetch -- the failure RESULTS.md §5 records. Separate
    graphs keep the arms independent, and are also required for a differing `dimension`.
    """

    name: str
    model: str
    dimension: int = 768
    use_prefixes: bool = False

    def embed_config(self, base_url: str) -> EmbedConfig:
        document_prefix, query_prefix = _prefixes_for(self.model) if self.use_prefixes else ("", "")
        return EmbedConfig(
            ollama_url=base_url,
            model=self.model,
            dimension=self.dimension,
            document_prefix=document_prefix,
            query_prefix=query_prefix,
        )

    def graph_name(self) -> str:
        return "calib_" + self.name.replace("-", "_").replace(".", "_")


def _prefixes_for(model: str) -> tuple[str, str]:
    """Every prefix a model's card documents, not only the ones currently applied.

    Deliberately `KNOWN_PREFIXES` rather than the shipped `_PREFIXES`: this harness exists
    to decide *whether* a prefix should ship, so it has to be able to measure one that
    does not. `EmbedConfig.from_env` consults the narrower table.
    """
    from ..config import KNOWN_PREFIXES

    for stem, pair in KNOWN_PREFIXES.items():
        if model.startswith(stem):
            return pair
    return "", ""


VARIANTS: dict[str, EmbedderVariant] = {
    v.name: v
    for v in [
        # The incumbent default: the number every existing figure was measured against.
        EmbedderVariant("embeddinggemma", "embeddinggemma:latest"),
        # The candidate. RESULTS.md §5 measured its prefixes making discrimination
        # slightly *worse* over two queries and flagged it as worth re-checking rather
        # than assuming -- hence both arms.
        EmbedderVariant("nomic_prefixed", "nomic-embed-text:latest", use_prefixes=True),
        EmbedderVariant("nomic_bare", "nomic-embed-text:latest", use_prefixes=False),
        # The shipped default, and the one new arm. mxbai is trained asymmetrically but
        # had no `_PREFIXES` entry until RESULTS.md §12, so `mxbai` here is what actually
        # ships today and `mxbai_prefixed` is what its model card asks for.
        EmbedderVariant("mxbai", "mxbai-embed-large:latest", dimension=1024),
        EmbedderVariant(
            "mxbai_prefixed", "mxbai-embed-large:latest", dimension=1024, use_prefixes=True
        ),
    ]
}


@dataclass
class Observation:
    variant: str
    fixture: str
    session: str
    query: str
    label: str
    top1_score: float
    top1_fact: str | None
    on_subject: bool
    n_hits: int
    latency_ms: float
    # The SAME hit's un-fused cosine. Both are recorded so one run can compare gating on
    # each without re-embedding -- `RecallConfig.gate_on`, RESULTS.md §12. Last and
    # defaulted so an older results JSON still re-analyses, scoring as if the two
    # coincided, which for `gate_on="fused"` is exactly what it was.
    top1_cosine: float = 0.0


@dataclass
class FloorPoint:
    floor: float
    tpr: float  # positives where the gate opens -- recall
    useful_tpr: float  # ...and the top hit is actually the right fact
    fpr_hard: float
    fpr_off_domain: float
    fpr_all: float
    youden_j: float


@dataclass
class Recommendation:
    """The operating point, chosen by constraint rather than by a symmetric score.

    Youden's J weights a false open and a false shut equally, and they are not equal: a
    false open puts a wrong memory into the prompt on a turn where the user never asked
    for memory, while a false shut leaves the turn exactly as it is today. That is the
    same asymmetry that makes `use_embedding_comparison` default False (RESULTS.md §6).
    So the floor is the *lowest* one holding off-domain false-opens under a stated budget
    in **every** regime -- which, because both TPR and FPR fall monotonically with the
    floor, is also the highest-recall floor that satisfies the budget. J is reported
    alongside, as a secondary read, not as the chooser.
    """

    fpr_budget: float
    floor: float
    # Floors that also satisfy the budget and stay within `tpr_tolerance` of the best
    # min-regime TPR. 24 chat positives move TPR in 4.2% steps, so a two-decimal
    # recommendation overstates the resolution the data has -- quote the band.
    band: tuple[float, float]
    tpr_tolerance: float
    feasible: bool
    per_regime: dict[str, FloorPoint] = field(default_factory=dict)


@dataclass
class VariantReport:
    variant: str
    model: str
    use_prefixes: bool
    # Which quantity the floor was swept against (`RecallConfig.gate_on`). A report is one
    # (variant, gate_on) pair, because a floor only means something next to both.
    gate_on: str = "fused"
    subject_check: bool = False
    counts: dict[str, dict[str, int]] = field(default_factory=dict)
    # Queries where the store returned nothing at all. `top1_score` records those as 0.0,
    # which on a negative is indistinguishable from the gate correctly staying shut -- so
    # a retrieval bug can masquerade as good precision. It has happened once already
    # (RESULTS.md §5, the HNSW under-return), so it is surfaced rather than inferred.
    zero_hit: dict[str, int] = field(default_factory=dict)
    # Per fixture, because the fixtures do not share an index-crowding condition: the
    # graph grows as fixtures are ingested in turn, and `_vector_arm` widens its
    # over-fetch as it does. Scores are unaffected (session filtering is in the Cypher
    # WHERE), but a pooled p95 would mix three different conditions.
    latency: dict[str, dict[str, float]] = field(default_factory=dict)
    score_summary: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)
    cosine_summary: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)
    curves: dict[str, list[FloorPoint]] = field(default_factory=dict)
    recommendation: Recommendation | None = None
    at_default_floor: dict[str, FloorPoint] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


# --- query set construction -------------------------------------------------------


@dataclass(frozen=True)
class LabelledQuery:
    query: str
    label: str
    # Fact texts that would count as a correct top hit -- every fact about the subject,
    # superseded ones included, because assembly (§7) surfaces those marked SUPERSEDED
    # rather than hiding them. That makes `useful_tpr` deliberately laxer than the
    # bench's `retrieval_hit`, which requires the top *live* fact to carry the gold
    # value. The two answer different questions and should not be quoted as one number.
    on_subject_facts: frozenset[str]


@dataclass(frozen=True)
class Fixture:
    name: str
    session: str
    candidates: list[CandidateFact]
    queries: list[LabelledQuery]


def _fold_of(subject_key: str, folds: int = 2) -> int:
    """Stable across runs and across variants -- the split must not move between arms."""
    digest = hashlib.sha1(subject_key.encode("utf-8")).digest()
    return digest[0] % folds


def map_questions_to_subjects(
    candidates: list[CandidateFact], questions: list[str], answers: list[list[str]]
) -> tuple[dict[int, str], list[str]]:
    """Question index -> the normalized subject key it asks about.

    The benchmark ships questions and gold answers but no link from a question to the
    fact that answers it, and the fold split is by subject, so the link has to be
    reconstructed. Two constraints do it between them:

      1. the gold answer is by construction the value of the subject's *newest* fact, so
         a candidate subject's newest fact must contain the gold answer;
      2. among those, the right subject is the one whose key shares the most content
         words with the question ("Which sport is goaltender associated with?" ->
         "associated goaltender sport").

    Questions that satisfy neither are dropped and counted, never silently guessed at.
    """
    by_key: dict[str, list[CandidateFact]] = {}
    for candidate in candidates:
        by_key.setdefault(normalize_subject(candidate.subject_hint), []).append(candidate)

    mapping: dict[int, str] = {}
    unmapped: list[str] = []
    for index, (question, gold) in enumerate(zip(questions, answers, strict=True)):
        question_tokens = set(normalize_subject(question).split())
        golds = [normalize_answer(g) for g in gold]
        best_key, best_score = None, 0.0
        for key, facts in by_key.items():
            newest = normalize_answer(facts[-1].fact)
            if not any(g and g in newest for g in golds):
                continue
            key_tokens = set(key.split())
            if not key_tokens:
                continue
            overlap = len(key_tokens & question_tokens) / len(key_tokens)
            if overlap > best_score:
                best_key, best_score = key, overlap
        # A subject whose key shares almost nothing with the question is a coincidence of
        # the gold value appearing somewhere, not the subject being asked about.
        if best_key is not None and best_score >= 0.5:
            mapping[index] = best_key
        else:
            unmapped.append(question)
    return mapping, unmapped


async def build_bench_fixtures(source: str, extractor_model: str) -> tuple[list[Fixture], list[str]]:
    sample = bench_data.load(source)
    extractor = KnowledgePoolExtractor(model=extractor_model)
    candidates = await extractor.candidates_for(source, sample.facts)
    await extractor.aclose()

    mapping, unmapped = map_questions_to_subjects(candidates, sample.questions, sample.answers)

    keys = {normalize_subject(c.subject_hint) for c in candidates}
    fold_of_key = {key: _fold_of(key) for key in keys}
    facts_by_key: dict[str, set[str]] = {}
    for candidate in candidates:
        facts_by_key.setdefault(normalize_subject(candidate.subject_hint), set()).add(candidate.fact)

    fixtures: list[Fixture] = []
    for fold in (0, 1):
        fold_candidates = [
            c for c in candidates if fold_of_key[normalize_subject(c.subject_hint)] == fold
        ]
        queries: list[LabelledQuery] = []
        for index, key in mapping.items():
            question = sample.questions[index]
            if fold_of_key[key] == fold:
                queries.append(LabelledQuery(question, POSITIVE, frozenset(facts_by_key[key])))
            else:
                queries.append(LabelledQuery(question, HARD_NEGATIVE, frozenset()))
        queries += [LabelledQuery(q, OFF_DOMAIN, frozenset()) for q in OFF_DOMAIN_NEGATIVES]
        fixtures.append(
            Fixture(
                name=f"bench_{source}_fold{fold}",
                session=f"calib-{source}-fold{fold}",
                candidates=fold_candidates,
                queries=queries,
            )
        )

    notes = [
        f"{len(mapping)}/{len(sample.questions)} bench questions mapped to a subject; "
        f"{len(unmapped)} dropped (no subject whose newest fact carries the gold answer)."
    ]
    return fixtures, notes


def _assert_positives_resolve(fixture: Fixture) -> None:
    """Every POSITIVE must name a fact the fixture actually stores.

    `on_subject_facts` is matched by exact fact STRING, and the facts are regenerated from
    the live extractor (`bench.gen_calib_fixtures`) while the positives are authored
    against indexes. A regeneration that shifts or rewords a fact does not raise -- it
    silently zeroes `on_subject`, and therefore `useful_tpr` and every `--subject-check`
    number, while the run still completes and prints plausible curves. So it is asserted
    at build time, where it is cheap and loud.
    """
    stored = {c.fact for c in fixture.candidates}
    broken = [
        q.query
        for q in fixture.queries
        if q.label == POSITIVE and not (q.on_subject_facts and q.on_subject_facts <= stored)
    ]
    if broken:
        raise SystemExit(
            f"{fixture.name}: {len(broken)} positive(s) point at a fact this fixture does not "
            f"store -- regenerate the positives' indexes alongside the facts: {broken}"
        )


def build_crowded_chat_fixture() -> Fixture:
    """Conversational, but with competing subjects -- see `calib_fixtures`."""
    candidates = [
        CandidateFact(
            fact=fact, type=FactType.PREFERENCE, confidence=1.0, valid_at=None,
            subject_hint=subject, attribute=attribute,
        )
        for fact, subject, attribute in CROWDED_CHAT_FACTS
    ]
    queries = [
        LabelledQuery(query, POSITIVE, frozenset({CROWDED_CHAT_FACTS[index][0]}))
        for query, index in CROWDED_CHAT_POSITIVES
    ]
    queries += [LabelledQuery(q, HARD_NEGATIVE, frozenset()) for q in CROWDED_CHAT_HARD_NEGATIVES]
    queries += [LabelledQuery(q, OFF_DOMAIN, frozenset()) for q in OFF_DOMAIN_NEGATIVES]
    fixture = Fixture(
        name="chat_crowded", session="calib-chat-crowded", candidates=candidates, queries=queries
    )
    _assert_positives_resolve(fixture)
    return fixture


def build_chat_fixture() -> Fixture:
    candidates = [
        CandidateFact(
            fact=fact,
            type=FactType.PREFERENCE,
            confidence=1.0,
            valid_at=None,
            subject_hint=subject,
            attribute=attribute,
        )
        for fact, subject, attribute in CHAT_FACTS
    ]
    queries = [
        LabelledQuery(query, POSITIVE, frozenset({CHAT_FACTS[index][0]}))
        for query, index in CHAT_POSITIVES
    ]
    queries += [LabelledQuery(q, OFF_DOMAIN, frozenset()) for q in OFF_DOMAIN_NEGATIVES]
    fixture = Fixture(name="chat", session="calib-chat", candidates=candidates, queries=queries)
    _assert_positives_resolve(fixture)
    return fixture


# --- measurement ------------------------------------------------------------------


async def measure_variant(
    variant: EmbedderVariant,
    fixtures: list[Fixture],
    store_config: StoreConfig,
    ollama_url: str,
    reingest: bool,
    subject_check: bool = False,
) -> list[Observation]:
    config = variant.embed_config(ollama_url)
    embedder = OllamaEmbedder(config)
    store = FalkorStore(
        StoreConfig(
            falkor_host=store_config.falkor_host,
            falkor_port=store_config.falkor_port,
            graph_name=variant.graph_name(),
        ),
        dimension=variant.dimension,
    )
    await store.connect()

    observations: list[Observation] = []
    try:
        for fixture in fixtures:
            stored = await store.count(fixture.session)
            if reingest or stored == 0:
                await store.clear_session(fixture.session)
                consolidator = DeterministicConsolidator(store, embedder)
                started = time.perf_counter()
                # One call per candidate: the fixtures are EXACTLY one fact per turn
                # (`gen_calib_fixtures`), and a `consolidate()` batch means one utterance
                # whose candidates cannot supersede each other (RESULTS.md §16). Chunking
                # 50 turns into one call would leave superseded facts live and quietly
                # recalibrate the floor against a store the write path never builds.
                # `prewarm` keeps the embedder batched, which is all the chunk was for.
                #
                # This does NOT owe a recalibration. Before the same-batch guard existed,
                # chunks of 50 and one-at-a-time produced identical stores, so the shipped
                # 0.57 was measured against the right one; this restores that behaviour
                # rather than changing it. The `score_floor`/embedder/`gate_on` invariant
                # is about changes that move the store, and this one is written to not.
                for i in range(0, len(fixture.candidates), 50):
                    window = fixture.candidates[i : i + 50]
                    await consolidator.prewarm([c.fact for c in window])
                    for candidate in window:
                        await consolidator.consolidate(fixture.session, [candidate])
                stored = await store.count(fixture.session)
                print(
                    f"  [{variant.name}] {fixture.name}: ingested {len(fixture.candidates)} "
                    f"-> {stored} live+superseded ({time.perf_counter() - started:.1f}s)"
                )
            else:
                print(f"  [{variant.name}] {fixture.name}: reusing {stored} stored facts")
            for labelled in fixture.queries:
                turn = TurnContext(session_id=fixture.session, user_message=labelled.query)
                result = await recall(turn, _open_gate(subject_check), store, embedder)
                hits = result.memories_used
                top = hits[0] if hits else None
                observations.append(
                    Observation(
                        variant=variant.name,
                        fixture=fixture.name,
                        session=fixture.session,
                        query=labelled.query,
                        label=labelled.label,
                        top1_score=top.score if top else 0.0,
                        top1_cosine=top.similarity if top else 0.0,
                        top1_fact=top.fact if top else None,
                        on_subject=bool(top and top.fact in labelled.on_subject_facts),
                        n_hits=len(hits),
                        latency_ms=result.latency_ms,
                    )
                )
            print(f"  [{variant.name}] {fixture.name}: {len(fixture.queries)} queries scored")
    finally:
        await embedder.aclose()
        await store.aclose()
    return observations


# --- analysis ---------------------------------------------------------------------


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)

    def pct(p: float) -> float:
        return ordered[min(len(ordered) - 1, max(0, int(round(p * (len(ordered) - 1)))))]

    return {
        "n": float(len(values)),
        "min": ordered[0],
        "p05": pct(0.05),
        "p50": pct(0.50),
        "mean": statistics.fmean(values),
        "p95": pct(0.95),
        "max": ordered[-1],
    }


GATE_QUANTITIES = ("fused", "cosine")


def _quantity(observation: Observation, gate_on: str) -> float:
    return observation.top1_cosine if gate_on == "cosine" else observation.top1_score


def sweep(
    observations: list[Observation], step: float = 0.01, gate_on: str = "fused"
) -> list[FloorPoint]:
    positives = [o for o in observations if o.label == POSITIVE]
    hard = [o for o in observations if o.label == HARD_NEGATIVE]
    off = [o for o in observations if o.label == OFF_DOMAIN]
    negatives = hard + off

    def rate(rows: list[Observation], floor: float, require_on_subject: bool = False) -> float:
        if not rows:
            return 0.0
        hit = sum(
            1
            for o in rows
            if _quantity(o, gate_on) >= floor and (o.on_subject or not require_on_subject)
        )
        return hit / len(rows)

    curve: list[FloorPoint] = []
    floor = 0.0
    while floor <= 1.0001:
        tpr = rate(positives, floor)
        fpr_hard = rate(hard, floor)
        fpr_off = rate(off, floor)
        fpr_all = rate(negatives, floor)
        curve.append(
            FloorPoint(
                floor=round(floor, 4),
                tpr=tpr,
                useful_tpr=rate(positives, floor, require_on_subject=True),
                fpr_hard=fpr_hard,
                fpr_off_domain=fpr_off,
                fpr_all=fpr_all,
                youden_j=tpr - fpr_all,
            )
        )
        floor += step
    return curve


def regime_of(fixture: str) -> str:
    """Which of the two corpora a fixture belongs to.

    The regimes are kept apart everywhere downstream. Pooling them would let the 99
    encyclopedic positives outvote the 24 conversational ones 4:1 and hand the
    recommendation to the corpus the gateway does *not* run against.
    """
    if fixture == "chat":
        return "chat"
    if fixture == "chat_crowded":
        return "chat_crowded"
    return "bench"


def _at(curve: list[FloorPoint], floor: float) -> FloorPoint:
    return min(curve, key=lambda point: abs(point.floor - floor))


def recommend(
    curves: dict[str, list[FloorPoint]],
    fpr_budget: float = 0.05,
    tpr_tolerance: float = 0.02,
) -> Recommendation:
    """Lowest floor whose off-domain false-open rate is within budget in EVERY regime.

    Off-domain is the binding constraint rather than the hard negatives: it is the
    negative a production turn actually looks like, and it is the one both regimes share.
    Hard negatives are reported separately -- if they cannot be held down at any floor
    with usable recall, that is a finding about what a scalar floor can do, not a knob to
    turn.

    Both TPR and FPR fall monotonically with the floor, so the smallest floor meeting the
    budget is also the highest-recall one that does: no search, and no tie-break needed.
    """
    regimes = [r for r, curve in curves.items() if r != "pooled" and curve]
    floors = [point.floor for point in next(iter(curves.values()))]

    def within_budget(floor: float) -> bool:
        return all(_at(curves[r], floor).fpr_off_domain <= fpr_budget for r in regimes)

    def min_tpr(floor: float) -> float:
        return min(_at(curves[r], floor).tpr for r in regimes)

    feasible = [f for f in floors if within_budget(f)]
    if not feasible:
        return Recommendation(
            fpr_budget=fpr_budget,
            floor=max(floors),
            band=(max(floors), max(floors)),
            tpr_tolerance=tpr_tolerance,
            feasible=False,
        )

    chosen = min(feasible)
    best_tpr = min_tpr(chosen)
    band = [f for f in feasible if min_tpr(f) >= best_tpr - tpr_tolerance]
    return Recommendation(
        fpr_budget=fpr_budget,
        floor=round(chosen, 4),
        band=(round(min(band), 4), round(max(band), 4)),
        tpr_tolerance=tpr_tolerance,
        feasible=True,
        per_regime={r: _at(curves[r], chosen) for r in curves},
    )


def build_report(
    variant: EmbedderVariant,
    observations: list[Observation],
    notes: list[str],
    fpr_budget: float = 0.05,
    subject_check: bool = False,
    gate_on: str = "fused",
) -> VariantReport:
    rows = [o for o in observations if o.variant == variant.name]
    by_regime: dict[str, list[Observation]] = {"pooled": rows}
    for row in rows:
        by_regime.setdefault(regime_of(row.fixture), []).append(row)

    curves = {regime: sweep(subset, gate_on=gate_on) for regime, subset in by_regime.items()}
    fixtures = sorted({o.fixture for o in rows})
    return VariantReport(
        variant=variant.name,
        model=variant.model,
        use_prefixes=variant.use_prefixes,
        gate_on=gate_on,
        subject_check=subject_check,
        counts={
            regime: {
                label: sum(1 for o in subset if o.label == label)
                for label in (POSITIVE, HARD_NEGATIVE, OFF_DOMAIN)
            }
            for regime, subset in by_regime.items()
        },
        latency={
            fixture: _summary([o.latency_ms for o in rows if o.fixture == fixture])
            for fixture in fixtures
        },
        zero_hit={
            label: sum(1 for o in rows if o.label == label and o.n_hits == 0)
            for label in (POSITIVE, HARD_NEGATIVE, OFF_DOMAIN)
            if any(o.label == label and o.n_hits == 0 for o in rows)
        },
        score_summary={
            regime: {
                label: _summary([o.top1_score for o in subset if o.label == label])
                for label in (POSITIVE, HARD_NEGATIVE, OFF_DOMAIN)
                if any(o.label == label for o in subset)
            }
            for regime, subset in by_regime.items()
        },
        cosine_summary={
            regime: {
                label: _summary([o.top1_cosine for o in subset if o.label == label])
                for label in (POSITIVE, HARD_NEGATIVE, OFF_DOMAIN)
                if any(o.label == label for o in subset)
            }
            for regime, subset in by_regime.items()
        },
        curves=curves,
        recommendation=recommend(curves, fpr_budget=fpr_budget),
        at_default_floor={
            regime: _at(curve, RecallConfig.score_floor) for regime, curve in curves.items()
        },
        notes=list(notes),
    )


def _point_line(label: str, point: FloorPoint) -> str:
    return (
        f"    {label:<22} TPR {point.tpr:.3f}  useful {point.useful_tpr:.3f}  "
        f"FPR off-domain {point.fpr_off_domain:.3f}  hard {point.fpr_hard:.3f}  J {point.youden_j:.3f}"
    )


def print_report(report: VariantReport) -> None:
    flag = "  subject_check=ON" if report.subject_check else ""
    print(
        f"\n=== {report.variant}  ({report.model}, prefixes={report.use_prefixes}) "
        f"gate_on={report.gate_on}{flag} ==="
    )
    for regime, counts in report.counts.items():
        print(
            f"  {regime:<7} {counts[POSITIVE]} positive / {counts[HARD_NEGATIVE]} hard-negative "
            f"/ {counts[OFF_DOMAIN]} off-domain"
        )
    if report.zero_hit:
        total = sum(report.zero_hit.values())
        if report.subject_check:
            # With the admission rule on, an empty result on a negative is the rule
            # WORKING -- it refused every hit. Only a positive returning nothing is
            # suspicious here, so the alarm is narrowed rather than silenced.
            positives = report.zero_hit.get(POSITIVE, 0)
            print(
                f"  {total} queries returned zero hits {report.zero_hit} -- expected with "
                "--subject-check, which refuses hits outright"
                + (f"; {positives} of them POSITIVES, which is not" if positives else "")
            )
        else:
            print(
                f"  !! {total} queries returned ZERO hits {report.zero_hit} "
                "-- scored 0.0, which on a negative looks like a correct shut. Investigate "
                "before trusting the floor (RESULTS.md §5)."
            )
    print("  recall() A-D latency, per fixture (graph grows as fixtures are ingested in turn)")
    for fixture, stats in report.latency.items():
        if stats.get("n"):
            print(f"    {fixture:<38} p50 {stats['p50']:6.1f}ms   p95 {stats['p95']:6.1f}ms")
    # The distribution of the quantity actually being swept, not always the fused one --
    # otherwise a `gate_on=cosine` report prints numbers its own curves are not derived
    # from, which is how a floor gets read off the wrong column.
    summary = report.cosine_summary if report.gate_on == "cosine" else report.score_summary
    print(f"  top-1 {report.gate_on} distribution")
    for regime, labels in summary.items():
        if regime == "pooled":
            continue
        for label, stats in labels.items():
            print(
                f"    {regime:<6} {label:<20} p05 {stats['p05']:.3f}  p50 {stats['p50']:.3f}  "
                f"mean {stats['mean']:.3f}  p95 {stats['p95']:.3f}  max {stats['max']:.3f}"
            )

    default = RecallConfig.score_floor
    print(f"  at the current default floor {default:.2f}")
    for regime, point in report.at_default_floor.items():
        if regime != "pooled":
            print(_point_line(regime, point))

    rec = report.recommendation
    if rec is None:
        return
    if not rec.feasible:
        print(
            f"  NO floor holds off-domain false opens at or under {rec.fpr_budget:.0%} "
            "in both regimes -- this model cannot be shipped on a scalar floor."
        )
        return
    print(
        f"  recommended floor {rec.floor:.2f}  (band {rec.band[0]:.2f}-{rec.band[1]:.2f} within "
        f"{rec.tpr_tolerance:.0%} TPR, off-domain FPR budget {rec.fpr_budget:.0%})"
    )
    for regime, point in rec.per_regime.items():
        if regime != "pooled":
            print(_point_line(regime, point))
    for note in report.notes:
        print(f"  note: {note}")


def reanalyze(path: Path, fpr_budget: float) -> list[VariantReport]:
    """Rebuild the reports from saved observations.

    The measurement is the expensive half and the analysis is the half that gets
    revised, so the raw per-query observations are the artifact of record. Changing how
    a floor is chosen must never require re-embedding a corpus.
    """
    payload = json.loads(path.read_text())
    observations = [Observation(**row) for row in payload["observations"]]
    notes = payload.get("reports", [{}])[0].get("notes", [])

    # `top1_cosine` defaults to 0.0 so a pre-§13 file still loads -- but scoring the
    # cosine arm off those zeros would report a floor of 0.0 opening on nothing, which
    # reads as a real curve rather than as missing data. Refuse instead: a wrong answer
    # that looks like a right one is the worst thing this harness could produce.
    quantities = GATE_QUANTITIES
    if any(o.top1_score > 0.0 for o in observations) and not any(
        o.top1_cosine > 0.0 for o in observations
    ):
        quantities = ("fused",)
        print(
            f"  {path.name} predates RESULTS.md §13 and carries no un-fused cosine; "
            "reporting the fused arm only. Re-run the measurement to compare gate "
            "quantities."
        )

    reports = []
    for name in dict.fromkeys(o.variant for o in observations):
        for gate_on in quantities:
            reports.append(
                build_report(
                    VARIANTS[name], observations, notes, fpr_budget=fpr_budget, gate_on=gate_on
                )
            )
    return reports


async def main_async(args: argparse.Namespace) -> None:
    if args.analyze:
        reports = reanalyze(Path(args.analyze), args.fpr_budget)
        for report in reports:
            print_report(report)
        if args.out:
            payload = json.loads(Path(args.analyze).read_text())
            payload["reports"] = [asdict(r) for r in reports]
            Path(args.out).write_text(json.dumps(payload, indent=2))
            print(f"\nwrote {args.out}")
        return

    names = list(VARIANTS) if args.configs == "all" else args.configs.split(",")
    unknown = [n for n in names if n not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variant(s) {unknown}; known: {sorted(VARIANTS)}")

    fixtures: list[Fixture] = []
    notes: list[str] = []
    if not args.chat_only:
        bench_fixtures, bench_notes = await build_bench_fixtures(args.source, args.extractor_model)
        fixtures += bench_fixtures
        notes += bench_notes
    fixtures.append(build_chat_fixture())
    fixtures.append(build_crowded_chat_fixture())
    for fixture in fixtures:
        print(f"fixture {fixture.name}: {len(fixture.candidates)} facts, {len(fixture.queries)} queries")

    store_config = StoreConfig.from_env()
    observations: list[Observation] = []
    reports: list[VariantReport] = []
    for name in names:
        variant = VARIANTS[name]
        rows = await measure_variant(
            variant,
            fixtures,
            store_config,
            args.ollama_url,
            reingest=args.reingest,
            subject_check=args.subject_check,
        )
        observations += rows
        # One measurement pass, both gate quantities. The scores are already in hand, so
        # comparing them costs no embedding and no store round-trip -- and crucially both
        # are scored on the SAME observations, so the comparison isolates the quantity.
        for gate_on in GATE_QUANTITIES:
            report = build_report(
                variant, rows, notes, fpr_budget=args.fpr_budget,
                subject_check=args.subject_check, gate_on=gate_on,
            )
            reports.append(report)
            print_report(report)

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "reports": [asdict(r) for r in reports],
                    "observations": [asdict(o) for o in observations],
                },
                indent=2,
            )
        )
        print(f"\nwrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="score_floor calibration across embedders")
    parser.add_argument("--configs", default="all", help=f"comma-separated, or 'all': {sorted(VARIANTS)}")
    parser.add_argument("--source", default="factconsolidation_sh_6k")
    parser.add_argument("--extractor-model", default=CACHED_EXTRACTOR_MODEL)
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--chat-only", action="store_true", help="skip the benchmark fixture")
    parser.add_argument("--reingest", action="store_true", help="rebuild stores even if populated")
    parser.add_argument(
        "--fpr-budget",
        type=float,
        default=0.05,
        help="max off-domain false-open rate the chosen floor may allow, in every regime",
    )
    parser.add_argument(
        "--no-subject-check",
        dest="subject_check",
        action="store_false",
        help="disable the wrong-subject admission rule (on by default, mirroring RecallConfig)",
    )
    parser.set_defaults(subject_check=RecallConfig.subject_check)
    parser.add_argument(
        "--analyze",
        default=None,
        help="re-derive reports from a saved run's observations instead of measuring",
    )
    parser.add_argument("--out", default="data/results/calibration.json")
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
