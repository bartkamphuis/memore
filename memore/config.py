"""Config surface -- single source of truth (recall-stage-spec.md §11,
recall-writepath-spec.md §4)."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class RecallConfig:
    """recall-stage-spec.md §11."""

    alpha: float = 0.7  # key-synthesis blend weight (§4)
    k: int = 12  # retrieval width (§5)
    lookup_timeout_ms: int = 180  # hard timeout on hybrid_search (§5)
    # Gate floor (§6). Calibrated for `EmbedConfig.model` AND for `gate_on` below -- the
    # three are one decision, not three constants. Re-run
    # `python -m memore.bench.calibrate` before changing any of them.
    #
    # The spec's 0.35 was carried over untested and is wrong for every model measured: it
    # opens the gate on 20% of off-domain conversational turns even with the
    # embeddinggemma it was written for.
    #
    # 0.48 was this project's first calibrated value (RESULTS.md §5) and is now retired.
    # It was measured against fixtures that had drifted from the write path -- terse
    # authored facts, and positives sharing word stems with them -- so it never saw the
    # paraphrase case that dominates real use. Re-measured against extractor-derived facts
    # and paraphrase-only positives, 0.48 lets off-domain false opens reach 0.077, over
    # the 5% budget it was supposed to satisfy.
    #
    # 0.57 is the lowest floor holding off-domain false opens under 5% in all three
    # measured regimes (band 0.57-0.57). RESULTS.md §13.
    score_floor: float = 0.57
    # WHICH quantity the floor is compared against, and it is one decision with the floor
    # exactly as the floor is one decision with the embedder -- each choice puts the
    # relevant/irrelevant boundary somewhere different, so a floor only means something
    # next to the quantity it was calibrated on.
    #
    #   "fused"   `cos·(1 + w·bm25)/(1 + w)` -- ranks well, but its range is
    #             `[cos/(1+w), cos]`, so a fact sharing no term with the query is docked
    #             ~23% before the floor sees it, and whether the gate opens then depends
    #             on wording and on what else is in the store (RESULTS.md §12).
    #   "cosine"  the un-fused similarity. Ranking still uses the fused score; only the
    #             open/shut test changes. Cannot inflate anything past the floor, since
    #             `fused <= cos` always, so it does not reintroduce what the
    #             multiplicative-fusion invariant guards against.
    #
    # Measured at the SAME recommended floor (0.57), so this is a like-for-like swap:
    #
    #                bench TPR/hard    chat TPR/off-domain   crowded TPR/hard
    #   fused        1.000 / 0.111     0.622 / 0.026         1.000 / 0.692
    #   cosine       1.000 / 0.263     0.784 / 0.026         1.000 / 0.692
    #
    # +16.2 points of conversational recall at an identical floor and identical off-domain
    # false-open rate. The cost is bench hard negatives (right relation, absent subject) --
    # and that is `subject_check`'s job, not the floor's: the gate keeps *off-topic* memory
    # out, never *wrong-subject* memory, because those distributions overlap and no scalar
    # threshold separates them (RESULTS.md §5, §9). Trading a metric the floor was never
    # able to own for recall it can is the right side of that division of labour.
    #
    # Re-run `python -m memore.bench.calibrate` before changing this OR the floor.
    gate_on: str = "cosine"
    inject_token_budget: int = 512  # hard cap on injected tokens (§6)
    enabled: bool = True  # master kill switch -> full no-op (§7)
    # Multi-hop chain expansion (`memore.chain`). 0 disables it entirely and recall
    # behaves exactly as before. This runs AFTER the gate, never before: the gate's
    # open/shut decision stays a pure function of similarity to the turn, so
    # `score_floor` keeps the meaning it was calibrated with. Expansion only decides what
    # else travels with an already-opened block, still under `inject_token_budget`.
    expansion_hops: int = 0
    expansion_fanout: int = 4  # max neighbours followed per fact, per hop
    # Subject admission (`memore.subjects`): refuse a hit whose subject is one of a crowd
    # of same-relation subjects when the query names none of what distinguishes it. This
    # is the only defence against wrong-subject recall -- right topic, wrong entity --
    # which no score threshold can catch, because the distributions overlap.
    # ON by default: measured a strict improvement in all three regimes at no cost to
    # recall (RESULTS.md §9). Wrong-subject false opens fell 0.758 -> 0.253 on the
    # benchmark and 0.846 -> 0.462 on a conversational session with competing subjects,
    # while TPR held at 1.000/0.920/1.000 and useful-recall *rose* 0.970 -> 0.990 --
    # refusing a wrong-subject top hit promotes the correct fact underneath it.
    subject_check: bool = True
    # NOT tuned for accuracy. `1` polices any subject with a single sibling; `2` requires
    # a genuine crowd. The measured trade, and the reason 2 ships:
    #
    #   2  strict improvement everywhere, no recall cost in any regime.
    #   1  additionally blocks 69% of wrong-entity queries in a two-subjects-per-relation
    #      chat store, but costs ~8% of conversational recall -- real paraphrase losses
    #      ("what kind of milk do I use?" against subject "coffee preference" shares no
    #      token, so a lexical check cannot confirm what similarity found).
    #
    # Raise to 1 only with eyes open, and re-run `memore.bench.calibrate --subject-check`.
    subject_min_competitors: int = 2
    # A subject token in this many live subjects or fewer counts as naming the entity
    # rather than the relation.
    subject_df_max: int = 2


# Generation model, and the context width it is served at. These two are ONE decision for
# an operational reason that has nothing to do with quality: Ollama reloads a model when a
# request asks for options it was not loaded with, and `num_ctx` is one of those. A default
# that disagrees with how the model is actually served turns every process boundary into a
# multi-second reload of an 18GB weight set. So the default matches the served pin:
#
#     gemma4:26b   32768 ctx   100% GPU   keep_alive Forever
#
# Change both together, or not at all. `MEMORE_LLM_NUM_CTX` exists so a differently-served
# host can align without a code edit -- it is not a tuning knob.
DEFAULT_LLM_MODEL = os.getenv("MEMORE_LLM_MODEL", "gemma4:26b")
DEFAULT_LLM_NUM_CTX = int(os.getenv("MEMORE_LLM_NUM_CTX", "32768"))

# `keep_alive` is a PER-REQUEST field: whatever a request says (or defaults to) governs the
# residency of the model it touched. -1 means "stay loaded", which preserves an operator's
# pin instead of quietly replacing it with Ollama's 5-minute default.
DEFAULT_KEEP_ALIVE: int | str = -1


@dataclass(frozen=True)
class WritePathConfig:
    """recall-writepath-spec.md §4."""

    extract_window_turns: int = 3
    min_extract_confidence: float = 0.6
    consolidation_k: int = 5
    match_floor: float = 0.5
    extractor_model: str = DEFAULT_LLM_MODEL
    enabled: bool = True


@dataclass(frozen=True)
class StoreConfig:
    """Connection settings. Defaults target the compose stack; env vars let the same
    code run under `uv run` on the host and inside the app container unchanged."""

    falkor_host: str = "localhost"
    falkor_port: int = 6379
    graph_name: str = "memore"
    # Hybrid fusion weights (see store/falkor.py for the normalization contract).
    vector_weight: float = 0.7
    text_weight: float = 0.3

    @classmethod
    def from_env(cls) -> StoreConfig:
        return cls(
            falkor_host=os.getenv("MEMORE_FALKOR_HOST", "localhost"),
            falkor_port=int(os.getenv("MEMORE_FALKOR_PORT", "6379")),
            graph_name=os.getenv("MEMORE_GRAPH", "memore"),
        )


# Asymmetric embedders want to know whether a string is being stored or searched for.
# nomic-embed-text is trained with these exact prefixes -- but measured over the
# calibration query set they make its discrimination *worse*, not better (RESULTS.md §5),
# so the mapping is kept for the record and no shipped default uses it.
# Prefixes a model's card documents, whether or not we use them. Reference data for
# `bench.calibrate`, which measures each as its own arm -- NOT applied by `from_env`.
KNOWN_PREFIXES = {
    "nomic-embed-text": ("search_document: ", "search_query: "),
    "mxbai-embed-large": ("", "Represent this sentence for searching relevant passages: "),
}

# Prefixes actually APPLIED. Separate from the table above on purpose: a card documents
# what a model was trained with, but whether the prefix helps *this* gate at *this* floor
# is an empirical question, and the two tables disagree.
#
# mxbai-embed-large is trained asymmetrically and its card specifies the query prefix
# above. Measured (RESULTS.md §13) it costs conversational recall at every operating
# point -- chat TPR 0.784 -> 0.730 gating on cosine, 0.622 -> 0.595 gating on fused, with
# no off-domain gain -- so it is deliberately NOT applied. Listing it here would enable it
# silently for every deployment, because `from_env` consults this table by model name.
_PREFIXES = {
    "nomic-embed-text": ("search_document: ", "search_query: "),
}

# Output width per model. The store's vector index is created with this dimension, so a
# mismatch is not a tuning error -- it is an index that cannot be built. Resolved from
# the model name so that setting MEMORE_EMBED_MODEL alone is sufficient; MEMORE_EMBED_DIM
# still overrides for a model not listed here.
_DIMENSIONS = {
    "embeddinggemma": 768,
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
}

DEFAULT_EMBED_MODEL = "mxbai-embed-large:latest"


def _lookup(table: dict, model: str, fallback):
    for stem, value in table.items():
        if model.startswith(stem):
            return value
    return fallback


@dataclass(frozen=True)
class EmbedConfig:
    """Embedding model choice, which is coupled to `RecallConfig.score_floor`.

    The model and the floor are one decision, not two: each model puts the boundary
    between a relevant and an irrelevant query in a different place, so a floor is only
    meaningful next to the model it was calibrated against. Both were chosen together
    over the calibration query set (`memore.bench.calibrate`, RESULTS.md §5) rather than
    from a handful of ad-hoc queries.

    `mxbai-embed-large` is the default. It is ~4x faster than the `embeddinggemma` it
    replaces (p95 A-D 83ms against 350ms, which is what brings the §14 200ms budget into
    reach) at essentially the same discrimination -- at their own calibrated floors,
    bench useful-recall 0.970 against 0.990 and chat recall 0.917 against 0.875.
    `nomic-embed-text` is the same speed but a measurably worse ranker: useful-recall
    0.869, and its FactConsolidation `retrieval_hit` drops 0.93 -> 0.77.

    Changing `model` REQUIRES recalibrating `RecallConfig.score_floor` and rebuilding the
    store: the vectors already written are the old model's, and the vector index is
    created at a fixed `dimension`.
    """

    ollama_url: str = "http://localhost:11434"
    model: str = DEFAULT_EMBED_MODEL
    dimension: int = 1024
    document_prefix: str = ""
    query_prefix: str = ""

    @classmethod
    def from_env(cls) -> EmbedConfig:
        model = os.getenv("MEMORE_EMBED_MODEL", DEFAULT_EMBED_MODEL)
        document_prefix, query_prefix = _lookup(_PREFIXES, model, ("", ""))
        return cls(
            ollama_url=os.getenv("MEMORE_OLLAMA_URL", "http://localhost:11434"),
            model=model,
            dimension=int(os.getenv("MEMORE_EMBED_DIM", _lookup(_DIMENSIONS, model, 768))),
            document_prefix=os.getenv("MEMORE_EMBED_DOC_PREFIX", document_prefix),
            query_prefix=os.getenv("MEMORE_EMBED_QUERY_PREFIX", query_prefix),
        )
