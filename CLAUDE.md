# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

Step 0 (the consolidation spike) is complete — both arms ran; see `RESULTS.md`, and read
it before quoting any number, because several of its findings correct earlier claims.

The §6 demo trace reproduces in the part that defines "it works": NEW at ordinal 1, gate
open on the ask-turn, CONTRADICTION resolved by freshness ordinal with the prior fact
superseded not deleted, both facts surfaced with correct `valid as of` / `SUPERSEDED`
framing, no write on the transient turn, gate shut on an unrelated one. What does *not*
match §6's illustrative text is P1's output shape, which is **extractor-model dependent**
and has changed once already: under `gemma4:26b` it emits `PREFERENCE` as the spec writes
(the `gemma4:12b` this repo defaulted to earlier emitted `IDENTITY`), but it still keeps
the turn's own phrasing ("The user deploys to staging by default") rather than the spec's
normalized "deploys to X by default", and reports `conf=1.00` against the spec's
illustrative 0.82. Extractor behaviour, unrelated to consolidation; don't claim the trace
matches verbatim, and re-check it after any extractor-model change rather than trusting
this paragraph.

That `conf=1.00` is not evidence that `min_extract_confidence` (0.6) has become a dead
filter — checked, because it would matter. `gemma4:26b` emits a graded 1.0 / 0.9 / 0.6
across turns of varying hedging; the demo turns are simply unambiguous. Genuinely tentative
turns ("I might switch to Postgres at some point, maybe") are dropped by P1's salience
judgement before the threshold sees them, which is the designed division of labour.

Steps 1–3 are built too: store, embedder, deterministic consolidator, P1 extraction,
`recall()` with the gate, context assembly, write-path orchestration, terminal demo,
bench harness, and the `score_floor` calibration harness (`memore/bench/calibrate.py`).

Multi-hop is built and measured too (RESULTS.md §8): a deterministic value→subject graph
walk in `memore/chain.py`, run *after* the gate, taking `factconsolidation_mh_6k` from
0.200 to **0.760** exact-match against a field in single digits. It is off by default
(`expansion_hops = 0`) — see the invariant below before turning it on.

Subject aliasing is built and measured (RESULTS.md §10): `memore/aliases.py` merges two
namings of one subject when they differ only by *generic relation words*, decided by
document frequency across the session's subjects. This is the gated form of the subset
merge §3 rejected, and it closes §3's last open item — sh_32k oracle 0.940 → **0.960**
with **zero** over-merges, reaching the score ungated subset-merging got while still
refusing every merge §3 objected to. On by default; inert below 200 subjects, so
conversational stores are untouched.

Still unbuilt, all deliberately deferred by `recall-poc-spec.md` §5: the async job
machinery, rolling-summary-vector key synthesis, the queryable audit log, and
cross-session recall. The §14 200ms P95 latency budget **is now met** (~96ms P95, ~146ms
with chain expansion on) after calibrating `score_floor` over a real query set and
switching the embedder — RESULTS.md §5. The open weakness is wrong-subject recall, not
latency: see the scalar-floor limit below.

## Commands

```bash
uv sync --extra dev --extra bench          # dev env; add --extra graphiti for spike arm (a)
docker compose up -d falkordb              # the store; app container is `memore` in compose.yaml
uv run pytest tests/ -q                    # full suite
uv run pytest tests/test_consolidation.py::test_contradiction_supersedes_without_deleting  # one test
uv run ruff check memore/ tests/

# What does the store actually hold? First thing to run when recall returns nothing --
# an empty session and a broken lookup look identical from the outside, and recall is
# session-scoped, so the bench's `bench-<source>` session is invisible to `demo`.
uv run memore inspect
uv run memore inspect --session <name> --query '...'

# Step-0 spike (needs FalkorDB up and Ollama on the host)
uv run python -m memore.bench.run --source factconsolidation_sh_6k --arm deterministic
uv run python -m memore.bench.oracle_run --source factconsolidation_sh_6k

# score_floor calibration -- re-run after ANY change to the embedder or the floor
uv run python -m memore.bench.calibrate --configs all --reingest
uv run python -m memore.bench.calibrate --analyze data/results/calibration.json  # re-score, no re-embed

# Multi-hop (RESULTS.md §8). --via-recall routes through the real recall stage (gate +
# budget) instead of raw top-k; with --expansion-hops 0 that isolates the gate's
# contribution from the walk's. --no-context is the parametric-knowledge control.
uv run python -m memore.bench.run --source factconsolidation_mh_6k --expansion-hops 3
uv run python -m memore.bench.run --source factconsolidation_mh_6k --via-recall
uv run python -m memore.bench.run --source factconsolidation_mh_6k --no-context
```

Each embedder needs its **own graph** (`MEMORE_GRAPH`): the vector index is created at a
fixed dimension, and stored vectors are only searchable by the model that wrote them.
`connect()` refuses to open a graph whose index width does not match the configured
embedder, so a stale graph from a previous model fails loudly at startup. The default
`memore` graph predates the switch to `mxbai-embed-large` and holds 768-dim vectors —
drop it once, or point elsewhere:

```bash
docker exec memore-falkordb redis-cli GRAPH.DELETE memore   # rebuild under the new embedder
MEMORE_GRAPH=memore_mxbai uv run python -m memore.cli demo  # or just use another graph
```

The bench needs `data/Conflict_Resolution.parquet`; `memore/bench/data.py` prints the
curl command to fetch it. Subject extraction is cached under `data/cache/` — delete the
cache to re-extract, and note that **both spike arms must consume the same cache file**
or the comparison stops isolating the consolidation decision.

Ollama runs on the **host**, not in compose (the GPUs already serve the models there);
the app container reaches it via `host.docker.internal`. `redis` is pinned `<8.1` — see
the comment in `pyproject.toml`, it is a real incompatibility, not caution.

### Match the served models, or every process start pays an 18GB reload

Ollama reloads a model when a request asks for options it was not loaded with — `num_ctx`
is one — and `keep_alive` is **per-request**, so a request that omits it replaces an
operator's `Forever` pin with the 5-minute default. Defaults therefore track how the host
actually serves them, and are one decision, not three:

```
gemma4:26b                32768 ctx   100% GPU   Forever   <- config.DEFAULT_LLM_MODEL / _NUM_CTX
mxbai-embed-large:latest    512 ctx   100% GPU   Forever   <- config.DEFAULT_EMBED_MODEL
```

`MEMORE_LLM_MODEL` / `MEMORE_LLM_NUM_CTX` align a differently-served host without a code
edit. Two call sites deliberately do *not* follow the default, both for measurement
reasons: `bench.extract.CACHED_EXTRACTOR_MODEL` is a **cache key** — every subject cache
and therefore every §3 oracle number is `gemma4:12b`'s, and repointing it re-extracts for
hours while changing the one variable those runs hold fixed — and `bench.reader.Reader`
now follows the default, which means end-to-end numbers produced from here on are a
different reader's than RESULTS.md §2–§3 (pass `--reader-model gemma4:12b` to reproduce
those). The oracle uses no LLM and is unaffected either way.

## The spec set — cited everywhere, not published

Source docstrings cite four design documents by bare filename:

- `recall-poc-spec.md` — scopes the standalone terminal PoC this repo builds. The entry point.
- `recall-stage-spec.md` — the read path (recall stage A–D), store choice, config, invariants (§13), definition of done (§14).
- `recall-writepath-spec.md` — the write path (P1 extract → P2 consolidate → P3 commit); expands stage-spec §8.
- `recall-stage-test-spec.md` — test suites and fixtures for the read path.

**These are not in the public repo.** They describe a private production integration target, so a `§`-reference in a docstring points at a document you will not find here. That is deliberate, not rot. Everything load-bearing from them — the invariants, the interfaces, the reasoning behind each — is restated below and in `RESULTS.md`, which is why those two files are long. Nothing in the code depends on the specs being present.

### The one live override — do not miss this

`recall-poc-spec.md` §4 **overrides** `recall-writepath-spec.md` §2.3. The writepath spec says to delegate NEW / DUPLICATE / CONTRADICTION / REFINEMENT to Graphiti's edge-invalidation. The PoC spec revokes that: Graphiti is **storage + hybrid retrieval only**, and consolidation is a deterministic freshness-ordinal primitive we own, in a `Consolidator` behind the `MemoryStore` boundary. Anyone reading the writepath spec alone will implement the wrong thing.

The reason is the whole thesis: Graphiti scores 7% on MemoryAgentBench FactConsolidation single-hop, field best *was* ~54%, and a deterministic freshness primitive is the claimed edge. Consolidation is the novelty; everything else is settled wheel to be reused.

**That framing is now partly overtaken — see RESULTS.md §0 before repeating it.** arXiv:2606.01435 (Reddy & Challaram, May 2026) published the same deterministic-freshness idea and scores 0.948 single-hop / 0.515 multi-hop with gpt-4o, so ~54% is no longer the bar and the primitive is no longer novel. The override above still stands unchanged — delegating consolidation to Graphiti's edge-invalidation is still the wrong build — but what is *ours* is narrower than "consolidation": the deterministic multi-hop chain walk (§8), subject-identity work (§9, §10), the write-time rather than read-time resolution, and running entirely on local models. Do not write "beats the field by 40 points" anywhere.

## Build order (PoC — reversed from the production spec)

Production (`recall-stage-spec.md` §12) is read-path-first. The PoC is **write-path-first**, because recall quality is downstream of store quality:

0. **Consolidation spike** — build both the Graphiti-delegated and the deterministic-primitive path, measure on FactConsolidation single-hop. This gates everything: if the deterministic one doesn't beat ~54%, the thesis is dead and further building is waste.
1. Write path: extract → deterministic consolidate → commit.
2. Read path: `recall()` — key synthesis → hybrid lookup → gate → assembly.
3. Demo REPL alternating learn-turns and ask-turns, printing the full trace (the target trace is `recall-poc-spec.md` §6 — that trace being correct is the definition of "it works").
4. Reproducible bench harness around the Step-0 measurement.

## Interfaces — verbatim, no ad-hoc shapes

Use the production-spec types exactly as written: `MemoryStore`, `recall()`, `TurnContext`, `RecallResult`, `MemoryHit`, `Episode`, `CandidateFact`, `Consolidator`, `RecallConfig`, `WritePathConfig`. This is the discipline that makes the PoC the production core rather than a throwaway — the test is whether the gateway could import `recall()` and the consolidator unchanged. Inventing quick shapes to move fast defeats the point of building standalone.

## Invariants that are easy to violate by accident

- **No LLM in recall components A–D, and none in the consolidation decision.** An LLM *is* the right tool for P1 extraction — that runs off the response path.
- **Value comparison in consolidation is normalized-string equality, and `use_embedding_comparison` defaults to False.** Do not "improve" this by turning embedding similarity back on: a false DUPLICATE discards an update permanently, while a false CONTRADICTION keeps both facts with the right one live. Measured at 32k, sentence embeddings put real value changes ("rugby union"→"rugby", 0.982) *above* the threshold and genuine paraphrases (0.877) *below* real contradictions (0.849–0.911), so no threshold separates the cases. RESULTS.md §6.
- `recall()` never raises. Store error or timeout → closed result, logged, turn proceeds.
- `enabled=False` on either path → full no-op.
- `inject_token_budget` is a hard cap, not a target.
- Session-scoped only; no cross-session recall in v1.
- No Graphiti types leak past the `MemoryStore` boundary.
- Supersede, never delete — superseded facts stay and surface to recall marked `SUPERSEDED`.
- **`score_floor` and `EmbedConfig.model` are ONE decision, not two.** Each model puts the relevant/irrelevant boundary somewhere different, so a floor only means something next to the model it was calibrated against. Current pairing: `mxbai-embed-large` + floor `0.48`. Changing either requires re-running `uv run python -m memore.bench.calibrate` and re-checking both regimes — never swap the embedder and keep the floor. The spec's 0.35 was never calibrated for *any* model: it opens the gate on 20% of off-domain conversational turns even with the `embeddinggemma` it was written for. RESULTS.md §5.
- **A different embedder also means a different vector-index width**, and a graph indexed for the old width cannot be searched with the new one. `FalkorStore.connect()` raises on the mismatch; do not "fix" that by catching it. Use a fresh `MEMORE_GRAPH` or rebuild. It is guarded because the failure is silent otherwise: the write succeeds, only the query raises, and `recall()` swallows store errors by design — leaving a gate shut forever. RESULTS.md §5.
- **The gate keeps *off-topic* memory out, not *wrong-subject* memory.** Measured at the calibrated floor: off-domain false-open ~0.03, but **hard**-negative (right topic, wrong subject) false-open **0.68–0.88** — and that is the *best* case, bought with recall; at 0.35 it is 1.000. The distributions overlap; no threshold separates them. Do not try to tune it away — fixing it needs a subject-key check the score cannot carry. RESULTS.md §5.
- **Subject aliasing merges on document frequency, and the threshold is a RATIO, never a count.** `affiliated` is df 9 at 6k and df 60 at 32k — the same relation word — so an absolute cutoff tuned on one corpus is either inert or catastrophic on the other. `AliasConfig.df_ratio` is `df / n_subjects`, defaulting to `0.015`, chosen in the *wide* gap between the hand-labelled bands (2.1x margin over `kingdom` at 0.0071) rather than the narrow one at 0.0083. Measured: 0.015 and 0.008 score identically, so the margin costs nothing. Re-run `python -m memore.bench.oracle_run --alias-df-ratio ...` on **both** corpora before touching it, and judge it on the printed merge log, never the score — RESULTS.md §3 rejected the ungated rule *despite* a better score. RESULTS.md §10.
- **`AliasConfig.min_subjects` (200) is a safety guard, not a performance knob.** A relative threshold degrades worse than an absolute one at small scale: in a 12-subject session `location` in two subjects reads as ratio 0.17, eleven times the threshold, so an unguarded rule fires hardest where its evidence is weakest. Nothing below 303 subjects has been validated. Lowering it merges conversational stores on evidence that does not exist. RESULTS.md §10.
- **Aliasing reads document frequency in ARRIVAL order, not from the finished corpus.** The rule is conservative when cold — `affiliated` is not yet a relation word on its first appearance — and converges as the session fills. `bench.oracle.build_groups` replays the same growing vocabulary deliberately: grouping a completed corpus in one pass credits the store with merges ingest never made and scores a system that was never run. If you add a caller, pass it the same `AliasConfig` the ingest used. RESULTS.md §10.
- **Chain expansion runs AFTER the gate, never before, and never seeds it.** A multi-hop answer fact shares no entity with the question — that is what makes it multi-hop — so it cannot clear a similarity floor and must not be asked to. Judging relevance on the seeds and letting the chain ride along is what keeps `score_floor` meaning what it was calibrated to mean. Moving expansion ahead of the gate would silently re-open the gate on turns that have nothing relevant, and would invalidate §5's calibration. Chain facts carry `score=0.0` because they were never ranked against the turn — do not invent a similarity for them. RESULTS.md §8.
- **Hybrid fusion is multiplicative** (`cos · (1 + w·bm25)/(1 + w)`), not additive. Additive fusion mixes an absolute score with a set-relative one and lets an irrelevant query clear `score_floor`. Full-text queries must be `|`-joined term unions, or the index ANDs them and the BM25 arm silently returns nothing. RESULTS.md §5.
- **Do not wrap this in MCP.** MCP is permanently the wrong shape for pre-fetch recall (it would put recall back behind a model decision, which is the pattern this design leaves behind). It is only the right shape for the deliberate model-initiated fallback lookup — see `recall-poc-spec.md` §5a.

Deferred PoC scope is listed in `recall-poc-spec.md` §5 (cross-session recall, async job machinery, rolling-summary key synthesis, provider abstraction, audit-log query path). Mark these in code with comments pointing at the production spec section rather than omitting them silently.
