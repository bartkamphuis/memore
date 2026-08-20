# memore

*Italian, from Latin* memor *— mindful, remembering.*

Deterministic memory for LLM agents. Facts are consolidated **when they are written**, so a
slot holds one live value at a time and "which of these two contradictory facts is current?"
is never asked at read time — or asked of a model at all.

**There is no LLM in the invalidation path, and there is a test that asserts it.**
Every comparable system puts one there. Here, which fact supersedes which is decided by
`(subject key, freshness ordinal, value equality)` — a pure function whose collaborators are
pinned by `test_no_llm_in_the_consolidation_decision`. The benchmark numbers below say the
determinism costs nothing: 0.990 / 0.960 oracle consolidation accuracy at 6k / 32k, zero gold
facts wrongly superseded.

**The limit is in the other lane, and it is measured.** An LLM still *extracts* candidate
facts, off the response path, and it names one assertion differently when the wording
changes: **0.583 subject agreement and 0.417 attribute agreement across four paraphrases of
one sentence**, against a self-agreement control of 12/12 (`RESULTS.md` §26). Subject
identity is exact match, so a differently-named subject is a split subject. That is the open
problem, it is stated precisely in `RESULTS.md` §29.3, and it is not solved here.

Memory is fetched *before* the model call and injected at prompt-assembly time, rather than
exposed as a tool the model chooses to invoke. Local models throughout: no cloud LLM
anywhere in the pipeline, and none at all in the recall path or the consolidation decision.

📄 **[`docs/explainer.md`](docs/explainer.md)** — the design walked through with diagrams.
📓 **[`RESULTS.md`](RESULTS.md)** — every run, including the ones that failed. Start at §29.
🚫 **[Do not re-litigate](CLAUDE.md#do-not-re-litigate--the-dead-ends-and-the-arithmetic-that-closed-them)** —
the dead ends, each with the measurement or the arithmetic that closed it. If you are
building one of these, this is probably the most useful section in the repo.

```
turn ──► recall()   A key synthesis ─► B hybrid lookup ─► C gate ─► D assembly ──► prompt
                    no LLM in A–D · 200ms P95 budget · 79ms conversational,
                                                        127-134ms at 2310 facts

turn ──► write path  P1 extract (LLM, off-path) ─► P2 consolidate ─► P3 commit
                                                   deterministic: (subject key,
                                                   freshness ordinal, value equality)
```

## Where this sits relative to published work

**Read this before quoting any number from this repo.**

The deterministic-freshness idea is **not** original here. Reddy & Challaram,
[*Don't Ask the LLM to Track Freshness: A Deterministic Recipe for Memory Conflict
Resolution*](https://arxiv.org/abs/2606.01435) (arXiv:2606.01435v1, May 2026), proposed
LLM candidate-extraction followed by a deterministic `max(serial)` and measured it on the
same benchmark. That paper is prior art for the core thesis, and this project was built
against it.

Two things are worth knowing about it. First, it was **revised in August 2026** and
retitled *Reliable Post-Retrieval Assembly for Agent Memory: Separating Evidence Extraction
from Policy Execution* — the authors now attribute the gain primarily to separating
evidence extraction from policy execution "rather than the freshness mechanism alone."
Second, this repo reached the same conclusion independently and from the other direction:
consolidation was correct on **every** subject group in every run at both corpus sizes, and
every single residual error was extraction naming one subject two different ways
(`RESULTS.md` §3). Freshness was never the bottleneck. Subject identity was.

Measured against their numbers on MemoryAgentBench FactConsolidation. Their §4.5 reports
**SubEM** (substring exact match), which is MemoryAgentBench's own metric, so that is the
matched row; exact-match is the stricter figure this repo prefers and they do not report it:

One reader throughout: `gemma4:26b`, measured 2026-08-21 through the shipped `recall()`
path. (`RESULTS.md` §8's older multi-hop row — 0.800 SubEM / 0.760 exact — is a `gemma4:12b`
reader on the same store, and must not be mixed into this table.)

| | memore (local `gemma4:26b`) | 2606.01435 (gpt-4o) |
|---|---|---|
| single-hop, SubEM | 1.000 @6k · 0.980 @32k | 0.948 |
| single-hop, exact-match | 1.000 @6k · 0.980 @32k | not reported |
| multi-hop, SubEM | **0.880** @6k | 0.515 |
| multi-hop, exact-match | **0.850** @6k | not reported |

Single-hop is a **modest edge at best and not a controlled comparison** — different reader,
different setup, and 2 of the 100 single-hop questions at 32k have gold answers that
contradict the benchmark's own "newer serial wins" rule, so the ceiling is 98 rather than
100. Treat it as parity.

The multi-hop row requires `--expansion-hops 3`. **The shipped conversational default is
`expansion_hops = 0` and scores 0.060 SubEM on that corpus** — worse than no gate at all —
because a multi-hop answer shares no entity with its question and so cannot clear a
similarity floor. That is the design working as specified, and it is why the hop count
belongs with the number every time (`RESULTS.md` §8, §25.3).

Multi-hop is where the margin is real, and the mechanism is different: a deterministic
value→subject graph walk over live facts, with no LLM and no embeddings, which has no
counterpart in the published recipe.

SubEM is a generous metric and rewards a hedging reader — `RESULTS.md` §3 shows a case
where "association football and basketball" scores 1.0 against gold `basketball`. That cuts
both ways in the table above, which is why exact-match is listed beside it.

The other architectural difference: consolidation here happens at **write** time, so the
store is always in a resolved state, versus assembling over retrieved candidates at read
time. That is what makes the read path free of any freshness reasoning.

Older comparisons you may see quoted — HippoRAG-v2 at 54%, Zep/Graphiti at 7% — are the
pre-2606.01435 field and are **no longer the state of the art.** `RESULTS.md` predates that
finding in places; §0 records the correction.

## What is actually novel here

- **A deterministic multi-hop chain walk** (`memore/chain.py`). Exact token containment of
  one fact's value in another fact's subject key, walked after the relevance gate. Takes
  multi-hop from 0.200 to 0.760. It is nearly free precisely *because* consolidation keeps
  the graph sparse — superseded facts are what would otherwise make hub entities explode
  the frontier.
- **Wrong-subject admission control** (`memore/subjects.py`). A similarity gate keeps
  *off-topic* memory out (~3% false opens) but cannot keep out memory about *someone else* —
  right relation, wrong entity — which cleared the calibrated floor 75.8% of the time. No
  threshold fixes it; the distributions overlap by construction. The separating signal is
  whether the query **names** the subject, decided from the session's own vocabulary with no
  LLM and no embedding. False opens fall to 0.253, and useful-recall *rises*.
- **DF-gated subject aliasing** (`memore/aliases.py`). Merges two namings of one subject
  when they differ only by generic relation words, decided by document frequency across
  subjects. Fixes the under-merges above without the over-merges that plain subset-merging
  causes. sh_32k 0.940 → 0.960, zero over-merges.
- **Negative results, recorded rather than buried — and this is the contribution.**
  Embedding-based duplicate detection silently *loses facts* and is off by default. A scalar
  relevance floor cannot separate "has this kind of fact" from "has this fact": a
  conversational positive and a wrong-subject hard negative sit 0.048 of median cosine apart,
  with the floor between them. Hybrid fusion must be multiplicative. The relevance floor and
  the embedder are one decision, not two. A cache that passes the benchmark can be useless in
  the regime the system is for. Four separate findings are the same lesson from four
  directions: **a fixture that cannot express a failure cannot be evidence about it.** All in
  `RESULTS.md`, with the runs behind them, and collected as a
  [do-not-re-litigate list](CLAUDE.md#do-not-re-litigate--the-dead-ends-and-the-arithmetic-that-closed-them).

## Honest limitations

- **One benchmark, one task family, two of its four splits.** 64k and 262k are unmeasured.
- **The benchmark's gold answer is arrival order, and the freshness primitive is arrival
  order.** On this task "newest wins" is the stated scoring rule, so a deterministic
  implementation of it is close to implementing the metric. The interesting finding is that
  systems *told* the rule still fail it — not that this one succeeds.
- **The comparison arm did not run.** Graphiti-delegated consolidation stored 0 edges in
  762s for 20 facts under local-model constraints. That is a documented negative about
  viability under *these* constraints, not a measurement of Graphiti's quality.
- **Different reader from published work**, uncontrolled.
- **Write-lane identity is stochastic under paraphrase, and this is the number.** Subject
  0.583, attribute 0.417, arity 0.833 unanimous across four wordings of one assertion, on a
  committed fixture, three identical runs (`RESULTS.md` §26). The failure mode is a *split*
  subject — both facts survive and recall can still reach them — not a lost one: cardinality
  agreement is 12/12. Two prompt-level attempts to close this class of error are on the
  record, both fixed the target and cost elsewhere, and a third is explicitly ruled out.
- **PoC scope.** No async job machinery, no cross-session recall, no queryable audit log,
  no rolling-summary key synthesis. Two further items are specified and deliberately not
  shipped — a k-sample vote over P1's fields and a cross-encoder reranker — each with its
  reason recorded in `RESULTS.md` §29.2 rather than left on a TODO list.

## Running it

Ollama runs on the host (it has the GPUs); FalkorDB runs in Compose.

```bash
docker compose up -d falkordb
uv sync --extra dev --extra bench

uv run memore demo        # interactive trace of both paths
uv run memore inspect     # what the store actually holds, by session
uv run pytest tests/ -q
```

Models are pinned to how the host serves them — mismatched `num_ctx` makes Ollama reload an
18GB model on every process start, and `keep_alive` is per-request, so omitting it silently
replaces a `Forever` pin with a 5-minute default:

```
gemma4:26b                32768 ctx   # extraction (P1) and the bench reader
mxbai-embed-large:latest    512 ctx   # embeddings; paired with score_floor 0.57
```

`MEMORE_LLM_MODEL` / `MEMORE_LLM_NUM_CTX` / `MEMORE_EMBED_MODEL` align a different host.
Each embedder needs its own graph (`MEMORE_GRAPH`) — the vector index is created at a fixed
width, and `connect()` refuses a graph whose width does not match.

### Getting memories in and out

**In** — every turn typed into `memore demo` runs the write path: P1 extracts durable facts
(transient turns store nothing, by design), P2 consolidates against the existing subject,
P3 commits.

**Out** — `recall()` runs before the model call and injects only when the gate opens. There
is no "fetch memory" tool for a model to call; that is the pattern this design leaves
behind.

**Recall is session-scoped**, which is the most common surprise: the bench writes to
`bench-<source>`, the demo defaults to `demo`, and a query against the wrong session
correctly finds nothing.

```bash
uv run memore inspect --session bench-factconsolidation_sh_6k
uv run memore inspect --session demo --query "what's my deploy setup?"
```

`inspect` groups facts by subject and shows superseded ones alongside live — "supersede,
never delete" is the design claim, so the inspector shows the evidence.

### Reproducing the benchmark

The corpus is not redistributed here; fetch it from the MemoryAgentBench dataset:

```bash
curl -sL -o data/Conflict_Resolution.parquet \
  'https://huggingface.co/datasets/ai-hyz/MemoryAgentBench/resolve/main/data/Conflict_Resolution-00000-of-00001.parquet'

uv run python -m memore.bench.run --source factconsolidation_sh_6k --arm deterministic
uv run python -m memore.bench.oracle_run --source factconsolidation_sh_6k
uv run python -m memore.bench.run --source factconsolidation_mh_6k --expansion-hops 3

# the write lane's own instruments -- no benchmark, no reader, no retrieval
MEMORE_GRAPH=memore_slots uv run python -m memore.bench.slots --runs 3
MEMORE_GRAPH=memore_slots uv run python -m memore.bench.paraphrase --runs 3
```

`oracle_run` is the honest instrument: it scores the consolidation decision directly, with
no retrieval and no reader, so a good number cannot come from the reader guessing.

The bench routes through the real `recall()` by default — gate, subject check and token
budget included. `--raw-topk` is the older path that ranked the store's top-k directly, and
is how any figure from before `RESULTS.md` §25 is reproduced. **Multi-hop needs
`--expansion-hops 3`**: the shipped conversational default is 0, and on that corpus it scores
below no gate at all, for the structural reason in §25.3.

## References

- Hu, Wang & McAuley, [*Evaluating Memory in LLM Agents via Incremental Multi-Turn
  Interactions*](https://arxiv.org/abs/2507.05257) (arXiv:2507.05257) — introduces
  MemoryAgentBench and the FactConsolidation task used here.
- Reddy & Challaram, [arXiv:2606.01435](https://arxiv.org/abs/2606.01435) — v1 *Don't Ask
  the LLM to Track Freshness*; v2 *Reliable Post-Retrieval Assembly for Agent Memory*. Prior
  art for the deterministic freshness primitive, and the current baseline to beat.
  ([companion repo](https://github.com/cvikasreddy/memory-conflict-resolution))
- Du, [*Memory for Autonomous LLM Agents: Mechanisms, Evaluation, and Emerging
  Frontiers*](https://arxiv.org/abs/2603.07670) (arXiv:2603.07670) — survey.
- Zep / Graphiti — bitemporal edges and hybrid retrieval. Used here for storage and
  retrieval only; consolidation is deliberately not delegated to it.
- MemGPT / Letta — the tool-call memory lineage this design moves away from.

Supersede-never-delete with `valid_at` / `invalid_at` is bitemporal modelling, standardised
in SQL:2011 and long predating any of this. None of the temporal machinery is new; the
claim is only about *who decides*, and when.

## License

Apache-2.0. See `LICENSE`.
