# Step-0 spike results — deterministic freshness vs. delegated consolidation

Reproduce with the commands in `CLAUDE.md`. Raw JSON in `data/results/`.

**Status: Step 0 complete.** Both arms have run; the deterministic primitive is measured
at two context lengths. The headline substring metric is inflated — read §3 and §7 before
quoting anything here.

## 0. Correction — the baseline this document argues against is out of date

Everything below was written against a field whose best FactConsolidation single-hop
result was **HippoRAG-v2 at 54%**, with Zep/Graphiti at 7%. That was correct when measured
and is **no longer the state of the art.** Every "against a field best of ~54%" framing in
this document should be read with the following in front of it.

Reddy & Challaram, arXiv:2606.01435 (v1, May 2026, *Don't Ask the LLM to Track Freshness:
A Deterministic Recipe for Memory Conflict Resolution*) proposed LLM candidate-extraction
followed by deterministic `max(serial)` — the same core idea as §2's primitive — and
reported:

| | single-hop | multi-hop |
|---|---|---|
| gpt-4o-mini | 0.780 | 0.302 |
| gpt-4o | **0.948** | **0.515** |

Their §4.5 states the metric: "We use SubEM (substring exact match), the metric used in the
MAB paper." That is the same metric §1 ports in `memore/bench/scoring.py` and reports as
`accuracy`, so the matched comparison is SubEM-to-SubEM — **not** against the exact-match
figures this document generally prefers:

| | memore SubEM | memore exact-match | 2606.01435 SubEM (gpt-4o) |
|---|---|---|---|
| sh_6k | 1.000 | 0.990 | 0.948 |
| sh_32k | 0.980 | 0.950 | 0.948 |
| mh_6k | **0.800** | **0.760** | 0.515 |

Against that baseline, this project's position changes from "beats the field by 40 points"
to something narrower and more defensible:

- **Single-hop is parity, not a win.** 0.980–1.000 SubEM against 0.948 is inside the noise
  of an uncontrolled setup comparison, and §2 already establishes that 2 of the 100 32k
  questions are unwinnable, putting the ceiling at 98.
- **Multi-hop remains a real margin.** 0.800 SubEM against 0.515, by a mechanism with no
  counterpart in their recipe (§8's deterministic chain walk, no LLM, no embeddings).
- **The write-time/read-time split is a genuine architectural difference.** Their recipe
  assembles over retrieved candidates at read time; consolidation here happens at write
  time, so the store is always in a resolved state and the read path does no freshness
  reasoning at all.

The second thing worth recording is that arXiv:2606.01435 was **revised on 2026-08-02** and
retitled *Reliable Post-Retrieval Assembly for Agent Memory: Separating Evidence Extraction
from Policy Execution*, its abstract now attributing the gain primarily to separating
evidence extraction from policy execution "rather than the freshness mechanism alone."

That is the same conclusion §3 reaches independently, from the opposite direction:
consolidation was correct on every subject group in every run at both corpus sizes, and
every residual error was extraction naming one subject two ways. Two separate measurements
agreeing that the bottleneck is subject identity rather than freshness is a stronger result
than either alone — and it is why §9 and §10, both of which attack subject identity, are
the parts of this work with the most left in them.

Nothing measured below is retracted. What changes is the comparison it is set against.

## 1. What was measured

MemoryAgentBench `Conflict_Resolution`, sub-dataset `factconsolidation_sh_6k`:
455 numbered facts where a higher serial number overrides an earlier fact about the same
subject, and 100 questions whose gold answer is the value from the newest matching fact.

The published field results on this task, single-hop, are the thing to beat:

| system | accuracy |
|---|---|
| HippoRAG-v2 | 54% (field best) |
| BM25 | 48% |
| Mem0 / Contriever | 18% |
| Zep/Graphiti | 7% |

"Accuracy" is `substring_exact_match` — confirmed from the benchmark's own README metric
table, and ported verbatim in `memore/bench/scoring.py`.

## 2. Arm (b) — deterministic freshness primitive

Ingest: 455 facts → **303 subject groups**, 152 of them holding more than one fact.

| case | count |
|---|---|
| NEW | 303 |
| CONTRADICTION | 152 |
| DUPLICATE | 0 |
| REFINEMENT | 0 |

303 facts left live — **exactly one per subject group**, no group with a stale fact still
live, and **zero cases of a gold-bearing fact wrongly superseded**. Ingest took ~8s for
the whole corpus with no LLM in the decision.

| metric | value | what it measures |
|---|---|---|
| `substring_exact_match` | **1.000** | the official metric; see the caveat in §3 |
| `exact_match` | **0.990** | stricter: the answer *is* the gold value |
| oracle consolidation accuracy | **0.990** | is the newest fact on the question's subject the one left live — no retrieval, no reader |
| `retrieval_hit` | 0.930 | top-ranked live fact contains the gold string |

The oracle and exact-match agree at 0.990 on the same single question, which is the
number I would defend. That agreement is the useful property: the oracle uses no
retrieval and no reader, so when it tracks end-to-end exact-match the accuracy is
attributable to consolidation rather than to the rest of the pipeline.

### It holds at 5× the corpus — with a caveat about the benchmark itself

`factconsolidation_sh_32k`, 2310 facts, same 100-question format:

| | sh_6k | sh_32k |
|---|---|---|
| facts | 455 | 2310 |
| subject groups | 303 | 1559 |
| groups with >1 fact | 152 | 751 |
| CONTRADICTION resolved | 152 | 751 |
| live facts (must equal groups) | 303 ✓ | 1559 ✓ |
| **oracle consolidation accuracy** | **0.990** | **0.940** |
| gold fact wrongly superseded | 0 | 0 |
| groups left with >1 live fact | 0 | 0 |
| end-to-end `exact_match` | 0.990 | **0.950** |
| end-to-end SubEM | 1.000 | 0.980 |
| `retrieval_hit` | 0.930 | 0.890 |
| ingest time (no LLM in the decision) | 8.4s | 42.3s |

At 32k end-to-end exact-match (0.950) comes in slightly *above* the oracle (0.940). That
is expected rather than contradictory: the oracle scores only the single top live fact,
while the reader sees the whole `k=12` block and can still answer correctly when an
under-merge left both facts live. The practical reading is that retrieval holds up at
1559 live facts — `retrieval_hit` slips 0.930 → 0.890, but not enough to cost accuracy.

Both structural invariants hold exactly at 5× the corpus: one live fact per subject,
nothing stale left live, nothing wrongly superseded.

The residual failures at 32k decompose as:

- **2 of the 6 are unwinnable.** The benchmark's gold contradicts its own stated rule.
  For "What is the capital of Papal States?" the corpus holds `#1291 … is Rome` and
  `#2016 … is Watertown`, and the gold is **Rome** — the *older* fact, when the task
  prompt says explicitly that "the newer fact has larger serial number". Same for "Which
  city did Noel Pemberton Billing work in?" (`#707 London`, `#1929 Washington, D.C.`,
  gold `London`). Any system obeying the stated rule must answer these "wrongly". sh_6k
  has zero such cases. The effective ceiling is 98, not 100.
- **The remaining 4 are under-merges** — P1 naming one subject two ways in a form that
  canonicalization does not catch.

So against the winnable set: **94/98 at 32k**, and 99/100 at 6k.

## 3. The 1.000 is inflated — read this before quoting it

On the failures the reader hedged instead of resolving, e.g.

> Q: Which sport is Tunisia national football team associated with? gold: `basketball`
> answer: `"association football and basketball"`

Substring matching scores that 1.0. Exact match scores it 0.0, correctly. The official
metric is generous in exactly this way, so a SubEM-to-SubEM comparison against the field
is still like-for-like — but the honest single number is **0.99**, not 1.00.

### Where the difficulty actually lives

Every failure, at both corpus sizes, has been an **under-merge** — never a wrong
freshness decision. P1 names one subject two ways, the two namings never collide, both
facts stay live, and the reader sees both. Consolidation itself has been correct on every
group it was given, in every run.

That is the useful result of the spike beyond the headline: the field's hard problem
(deciding which of two conflicting facts is current) is solved by a deterministic ordinal
and stays solved at 5× scale, while the *remaining* difficulty has moved somewhere else
entirely — into naming subjects consistently. Those are different problems with different
fixes.

### What actually fixed it: order-insensitive subject keys

The residual errors were P1 naming one subject two ways, and inspection showed the two
namings are almost always the *same content words in a different order* —
`sport associated with tunisia national football team` vs
`sport tunisia national football team is associated with`. So `normalize_subject` now
drops function words and **sorts the remaining tokens**. Still an exact match, still no
threshold, no embedding, no LLM — just a stronger normalization.

| oracle | before | after |
|---|---|---|
| sh_6k | 0.950 | **0.990** |
| sh_32k | 0.870 | **0.940** |
| over-merges | 0 | 0 |

End-to-end exact-match at 6k moved 0.950 → **0.990** with it. All 55 merges at 32k were
inspected and every one was a word-order or function-word variant of a single subject.

What remains: variants where one naming carries an *extra content word*
(`religion of Karen Armstrong` vs `religion Karen Armstrong is affiliated with`).
Sorting cannot collapse those.

### Subset-merging is the obvious next step, and it is wrong

Merging key A into B when `tokens(A) ⊂ tokens(B)` scores better — sh_32k 94 → 96, which
saturates the winnable ceiling. It is still the wrong change. Most of its 62 merges add a
relation word, but some add an *entity* word and narrow the subject:

```
buddhism founder  ⊆  buddhism founder shingon
```

`#696 Buddhism was founded by Gautama Buddha` … `#2295 Shingon Buddhism was founded by
William Waynflete` are different subjects with different answer chains. Merging them
supersedes a correct fact and loses it forever. It happens not to cost accuracy on these
100 questions, which is exactly why an aggregate number should not decide it — over-merge
is the unrecoverable direction. Pinned by `test_narrower_entity_is_not_the_same_subject`.

A principled version would gate the merge on the extra tokens being *generic relation
words* — high document frequency across subjects — rather than rare entity words. That
needs corpus statistics in the identity decision and has not been built.

**Built now — see §10.** The gated rule reaches the same 96 as ungated subset-merging
while refusing every merge this section objected to, `buddhism founder ⊆ buddhism founder
shingon` included. The paragraph above stands as the reason the *ungated* form stays
rejected; it is no longer a description of unfinished work.

### Feeding known subjects back to P1 is implemented but is NOT in these numbers

`WritePath` now reads the session's existing subject keys out of the store and shows them
to P1, so the extractor can reuse a key instead of coining a synonym. It demonstrably
works on the demo path — `deployment environment` is reused across turns 1 and 3, which
is what makes the §6 trace resolve as a CONTRADICTION rather than two unrelated facts.

**It does not affect the benchmark numbers above.** The bench uses
`KnowledgePoolExtractor`, which batches 25 facts per call, runs the batches concurrently,
and never goes through `WritePath` — no batch sees what any other batch decided. So the
0.95 was achieved *without* subject feedback, and the headroom from applying it at
benchmark scale is unmeasured. Wiring it in means giving the batched extractor a shared,
growing subject vocabulary, which serializes the batches; that trade has not been made.

## 4. Arm (a) — Graphiti-delegated consolidation

It ran, and it did not work. On 20 facts (2 episodes of 10) with `gemma4:12b` behind
graphiti-core's OpenAI-compatible client:

| | |
|---|---|
| ingest time | **762s for 20 facts** — extrapolates to ~4.8 hours for the 455-fact corpus |
| edges stored | **0** |
| accuracy / retrieval_hit | 0.000 |

Two failure modes in the log:

```
Error in generating LLM response: Extra data: line 3 column 7 (char 79)
Target entity not found in nodes for edge relation: PLAYS_POSITION_OF
```

The first is the local model emitting JSON that graphiti's structured-output parser
rejects; the second is edge extraction referencing entities the node pass never created,
so the edges are discarded. Nothing reached the graph, so there was nothing to
contradict and nothing to retrieve.

**What this does and does not license saying.** It is *not* a measurement of Graphiti's
quality — the published 7% comes from their own setup with a frontier model, and this
result would look different with one. What it does establish is narrower and still
decision-relevant: **under this project's stated constraints — no cloud LLM in this path
(`recall-poc-spec.md` §7), local hardware — the delegated design is not viable.** It is
three orders of magnitude slower than the deterministic path (762s for 20 facts vs 8.0s
for 455) and its correctness depends on a local model reliably emitting schema-valid
JSON across many chained calls, which it did not.

That is enough to clear §3's Step-0 gate as a documented negative: the deterministic
primitive is the path forward, and the reason is recorded rather than assumed.

## 5. Things the build surfaced that the spec did not anticipate

### The BM25 arm was dead, and the first numbers were vector-only

The full-text index intersects space-separated terms, so passing a natural question
through verbatim demands that *every* word appear in the fact. It never does.
`_text_arm` returned **zero hits on every query** — measured directly, not inferred.
"Hybrid retrieval" was aspirational: ranking was `0.7 × cosine` and nothing else.

This matters for provenance, not just correctness: the §2 numbers were **first measured
with vector-only retrieval**. Joining query terms with `|` asks for a union and the arm
comes alive. §2's figures are from the re-run after the fix.

Measured in isolation — before the subject-key canonicalization of §3, so these are not
the numbers in §2 — turning the arm on did **not** improve the benchmark:

| sh_6k, text arm only | before | after |
|---|---|---|
| `exact_match` | 0.950 | 0.950 |
| SubEM | 1.000 | 1.000 |
| `retrieval_hit` | 0.940 | 0.900 |
| oracle | 0.950 | 0.950 |

On this corpus every fact of a given relation shares most of its wording ("X is
associated with the sport of Y"), so keyword overlap boosts near-miss facts about the
*wrong* subject. The oracle held still across the change, which is what makes the
attribution clean: it never touches retrieval, so a moved `retrieval_hit` beside an
unchanged oracle isolates the delta to ranking.

(The later canonicalization change then moved `exact_match` 0.950 → 0.990 and
`retrieval_hit` 0.900 → 0.930 — a separate change with a separate cause. §2 carries the
current figures.)

The text arm earns its place on gate precision (below), not on this benchmark.

### Fixing it broke the gate, which forced the fusion to change

With the text arm alive, additive fusion (`0.7·cos + 0.3·bm25_norm`) let an unrelated
query — "what is the weather in Paris?" — score **0.419** against a deploy fact, clearing
the 0.35 floor. The cause is mixing an absolute term with a set-relative one: BM25's
max-normalization hands the best of two terrible matches a full 1.0. Fusion is now
multiplicative, `cos · (1 + w·bm25_norm) / (1 + w)`, which keeps the score anchored on
cosine so `score_floor` retains meaning. Same query now scores 0.170 and the gate stays
shut. Pinned by `test_irrelevant_query_stays_below_the_gate_floor`.

### The latency model is inverted — and the 200ms budget is now met

`recall-stage-spec.md` §1 budgets ~150ms for the store lookup (B) and "low single-digit
milliseconds" for key synthesis (A). Measured, warm:

| component | spec expectation | measured |
|---|---|---|
| A — embedding + blend | a few ms | **295–323 ms** |
| B — hybrid lookup | ~150 ms (dominant) | **2.6–4.3 ms** |

FalkorDB is ~50× faster than budgeted; key synthesis was ~100× slower than assumed and
consumed the entire budget by itself. §14's "A–D P95 ≤ 200ms" was **not met** at
~300–340ms end to end.

**This is now fixed** by changing the embedder, which required calibrating `score_floor`
first (next subsection). With `mxbai-embed-large` at floor 0.48, measured over the
calibration query set through the real `recall()` path:

| fixture | A–D p50 | A–D p95 |
|---|---|---|
| bench fold0 (219 facts) | 84.5 ms | 91.0 ms |
| bench fold1 (236 facts) | 83.8 ms | 90.0 ms |
| chat (12 facts) | 80.6 ms | 95.8 ms |

The §6 demo trace runs at 63–74ms end to end. **§14's A–D P95 ≤ 200ms is met**, with
roughly 2× headroom. The inversion itself stands as a finding for the production spec:
the budget is spent almost entirely in the embedder, and the store is nearly free.

**It is not HTTP overhead, and it is not inference.** Measured against `/api/embed`,
which reports its own timings: transport costs **1.3–1.5 ms**, and Ollama reports
`load_duration ≈ 250 ms` on *every* `embeddinggemma` call — back to back, on a warm
connection, with `keep_alive` set explicitly. `/api/ps` shows the model at 0 GB VRAM
between calls. Other embedding models on the same daemon do not do this:

| model | warm call | of which "load" | dim |
|---|---|---|---|
| nomic-embed-text | 55–75 ms | 42–46 ms | 768 |
| mxbai-embed-large | 66–71 ms | 46–52 ms | 1024 |
| embeddinggemma | 300–360 ms | 244–266 ms | 768 |

So the fix is not an in-process embedder and not a torch dependency — it is a different
model, worth ~5×. That is what the switch below does.

**Why the switch needed a calibration first.** Swapping the model moves the whole score
distribution, and `score_floor` is calibrated to it. An early two-query probe suggested
the faster models were comparable discriminators that simply compressed similarities
upward, landing an unrelated query at ~0.35 — exactly the default floor. Switching on
that basis would have silently opened the gate on irrelevant turns, and the gate is the
design's stated differentiator (§6). **`score_floor` being embedder-dependent is a
finding for the production spec, which treats 0.35 as a constant.**

### Calibrating the floor over a real query set

`memore/bench/calibrate.py` replaces those two ad-hoc queries. It scores every query
through the **actual `recall()` path** with the floor dropped to 0.0, so it yields the
top-1 fused score the gate would have tested *and* the true A–D latency, on the code that
will use the result. Two regimes, kept separate throughout — pooling them lets the
larger corpus outvote the one the gateway actually runs against:

- **bench** — FactConsolidation sh_6k, its ~300 subject groups split in half by a stable
  hash into two folds, each ingested into its own session. A subject lands wholly in one fold,
  so its update chain is never cut. Each of the 100 questions is then a positive against
  the fold holding its subject and a **hard negative** against the other — same corpus,
  same relation templates, subject simply absent. 99/100 questions mapped (1 dropped, no
  subject whose newest fact carries the gold answer).
- **chat** — an authored 12-fact conversational session with 25 positives. Authored, and
  labelled as such: there is no public corpus of gateway turns against a personal memory
  store.

39 off-domain negatives run against every session in both regimes. One was reclassified
after the first run and the numbers here are from the re-run: "run the tests again"
against a store holding "runs tests with pytest" was labelled a negative, and was the
single largest false-open for all three models. That was a mislabel — `recall-poc-spec.md`
§6 uses that turn to show the *write* path storing nothing, which says nothing about the
read path, and surfacing the test runner there is a correct recall. Corrected rather than
left in to flatter the floor.

Operating point is chosen by constraint, not by Youden's J: **the lowest floor holding off-domain false
opens under 5% in every regime.** A false open puts a wrong memory into a prompt the user
never asked memory for; a false shut leaves the turn as it is today. That asymmetry is
the same one that makes `use_embedding_comparison` default False (§6).

Each variant at **its own** calibrated floor — comparing a calibrated floor against an
uncalibrated one would prove nothing:

| model | floor (band) | bench useful-recall | chat recall | `retrieval_hit` | A–D p95 |
|---|---|---|---|---|---|
| embeddinggemma | 0.47 (0.47–0.50) | 0.990 | 0.880 | 0.930 | 365 ms |
| **mxbai-embed-large** | **0.48 (0.48–0.51)** | **0.970** | **0.920** | **0.890** | **96 ms** |
| nomic-embed-text | 0.53 (0.53–0.54) | 0.869 | 0.840 | 0.760 | 92 ms |
| nomic + its prefixes | 0.61 (0.61) | 0.818 | 0.600 | — | 89 ms |

`mxbai-embed-large` is now the default at floor 0.48: ~4× faster than embeddinggemma at
essentially the same discrimination. Bands are quoted because 25 chat positives move TPR
in 4% steps — a two-decimal floor overstates the resolution the data has.

Three things this corrected:

**0.35 was wrong for every model, including the one it was written for.** At 0.35,
embeddinggemma opens the gate on **20.5%** of off-domain chat turns (mxbai 53.8%, nomic
46.2%), and on **100%** of bench hard negatives for all four. The earlier framing — 0.35
fine for embeddinggemma, risky only for the faster models — was an artifact of testing one
irrelevant query. The floor was never calibrated for anything.

**`nomic-embed-text` was the wrong candidate for the latency win.** It gets the speed but
is a measurably worse ranker: `retrieval_hit` 0.93 → 0.76 on the benchmark, useful-recall
0.869. `mxbai-embed-large` gets the same speed at 0.890 / 0.970. Note `exact_match`
(0.990) and SubEM (1.000) are identical across all three — end-to-end accuracy hides the
ranking difference, because k=12 usually puts the right fact *somewhere* in the block even
when it is not top-1. Only `retrieval_hit` and useful-recall see it.

**nomic's asymmetric prefixes hurt, confirmed at scale.** The earlier note that they made
discrimination slightly worse over two queries was flagged as worth re-checking rather
than assuming. Re-checked over 300+ queries: they force the floor to 0.61 and drop chat
recall to 0.600. No shipped default uses them.

### A scalar floor cannot separate "has this kind of fact" from "has this fact"

The most interesting result is the one the floor cannot fix. At each model's calibrated
floor the **hard**-negative false-open rate is **0.68–0.88** — against 0.00–0.04 for
off-domain negatives at that same floor — and it only reaches even that by giving up
recall; at 0.35 it is **1.000** for all four. The distributions overlap by construction:
for mxbai, bench positives sit at p05 0.621 and hard negatives at p95 0.675.

A hard negative here is "Which sport is X associated with?" against a store full of
sport facts about *other* subjects. Cosine similarity measures topical fit, and the store
genuinely is topically about that — so the gate opens and injects a confidently wrong
fact. This is not a tuning problem and no threshold fixes it: separating them needs a
signal the score does not carry, such as checking the query's subject against the
`subject_key`s actually held (the write path already computes exactly that key).

Worth flagging for the production spec on the same footing as the embedder-dependence
finding, and it bounds what §6's gate can promise: it reliably keeps *off-topic* memory
out of the prompt, not *wrong-subject* memory.

### An embedder change could silently kill recall, and now cannot

Found while switching models. FalkorDB's vector index is created at a fixed dimension.
Pointing a 1024-dim embedder at a graph indexed for 768 raises "already indexed" on index
creation — which `_ensure_indexes` correctly swallows — after which `add_fact` writes
wrong-width vectors without complaint and only the *query* raises. `recall()` catches
every store exception by design (§3.1: a store failure must never fail the turn), so the
whole thing degrades to **a gate shut forever**, one WARNING per turn, memory quietly
dead.

It was one step away: `MEMORE_EMBED_MODEL` is the documented way to change embedder and
the graph name does not change with it. `FalkorStore.connect()` now compares the existing
index dimension against its own and raises with the remedy; pinned by
`test_embedder_dimension_mismatch_fails_at_connect`. Same shape as the session-scoping
bug below — a store that returns nothing looks exactly like a store with nothing in it.

## 6. Two things about consolidation itself

### Embedding-based duplicate detection loses facts — it is now off by default

`ConsolidationConfig.duplicate_similarity` was 0.97 with embedding comparison **on**.
Measured against the real embedder (`embeddinggemma`):

| pair | cosine |
|---|---|
| identical strings | 1.000 |
| "Catholic bishop … Catholic Church" → "… Catholicism" (**real change**) | 0.986 |
| "flanker … rugby union" → "… rugby" (**real change**) | 0.982 |
| "Hines Ward plays … wide receiver" vs "… cornerback" (contradiction) | 0.911 |
| "capital of Germany is Berlin" vs "… is Bonn" (contradiction) | 0.902 |
| "deploys to staging by default" vs "the default deploy target is staging" (paraphrase) | 0.877 |
| "deploys to staging by default" vs "deploys to prod by default" (contradiction) | 0.849 |

An earlier draft of this document claimed 0.97 was "safe in the direction that matters —
no real contradiction is ever swallowed as a duplicate." **That was wrong**, and sh_32k
falsified it: the top two rows are genuine value updates that scored *above* the
threshold, were classified DUPLICATE, and were discarded. The store kept the stale value
and answered wrongly. It held at 6k only because no such near-identical pair occurred
there — a sample-size artifact I generalized from.

The two error directions are not symmetric, which is what makes this a design rule rather
than a tuning problem:

- a false **DUPLICATE** discards the update — the store is permanently wrong;
- a false **CONTRADICTION** keeps both facts with the newer live — the answer is still
  right, at the cost of store size and a `SUPERSEDED` line.

And no threshold separates the cases anyway: real updates land at 0.98+ while genuine
paraphrases sit at 0.877, *below* real contradictions at 0.849–0.911. A single sentence
embedding cannot see that only the value changed.

`use_embedding_comparison` therefore defaults to **False**. Exact normalized-string
equality catches the duplicates that matter and cannot lose a fact. Effect: sh_32k oracle
0.850 → **0.870**, over-merges 2 → 0, with sh_6k unchanged at 0.950. This is
`recall-poc-spec.md` §4.3's escalation point, and the finding is that the escalation is
needed for *duplicate* detection specifically — contradiction detection is fine without
it.

### Session scoping degrades silently on a shared vector index

FalkorDB's vector index is global per (label, property) and its query procedure takes no
filter, so `session_id` can only be applied *after* the ANN fetch. With a fixed
over-fetch, a session holding one fact silently returns nothing once other sessions crowd
the index — observed directly: 914 facts in the graph, demo session holding 1, global
top-48 never reaching it. `_vector_arm` widens the fetch until it has enough in-session
hits, and `test_session_scoping_survives_a_crowded_index` pins it.

**The first fix was incomplete, and the calibration run is what exposed it.** The
widening was bounded by the graph's own fact count — stop once `fetch` covers every node
there is. That bound is wrong, because **HNSW returns fewer nodes than it is asked for**.
Measured on the 467-fact calibration graph: ask the index for 467, get **182** back, of
which **zero** belonged to the 12-fact session being queried; ask for 2000 and all 12
arrive. "I asked for the whole graph" is not "I saw the whole graph", so the bound
reintroduced the exact silent-empty-result it was added to prevent.

It surfaced because the harness records `n_hits`: 5 of 1360 observations returned no hits
at all, at normal latency rather than the timeout. All 5 were off-domain negatives in the
two `nomic` variants, where a spurious empty result is *indistinguishable from the gate
correctly staying shut* — which is what makes this class of bug worth instrumenting for
rather than eyeballing.

The whole calibration was re-run on the fixed code and the figures above are from that
run; zero-hit observations went 5 → **0**. The shipped floor did not move (mxbai 0.48,
unchanged, with identical recall and FPR), but `embeddinggemma`'s did, 0.45 → 0.47 — so
the artifact was not cosmetic, and quoting the pre-fix numbers would have been quoting a
bug. `print_report` now flags any zero-hit observation, because a retrieval failure that
reads as good precision is exactly the thing this harness must not launder.

`_vector_arm` is now bounded by `_MAX_VECTOR_FETCH` rather than by the graph size, and
stops on the *session's* fact count rather than on `k`, so a session holding 3 facts does
not widen forever chasing a 12th hit that does not exist.
`test_widening_is_not_bounded_by_the_graph_size` pins it.

Worth flagging for the production design: with genuinely many sessions this wants a
filtered ANN or a per-session index, not a widening scan. The scan is a PoC-scale
workaround whose cost grows with total corpus size while the useful result stays
session-sized.

## 7. What this does not yet show

- **Arm (a) never ingested a full corpus** (§4). The gate is cleared on
  "not viable under our constraints", not on a like-for-like quality comparison.
- **`sh_6k` and `sh_32k` only.** 64k and 262k (4580 / 18332 facts) are unmeasured; 262k
  alone is ~12 hours of extraction at the observed rate. 32k already answers whether 6k
  was a lucky artifact, so the marginal value of the longer two is low until the P1
  canonicalization bottleneck is addressed.
- **Only the oracle ran at 32k.** The end-to-end reader numbers (SubEM / exact-match) are
  from 6k. `retrieval_hit` will fall at 32k — 1614 live facts is far more retrieval
  pressure — and that is a separate claim from the consolidation result.
- **Different reader from the field.** These runs answer with a local `gemma4:12b`; the
  published numbers use GPT-4o / GPT-4.1-mini. This cuts against us if anything, but it
  is not a controlled comparison.
- **Retrieval differs between the arms and cannot be made identical.** Graphiti's ingest
  *is* its edge-invalidation, so it cannot be run against our FalkorStore. Any arm-vs-arm
  delta compares whole designs, not the resolution rule in isolation. Both arms do
  consume an identical cached `CandidateFact` list, so extraction quality cancels out.
- ~~**Multi-hop untested.**~~ Now measured -- see §8.
- **The bench does not exercise `recall()`.** `run_deterministic` calls
  `store.hybrid_search` and `build_block` directly, so `score_floor`, the lookup timeout
  and the failure-safety path play no part in these numbers. That is deliberate — the
  spike measures store + consolidation, not the recall stage — but it means the gate is
  covered only by unit tests against fakes, and by the §6 demo trace.

## 8. Multi-hop — the deterministic chain walk

`factconsolidation_mh_6k` is the split where every published system scores single digits.
It ships the **identical 455-fact corpus** as sh_6k (verified byte-for-byte) with 100
harder questions, so store and consolidation are already known-correct on it and the
entire delta is retrieval and composition. The same cached subject extraction is reused
for exactly that reason.

### Why single-shot retrieval cannot work here, structurally

"In which location did the spouse of Igor of Kiev pass away?" is answered by *"Olga of
Kiev died in the city of Rodez."* — a fact sharing **no entity with the question**. "Olga
of Kiev" only exists once hop 1 resolves. Measured: the top-ranked live hit carried the
gold on **0 of 100** questions. This is not a ranking bug to tune; §5 already established
that cosine is specifically blind to entity identity.

And every hop is a *consolidation* decision:

```
The author of Our Mutual Friend is Charles Dickens.   ← superseded
The author of Our Mutual Friend is Charles Darwin.    ← live
Charles Darwin is married to Amala Paul.
Amala Paul is a citizen of India.                     ← superseded
Amala Paul is a citizen of Belgium.                   ← live, and the gold
```

Take the stale fact at hop 1 and the chain dead-ends at Dickens. So multi-hop
FactConsolidation is precisely where a freshness primitive should pay off twice.

### The walk, and why it is nearly free

`memore/chain.py`: an edge exists when one fact's **value** names another fact's
**subject**, decided by exact token containment over the *same* `normalize_subject` key
that decides subject identity for consolidation. No LLM (§13 forbids one in A–D), no
embeddings, no threshold. Nothing is stored — a fact already carries its text and its
`subject_key`, so the value is derived at read time. No schema change, no type change, no
P1 change.

Measured **before** building it, on the fact strings alone:

| | depth 0 | ≤1 hop | ≤2 hops | ≤3 hops |
|---|---|---|---|---|
| clean terminus (n=63) | 33% | 70% | 90% | **97%** |
| ambiguous (n=37) | 38% | 86% | 92% | 95% |
| all | 35% | 76% | 91% | **96%** |

Branching factor: **mean 1.0, p95 3, max 4**. That is the surprise, and it is
consolidation doing a second job: walking **live facts only** collapses 455 facts to 303
before the walk starts, so a hub value like "Italy" points at the handful of subjects
*currently* asserting something about Italy rather than at every historical assertion.
Superseded facts are exactly what would make the frontier explode.

### Result: the gate is catastrophic alone, and the walk fixes it

Three arms, differing only in retrieval (identical store, identical reader):

| arm | facts injected | `retrieval_any` | clean (n=63) |
|---|---|---|---|
| A — raw top-k, no gate | 12 | 0.400 | 0.349 |
| B — via `recall()`, gate only | ~2.9 | **0.050** | 0.048 |
| C — gate + 3-hop walk | ~5.4 | **0.910** | 0.889 |

The gate *destroys* multi-hop recall on its own — 0.400 → 0.050 — for the structural
reason above. The walk takes it to 0.910 while injecting **fewer facts than ungated
top-12**. End to end, with MemoryAgentBench's own reader and scorer:

| mh_6k | SubEM | exact-match |
|---|---|---|
| parametric floor (no context at all) | **0.000** | **0.000** |
| baseline (raw top-k, no walk) | 0.220 | 0.200 |
| **gate + 3-hop walk** | **0.800** | **0.760** |

The parametric control matters: with no recalled block the reader scores **zero**,
because the corpus deliberately corrupts real-world facts, so world knowledge actively
produces wrong answers. Nothing here is the model knowing the answer; the whole 0.760 is
attributable to the memory system.

`retrieval_any` is reported LIVE-only. Counting any hit would hand free credit to a
superseded fact carrying the same value — on this benchmark, the one contamination that
would invalidate the result.

### What it costs, and what still bounds it

- **Latency** +46ms: A–D p95 94 → **146ms**, still inside §14's 200ms but at ~1.4×
  headroom rather than 2×. ~22ms of that is identifiable (14ms session-wide query, 8ms
  adjacency rebuilt per turn); both disappear under materialized edges, which is the
  production shape flagged in `live_chain_view`.
- **The calibrated floor costs ~5pp of multi-hop recall.** At floor 0.48 the walk reaches
  91/100; at 0.30 it reaches **96/100 — exactly the offline ceiling**. The gate closes on
  only 3/100 turns, so this is a ranking cost, not a gating one. The floor is *not*
  loosened for it: 0.48 is what conversational precision requires (§5), and multi-hop
  recall is not worth reopening the gate on 54% of off-domain chat turns.
- **The bottleneck has moved to the reader.** The store delivers the chain 91% of the
  time; the reader converts it to a correct answer 80% (SubEM) / 76% (EM). Inspected
  failures are hedges ("the knowledge pool does not contain…") and format misses
  ("Japan, Shinzō Abe" against gold "Shinzō Abe"), not missing facts. The reader prompt is
  MemoryAgentBench's own `factconsolidation` template, written for single-fact lookup with
  `max_tokens=32`; it was never designed for 3-hop composition. Left untouched, because
  tuning it would break comparability with every single-hop number here.

### Why `expansion_hops` still defaults to 0

Measured in all three regimes before deciding:

| regime | effect of the walk |
|---|---|
| multi-hop | 0.200 → 0.760 exact-match |
| single-hop sh_6k | **no change** — SubEM 1.000, EM 1.000, `retrieval_hit` 0.890 → 0.890 |
| conversational (calibration chat fixture) | **complete no-op** — 0 chain facts added, gate decisions identical, +11ms |

`retrieval_hit` holding at exactly 0.890 on sh_6k is the check that mattered: chain facts
rank below the seeds that found them, so top-1 must not move, and it does not.

The walk is therefore safe everywhere and transformative on one workload — but in the
regime this PoC actually targets (a chat gateway, `recall-poc-spec.md` §2) it is a
measured no-op that costs latency, because a personal memory store has few entity chains.
Flipping a default needs positive evidence *in the target regime*, not merely absence of
harm elsewhere, so the shipped default stays `expansion_hops = 0` and multi-hop workloads
opt in. Same discipline `score_floor` got in §5.

## 9. Wrong-subject recall — the gate's other half

§5 closed on a limit rather than a fix: the gate reliably keeps *off-topic* memory out
(~3% false opens) but not *wrong-subject* memory — right relation, wrong entity — which
cleared the calibrated floor **75.8%** of the time on the benchmark. No threshold catches
it, because the distributions overlap by construction (bench positives p05 0.621, hard
negatives p95 0.675). Cosine measures topical fit and the store genuinely *is* topically
about that.

The separating signal is not similarity at all: it is whether the query **names** the
subject. `memore/subjects.py` decides that deterministically — no LLM, no embedding, no
threshold on a score — from the session's own subject-key vocabulary, where an entity is
simply a low-document-frequency token (a relation word like "sport" or "capital" appears
in many subjects; an entity in one or two).

### The naive rule had to be measured, because it fails

Requiring the query to share a token with the subject blocks **44% of conversational
positives**. A chat subject is an abstract label and the query is a paraphrase: "what city
am I in?" against subject `location` shares nothing. That would have answered the
hard-negative problem by destroying the case the gate exists to serve — and it is only
visible if you measure the *conversational* regime separately, which is the same reason
§5 keeps the regimes apart.

What makes it safe is the **competitor precondition**. The failure needs a crowded
relation: many subjects differing only by entity. A subject with no competitors is never
policed, so paraphrase survives untouched.

### It is a veto, never a promotion

The check runs *after* the floor test and can only remove hits the score already
admitted, so `score_floor` keeps exactly the meaning §5 calibrated. A failure to load the
subject vocabulary degrades to *no check*, not to a closed gate — this is a precision
refinement and losing it must not cost recall (§3.1).

### Measured, all three regimes, floor 0.48

A third fixture was added for this: `chat_crowded`, a conversational session whose
subjects compete (two services with deploy targets, two teams with leads, two pets with
breeds) and which carries genuine hard negatives — "where does the mobile app deploy?"
against a store holding the web app and the api. The original `chat` fixture could not
measure this at all: one fact per subject means the failure cannot occur there.

| regime | metric | off | on |
|---|---|---|---|
| bench | hard-negative FPR | 0.758 | **0.253** |
| bench | off-domain FPR | 0.038 | **0.000** |
| bench | useful-recall | 0.970 | **0.990** |
| bench | TPR | 1.000 | 1.000 |
| **chat_crowded** | **hard-negative FPR** | **0.846** | **0.462** |
| chat_crowded | TPR / useful-recall | 1.000 / 1.000 | 1.000 / 1.000 |
| chat | TPR / off-domain FPR | 0.920 / 0.026 | 0.920 / 0.026 |

Two things worth reading twice. **The failure is real in conversation, not just on the
benchmark** — 84.6% of wrong-entity chat queries opened the gate before this. And
**useful-recall went up**: refusing a wrong-subject top hit promotes the correct fact
underneath it. Both bench positives the rule "cost" were inspected and in each the top hit
was already the wrong subject ("Which continent is India located in?" matching a fact
about Hyderabad), so it rejected a wrong fact rather than a right one.

`subject_check` therefore ships **on** — a strict improvement in every measured regime at
no cost to recall, unlike `expansion_hops`, which is a no-op in the target regime. Cost is
~7ms and one extra store query per gate-open turn; A–D p95 stays ~94ms against the 200ms
budget.

### What it does not fix, and the knob not to turn casually

0.462 is better than 0.846 but it is not a solution. The residual cause is structural:
`subject_min_competitors = 2` requires a genuine crowd, and most conversational relations
hold exactly **two** subjects (two pets, two machines, two teams), giving one competitor
each — below the threshold, so unpoliced.

Setting it to `1` is the obvious response and it is a real trade, not a free win:

| `subject_min_competitors` | crowded-chat wrong-entity blocked | conversational recall |
|---|---|---|
| 2 (default) | partial | **no loss** |
| 1 | 69% fully blocked | **−8%** |

The 8% are genuine paraphrase losses, inspected: "what kind of milk do I use?" against
subject `coffee preference` shares no token, so a lexical check cannot confirm what
similarity correctly found. That is the irreducible tension — a query that names nothing
can only be matched semantically, and a rule that demands naming will refuse it. Which
side to take is an operator decision, which is why it is a config knob and why `2` is the
default: it is the setting with no measured downside.

Both remaining directions need something this rule does not have: a signal that "milk" and
"coffee preference" refer to the same subject while "mobile app" and "web app" do not.
That is a subject-identity question, not a retrieval one — the same question §3's
under-merge analysis runs into from the other side.

## 10. Subject aliasing — the gated subset merge §3 left open

§3 ends on a refusal with a recipe attached: subset-merging subject keys scores better
(sh_32k 94 → 96) and is still wrong, because some extra tokens are entities rather than
relation words, and merging those supersedes a correct fact permanently. The recipe was
"gate the merge on the extra tokens being generic relation words — high document frequency
across subjects". `memore/aliases.py` is that gate.

The result is the one worth having: **the same 96, without any of the merges §3 objected
to.** The score §3 declined to buy was available at a price it was willing to pay.

| oracle | alias off | alias on (0.015) | alias on (0.008) |
|---|---|---|---|
| sh_6k | 0.990 | 0.990 | 0.990 |
| sh_32k | **0.940** | **0.960** | **0.960** |
| over-merges, both corpora | 0 | **0** | **0** |
| groups left with >1 live fact | 0 | 0 | 0 |
| merges made (6k / 32k) | 0 / 0 | 2 / 36 | 3 / 39 |

6k has no headroom — it was already at the 0.990 ceiling §3 describes, where the one
remaining failure is a reader hedge, not a consolidation error. The 32k corpus is where
under-merges lived and where the rule pays.

**6k is not a second independent validation, and should not be quoted as one.** With 303
subjects against a 200-subject activation floor, the rule is inert for the first 243 of its
455 facts and live for only 47% of the ingest — against 91% at 32k. What 6k establishes is
the ratio's *portability* (`affiliated` reads as a relation word at both scales, which is
the claim an absolute threshold fails) and that the rule costs nothing where there is
nothing to gain. The threshold itself rests on 32k.

### The four remaining 32k failures contain no consolidation error

§2 decomposed the six baseline failures as 2 unwinnable + 4 under-merges. The rule removes
exactly the two it targets — `Morris Iemma` and `Maria Amalia of Naples and Sicily`, both
`religion of X` vs `religion X is affiliated with`. What is left is more interesting than
the score:

- **2 unwinnable** (§2): `Papal States` and `Noel Pemberton Billing`, where the benchmark's
  gold is the *older* fact and contradicts the rule its own prompt states.
- **2 oracle-matcher artifacts**, not store errors. For "Which city is the headquarter of
  Microsoft located in?" the store is *correct* — group `headquarters microsoft` holds
  `#343 … Redmond` and `#656 … Beverly Hills`, and #656 (the gold) is live. The oracle's
  token-F1 matcher assigned the question to `city headquarters ilyushin located` instead,
  because `headquarter` and `headquarters` are different tokens, so the question's only
  real signal (`microsoft`) is outscored by two incidental matches (`city`, `located`).
  Same for MTV.

So at 32k the oracle's own question-to-group matcher is now a larger error source than
consolidation, and **0.960 understates the consolidation result** rather than flattering
it. The matcher is deliberately left alone: stemming it would raise the reported number
by changing the measuring instrument, which would break comparability with every figure
recorded above. Recorded here instead, which is the honest form of the same information.

### The threshold has to be relative, and that was measured rather than assumed

The obvious form of the rule — "generic means df ≥ N" — is wrong, and the two corpora
refute it outright. `affiliated` is the same relation word in both:

```
sh_6k    df  9 / 303 subjects  = 0.0297
sh_32k   df 60 / 1559 subjects = 0.0385
```

An absolute cutoff tuned at 32k is inert at 6k; one tuned at 6k merges `rugby union` into
`rugby` at 32k. The *ratio* is stable across the 5× scale change where the count is not, so
the rule reads `df / n_subjects`. This is the same trap as `score_floor` across embedders
(§5): a threshold means nothing without the thing it was calibrated against.

### Where 0.015 came from — all 68 subset pairs, hand-labelled

Both corpora were enumerated exhaustively (64 pairs at 32k, 4 at 6k — small enough to read
every one) and each labelled by whether the two keys name the same subject. Sorted by the
minimum extra-token ratio, the labels separate cleanly:

```
must NOT merge   kingdom   0.0071   Kingdom of England is not England
                 ireland   0.0045   UK of Great Britain and Ireland is not Great Britain
                 union     0.0038   rugby union is not rugby
                 rock      0.0026   rock music is not country music
                 hip/hop   0.0013   hip hop is not country music
                 shingon   0.0006   Shingon Buddhism is not Buddhism
------------------------------ decision boundary ------------------------------
genuine merge    written   0.0083   |  location 0.0083 / 0.0099  |  famous  0.0186
                 current   0.0244   |  located  0.0334           |  city    0.0673
                 affiliated 0.0297 / 0.0385
```

Two bands separate the labels. `(0.0071, 0.0083]` captures everything but leaves a 1.17×
margin over the worst unsafe case; `(0.0083, 0.0186]` gives up `location` and `written` for
a **2.1× margin**. A 17% margin derived from two corpora is not a margin, and the error
asymmetry already recorded in `ConsolidationConfig` — a false merge discards a correct fact
forever, a missed merge leaves the status quo — makes this not a close call.

The measurement then made it free: **0.015 and 0.008 score identically** (0.960 / 0.990).
The three extra merges 0.008 buys change no answer. That is the strongest available
argument for the conservative setting, and it is why it ships as the default.

`Kingdom of England` is the case that decides the boundary, so it is worth seeing:

```
#370  England is located in the continent of Europe.
#1850 England is located in the continent of South America.   <- live, correct
#2207 Kingdom of England is located in the continent of Europe.
```

Merging on `kingdom` (0.0071 — higher than several tokens a naive threshold would trust)
would supersede #2207 with #1850 and lose an answer. It is refused, and pinned by
`test_boundary_entity_that_a_naive_threshold_would_have_merged`.

### Every merge, inspected — which is the acceptance criterion, not the score

§3 rejected the ungated rule *despite* a better score, because an aggregate cannot tell a
merge that fixed an under-merge from one that destroyed a fact. So the log is part of the
output (`SubjectVocabulary.merges`, printed in full by `oracle_run`), and all 36 merges at
32k were read:

```
affiliated   31    religion of X            <-> religion X is affiliated with
current       2    head of state of X       <-> current head of state of X
city located  2    headquarters of X        <-> city where X is headquartered
famous        1    work of X                <-> famous work of X
```

Every one is a relation-word variant of a single subject. Zero narrow an entity. The two
directions both appear (`affiliated X` → `X` and `X` → `affiliated X`) because the surviving
key is whichever arrived first — group *membership* is what consolidation acts on, and that
is order-independent.

### What it cannot do, so nobody tunes toward it

Two genuine relation words at 32k have df=1: `employed` (`employer luther martin` vs
`employed employer luther martin`) and `married` (`elvis presley spouse` vs `elvis married
presley spouse`). Both are real merges this rule refuses, and no threshold recovers them —
lowering the bar far enough to catch a df=1 relation word first sweeps in `shingon`,
`gaelic` and `nazi`, which sit at exactly the same frequency. A rule reading surface
statistics cannot recognise a relation word that appears once.

That is the same wall §9 hits from the other side (`milk` vs `coffee preference`): telling
whether two namings denote one subject is an identity question, not a frequency one. Both
residuals now point at the same missing primitive, which is a more useful place to be than
two unrelated open problems. Pinned as a known miss by
`test_rare_relation_word_is_a_known_miss_not_a_merge`.

### Two things that are easy to get wrong here

**The rule is inert below `min_subjects` (200), and that is load-bearing.** A relative
threshold degrades *worse* than an absolute one at small scale: in a 12-subject personal
session, `location` appearing in two subjects reads as ratio 0.17 — eleven times the
threshold — so an unguarded relative rule fires hardest exactly where its evidence is
weakest. The measured corpora are 303 and 1559 subjects; nothing smaller has been
validated, so nothing smaller is policed. Conversational stores are unaffected by this
change, which is also why it is safe to ship on.

**Document frequency is read in arrival order, not from the finished corpus.** The
consolidator sees subjects one at a time, so `affiliated` is not yet a relation word on its
first appearance; the rule is conservative when cold and converges as the session fills.
`oracle.build_groups` replays the same growing vocabulary for this reason: grouping the
completed corpus in one pass would credit the store with merges it never made and score a
system that was never run.

The size of that effect was measured rather than assumed, by diffing the finished-corpus
prediction against the merges ingest actually made. At 32k, **37 pairs are eligible at
final DF and ingest merged 36**. The single miss is exactly the case the story predicts:

```
innocent religion xi              first seen at serial 80
affiliated innocent religion xi   first seen at serial 197
rule activates (200th subject)    at serial 205
```

Both namings arrived before the rule was allowed to fire at all, so nothing was lost to DF
*convergence* — the whole gap is the `min_subjects` cold-start guard. No merge was made
that the finished-corpus view does not also endorse.

Cost: ~21µs per fact at 1559 subjects (59ms vs 10ms to resolve the whole 32k corpus), plus
one store query per session to seed the vocabulary. It is on the write path, off the
response path, and nowhere near recall's latency budget.
