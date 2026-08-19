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

---

## 11. One subject is not one slot — the defect the benchmark cannot see

Three parallel gateway sessions, same 17 turns sent to three separate chats, `gemma4:26b`
throughout. Across all three, **18 facts were marked SUPERSEDED and 1 of those was
correct.**

The one correct supersede was `capital of the Netherlands: Amsterdam → Den Haag`. Every
other one looked like this (session 1, subject `current memory system`):

```
#10  superseded  the current memory system is written in Python
#11  superseded  the memory system extraction process is asynchronous...
#12  superseded  the memory system extraction is one turn behind...
#13  superseded  the memory system extraction returns structured data
#14  superseded  the memory system uses `SUPERSEDED` tags...
#18  live        the memory system fact lookup times range from 70-90ms
```

Six facts, all six simultaneously true, five marked stale. Two turns later the model was
asked what language the system was written in and had to answer from a fact the store had
labelled superseded. The cleanest single instance is `the user is a software engineer`,
which was superseded in **all three** sessions by `Bart specialises in memory systems for
LLMs` — same subject, no disagreement whatsoever.

### The cause is one line, and it is the design, not a bug

`_classify` returned CONTRADICTION for any same-subject fact whose normalized text
differed and was not a superset, and the supersede loop then ran over `live` — every live
fact on the subject. Together those enforce **exactly one live fact per subject key**.

That is correct only if a subject holds exactly one value. It does not. `subject_hint`
answers *"what is this fact about"*, and the answer is a **topic**; a topic accumulates
many properties that are all true at once. The consolidator was reading a topic as if it
were a slot.

Two forces made it worse rather than better. `subject_labels` was fed back into P1 with
"reuse that exact subject string" — the RESULTS.md §3 fix for *missed* contradictions —
which actively pushed the extractor toward coarse subjects. Subject identity was being
asked to do two incompatible jobs at once: *"same topic, retrieve together"* and *"same
slot, so contradict"*. No single granularity satisfies both, because they pull opposite
ways.

### Why sh_6k / sh_32k / mh_6k report nothing

MemoryAgentBench FactConsolidation is constructed as repeated updates to **one attribute
per subject**. "One live fact per subject" is true there *by construction*, so the defect
cannot occur in the corpus and cannot appear in the score. §2–§3's numbers were measured
on data that structurally cannot express it.

This is worth stating plainly: those numbers are not wrong, but they measure the primitive
on the one shape where its weakest assumption always holds. Conversational memory is a
different shape, and nothing in the bench suite was ever going to say so.

### The fix: subject is the topic, attribute is the slot

`CandidateFact.attribute` / `StoredFact.attribute` name the single property the fact gives
a value for. Consolidation competes on **(subject_key, attribute)**; only a fact in the
same slot can supersede. `subject_key` is untouched, so `aliases.py`, `chain.py` and the
subject vocabulary all keep working on the topic exactly as before.

The decision stays deterministic — exact match on two normalized keys, then the freshness
ordinal. No LLM and no embedding entered the decision. P1 already chose which facts
collide, by choosing `subject_hint`; it is now asked the question that actually determines
that, instead of one that only correlates with it.

`""` means unspecified and collides with everything, in both directions — an old fact with
no attribute, or a candidate with none, behaves exactly as it did before this field
existed. That keeps old graphs and the bench harness (whose cached subject extraction
carries no attribute) inert, and it keeps the fallback on the safe side of the asymmetry:
with no slot information we over-supersede, which mislabels a neighbour but leaves the
right answer live, rather than under-superseding and presenting a dead fact as current.

### Measured after, same turns, same model

```
#3   live        [the user :: occupation]              The user is a software engineer
#4   live        [the user :: specialisation]          ...specialises in memory systems for LLMs
#5   live        [the memory system :: implementation language]  ...written in Python
#7   live        [the memory system :: latency lookup]  ...lookup times range from 70-90ms
#9   SUPERSEDED  [the Netherlands :: capital city]      ...is Amsterdam
#10  live        [the Netherlands :: capital city]      ...is Den Haag
#12  SUPERSEDED  [the user :: default deployment target] deploys to staging by default
#13  live        [the user :: default deployment target] ...is now production
```

**3 supersedes, 2 correct**, against 18 supersedes and 1 correct before. Both genuine
contradictions still fire, including the §6 pair — P1 emitted the identical attribute
(`default deployment target`) for "deploys to staging by default" and "the default
deployment target is now production", which was the one thing this change could not afford
to break.

The remaining error is a P1 labelling mistake, not a consolidation one: `the user was born
in Den Haag` was filed under `creation location`, the slot already holding `the user wrote
the memory system in Den Haag`, and superseded it. Slotting moves this class of error to
where an LLM can be asked to fix it; it does not eliminate it.

### What this does not fix

Attribute identity has both of the identity errors subject identity has, and they are the
residual failure modes of the design rather than incidental:

- **Slot split** — two namings of one property (`deploy target` vs `deployment
  environment`) become two slots, and the contradiction between them is missed. A stale
  fact stays live. This is §3's synonym problem one level down.
- **Slot collision** — two genuinely different properties get the same slot name, and the
  newer one supersedes a fact that is still true. **This is the direction that loses
  information**, and it is the one that survived into the measured run above: the
  `creation location` case is not a one-off outlier, it is this failure mode's general
  shape. Attribute-slotting narrows the over-superseding defect from "every fact about a
  topic" to "facts P1 mislabels into one slot" — it does not close it.

Slot split is the narrower problem of the two — the candidate set is one subject's slots
rather than the whole session, `subject_slots` shows P1 the properties already held per
subject so it can reuse one, and sorted-token normalization collapses word-order variants
as it does for subjects. Slot collision has no such lever: nothing deterministic can tell
that "born in" and "wrote it in" are different properties.

Whether either needs the §10 document-frequency treatment is unmeasured, and there is no
corpus that would answer it — for exactly the reason there is none that shows the defect.

---

## 12. The BM25 arm is a penalty, and the floor sits on the penalised score

Reported from real console use: the gate often stays shut on turns the store can answer,
and whether it opens seems to depend on *wording* — "my / me / I / the" — rather than on
whether the memory exists.

That is correct, and the cause is not the embedder.

### The embedding is fine; the fusion is not

`mxbai-embed-large` puts "how old am i" at cosine **0.608** against `The user is 58 years
old.` — comfortably over the 0.48 floor. Across 15 real trace queries whose answer is in
the store, raw cosine clears the floor on **15/15**. First-person query against
third-person stored fact is not the problem.

The fused score is, and the mechanism is arithmetic. Fusion is

```
fused = cos · (1 + w·bm25_norm) / (1 + w)          w = text_weight = 0.3
```

`bm25_norm` is `raw / best_text`, so it is 1.0 only for the single best lexical match in
the result set and 0.0 for anything sharing no terms with the query. The range of `fused`
is therefore `[cos/1.3, cos]` — **BM25 can only ever deduct**, never add, and a fact with
no lexical overlap loses exactly 23% of its score before the floor sees it:

```
  query                              cos    fused   ratio   gate
  what am i called                  0.555   0.427   0.769   shut
  What are my profile details?      0.568   0.437   0.769   shut
  who am i?                         0.576   0.443   0.769   shut
  what do i do for work?            0.489   0.427   0.874   shut
  how old am i                      0.608   0.608   1.000   OPEN
  do i write tests?                 0.767   0.767   1.000   OPEN
```

`what am i called` is the clearest case: `The user's name is Bart.` is the top semantic
match at 0.555 and shares not one word with the query — "called" versus "name" — so it is
docked to 0.427 and refused. The user's diagnosis was right; the word matching is BM25's,
not the embedder's.

This also explains why the same question behaves differently in different sessions. The
penalty depends on `best_text`, which depends on what *else* is in the store, so identical
phrasing can open the gate in one session and not another.

### It also inverts positives and negatives

The worst off-domain negative in the set scores **fused 0.580** — higher than three
genuinely answerable questions score after the deduction. The fused score is not merely
noisier than cosine here, it ranks a false positive above a true one.

### Floor sweep, both quantities, real P1 phrasing

15 positives (answer is in the store) against the 39 off-domain negatives from
`calib_fixtures`, gating on the shipped fused score versus on raw cosine:

```
   floor |    FUSED (shipped)    |   COSINE (gate only)
         |  recall  false-open   |  recall  false-open
    0.48 |    0.73       0.077   |    1.00       0.282   <- shipped
    0.52 |    0.67       0.051   |    0.87       0.103
    0.54 |    0.67       0.026   |    0.87       0.077
    0.56 |    0.67       0.026   |    0.80       0.051
    0.60 |    0.47       0.000   |    0.53       0.000
```

Cosine-gating **dominates at every matched false-open budget** in the useful range: 0.73 →
0.87 recall at FO 0.077, 0.67 → 0.80 at FO 0.051. Gating on cosine while continuing to
*rank* on the fused score is the change this points at, and it does not reintroduce what
the multiplicative-fusion invariant guards against — that invariant exists to stop BM25
*inflating* an irrelevant hit over the floor, and cosine-gating cannot inflate anything,
because `fused ≤ cos` always. It removes a deduction; it adds no bonus.

### Why it is not shipped here

Two reasons, both measured rather than cautious.

**The floor moves with the gated quantity.** At the shipped 0.48 on raw cosine, off-domain
false-open is 0.282 — a quarter of unrelated chit-chat opens the gate. `score_floor` and
the gated quantity are one decision in exactly the way `score_floor` and the embedder are.
This needs `bench.calibrate`, not a hand-picked number.

**The calibration fixture does not contain this failure.** `CHAT_FACTS` stores terse
fragments — `"deploys to staging by default"` — while real P1 stores
`"The user deploys to staging by default"`, and the fixture's positives share stems with
its facts (`deploy setup`/`deploys`, `test framework`/`tests`). The 1/(1+w) deduction
therefore almost never bit during the run that chose 0.48. Recalibrating against that
fixture would re-bake the blind spot whichever quantity is gated on. The fixtures have to
be regenerated from the real extractor first, so they track P1 instead of drifting from it
again.

And the honest limit: the distributions overlap. Positives run 0.489–0.767 while the worst
off-domain negative reaches 0.581, so **six real positives sit below the worst negative**.
No floor on either quantity separates them. Cosine-gating is a better operating curve, not
a fix — the same structural result §5 and §9 reached from the other direction.

### `mxbai-embed-large` has no prefix entry

`_PREFIXES` covers `nomic-embed-text` only, so the current default embedder is used
symmetrically although it is trained asymmetrically. Measured over this set the query
prefix is roughly neutral on positives and cut off-domain false-open 9/39 → 6/39 on one
arm — enough to belong in the sweep as a variant, not enough to change by hand. Same
coupling: adding it changes the query-side distribution and therefore the floor.

## 12a. An explicit "remember that X" was being silently refused

Separate defect, no calibration involved. P1's salience gate dropped
`"The chair is against the wall"` — defensible, it is mundane — but it also dropped
`"Remember that the chair is against the wall"`, which is not a salience judgement to
make. The user had already made it.

Fixed in the P1 prompt: an explicit remember-request overrides the salience rules, and the
one rule it does not override is the question rule (`"can you remember what I said
earlier?"` is still a question). Verified against `gemma4:26b` — six remember-request
phrasings now all store, and six transient turns including two memory-related questions
still all drop.

---

## 13. Recalibration — the fixtures were the instrument, and it was bent

§12 identified the BM25 deduction and stopped short of shipping a change, for two stated
reasons: the floor moves with the gated quantity, and the calibration fixtures could not
express the failure. Both were addressed before re-measuring.

### The fixtures are now derived from the write path, not authored

`bench.gen_calib_fixtures` runs the real `OllamaExtractor` over authored learn-turns and
keeps what it emits, so `CHAT_FACTS` holds `"the user deploys to staging by default"`
rather than the hand-written `"deploys to staging by default"`. Two things fell out of
that immediately, and both are the point:

- **The old fixture contained a fact the write path would never produce.** `"flies out to
  Lisbon on the 14th"` was authored as a stored fact; P1 refuses it, correctly, as a dated
  one-off. The generator's one-fact-per-turn assertion caught it. It is now a recurring
  trip.
- **`"the user's work laptop"`**, not `"the work laptop"` — P1 attaches the possessor. A
  fixture written by hand does not guess that.

The generator asserts what silently breaks otherwise: exactly one fact per turn (the
positives are `(query, index)` pairs, so a turn yielding two shifts every later index and
zeroes `useful_tpr` without raising), and that each entity survives extraction (the
crowded fixture's hard negatives are near-misses *by construction*; rename the entities
and that fixture stops measuring anything). `build_chat_fixture` additionally asserts at
build time that every positive names a fact the fixture actually stores.

Regeneration is **not** bit-reproducible even at `temperature=0.0` — one run emitted
`"the prod cluster is in us-east-1"`, the next `"the production cluster is located in
us-east-1"`. So the generated literals are checked in rather than derived at import: a
floor has to be calibrated against a fixed object.

### The positives had the same drift, one level up

Every original positive shared a word stem with its fact — `deploy setup`/`deploys`,
`test framework`/`tests`, `milk`/`oat milk` — so the BM25 arm found something on every
one, and the 1/(1+w) deduction almost never bit. **A fixture that cannot express the
failure cannot calibrate against it.** Twelve paraphrase-only positives were added, one
per fact, sharing no content word with what they target (`"who employs me?"`,
`"what goes in my flat white?"`, `"anything I can't eat?"`).

That single change moved the measured chat TPR of the *shipped* config from 0.920 to
0.730. The gate was always this leaky; the instrument could not see it.

### Result — cosine dominates fused at every floor in the chat regime

`mxbai-embed-large`, `subject_check=ON`, 37 chat positives / 39 off-domain:

```
   floor |  FUSED tpr / off-dom   |  COSINE tpr / off-dom
    0.48 |      0.730       0.077 |      0.892       0.205   <- old shipped (fused)
    0.52 |      0.649       0.051 |      0.838       0.103
    0.54 |      0.622       0.051 |      0.784       0.051
    0.57 |      0.622       0.026 |      0.784       0.026   <- NEW shipped (cosine)
    0.60 |      0.514       0.000 |      0.649       0.000
```

At matched false-open rates cosine is worth a consistent **+16 points of conversational
recall**. And the shipped move improves *both* axes at once:

| | chat TPR | chat off-domain FPR |
|---|---|---|
| old: fused @ 0.48 | 0.730 | **0.077** — over the 5% budget it was chosen to satisfy |
| new: cosine @ 0.57 | **0.784** | **0.026** |

All three regimes at the new pairing: bench TPR 1.000 / useful 0.990 / off-domain 0.000;
chat 0.784 / 0.026; crowded 1.000 / 0.000. Latency is unchanged — p50 ~84ms chat, ~106ms
bench, well inside §14's 200ms P95 budget, because nothing new is computed: the un-fused
cosine was already in hand and is now simply carried on the hit.

**What it costs.** Bench hard negatives rise 0.111 → 0.263 (right relation, absent
subject). That is `subject_check`'s job, not the floor's — §5 and §9 both established that
the gate keeps *off-topic* memory out and never *wrong-subject* memory, because those
distributions overlap and no scalar threshold separates them. Trading a metric the floor
was never able to own for recall it can is the right side of that division of labour.

### The query prefix is measured and rejected

`mxbai-embed-large` is trained asymmetrically and its card specifies a query prefix. Using
it costs conversational recall at every operating point — chat TPR 0.784 → 0.730 on
cosine, 0.622 → 0.595 on fused — for no off-domain gain. It buys bench hard-negative
precision (0.263 → 0.253 cosine, 0.111 → 0.081 fused), which is again not the floor's job.

So it is **not** applied. `config.KNOWN_PREFIXES` records what the card documents and
`_PREFIXES` records what ships; they are separate tables precisely because listing mxbai
in the latter would have enabled it silently for every deployment.

### What this does not fix, stated against the original report

`"what am i called"` — cosine 0.555 against `The user's name is Bart.` — **still shuts**,
because 0.555 is below 0.57. On a 12-query ad-hoc probe drawn from the original traces the
new pairing fixed one query and lost two; that is small-sample noise against the 37-positive
calibrated set where it wins on both axes, but it is not nothing, and the reason is the one
§12 already gave: the distributions overlap. Chat positives run down to p05 0.404 while
off-domain negatives reach 0.575. **No floor on either quantity separates them.**

Two knobs remain for anyone who wants the other trade, both measured above: cosine at 0.55
buys back `what am i called` at off-domain 0.051, and cosine at 0.52 gives chat TPR 0.838
at off-domain 0.103. Neither meets the 5% budget `recommend()` enforces, which is why
neither ships by default; on a personal console, where a spurious recalled fact costs
almost nothing, 0.52 is a defensible local override.

---

## 14. The hint list was feeding P1 scrambled slot names

Live testing across three parallel sessions made §11's two residual failure modes the
dominant error. The three stores diverged sharply while consolidation itself was correct in
every one — the variable was the attribute string P1 coined.

The clearest case, same script, same model, same config:

```
session B   #15 superseded  preference technical   the user likes coding in Python
            #17 live        preference technical   the user prefers Ruby        <- resolved

session C   #13 live        language primary       the user likes coding in Python
            #15 live        language preference    the user prefers Ruby
            #16 live        affinity language      the user likes Python        <- 3 live
```

### The mechanism, and it is not the model's judgement

`subject_slots` shows P1 the slots a subject already holds so it can reuse one. What it was
showing was the **normalized key** — and the key is sorted content words. Every multi-word
attribute in the live store came back scrambled:

```
  natural phrasing                        what P1 was shown to reuse
  lookup latency                     ->   latency lookup
  medical issues                     ->   issues medical
  todo list                          ->   list todo
  main office location               ->   location main office
  favourite city to write python code->   city code favourite python write
```

The instruction says *"reuse that exact property string"*. Given `city code favourite python
write`, no model does — it coins a fresh slot, and the contradiction never fires. This is the
slot-split failure arriving through the hint list rather than through the model's judgement.

`StoredFact.subject_label` exists for exactly this reason and its docstring says so. §11
argued explicitly that attributes needed no counterpart, because sorting makes reuse
order-insensitive. **That argument is right for matching and wrong for prompting**, and the
old reasoning has been removed rather than left standing.

### The fix

`StoredFact.attribute_label` holds the first natural phrasing; `subject_slots` shows it. The
key is untouched, so identity still decides on sorted tokens and a rephrasing still lands in
the same slot. Verified end to end:

```
  the hint list P1 now receives          the keys that decide identity
  the memory system -> lookup latency    memory system  'latency lookup'
  the user -> todo list                  user           'list todo'
  Supreme Data Systems ->                data supreme systems
      main office location                              'location main office'
```

A test pins that the label never participates in matching — `deploy target` and
`target deploy` must still collide — because §11's own tests pass attributes explicitly and
would not catch a regression there.

### The instrument, and what it could not show

`memore/bench/slots.py` scores a fixed 25-turn script on three axes that move independently:

| | measures | baseline (3 runs) |
|---|---|---|
| must-collide resolved | slot **split** | 17/17 |
| must-coexist intact | slot **collision** | 21/21 |
| one-subject coherent | **subject** split | 37/39 |

Both attribute axes are clean at this length, including on deliberately long attribute names.
Re-run after the fix: **17/17, 21/21, 37/39 — identical**. That is the expected result and it
is worth stating as measured rather than predicted: the change is a no-op on this script in
both directions, so it is confirmed not to have caused collisions while buying nothing here.
The harness **does not reproduce the live failure**, and this fix rests on the mechanism above
plus the live stores, not on a moved number. Stated plainly rather than papered over: a
25-turn script is not a 50-turn session with long analytical assistant replies feeding P1, and
closing that gap honestly needs a longer script.

What the harness *does* catch is subject splitting — `Lisa` versus `the user's sister Lisa`
across runs of identical input. That is co-reference, not phrasing, and `attribute_label` does
nothing for it. It is the next thing to attack.

The three axes are reported separately on purpose. Making P1 reuse attributes harder fixes
splits and causes collisions; a single "fragmentation" count would hide exactly that trade.

---

## 15. Subject co-reference — one entity, two names

§14 closed with the honest observation that the slot harness caught only one thing:
`Lisa` versus `the user's sister Lisa` on identical input, 2 failures in 39. That is
co-reference, not phrasing, and no amount of attribute work touches it.

The first move was **not** to fix it. It was to establish that the instrument could see
it, because at 2-in-39 with a non-deterministic extractor, a move to 39/39 is
indistinguishable from noise — and §11 is the standing lesson about fixing a defect the
measurement is structurally blind to.

### Read the failure before designing for it

A clean baseline run of the 25-turn script produced **36/39** one-subject coherent, and
the three failures were not one shape but two:

```
run 1   FAIL [14, 15]   SUBJECT-SPLIT   lisa | user
run 2   FAIL [16, 17]   SUBJECT-SPLIT   lisa | lisa sister user
run 3   FAIL [14, 15]   SUBJECT-SPLIT   lisa | user
```

That distinction decided the whole approach. `lisa` versus `lisa sister user` share the
proper noun, so a surface merge rule could in principle reach it. `lisa` versus `user`
share **nothing** — P1 filed a fact *about Lisa* under the subject *the user* — and no
deterministic rule over subject keys recovers that. It is §9's wall from a third side.

Two of the three failures were the unreachable shape. A matcher was the wrong build.

### The instruction conflict

`extract.py` had been telling P1 two incompatible things in one sentence:

> `subject_hint` is the TOPIC the fact is about … **Prefer the shortest natural noun
> phrase**, and **reuse a subject you have already used** rather than rephrasing it.

Those clauses fight precisely when the stored label is long. Turn 14 ("My sister Lisa
lives in Amsterdam") invites `the user's sister Lisa`; turn 17 ("Lisa's birthday is …")
invites the shortest form, `Lisa`. Both instructions were followed; the store split.

This is §14's shape exactly — P1 shown one thing and instructed toward another — and it
takes §14's remedy: change what P1 is told, not what the store does afterwards.

### Extending the harness first

The 25-turn script could not carry the fix, for two independent reasons. One Lisa entity
at 2 failures in 39 is noise-adjacent. And there was **no over-merge trap in it at all** —
nothing like "Lisa's daughter Fien" — so `MUST_COREFER` alone would have scored a rule
that merges everything at 100%.

Nine turns and one axis were added before any prompt was touched:

- three co-reference entities, not one: Tom (descriptor first, name later), Miso (name
  first, descriptor later), Fien (introduced *through* Lisa);
- `MUST_DISTINGUISH`, the refuse-list — pairs that must NOT share a subject: Lisa/Fien
  (share a proper noun), Tom/Tom Bakker (share a first name), the memory system/its test
  suite (part-of), Pixel/Miso (both "the user's pet" if the descriptor is generalised).

Turns were **appended**, never interleaved, so every index above 24 is new and the
collide/coexist numbers stay comparable with §11 and §14.

The refuse-list is written before the fix on the precedent of §3 and §10, both of which
hand-labelled the pairs a rule must decline before letting an aggregate score decide
anything. The asymmetry is the same one `AliasConfig` records and it is not symmetric
book-keeping: **a split costs recall and is recoverable — both facts are in the store,
correctly filed. A merge destroys a fact permanently.** A fix that trades one over-merge
for one recovered split is a regression, not a wash.

The extended harness immediately showed how blind the old one had been. Where 25 turns
gave 2 failures in 39 across three runs, 34 turns gave **12 in 51 — the same failures in
every run**, plus an over-merge the old script could not express:

```
FAIL [25, 26, 27]     SUBJECT-SPLIT   colleague tom user | tom      3/3 runs
FAIL [28, 29]         SUBJECT-SPLIT   cat miso user | user          3/3 runs
FAIL [14,15,16,17]    SUBJECT-SPLIT   lisa | lisa sister user       3/3 runs
FAIL (6, 32)          OVER-MERGED     memory system                 3/3 runs
```

Reproducible in every run, not intermittent. The defect had been real all along; the
25-turn script was measuring one entity and calling the variance noise.

### The fix: five ordered naming rules

`subject_hint` now gets an ordered procedure instead of two competing preferences:

1. name the entity the fact is **about**, never the person who mentioned it;
2. the speaker is **always** "the user", even once you learn their name;
3. any other named entity takes **that name alone** — `Lisa`, not `the user's sister
   Lisa` — keeping whatever part of the name tells two bearers apart (`Tom Bakker`);
4. an unnamed entity takes the shortest natural noun phrase;
5. a subject is a thing, never a thing's property.

Rule 2 exists because rule 3 without it is actively harmful: `My name is Bart` would make
`Bart` a subject and split the store's single most-loaded one. That was caught by reading
the rule against turn 0, before running anything.

Rule 5 came from the live gateway store rather than the harness, which passes the case.
The console's graph held `netherlands` **and** `capital netherlands` as separate subjects —
the property absorbed into the subject, leaving four facts with an empty attribute and
nothing for a correction to collide with. It went through two versions; the second is
measured separately below, because it moves two axes in opposite directions.

### Measured

Three runs each, same **34-turn** script, same model and config, extractor prompt the only
variable. (The script grew to 36 turns afterwards, to settle a question about rule 5; the
next section re-measures both prompts on it and those totals are not comparable with
these.)

| | measures | before | rules 1–4 |
|---|---|---|---|
| must-collide resolved | slot **split** | 17/17 | 18/18 |
| must-coexist intact | slot **collision** | 27/27 | 27/27 |
| one-subject coherent | subject **split** | **39/51** | **51/51** |
| distinct kept apart | subject **over-merge** | 9/12 | 9/12 |

Every subject split closed, in all three runs, and **the refuse-list did not move** — the
fix bought the split axis without spending anything on the axis built to catch its price.
The subject vocabulary collapsed to one name per entity:

```
before   lisa | lisa sister user | sister user | colleague tom user | cat miso user |
         dog pixel user | user
after    the user | Lisa | Tom | Tom Bakker | Pixel | Miso | Fien | the memory system |
         the Netherlands
```

`Tom` and `Tom Bakker` stayed apart in all three runs, which is the trap a merge rule
keyed on a shared proper noun fails by construction. That is the argument for having
fixed this at P1 rather than with a matcher, stated as a measurement rather than a
preference.

### Rule 5, measured twice, and a prediction that did not survive

Rules 1–4 left `(6, 32)` failing exactly as before: P1 filed the test suite's runtime
under `the memory system`, in all three runs. What that over-merge cost on the 34-turn
script was **nothing measurable**:

```
memory system   latency lookup, implementation language, execution suite test time
```

The fact landed in the wrong *subject* but in its own *attribute*, so it competed with
nothing and must-coexist stayed 27/27. That is a real property of the two-level key of
§11: merging subjects only destroys a fact when the attribute collides too, so a subject
over-merge is **contained** rather than automatically catastrophic.

Rule 5 was rewritten on the theory that the containment was luck running out slowly —
that as first written it said "if you are about to write `the X of Y`, the subject is Y",
which is right for a property (`the capital of the Netherlands` is a value the Netherlands
has) and wrong for a part (`the test suite` has a runtime and a framework of its own).
The predicted bill was concrete: **one more fact about the test suite would ask for
`implementation language` under `memory system`, collide with "the memory system is
written in Python", and kill a true fact.**

Two turns were added to test exactly that, rather than leaving the claim as prose — turn
34 `"The test suite is written in pytest"`, paired with turn 6 on both `MUST_COEXIST` and
`MUST_DISTINGUISH`. Both prompts were then run against the same 36-turn script, three runs
each:

| | rule 5 **v1** | rule 5 **v2** (shipped) |
|---|---|---|
| must-collide resolved | 18/18 | 18/18 |
| must-coexist intact | **33/33** | **33/33** |
| one-subject coherent | 54/57 | 54/57 |
| distinct kept apart | 12/15 | **15/15** |
| failing | `[32, 34]` ×3, `(6, 32)` ×3 | `[3, 4]` ×3 |

**The predicted supersede did not happen.** must-coexist is 33/33 under *both* prompts.
Under v1, turn 34 took `the test suite` as its subject anyway — the sentence has no
"the X of Y" shape for v1's rule to misfire on, so the trigger simply never fired. The
latent-defect argument was wrong, and it is recorded here rather than quietly dropped
because it was the argument this rule shipped on.

What the ablation *did* establish is better than what it was built to show. v1 does not
merge turn 34 into the system — it **splits the test suite in two**, `memory system` for
turn 32 and `suite test` for turn 34. So at 36 turns the two prompts tie on subject
coherence at 54/57, and v2 wins the refuse-list outright, 15/15 against 12/15.

That also retires the trade recorded one column earlier. On the 34-turn script v2 looked
like it cost 3 points of coherence for 3 points of distinctness; that apparent cost was an
artifact of the script holding **one** fact about the test suite. Give the entity a second
fact and v1's advantage disappears, because v1 is the prompt that cannot keep the entity
together. v2 dominates on every axis at 36 turns, and ships on that rather than on the
asymmetry argument.

The `[3, 4]` assertion v2 breaks was **not** relaxed to accommodate it, though the case is
genuinely arguable — "I wrote the memory system in Den Haag" is about the system's origin
as much as about the user. Editing an assertion after seeing a change fail it is what §3
and §10 refuse, so it stands as a standing failure, flagged in the harness source. It also
stopped exercising §11's same-value-different-property case once its two facts landed on
different subjects, since subjects that never meet cannot compete — so turn 35 (`"I still
live in Den Haag"`, paired with turn 3) was added to restore that coverage on a subject
rule 5 cannot move.

### What is still unfixed, stated plainly

- The `lisa | user` shape — a fact about one entity filed under another — is addressed by
  rule 1 and did not recur in three runs. Three runs is not proof; it is what was measured.
- `[3, 4]` splits 3/3 under the shipped prompt, as above.
- A subject over-merge has now measured harmless **twice**, under both rule-5 variants,
  and the mechanism by which it should eventually cost a fact was predicted and then not
  observed. The containment argument in §11's two-level key is real; the claim that it
  runs out with the second fact about the merged-in entity is **unproven**, and should
  not be repeated as if it were measured.
- Nothing here is validated on a store built from a real session with long assistant
  replies. §14's caveat stands: 36 scripted turns is not a 50-turn working session.

## 16. Two ways a live fact got mislabelled, and one rule that was refused

Source: the two-column gateway console run of **2026-08-17** — 24 turns per column,
identical input, two independent stores (`logs/memore/console-c1-*.jsonl`, `-c2-*`). It is
the first trace from a real typed session rather than a scripted harness, and it surfaced
two defects that §11's slot work left standing. Both cost a live fact. Neither is
expressible in FactConsolidation, for the same structural reason §11 gives: the bench feeds
one fact per turn, so a batch is always of size one.

Nothing was ever deleted — supersede-never-delete held throughout, and both facts came back
from recall. The damage in both cases is a **still-true fact labelled SUPERSEDED**, which
hands the reader a false temporal claim ("Lisa *used to* live in Amsterdam"). Mislabelling,
not loss.

### 16.1 Same-batch supersede — the ordinal that meant nothing

Turn 4 fired in **both columns**, which is what makes it the headline: it is deterministic,
not P1 variance. One `consolidate()` call, three candidates:

```
NEW           [the user|tea preference] ord=11  the user does not like milk in their tea
CONTRADICTION [the user|tea preference] ord=12  the user likes green tea      -> retires 11
```

Both are true. Turn 3 in c2 is the same shape (`lolly preference`, blue/green).

The mechanism is that `consolidate()` writes through per candidate, so candidate *n+1* sees
candidate *n* in `live`. Their ordinals differ only by **position in P1's output array**,
which carries no information about which the user meant later. The freshness primitive was
being applied where freshness does not exist.

The guard withholds `CONTRADICTION` when the incumbent was written by the same batch, and
sits **after** the containment branches on purpose: `REFINEMENT` and the restatement branch
below are decided by what the strings say, which needs no ordering and is just as valid
between batch siblings. Only `CONTRADICTION` rests on recency, so only `CONTRADICTION` is
withheld. The exact-`DUPLICATE` scan is untouched, or the fix would trade a mislabelled
fact for a second copy.

### 16.2 The bill for coexistence, paid before it arrived

Letting two facts share a slot breaks the one-live-fact-per-slot invariant on purpose, and
that invariant was load-bearing elsewhere: a `CONTRADICTION` superseded the *whole*
competing set, justified as self-healing against corruption. Left alone, that turns the fix
into a one-turn delay —

```
turn  4   "no milk in tea" + "likes green tea"    coexist, both live       (the fix)
turn 30   "I hate green tea now"                  retires BOTH             (the bill)
```

So a contradiction now retires the incumbent plus any live fact from a **different** batch.
The incumbent's own batch siblings were never ordered against it, so disagreeing with the
incumbent says nothing about them. `source_episode_id` — the spec's field for exactly this
(`Episode`, writepath §3), previously written empty — carries the batch id.

Facts with `source_episode_id == ""` fall back to the old wholesale behaviour, on the same
argument `_competing` makes for `attribute == ""`: with no batch information, over-
superseding is the safe error, and pre-existing graphs and the bench keep the behaviour
they were measured with rather than silently acquiring a rule their data cannot support.

### 16.3 Subsumption inversion — a restatement that said less and won

Cross-turn, column 2, and a separate mechanism:

```
turn 11  [Bud|seating]  Bud sits in the Red chair in the Whangarei office
turn 16  [Bud|seating]  Bud sits in the red chair          CONTRADICTION -> retires turn 11
```

The arriving fact is a strict substring of the stored one. It disagrees about nothing, and
"in the Whangarei office" left the live set on recency alone. `REFINEMENT` already handled
the other direction (incumbent ⊂ candidate); this direction fell through to the money case.

**It coexists rather than answering DUPLICATE, and that is measured, not cautious.**
Containment this way round does *not* imply "adds nothing". Scanning both corpora for
same-subject pairs where a later fact is strictly contained in an earlier one:

```
sh_6k    455 facts, 303 subjects    0 pairs
sh_32k  2310 facts, 1559 subjects   1 pair
    "flanker is associated with the sport of rugby union"  ->  "... of rugby"
```

That single pair is a real value change — the same one §6 records the embedding threshold
swallowing at cos 0.982. A `DUPLICATE` verdict there discards an update permanently, the
unrecoverable direction `ConsolidationConfig` exists to warn about. And no surface rule
separates it from the Whangarei case: both are strict prefixes, differing only in whether
the trailing phrase opens with a preposition. A closed-class preposition test tuned on two
examples is exactly the kind of threshold §6 refused. So neither fact is retired and neither
is dropped — both stay live, the detail survives, and the reader gets both instead of one
wrong one.

The branch additionally requires a real slot on **both** sides. With no attribute there is
no "same property" for a restatement to be a restatement *of*, which is also what keeps the
bench inert: `bench/extract.py` supplies no attribute, so FactConsolidation takes the path
it always took.

### 16.4 The bench was asserting a simultaneity the corpus does not have

The guards were expected to be inert on FactConsolidation — no attribute, no containment
pairs at 6k, and one fact per turn. They were not, and the reason is worth more than the
regression check it came out of.

Both harnesses ingested in **chunks of 50**:

```python
for i in range(0, len(candidates), 50):
    await consolidator.consolidate(session, candidates[i : i + 50])
```

That chunk exists to batch the *embedder*. But a `consolidate()` batch is a semantic unit —
after §16.1 it means "one utterance, no internal freshness order" — so handing it 50
independent turns declared 50 facts simultaneous. Any two facts about one subject landing in
the same chunk stopped superseding each other:

```
             baseline (alias015)      chunks of 50        one at a time
sh_6k        NEW 301  CONTRA 154      NEW 319  CONTRA 136  NEW 301  CONTRA 154
sh_32k       NEW 1523 CONTRA 787      NEW 1539 CONTRA 771  NEW 1523 CONTRA 787
```

18 supersedes lost at 6k, 16 at 32k — purely from where the chunk boundaries fell. Note what
this did *not* move: oracle accuracy stayed 0.990 and 0.960 either way, and
`gold_fact_superseded` stayed 0. A harness bug of exactly this shape was invisible to the
score, which is the same lesson §11 records about aggregates.

The harnesses now prewarm the embedder over the chunk and consolidate one fact at a time
(`DeterministicConsolidator.prewarm`), which restores the batching where it belongs and
makes the arrival order the corpus actually has explicit. With that:

```
sh_6k   0.990 (99/100)   NEW 301  CONTRADICTION 154   >1 live fact per group: 0
sh_32k  0.960 (96/100)   NEW 1523 CONTRADICTION 787   >1 live fact per group: 0
```

— identical to the alias015 baselines in every field. So the bench is a regression guard
here and nothing more: an unchanged number is evidence of nothing breaking, not evidence the
fixes work. The tests that carry that load are trace-derived, at the end of
`tests/test_consolidation.py`, built from the console run's verbatim candidate texts.

**Anyone adding a caller must know this.** `consolidate(session, candidates)` is one
utterance. Facts that genuinely arrived separately go in separate calls, or they silently
stop resolving against each other.

### 16.5 The rule that was refused: snapping attribute names

The obvious third fix was to canonicalise drifting slot names the way §10 canonicalises
subjects — merge two attributes of one subject that differ only by generic relation words,
gated on document frequency. It was measured against this run's own attribute vocabulary
and **refused**, on two independent grounds.

**The df gate is degenerate at conversational scale.** The session produced 30–34 distinct
attributes. Every token appearing even once reads as ratio 0.029–0.033 — above the 0.015
threshold `AliasConfig` ships. Not "unreliable"; there is no vocabulary in which any token
is rare. This is §10's `min_subjects` lesson arriving on a second axis, and it says the same
thing: a relative threshold fires hardest exactly where its evidence is weakest.

**Its only reachable action in 24 turns would have been the bug.** The single
subset pair among same-subject attributes in either column is on Lisa in c1:

```
location  ⊂  work location        extra token "work", df 1, ratio 0.029 > 0.015 -> merge
```

Merging those makes "Lisa lives in Amsterdam" compete with "Lisa works in Amsterdam" — which
is precisely the fact that ended up mislabelled in c2. A rule whose one available move
reproduces the defect being fixed is refuted, not untuned. Every other multi-slot pair in
the run (`the red chair | position` vs `location`, `Bud | seating` vs `preference`,
`the user | programming language preference 1` vs `2`) is two genuinely different properties
that must stay apart.

**And the premise was partly wrong.** Attribute drift is **between runs, not within a
session**. Within c2, `Bud|seating` was correctly reused across turns 11 and 16 — the
`subject_slots` feedback loop worked. The pairs that look like drift (`chair`/`seating`,
`database architecture`/`database stack`) are *cross-column*: two independent stores, each
internally consistent. That is run-to-run variance in P1 at temperature 0 — the finding §15
already records — and it matters for reproducing an experiment, not for answering a question
inside one conversation. The one genuine within-session split in 48 turns is c1's
`Bud|chair` → `Bud|location` at turn 16.

### 16.6 Still open

- **The write path re-ingests what the model was shown.** On question turns where recall
  fires, P1 re-emits recalled facts *verbatim* — 4 of the run's 7 DUPLICATEs are the exact
  string recall injected on that same turn (t23 and t24 in c1, t19 and t24 in c2), and t24
  fired in both columns. The only thing containing it is the exact-normalized-string
  DUPLICATE scan, which catches neither a paraphrase nor a *join over two* recalled facts.
  Turn 23 c1, a pure question ("When is my flight to lisbon?"), stored
  `NEW [the user's trip to Lisbon|coincides with] :: coincides with Lisa's birthday`.
  Not fixed here — measured properly in §17.

  Corrected from the first draft of this section, which claimed history was an alternative
  channel: `extract_window_turns` is **3**, and it slices *messages*, so at turn 23 the
  window is roughly turns 21–22 and turn 6 (which did state the join) is 19 turns outside
  it. History is ruled out. The same draft called the join "present in neither turn 23 nor
  the store", which overstates it — both components are stored (Lisbon trip Aug 29, Lisa's
  birthday Aug 29), so it is derivable from the store even though no single fact holds it.
- **Latency**: recall median 99ms (c1) / 105ms (c2), consistent with §5. c1's p95 is **202ms**,
  at the §14 budget rather than under it — a cold-start artifact (turns 1–2 at 202/213ms,
  everything after at 87–120ms) over a 24-sample window where p95 is near the worst sample.
  Reported as measured.
- Gate decisions were **identical per-turn across both columns** (19/24 open, same turns)
  despite the stores having diverged — the gate is stable under store drift.

## 17. The assistant's reply was inside the turn being extracted from

§16.6 left this open: on question turns, P1 was storing things the user never said. It is
one channel, it is structural, and it is closed by moving text rather than by adding a rule.

### 17.1 What the console run actually shows

Not one fabrication — a loop that ran on every question turn where recall fired. Four of
the run's seven DUPLICATEs are the **exact string recall injected on that same turn**:

```
t23 c1   recalled "the user has a trip to Lisbon on August 29th, 2026" (0.791)  -> DUPLICATE
t24 c1   recalled "Lisa likes red tulips"                              (0.541)  -> DUPLICATE
t24 c2   same fact, same score                                                  -> DUPLICATE
t19 c2   recalled "Lisa leaves Amsterdam at 3:00 PM on Tuesday, ..."             -> DUPLICATE
```

The only thing containing that is the exact-normalized-string DUPLICATE scan, which catches
neither a paraphrase nor a **join over two** recalled facts. Turn 23 c1 is the join:
`NEW [the user's trip to Lisbon|coincides with] :: coincides with Lisa's birthday`.

### 17.2 Isolating the channel

Three candidates: prior turns, the `subject -> properties` hint list, and the assistant's
reply. Prior turns fall out on inspection — `extract_window_turns` is **3**, and it slices
*messages*, so at turn 23 the window is roughly turns 21–22 while turn 6 (which did state
the join) is nineteen turns outside it.

The other two were separated by running the real extractor on the real turn-23 text, with
the hint list reconstructed from c1's write log as it stood entering that turn, twice each:

```
A  bare turn, no hints, no reply                    0 facts, 0 facts
B  turn + hint list + real 3-message window         0 facts, 0 facts
C  B + an assistant reply stating the join          2 facts, 2 facts
     [the user|upcoming trip schedule] the user's flight to Lisbon is on August 29th, 2026
     [Lisa|birthday]                   Lisa's birthday is on August 29th
```

The hint list is **not** a channel — B is silent. The reply is, deterministically.

Note what C's two facts would have done downstream. `"the user's flight to Lisbon is on
August 29th, 2026"` lands on the slot already holding `"the user has a trip to Lisbon on
August 29th, 2026"`; not string-equal, no containment, so **CONTRADICTION** — a question
turn silently retiring the user's own phrasing in favour of the model's. The second is
saved only by §16.3: `"Lisa's birthday is on August 29th"` is a strict substring of the
stored `"...August 29th, 2026"`, so it now coexists where before it would have retired the
year.

### 17.3 Why it happened, and the fix

`assistant_response` sat inside `THE TURN TO EXTRACT FROM:` with equal standing to the
user's own words, while prior turns were explicitly labelled "context only". So the prompt's
existing rule — *"extract only what THIS turn asserts"* — was being obeyed. The reply **was**
this turn.

That is the asymmetry: the reply is generated *after* the user's message, so unlike prior
turns it can never be needed to resolve the user's references, and the reader that produced
it holds the entire conversation against the three messages P1 is shown. Anything from
anywhere in the session could launder into "what this turn asserts".

The fix moves the reply into its own labelled context block. `_SYSTEM` is **untouched** — no
rule was added, and adding one was measured and discarded:

```
C  reply inside the extraction block   question turn: 2, 2      learn turn: 1, 1
D  reply demoted to context            question turn: 0, 0      learn turn: 1, 1
E  D + an explicit system rule         question turn: 0, 0      learn turn: 1, 1
```

E buys nothing over D, and every word added to `_SYSTEM` is a word that can perturb §15's
five naming rules. D ships. The learn-turn column is the control that the fix does not
simply mute the channel: a genuine user assertion delivered *with* a reply still extracts.

### 17.4 The harness could not express this, so it was extended first

`slots.py` called `write_path.run(session, turn, "", history)` — assistant response always
empty, history always `"Understood."`. Per §13's rule, a fixture that cannot express the
failure cannot measure the fix, so the harness was extended **before** the change, as §15
did with its refuse-list:

- `REPLIES`, per-turn and only on the new turns. Turns 0–35 keep `""`, so the collide,
  coexist, subject and distinct numbers stay comparable with §11, §14 and §15.
- turns 36–37, questions whose replies restate stored memory, scored by `MUST_NOT_EXTRACT`.
- turn 38, a real user assertion delivered with a leaky reply, paired with 24 on
  `MUST_COLLIDE`. This is the guard that stops the axis being passed by muting P1: a
  question-suppressor would score 2/2 on the new axis and walk straight back into §12a,
  where an explicit "remember that X" was silently refused.

```
                          before          after       §15 reference
  must-collide resolved   18/21           18/21       18/18 on the original 6 pairs
  must-coexist intact     31/33           33/33       33/33
  one-subject coherent    54/57           54/57       54/57
  distinct kept apart     15/15           15/15       15/15
  no reply leak            3/6             6/6        (new axis)
```

Stable rather than marginal: 2/2 silent in each of the three runs, where the baseline leaked
in three of six.

**The recovery on must-coexist is the real cost of the leak, and it was not predicted.**
31/33 → 33/33 because turn 37's reply ("She's 45, and her birthday is on the 24th of
August") was re-extracted and collided with the facts turns 16 and 17 had already stored —
`[16, 17] COLLIDED, 1 superseded — lisa::age, lisa::birthday`. The channel was not merely
adding noise; it was destroying a true fact through the ordinary slot machinery. Closing it
restored §15's number exactly.

### 17.5 A new standing failure, left standing

`(24, 38)` reports SPLIT in **all three runs before the fix and all three after**:
`user::deployment workflow` vs `user::deploy target`. It is ordinary attribute-naming
variance — turn 38's text says "deploys", turn 24's said "deploy straight to production" —
and it measures nothing about the reply channel, since it is identical on both sides. Left
failing rather than relaxed, next to `[3, 4]`, per §3 and §10: an assertion edited because a
change broke it stops being evidence. Read two known failures into the collide total.

### 17.6 What this does not fix

- **Only the reply channel is closed.** Nothing prevents the *user* restating a fact the
  model just showed them, and that is indistinguishable from the user asserting it — as it
  should be.
- **Facts the assistant alone asserts are now never stored.** That is deliberate and it is
  a scope decision, not an oversight: nothing in the console run wanted it, and the reply
  is by construction downstream of everything the user said.
- **The write path still runs P1 on every turn**, question or not, at 2.1s a turn. The
  salience gate is doing its job (turns 14/15/17/20 stored nothing), but the cost is paid
  before the gate is consulted.

## 18. The attribute was named after the fact's TYPE, not the property

A second typed console run on 2026-08-17 (two columns, 27 turns, identical input, fresh
graph) showed Bud's preferences retiring each other:

```
#36 SUPERSEDED (preference) Bud likes Lisa.
#37 SUPERSEDED (preference) Bud likes beer.
#38 SUPERSEDED (preference) Bud likes the first Matrix movie.
#39 live       (preference) Bud likes red gaming chairs.
```

Nothing was deleted — all four are in the store and surface to recall marked `SUPERSEDED`,
per the invariant. But three true facts are presented as no-longer-true, which the reader
will act on, so the cost is real.

**P2 did exactly what the slot told it.** The defect is upstream: `preference` is a fact
*category* — it is literally a `FactType` enum value — not a property that holds one value
at a time. The prompt already contains the test that rules it out ("Could BOTH facts be
true at the same time? YES -> different attributes"), and P1 violated it four times.

### 18.1 The other column is the proof

Same model, same temperature, same 27 turns, and c2 named the same four slots differently:

| fact | c1 | c2 |
|---|---|---|
| likes Lisa | `preference` | `likes` |
| likes beer | `preference` | `beverage preference` |
| Matrix movie | `preference` | `interests` |
| red gaming chairs | `preference` | `interests` |
| **wrong supersedes** | **3** | **1** |

No code differs between the columns. This is P1 naming variance, and it is the whole
finding — a consolidator rule cannot be the cause of a divergence that has no code in it.

### 18.2 That run was measuring stale code — check the install first

`llm_gateway` pins `memore @ git+https://github.com/bartkamphuis/memore`, and the console
was running commit `37593fa`: no §16 same-batch guard, no §16 containment branch, no §17
reply demotion. The fingerprint is visible in the store — `source_episode_id` is `""` on
every fact, and §16 writes a batch id there.

So **7 of the run's 11 wrong supersedes were already fixed and merely not installed**. Do
this before reading any console trace as evidence:

```bash
uv pip install --python /home/marvin/code/llm_gateway/.venv/bin/python \
  --no-deps --force-reinstall "memore @ git+https://github.com/bartkamphuis/memore"
```

Note the `--python`: with `VIRTUAL_ENV` unset (which is how you work around this machine's
stale `memo_re` export), `uv pip install` does **not** discover `./.venv` — it silently
targets the default interpreter and reports success with the correct git sha. Verify by
grepping the installed file, not by reading `dist-info/direct_url.json`, which kept showing
the old commit id after a successful reinstall.

### 18.3 The replay — 27 turns, six samples, HEAD

The console run's assistant replies were never persisted (`logs/memore/*.jsonl` records
`memory.turn` / `recall` / `write` only; every `llm_audit.ndjson` row in the window has
`output_text: None`). So the replay holds the one thing that can be held fixed — the 27
user turns, verified byte-identical across both columns — and re-runs the write path at
HEAD with `assistant_response=""`, six independent sessions.

```
sample     facts   NEW  CONTRA   DUP
replay-1      41    41       0     0     <- outlier
replay-2      45    41       3     1
replay-3      44    40       3     1
replay-4      44    40       3     1
replay-5      44    40       3     1
replay-6      44    40       3     1
```

**§16 holds.** Zero recurrence in 6/6 for every same-utterance case:

| original defect | replay |
|---|---|
| blue lollies -> green lollies | 0/6 |
| milk-in-tea -> green tea (§16's own example) | 0/6 |
| Matrix -> gaming chairs, same turn | 0/6 |
| Python -> Ruby, same turn | 0/6 |

The guard is visible in the trace — same turn, same slot, both stored:

```
replay-4 turn=3 NEW ord=12 attr='tea preference' | the user does not like milk in their tea
replay-4 turn=3 NEW ord=13 attr='tea preference' | the user likes green tea
```

Two further cases from the console run also did not recur — Lisa `location` (lives vs
works) and Memore `development origin` (authorship vs conception place), both 0/6. **Not
because anything fixed them**: P1 simply named the slots apart every time
(`location` vs `employer` / `employer/location relation`). They were one-off naming
collisions, and grouping them with Bud in the first analysis was wrong.

**Bud is systematic.** 5 of 6 samples reproduce it, 2 wrong supersedes each. Net across
the six: 15 supersedes, 5 correct (all `capital city` Amsterdam -> Den Haag), 10 wrong (all
`Bud::preference`).

### 18.4 The mechanism: the hint list is a magnet

`subject_slots` returns `label -> attr1, attr2` and the prompt tells P1 that if the turn
"gives a new value for a property already listed under that subject, reuse that exact
property string too, so the two collide". That instruction is right, and it is what §14
built. But it has no way to tell a *property* from a *category*, so once `preference`
enters a subject's vocabulary it is recommended back to P1 for every later preference on
that subject, and they all land in one slot.

The split between the two halves is sharp, and the rates are kept per script rather than
pooled — the two `slots.py` baselines ran against different fixture versions (47 turns and
then 46, after §18.7 removed a pair), so a single denominator over all twelve would hide
which run each observation came from:

```
                                                  replay      slots A     slots B
                                                 (27 turns)   (47 turns)  (46 turns)
first Bud preference named `preference`             6/6          3/3         3/3
that name then REUSED for a later preference        5/6          2/3         1/3
```

P1 does not sometimes get this right and sometimes wrong. It coins the bare category
**every time, in every run of both scripts**; what varies is only whether it then escapes
by coining something specific for the next one. Note that escaping is not all-or-nothing
either — slots B run 1 reuses `preference` once and then escapes on the third preference
(`preference, preference, chairs gaming preference`), which is why that run still fails the
coexist group with one superseded fact rather than two.

### 18.5 The rate is ~2/3, and `--runs 3` cannot see it

The replay failed 5 of 6. The two `slots.py` baselines failed `[39, 40, 41, 43]` **2 of 3
and then 1 of 3** — and they differ only by two trailing turns that come *after* every turn
under test, so nothing in that edit can reach turns 39-43. That spread is P1's
temperature-0 variance and it is wide enough to swallow a fix: at this rate a three-run
sample cannot distinguish "fixed" from "got lucky". Any candidate fix needs more runs than
the standard three, and the two baselines are the reason. Do not read the shipped 1/3 as
the defect being milder than §18.1 showed; read it as the sample being too small.

### 18.6 The harness turns, and the refuse-list written before the fix

Appended as turns 39-45, indices 0-38 untouched, so §11/§14/§15/§17 numbers stay
comparable. No replies on the new turns: they measure how P1 *names* a slot, and a reply
would confound that with §17.

```
39-41  MUST_COEXIST [39,40,41,43]   Bud likes beer / the first Matrix movie / red gaming
                                     chairs. Three SEPARATE turns -- inside one utterance
                                     §16's guard already withholds CONTRADICTION, so a
                                     single-turn version passes while the defect stands.
                                     Four disjoint domains, so nothing here can compete
                                     legitimately.
42-43  MUST_COLLIDE (42, 43)         Bud's favourite programming language, Go -> Rust.
44     MUST_COEXIST [44]             one utterance, several true facts -- §16's guard.
45     MUST_COREFER                  Bud's seating.
```

`(42, 43)` is the refuse-list, and it is shaped against the fix that suggests itself
first. A `COLLECTION_TYPES` keyword list in the consolidator — "never supersede a
preference" — scores 3/3 on 39-41 and **breaks this pair**, because a favourite is a
preference that holds exactly one value at a time. It resolves 3/3 today, so the axis is
live. Three further reasons that fix is refused: it puts hand-written English in the
deterministic core (the shape §16.5 already refused); the bench carries `attribute=""`, so
a rule keyed on the attribute string is inert on the corpus that guards every other
regression; and a slot that can *never* resolve is the FactConsolidation task failing,
which is the one thing this project may not trade away.

`43` sits inside the coexist group deliberately: it is the survivor of the collide pair, so
if the magnet swallows the favourite-language slot too, the later correction retires the
three preferences with it and the group catches that.

Baseline, three runs, read separately:

```
                      run 1   run 2   run 3   total
must-collide          7/8     7/8     7/8     21/24   (2 known standing failures)
must-coexist         12/13   13/13   13/13    38/39
one-subject          19/20   19/20   19/20    57/60
distinct              5/5     5/5     5/5     15/15
no reply leak         2/2     2/2     2/2       6/6
```

The single coexist failure is the defect:

```
FAIL [39, 40, 41, 43] COLLIDED  1 superseded --
     bud::preference, bud::preference, bud::chairs gaming preference, bud::favourite language programming
```

### 18.7 A fixture that was removed for being vacuous

§16's containment branch (a restatement that says LESS must coexist) was first fixtured as
two turns — "Bud sits on the red chair in the Whangarei office", then "Bud sits on the red
chair". P1 emitted **nothing** for the second in 3 runs of 3. A pure restatement asserts no
new fact and the salience gate is right to drop it, so the branch is not reachable from a
turn that merely repeats; it needs one that reads as news while saying less, which no
scripted turn here produced. The pair was scoring two easy passes and testing nothing, so
it was removed rather than kept — a fixture that cannot express the failure cannot guard
against it, the same argument §13 makes about the calibration fixtures. The branch stays
guarded where it can be constructed directly: the trace-derived tests at the end of
`tests/test_consolidation.py`.

### 18.8 A P1 salience finding, deliberately not made into an axis

On the replay, "Bud likes red gaming chairs" was dropped by P1 in **6 runs of 6**, while
the same turn's "the red gaming chair cost $560" landed every time. One observation of one
turn does not justify a `MUST_EXTRACT` axis — it would be over-fitted to exactly this
sentence. But it must be *visible*, because every existing axis is blind to it: a coexist
group whose member wrote no fact still reads OK, since there is no dead ordinal to find. So
a change that quietly mutes P1 improves `no-reply-leak` and disturbs nothing else, which is
§17.4's warning with no instrument behind it. `print_run` now prints an unscored line:

```
wrote nothing []   (diagnostic only -- not scored)
```

### 18.9 What this does not fix, and what is still open

- **Nothing here fixes the defect.** §18 adds the instrument and the refuse-list; the fix
  is unbuilt. `git diff 37593fa..HEAD -- memore/extract.py` touches only §17's reply
  demotion — `_SYSTEM` and the hint list are unchanged, so nothing shipped today affects
  category naming.
- **Do not fix it by adding a rule to `_SYSTEM`.** The rule is already there verbatim and
  P1 violated it four times in one column. That is arm E, which §17 measured and recorded
  as buying nothing, and every word added there can perturb §15's five naming rules.
- **The replay does not exercise §17 at all.** No replies exist to demote, so the DUPLICATE
  re-ingestions cannot reproduce either way. That needs a live console run against a
  correctly installed memore.
- **The replay script is confounded by its own source.** Turn 10 pastes the previous run's
  store dump verbatim and turn 13 describes this defect in words; both sit inside
  `extract_window_turns=3` for turns 14-16, which are the turns that name Bud's slots.
  Faithful to the original, but it means P1 is measured on a script that warns it about the
  failure mid-way. The `slots.py` turns are the clean instrument.

### 18.10 The fix — ask P1 the question instead of inferring it from the name

The judgment that separates the two cases is not in the strings. "The capital of the
Netherlands is Amsterdam / Den Haag" and "Bud likes Lisa / beer" are structurally
identical pairs — same subject, same slot, different value, neither containing the other.
§16 hit that wall with surface rules and §6 hit it with embeddings, and §18.6 is the third
form of it: a `COLLECTION_TYPES` keyword list cannot separate "likes beer" from "favourite
beer is Guinness" either.

So it is asked of the only component allowed to judge. `CandidateFact.single_valued` is a
required boolean in the P1 schema:

```
- `single_valued` asks ONE thing about the attribute you just named: can this subject
  have only ONE value for it at a time?
    true   "age", "capital city", "deploy target", "favourite programming language"
    false  "likes", "interests", "skills", "languages spoken", "allergies"
```

`_classify` consumes it as a field lookup, placed with the same-batch guard and for the
same reason — only CONTRADICTION rests on recency, so only CONTRADICTION is withheld:

```python
if not candidate.single_valued:
    return ConsolidationCase.NEW, None, []
```

**No LLM runs in the consolidation decision.** The field is read exactly as `attribute` and
`subject_hint` are, `_classify` remains a pure function of its arguments, and
`test_no_llm_in_the_consolidation_decision` — which asserts the consolidator's
collaborators are `{store, embedder, config}` — still holds unchanged. What moved is
*where the judgment is made*, not whether the decision is deterministic. P1's output has
always been an input to this function; this adds one more field to it.

It sits **below** the exact-DUPLICATE scan deliberately. A collection that accumulates
copies of the same sentence is the store bloat writepath §2.2 case 2 exists to prevent,
and nothing about a slot holding several values makes a repeat of one of them news.

**Defaults True**, which is the pre-§18 code path exactly. A store written before the
field, and the bench's cached extraction (`bench/extract.py`, which supplies no such
field), behave as they always did — so every §3 oracle number stays reproducible. Same
inertness argument as `attribute == ""`.

### 18.11 Nine runs, and the standing failures are the only failures

`--runs 9` rather than the usual 3, because §18.5 established that at this defect's rate a
three-run sample cannot tell a fix from luck.

```
                        BEFORE (3 runs)              AFTER (9 runs)
must-coexist            12/13, 13/13, 13/13          13/13  x9      117/117
must-collide             7/8,   7/8,   7/8            7/8   x9       63/72
one-subject             19/20, 19/20, 19/20          19/20  x9      171/180
distinct                 5/5,   5/5,   5/5            5/5   x9       45/45
no reply leak            2/2,   2/2,   2/2            2/2   x9       18/18
```

All nine runs are identical, and **the only two failures anywhere in them are the two
pre-existing standing failures** — `(24, 38)` on collide and `[3, 4]` on one-subject, both
documented in §15 and §17.5 and both unchanged in text. No new failure appeared on any
axis.

Read the *before* column as the two separate baselines it is: the defect fired 2/3 on the
47-turn fixture version and 1/3 on the shipped 46-turn one (§18.5), and 9 runs of new code
against 3 of old is an asymmetric comparison. What survives that caveat is the direction
and the spread — the pre-fix runs disagreed with each other and the post-fix runs do not.

The number that decides this is **`must-collide`, not `must-coexist`.** The fix can only
ever *withhold* CONTRADICTION, so `[39, 40, 41, 43]` reaching 13/13 proves nothing on its
own: `single_valued: false` on every fact would score that perfectly while destroying the
store. `(42, 43)` resolving 9 times out of 9 is what says otherwise — a favourite IS a
preference, it holds one value at a time, and it still supersedes. That pair was written
into the harness *before* the fix, for exactly this.

A second thing the nine runs show, unprompted: the fix removed the **variance**, not only
the failures. Pre-fix, identical input scored 2/3 and then 1/3; post-fix, nine identical
runs. Encoding the judgment in the attribute's *name* left it to P1's phrasing, which
varies at temperature 0; answering a boolean does not.

### 18.12 The first version of the bullet caused a subject split, 9 times out of 9

The first `false` example list was `"likes", "interests", "skills", "office locations",
"todo list items"`. Nine runs with it:

```
must-coexist  13/13 x9        <- target axis, already clean
must-collide   7/8  x9        <- held
distinct       5/5  x9        <- held
one-subject   18/20 x8, 17/20 x1     <- REGRESSION, was 19/20
```

One group, failing 9 times out of 9:

```
FAIL [28, 29]  SUBJECT-SPLIT  miso | user
user -> ..., pets, ...
```

"I have a cat called Miso" filed as `the user :: pets`; "My cat Miso is 3 years old" filed
under `Miso`. The cause is in the example list: *"office locations"* and *"todo list
items"* are both collections **on the owner**, and P1 generalised the shape to a pet. That
is §15's rule 5 — a thing Y *contains*, which has properties of its own, is its own
subject — being pulled sideways by an example that contradicts it.

Replacing them with attributes belonging to the subject itself (`"languages spoken"`,
`"allergies"`), plus one sentence stating the answer never changes which subject a fact
belongs to, restored `one-subject` to 19/20 in all nine runs.

Note what that sentence is **not**: another restatement of a rule P1 is already ignoring,
which is arm E and which §17 measured as buying nothing. The subject rules were being
actively *contradicted* by an example in the same prompt, and the fix is to stop
contradicting them. That distinction is the difference between this working and arm E not.

The trade was also refused on principle, not only because it was cheap to fix. A split is
the recoverable direction — both facts survive and recall finds half an entity — against a
collision that leaves a true fact marked SUPERSEDED. §15 is explicit that trading one for
one is a regression rather than a wash, and banking a 9/9 split because the axis it paid
for looked good is exactly what that invariant forbids.

### 18.13 What this does not settle

- **`slots.py` is not the console.** The defect was found in a real typed session and the
  fix is measured on a 46-turn script. It needs a live two-column run against a correctly
  installed memore — which would also exercise §17, something no replay here could.
- **The bench numbers were argued inert, not re-measured.** `single_valued` defaults True
  and `bench/extract.py` supplies no such field, so the §3 oracle path is the pre-§18 code
  path by construction, and there is a test pinning it. That is an argument plus a unit
  test, not a re-run of sh_6k/sh_32k.
- **Nothing here measures a non-English turn.** `slots.py` is 46 English turns. The
  `single_valued` judgement and §15's subject rules both lean on English cues, so a report
  that another language "works" is not evidence either way until it has its own axis,
  scored separately so the numbers above stay comparable.
- **`(24, 38)` and `[3, 4]` are still failing**, unchanged, as they were before this work
  and for reasons that have nothing to do with it. §15 and §17.5.

### 18.14 The live two-column run — §16, §17 and §18 together, on real traffic

§18.13 listed "measured on a 46-turn script, not a live console run" as the open item.
This closes it. The same 27 user turns, driven through `POST /chat/stream` with
`memore: true` against a console restarted on a fresh graph (`memore_gateway4`) and a
verified `2a7097d` install — so the reader generated genuine replies and the write path
saw them. That is the first run to exercise §17 as well: the original run's replies were
never persisted, so every replay before this one passed `assistant_response=""`.

```
                        ORIGINAL RUN (37593fa)        THIS RUN (2a7097d)
                        c1          c2                c1          c2
CONTRADICTION            9           6                 1           1
  of which correct       2           2                 1           1
  of which WRONG         7           4                 0           0
DUPLICATE                3           1                 5           2
facts stored            48          46                49          50
```

**Zero wrong supersedes in either column.** The single contradiction per column is the
capital of the Netherlands, Amsterdam -> Den Haag, which is the one genuine correction in
the script.

The defect that started this, in both columns:

```
[Bud]  <-- 1 slot(s) with >1 live: likes
  #35 live (likes) Bud likes Lisa.
  #36 live (likes) Bud likes beer
  #37 live (likes) Bud likes the first Matrix movie
  #38 live (likes) Bud likes red gaming chairs
```

Note the slot is still named `likes` — a category, exactly as §18.4 predicted P1 would
name it. The fix does not stop that and was never meant to: `single_valued` makes the
naming stop mattering, which is why it was preferred over trying to make P1 name better.

The tea and lolly facts are all live in both columns (§16's same-batch guard), as are
Lisa's `location` and the Memore origin facts that collided once each in the original run.

**The DUPLICATEs are not §17 recurrences.** All five in c1 fire at one instant, 21:05:51,
alongside the two NEW facts of the same turn — turn 18, where the *user pastes the store
dump verbatim* into their message. P1 re-extracted the user's own text and consolidation
correctly answered DUPLICATE. §17.6 says this explicitly: nothing prevents the user
restating a fact the model showed them, and that is indistinguishable from the user
asserting it, as it should be. The remaining DUPLICATEs are turns 21-22 restating Den Haag.
No question turn stored anything in either column.

**A correction to §18.2's count.** The original run's `memory tools` supersede was scored
CORRECT there and it is not. Turn 25 ("I've turned on another set of memory tools that can
be used in parallel") and turn 26 ("they are separate for now... it is currently separate")
are compatible — one is about enabling a second set, the other about their sync state — so
retiring the first was wrong in both columns. The original run's tally is therefore
**2 correct / 13 wrong**, not 4 / 11. In this run they coexist, in c1 under one slot and in
c2 on two subjects.

### 18.15 What the live run shows that the harness does not

The opposite-axis pressure is now visible in a real session, and it is worth recording
because `slots.py` scores it but this is the first time it has been seen outside the
fixture:

- **Slot split, c1.** Bud's seating is stored twice — `#34 (chair) Bud sits in the Red
  chair in the Whangarei office.` and `#40 (seating_assignment) Bud sits on the Red chair
  in the Whangarei office.` Near-identical text, two slot names, both live. Lisa has the
  same shape: `#28 (employer) Lisa works for DDS in Amsterdam.` and `#29 (location work)
  Lisa works at the DDS office in Amsterdam.` A split costs recall and keeps both facts,
  which is the recoverable direction, but it is the cost side of §18 and it is real.
- **Subject split, c1.** `the red gaming chair` (#39, cost) and `the red chair` (#47,
  position) are one object under two subject keys.
- **A corrupted proper noun.** c1 #41 reads "Bud's work location is the **Whangacia**
  office." The turn says Whangarei and every other fact in both columns spells it
  correctly. This is a reader/extractor transcription error, nothing to do with
  consolidation, and it is the kind of defect no axis in `slots.py` can see — the fact is
  live, well-slotted, on the right subject, and wrong.

None of these is a consolidation failure. All three are arguments for the same thing: the
harness measures what it was built to measure, and a live run still surfaces classes of
error it cannot express.

## 19. Temporal expiry — spec, harness turns and refuse-list. Nothing built.

A fact can stop being *upcoming* without anybody contradicting it. "The user has a trip to
Lisbon on 29 August 2026" is true on the 17th and misleading on the 30th, and no later turn
will ever correct it — the correction is the calendar. Consolidation cannot see this:
freshness ordinals order *arrivals*, and nothing arrived.

This section specifies the fix and puts the assertions on record. **No code is written**,
by the §18 pattern: the instrument and the refuse-list first, the fix after.

### 19.1 Why it fits, and where it must NOT go

The comparison is arithmetic — a stored date against `now` — so it is legal in recall A–D
under the no-LLM invariant, exactly as `single_valued` is legal in `_classify`.

**It is a READ-time label, never a write-time state change.** A fact does not become false
when its date passes; its relation to *now* changes, and *now* is different on every read.
Storing an `expired` flag bakes in the moment it was computed and is wrong on the next
lookup. It would also collide with `invalid_at`, which means *superseded* — a specific
claim that a newer fact replaced this one, which is not what happened here.

So the label belongs in `assemble.py:27-29`, beside the two that exist:

```
[SUPERSEDED - was valid <from> to <to>]     # exists -- a newer fact replaced this
[valid as of <day>]                         # exists
[PAST - occurred <day>]                     # proposed -- the calendar moved, not the store
```

**PAST must not shut the gate, drop the fact, or reduce its score.** "Where did I go in
August?" needs that trip. Expiry that suppresses recall is deletion wearing a different
hat, and supersede-never-delete is the invariant this project is built on. The only thing
that changes is the framing handed to the reader.

### 19.2 The fields

Two, both answered by P1, both consumed mechanically — the `single_valued` shape:

```
occurs_at   ISO-8601 date, or null. The date the fact's EVENT happens. Null when the
            fact is not about a dated event -- which includes every standing property,
            and every fact whose text merely contains a number that looks like a year.
recurring   true when the event repeats (birthday, monthly renewal, weekly retro).
            A recurring event has no single date and can never be PAST.
```

The label applies when `occurs_at` is non-null **and** `recurring` is false **and**
`occurs_at < today`. Anything else is untouched.

Both default to the inert value — `null` and `false` — so a store written before the
fields, and the bench's cached extraction, behave exactly as they do now. Same argument as
`attribute == ""` and `single_valued = True`.

### 19.3 Why a field and not a parse of the fact text

Because the text varies, and that is measured. The 2026-08-17 live run gave one column each
way from **the same turn**:

```
c1  #18 (trip upcoming)      the user has a trip to Lisbon on 29th August 2026
c2  #47 (visiting_schedule)  Lisa is visiting the user on 2026-08-18      <- from "tomorrow"
```

c2 resolved a relative date to ISO; c1 left prose. That is the same failure as naming a
slot `preference` — the information exists, it is in a string, and the string varies. n=1
each way, so it is an argument rather than a rate.

The encouraging half: **P1 already resolves "tomorrow" against the turn**, and already
distinguishes recurring from one-shot implicitly — in *both* columns the birthday carried
no year and the trip did. The judgement is available; only its shape is unreliable.

### 19.4 The harness turns, and the rule that governs them

`slots.py` turns 46-50, appended, indices 0-45 untouched. **Nothing is scored yet** and
that is deliberate: `occurs_at` does not exist, so scoring written today would be scoring
against an interface nobody has seen output from — which is exactly what made the turn-45
containment pair vacuous (§18.7).

```
46-47  MUST_BE_PAST      "I'm flying to Porto on 12 May 2026"
                         "My passport expires on 4 April 2026"
48-50  MUST_NOT_BE_PAST  "My gym membership renews on the 1st of every month"
                         "Our team retro is on the last Friday of every month"
                         "My car is a 2019 Subaru"
   3   MUST_NOT_BE_PAST  "I was born in Den Haag"   (reused -- undated, permanently true)
```

`MUST_BE_PAST` is the direct analogue of `(42, 43)` in §18 and the reason this refuse-list
is not one-sided. Without a turn that must fire, `occurs_at: null` on every fact scores a
**perfect** refuse-list while implementing nothing — the same hole that made
`[39, 40, 41, 43]` meaningless on its own. Both turns are durable and both were framed as
upcoming when asserted, which is the case the feature exists for.

**THE DATE RULE, and it governs every turn ever added to this axis:**

> `MUST_BE_PAST` turns carry a FIXED date that is safely historical, so their verdict never
> changes. `MUST_NOT_BE_PAST` turns are RECURRING or UNDATED, never fixed-future.

A fixed future date expires eventually, and the harness would then start failing on a
calendar rather than on a code change. This is why turn 5 ("trip to Lisbon on the 29th
August 2026") and turn 17 ("Lisa's birthday is on the 24th of August") are *not* used here
despite being the obvious candidates: both are fixed dates that age, and turn 17 flips from
future to past on 24 August.

Turn 50 is the trap for the cheap implementation: a rule that scrapes four digits out of
the fact text reads "2019" as an event date and marks a Subaru PAST forever.

### 19.5 The open question — do not resolve it in `_SYSTEM`

**Should a completed, durable event carry PAST?** "The user started at DDS on 1 February
2026" is a dated event whose date has gone, and both readings are defensible:

- **Yes** — it is factually past, the label is accurate, and the reader gains the tense.
- **No** — it is a biographical record that is permanently true, and PAST framing invites
  the reader to treat it as stale the way SUPERSEDED does.

This is a decision, not a derivation. It is written here rather than settled because
committing a tentative answer as an *example* in `_SYSTEM` is precisely how §18.12 happened
— two innocuous example attributes split a subject in 9 runs out of 9. Decide it with a
measurement, then write the example.

### 19.6 Two smaller things that need stating, not defaulting

- **Day boundaries.** The user is in NZ (UTC+12/+13); "tomorrow" in the live run resolved
  to NZ-local. A date-only `occurs_at` compared against a UTC `now` is wrong by up to a
  day at the boundary, in the direction of marking things PAST early. The comparison needs
  a stated timezone rule, not whatever `datetime.now()` happens to give.
- **The motivating case is partly deferred scope.** "A new session past that date" is
  cross-session recall, which `recall-poc-spec.md` §5 defers; v1 is session-scoped. Within
  v1 this lands on long-running or resumed sessions. Worth knowing before it is built for
  a scenario the read path cannot reach yet.

### 19.7 What the pre-fix diagnostic does and does not tell you

`print_run` now prints the temporal turns and what P1 stored for them. It measures whether
a date survives into the fact text and in what shape — **not** `occurs_at`, which does not
exist. Do not read it as a baseline for the field itself.

Three runs, and the output was **identical in all three**:

```
[ 3] must-NOT-past  the user was born in Den Haag
[46] must-be-PAST   the user is flying to Porto on 12 May 2026
[47] must-be-PAST   the user's passport expires on 4 April 2026
[48] must-NOT-past  the user's gym membership renews on the 1st of every month
[49] must-NOT-past  the team retro occurs on the last Friday of every month
[50] must-NOT-past  the user's car is a 2019 Subaru
```

Two things worth carrying into the build:

- **All six turns extract, 3/3.** The fixtures are not vacuous — the check that removed the
  turn-45 pair (§18.7) passes here before any code is written. Turn 50 in particular
  survives P1's salience gate, so the "2019 is not an event date" trap is live rather than
  theoretical.
- **P1 passes explicit dates through as prose and normalises only when it must resolve.**
  Every date here stayed in the user's phrasing ("12 May 2026", "the 1st of every month"),
  where the live run's c2 emitted ISO for *"tomorrow"* — a relative date it had no choice
  but to compute. That refines §19.3: the variance is not random, it tracks whether P1 had
  to do arithmetic. Neither shape is a date the read path can compare, which is the point.

The five scored axes were unmoved by the new turns — `21/24, 39/39, 57/60, 15/15, 6/6`
across the three runs, with the same two standing failures and nothing else. Appending at
the end kept every earlier index meaning what it meant.

### 19.8 Built. What P1 actually emits, over 9 runs

`occurs_at` / `recurring` are in the P1 schema and consumed mechanically; the label lives
in `assemble.render_hit` behind `assemble.is_past`. The temporal axis is **scored now**,
which it was not in §19.4 — the order was field → 9 runs → read the output → *then* the
scorer, because scoring an unseen interface is what made the turn-45 pair vacuous.

Nine runs, and the field is the most stable thing in this harness:

```
[ 3] must-NOT-past   no date      9/9      the user was born in Den Haag
[46] must-be-PAST    2026-05-12   9/9      the user is flying to Porto
[47] must-be-PAST    2026-04-04   9/9      the user's passport expires on 2026-04-04
[48] must-NOT-past   recurring    9/9      gym membership renews on the 1st of every month
[49] must-NOT-past   recurring    9/9      the team retro occurs on the last Friday
[50] must-NOT-past   no date      8/9      the user's car is a 2019 Subaru   (1 stored nothing)
```

`MUST_BE_PAST` is **18/18** (2 turns × 9 runs) and `MUST_NOT_BE_PAST` **35/35** — 4 turns
× 9 runs = 36 rows, less the one row where turn 50 stored nothing and is reported
`NOTHING-STORED` rather than passed. 53 scorable rows, 53 correct, **zero** wrong dates
anywhere. Turn 50 — the trap for a four-digit scrape — never once
acquired an event date from "2019"; its single miss is P1 dropping the fact on salience,
which the `wrote nothing` diagnostic already covers and which is not a temporal failure.

**§19.3's variance argument is now settled in the field's favour, and visibly so.** The
pre-fix diagnostic recorded turn 46 as `the user is flying to Porto on 12 May 2026` —
the date living in prose. Post-fix it is `the user is flying to Porto` with `2026-05-12`
in `occurs_at`. The date moved out of the sentence and into a field the read path can
compare, which is exactly what the section asked for.

### 19.9 §19.5 is answered YES, by construction — recorded, not defaulted

§19.2's rule ("non-null AND not recurring AND `occurs_at < today`") applied to a completed
durable event — "the user started at DDS on 1 February 2026" — yields PAST. So
**implementing §19.2 verbatim IS the "Yes" branch of §19.5**; the two are not independent,
and shipping the code without saying so would have settled an open question silently.

Taken deliberately: nothing in the two fields distinguishes a biographical record from a
flight, so "No" needs a third signal from P1 that §19.2 does not have. No example was
written into `_SYSTEM` either way — that constraint from §19.5 still holds, and §18.12 is
why.

### 19.10 The control: `[9->10]` is NOT §19's, and the §18 baseline has drifted

Three runs in, a failure appeared that CLAUDE.md's baseline says should not exist:
`[9->10] MISSED user::likes vs user::likes`. Two new required fields in the P1 schema are
exactly the kind of change §18.12 showed can perturb §15's naming, so it was controlled
rather than argued: the same harness, `--runs 9`, on `HEAD~1` in a worktree, separate
graph.

**The arms provably ran different code**, which is worth showing rather than asserting:
the worktree was driven by the *treatment* venv's interpreter, and that venv carries an
editable `.pth` pointing at the working tree. `python -m` puts the CWD at `sys.path[0]`,
ahead of the `.pth`, so the worktree's own sources won — and the output proves it, twice
over. The baseline printed the pre-§19 header (`axis not scored until occurs_at lands`)
and the old two-column diagnostic with no date column, strings that exist only at
`HEAD~1`; and its turn 46 came back as prose, `the user is flying to Porto on 12 May
2026`, which the §19 prompt does not produce. Check this whenever a worktree control
shares a venv with its treatment — the failure is silent and would make both arms the
same code.

| axis (9 runs each)      | pre-§19 | §19       |
|-------------------------|---------|-----------|
| must-collide            | 58/72   | **60/72** |
| must-coexist            | 117/117 | 117/117   |
| one-subject coherent    | 171/180 | 171/180   |
| distinct (over-merge)   | 45/45   | 45/45     |
| no-reply-leak           | 2/2 ×9  | 2/2 ×9    |
| `[9->10]` failures      | **5/9** | **3/9**   |

**§19 is exonerated.** `[9->10]` fails *more often* without it, and the pre-§19 arm
produces the identical `likes`/MISSED mechanism in 3 of its 9 runs. §19 moves no axis
down and collide up by 2.

Two things this turned up that outlive §19:

- **CLAUDE.md's "the only two failures are `(24, 38)` and `[3, 4]`" no longer reproduces
  at `HEAD~1`.** `[9->10]` fails 5/9 *before* §19. That claim was true when §18 measured
  it and is not true now; it is a standing failure that postdates §18's measurement, and
  it should be read as such rather than as a regression in whatever ran last.
- **`(9, 10)` is a fixture that still depends on naming**, which is what §18 set out to
  remove. It resolves only when P1 happens to coin something single-valued
  (`language preference programming`); when it coins `likes`, `single_valued=false` is the
  *correct* answer by `_SYSTEM`'s own example list, and the pair becomes unsatisfiable.
  The two failure modes are one defect wearing two hats: `SPLIT` (2/9 baseline) when the
  namings differ, `MISSED` (3/9 baseline, 3/9 §19) when they agree on `likes`. Do not
  "fix" it by deleting `likes` from the example list without re-running both arms — that
  list is load-bearing for `[10, 11]` and `[39, 40, 41, 43]`.

The contention is worth stating: §19's runs 1–5 ran alone and 6–9 alongside the baseline,
which ran mostly contended. Same model at temperature 0, so this costs power rather than
biasing direction — but the two rates were not collected under identical conditions.

## 20. The `[9->10]` fixture: diagnosed, twice fixed, neither fix shipped

§19.10 left `(9, 10)` as a fixture that still turns on naming. This section diagnoses it,
records two prompt edits that **do** fix it, and explains why neither is in the tree.

### 20.1 Why the pair is unsatisfiable when P1 says `likes`

```
 9  "I like coding in Python"
10  "Actually I've changed my mind, I prefer Ruby now"     -> MUST_COLLIDE
11  "I hate green sweets"                                  -> MUST_COEXIST with 10
```

A probe replaying turns 0-11 (`scratchpad/probe910.py`) shows the resolving shape:

```
[ 9] NEW            attr='programming language preference'  sv=False   the user likes coding in Python
[10] CONTRADICTION  attr='programming language preference'  sv=True    ...preference is Ruby
```

`_classify` reads the *incoming* candidate's `single_valued`, so turn 10's answer decides.
Two things must hold: both turns name the same slot, and turn 10 answers `true`.

When turn 9 instead names the slot `likes`, both fail *correctly*. Turn 10 reuses `likes`
because the prompt asks for exact reuse, and then answers `single_valued=false` — which is
right for a slot called `likes`, and is what `_SYSTEM`'s own example list teaches. The pair
becomes unsatisfiable by construction. That is the whole defect, and it is **§18's category
attribute wearing new clothes**: `likes` and `interests` are `FactType` values sitting in
the `false` example list, in a prompt whose own rules say an attribute names a property and
never a category.

The parallel pair `(42, 43)` — "Bud's favourite programming language is Go" → "changed his
mind... Rust" — resolves reliably for exactly the reason 9/10 does not: turn 42 says
"favourite programming language", so P1 has no invitation to write `likes`.

### 20.2 Two arms, 9 runs each, against HEAD as arm A

- **B** replaced the `false` examples entirely: `languages spoken`, `allergies`,
  `qualifications`, `dietary restrictions`, and a non-`likes` closing line.
- **B2** was the minimal edit: delete `likes` and `interests`, keep `skills`,
  `languages spoken`, `allergies` and the original closing line.

| axis (9 runs)         | A (HEAD) | B        | B2         |
|-----------------------|----------|----------|------------|
| must-collide          | 60/72    | **63/72**| **63/72**  |
| must-coexist          | 117/117  | 117/117  | 117/117    |
| one-subject coherent  | **171/180** | 167/180 | 167/180  |
| distinct (over-merge) | **45/45**| **45/45**| **43/45**  |
| `[9->10]` failures    | 3/9      | **0/9**  | **0/9**    |
| new failures          | —        | `[32,34]` split 4/9 | `[30,31]` 2/9, `[28,29]` 2/9, **`(15,30)` OVER-MERGE 2/9** |

**Both edits fix the target completely** — 0/9, from 3/9 — and both lift collide to 63/72.
The diagnosis in §20.1 is therefore confirmed: the example list *is* the cause.

One figure in that table is not machine-printed: **arm B's `117/117` is summed by hand from
its nine per-run lines**, because arm B ran before the bug in §20.4 was found and its own
printed summary said `50/117`. The reconstruction rests on nine `13/13 intact` lines and the
absence of any `COLLIDED` line in the run, which is the same evidence that exposed the bug.
Arms A and B2 are as printed — A predates the temporal block, B2 postdates the fix.

### 20.3 Why neither shipped

Acceptance criteria were written down before the runs, which is the only reason this
section is not a rationalisation:

- **B2 is disqualified outright.** `(15, 30) OVER-MERGED lisa` in 2/9 runs, against 0/9 for
  A and B. A merge destroys a fact where a split only costs recall, and CLAUDE.md's
  asymmetry rule makes trading one for the other a regression, not a wash. The minimal edit
  was *worse* than the wholesale one, which is not what anyone predicted.
- **B fails the pre-registered coherence bar** (`one-subject >= 171/180`; it scores
  167/180, a new `[32, 34]` subject split in 4/9). It destroys nothing — coexist and
  distinct are untouched — so this is split-for-split: `[9->10]` leaves a contradicted fact
  live where `[32, 34]` scatters one subject across two keys. Roughly one-for-one in rate,
  opposite in kind, and no clear net gain.

So HEAD is unchanged. This is recorded rather than shipped because the *finding* is durable
and the *fix* is not: any edit to `_SYSTEM` perturbs subject naming somewhere else, which
is §18.12 for the third time, and two examples were enough to move an axis four points.

**Do not re-run this experiment expecting a different answer.** If `(9, 10)` is worth
fixing, the lever is turn 9's own ambiguity — "I like coding in Python" is genuinely both a
collection membership and a scalar preference, and only turn 10 disambiguates it — not
another pass at the example list. Both passes have been made and are on the record above.

### 20.4 A harness bug found by the same run, and worth more than the experiment

Arm B's summary printed `must-coexist intact 50/117` while all nine of its own per-run
lines said `13/13` and not one `COLLIDED` line existed. That is a **reporting bug** in
`print_run`, introduced with §19's temporal block: the block reused the name `ok`, still
live from the coexist block and read positionally by the return tuple, so `totals[2]`
accumulated the temporal count instead. 50 is 9 runs × ~5.5 temporal rows.

It only surfaced because the summary disagreed with its own detail — 50/117 is exactly the
shape of a catastrophic false-supersede result, and would have been reported as one. Fixed,
and now guarded by `test_print_run_reports_the_coexist_axis_it_computed`, which was written
vacuous first (an empty report makes both counts zero and passes either way, §18.7 yet
again) and then made to fail `assert 0 == 13` against the bug before being kept.

## 21. Where this stands, and what is left

Written 2026-08-18, to answer "is this useful / an advance, and what is left" without
re-deriving it from twenty sections. Nothing here is new measurement — it is §0, §3, §7,
§9 and §20 collected and ranked. Read this first in a new session; read the cited sections
before acting on any of it.

### 21.1 The three-tier verdict

Do not compress these into one claim. The earlier "beats the field by 40 points" framing
came from exactly that compression.

- **Not an advance: the deterministic freshness primitive.** arXiv:2606.01435 published the
  same core idea (LLM extraction + deterministic `max(serial)`) in May 2026. Single-hop is
  **parity** — 0.980–1.000 SubEM against 0.948, which §0 calls inside the noise of an
  uncontrolled setup comparison, with the ceiling at 98 because 2 of 100 32k questions are
  unwinnable.
- **A real margin: multi-hop.** 0.800 SubEM against 0.515, by a mechanism with no
  counterpart in their recipe (§8's chain walk — no LLM, no embeddings). Three caveats
  belong with the number every time it is quoted: it is `mh_6k` only; the reader is a local
  `gemma4:12b` against their gpt-4o (which cuts *for* us, but is not controlled); and
  **chain expansion ships off** (`expansion_hops = 0`), so this is not the shipped
  conversational default's number. §9 calls expansion a no-op in the target regime.
- **Genuinely ours and defensible:** write-time rather than read-time resolution (the store
  is always in a resolved state; the read path does no freshness reasoning at all), running
  entirely on local models, and the subject-identity work in §9 and §10.

### 21.2 The headline finding is the convergence, not a benchmark number

arXiv:2606.01435 was revised on 2026-08-02 to attribute its gains primarily to separating
evidence extraction from policy execution "rather than the freshness mechanism alone."

§3 reached the same conclusion independently and from the opposite direction: consolidation
was correct on every subject group in every run at both corpus sizes, and every residual
error was extraction naming one subject two ways.

**Two measurements from different directions agreeing that the bottleneck is subject
identity rather than freshness is a stronger result than either alone.** It is the answer
to "what did this project find", and it is what ranks the open work below.

### 21.3 It works well *on the fixtures it has*

The single largest untested surface in the shipped read path, from §7: **the bench never
exercises `recall()`.** `run_deterministic` calls `store.hybrid_search` and `build_block`
directly, so `score_floor`, the lookup timeout and the failure-safety path play no part in
any headline number. The gate is covered only by unit tests against fakes and the §6 demo
trace. A production integration hits this first.

### 21.4 What is left, ranked by evidence rather than by ease

1. **Subject identity.** Named by two independent measurements (§21.2) as *the* bottleneck.
   §9's residual is structural, not a tuning miss: `subject_min_competitors = 2` requires a
   genuine crowd, and most conversational relations hold exactly two subjects — one
   competitor each, below the threshold, unpoliced. Crowded-chat wrong-entity FPR sits at
   **0.462** (from 0.846). Setting the knob to 1 blocks 69% and costs **8% conversational
   recall**, measured, and the losses are genuine paraphrases. §9's closing line names what
   is actually missing: a signal that "milk" and "coffee preference" are the same subject
   while "mobile app" and "web app" are not. Neither similarity nor lexical overlap carries
   it.
2. **Run `recall()` under the bench**, closing §21.3. Smallest of the four and the one a
   gateway integration would want first.
3. **Scale.** 64k/262k unmeasured; only the oracle ran at 32k, and `retrieval_hit` is
   expected to fall there (§7). §7 argues the marginal value stays low until the P1
   canonicalization bottleneck is addressed — which is item 1 again.
4. **Cross-session recall**, deferred by `recall-poc-spec.md` §5, along with the async job
   machinery, rolling-summary key synthesis and the queryable audit log. §19.6 flags that
   temporal expiry's motivating case ("a new session past that date") lands here, so §19 is
   built against a scenario the read path cannot fully reach yet.

### 21.5 The standing harness failures are evidence for that ranking

`[3, 4]` (9/9), `(24, 38)` (9/9) and `[9->10]` (3/9) are **all** subject- or slot-naming
failures. They are instances of item 1 rather than a separate defect list, which is itself
support for the ranking. §20 further established that `[9->10]` cannot be fixed from the
`_SYSTEM` example list without paying for it elsewhere — both passes are on the record, so
start from turn 9's own ambiguity or not at all.

---

## 22. A1: `single_valued` moves onto the slot. Built, controlled, and inert.

`identity-and-gate-spec.md` A1. `single_valued` was a schema property of the slot being
re-derived by P1 every turn, so a slot's arity was only as stable as a model §18.5
measured disagreeing with itself. §18.11 removed that variance by asking the boolean
instead of encoding it in the attribute's name; A1 removes the re-derivation by recording
the first answer and reading it thereafter.

### 22.1 What shipped

A `:Slot` node keyed on `(session_id, subject_key, attribute)` — the same pair `_competing`
filters on, because that pair *is* the slot whose arity this is. The key was the one design
question worth settling from evidence rather than taste, and §11 settles it: the residual
error there is `the user :: creation location` holding both "born in Den Haag" and "wrote
the memory system in Den Haag". What needs correcting is **that slot on that subject** —
not `creation location` wherever it appears, and not everything about `the user`. A1's own
phrase "correctable in one place" means one node instead of a re-derivation per turn.

- `ensure_slot_schema` is `ON CREATE SET` in Cypher, so A1's "do not let a later fact
  overwrite the stored value implicitly" is enforced by the query rather than asserted in
  a comment. `set_slot_schema` is the single deliberate correction path.
- `_classify` takes the arity as a plain argument (`None` → fall back to the candidate's
  own value), not as a mutated candidate. It stays a pure function of its arguments, no
  LLM enters the decision, and `test_no_llm_in_the_consolidation_decision` passes
  unchanged — the same four things §18 asked not to undo.
- A correction rewrites no fact. Arity is read at classification time and never stored on
  a `StoredFact`, so it takes effect on the next fact in the slot and leaves the ones
  already classified alone. Re-deciding those would be exactly the implicit revision
  `ensure_slot_schema` refuses.
- Disagreements between the record and what P1 emitted are logged at INFO and **not acted
  on**, per A1.
- Inert where it has no evidence, the same argument `attribute == ""` and
  `single_valued = True` already make: an empty attribute records nothing and looks nothing
  up, so old graphs and `bench/extract.py` — which name no attribute at all — reach neither
  path, and every §3 oracle number stands unchanged. A store without the three methods
  degrades to the pre-A1 behaviour with one INFO line (spec invariant 2), proven against a
  store class that genuinely lacks them rather than a monkeypatch.
- `clear_session` deletes `:Slot` nodes too. Nothing else lists them — `sessions()` counts
  Facts — so orphans in a shared graph would accumulate completely invisibly.

Covered by six unit tests and a real-graph round trip proving `ON CREATE SET` refuses the
implicit overwrite while `set_slot_schema` performs the deliberate one.

### 22.2 It is behaviourally inert on the 51-turn script, and that is the result

```
slot arity   40 recorded (5 multi-valued), 0 P1 disagreements     x9 runs
```

Per run: 52 candidate facts named a slot, across 40 distinct slots — so **12 reuses per
run, 108 across nine runs, and zero disagreements.** The recorded value was never once
different from what P1 would have supplied anyway.

Report the denominator, not the bare zero. 0-of-108 is a finding; 0 alone is consistent
with the measurement being broken. The specific way it could have been broken was checked:
if `_slots_for` had returned `None` — a store capability check failing — the lookup is
skipped and no disagreement can *ever* be logged, which looks exactly like a genuine zero.
Neither fallback line (`store has no slot_schemas`, `could not read slot schemas`) appears
anywhere in the nine runs, and 40 slots were genuinely recorded per run, so the zero is
P1 agreeing rather than A1 not running.

So §18.11's finding is stronger than it claimed: asking the boolean did not merely reduce
the variance in `single_valued`, it removed it. A1 removes a re-derivation that turns out
to be stable. Its deliverable is therefore **permanence and the correction path**, not a
score — and the two axes A1 could have moved are `must-coexist` and `distinct`, which
CLAUDE.md already warns are the one-sided ones scored perfectly by doing nothing.

### 22.3 The baseline moved, and it is not A1's

Nine runs of A1 came in below §19's published figures on two axes. **A control at the
pre-A1 commit (`5809b79`, worktree, separate graph, `--runs 9`) reproduces the treatment
bit-identically** — every axis, every run, every failing pair:

| axis (9 runs each)   | §19 published | pre-A1 control | A1 |
|----------------------|---------------|----------------|-------------|
| must-collide         | 60/72         | **54/72**      | **54/72**   |
| must-coexist         | 117/117       | 117/117        | 117/117     |
| one-subject coherent | 171/180       | **162/180**    | **162/180** |
| distinct (over-merge)| 45/45         | 45/45          | 45/45       |
| no-reply-leak        | 18/18         | 18/18          | 18/18       |
| temporal             | 6/6 ×9        | 6/6 ×9         | 6/6 ×9      |

**A1 passes its acceptance criterion against the control, which is the comparison that
means something.** A reader diffing 54/72 against A1's quoted 63/72 would read a regression
that does not exist — and note that 63/72 was already stale when A1 was written: §19.10's
own control re-measured the shipped code at 60/72. The bar is now pre-registered in
`slots.py` so the next change does not repeat this.

The arms provably ran different code, and the check was designed *before* the runs rather
than reconstructed after, which is §19.10's lesson applied: pre-A1 `slots.py` cannot print
the `slot arity` diagnostic line, and the control's output contains none. That is a
positive tell from the output itself, not an assertion about interpreters.

Three independent arguments say A1 is not the cause, and they do not depend on each other:

1. It changed **zero** classification decisions (§22.2). A decision can only differ where
   the record differs from P1, and none did.
2. `one-subject` is **structurally beyond its reach**. It measures which subject P1 *names*;
   A1 touches only the arity `_classify` consumes, downstream of naming.
3. The **pre-A1 control produces the same numbers.**

### 22.4 §19's numbers do not reproduce, and "drift" is not the established explanation

Say only what was measured: **the numbers do not reproduce**. The serving state is ruled
out as the cause; whether this is a change over time or a difference in *sampling regime*
is open, and the evidence leans to the second.

The obvious explanation is ruled out. Ollama is serving exactly what CLAUDE.md documents —
`gemma4:26b` at 32768 ctx and `mxbai-embed-large` at 512, both pinned — so this is not the
`num_ctx` / `keep_alive` reload the commands section warns about. `_SYSTEM` is unchanged in
the tree, and still carries §18.12's fix verbatim (`"languages spoken"`, `"allergies"`, and
the sentence that the arity answer never changes which subject a fact belongs to).

Two failures account for the whole delta, both 9/9 under *both* arms:

- **`[9->10]` MISSED**, `user::likes` vs `user::likes`, at **9/9**. §19.10 measured this
  same mechanism at 3/9 with §19 and 5/9 without. Same failure, higher rate.
- **`[28, 29]` SUBJECT-SPLIT, `miso | user`**, with a `pets` slot on `the user` — 9/9. This
  is §18.12's failure exactly, down to the slot name. §18.12 attributed it to the
  `single_valued` example list and reported it fixed 9/9 after replacing the two
  owner-collection examples.

**§18.12's claim does not currently reproduce, and the reason is now open.** Three
readings, and the third is not the obvious one:

1. P1 drifted over the two days.
2. The example list was never the mechanism, and 9/9-clean was luck at n=9.
3. **§18.12's fix was validated against a mixture and is now evaluated at a point.** See
   below — today's nine runs are one deterministic sample, and §19's were not. A fix that
   was right on the samples it saw can land on a different single point without either
   drifting or having been wrong.

Do not resolve this by editing the example list: §20 put both those arms permanently out
of bounds, and this section adds no evidence that reopens them.

**All nine runs were byte-identical on every scored line, in both arms** — and that is
what makes reading 3 above the better one. §18.5's whole argument for `--runs 9` was that
identical input scores differently run to run. That variance is currently absent while the
*level* has shifted, which is the signature of a model pinned in memory (temperature 0, no
reload) rather than of a stable extractor.

It follows that today's nine runs are closer to n=1 than to n=9 — and, crucially, that
**§19's figures and today's are not the same kind of number.** §19.10's control reported
`[9->10]` at 5/9 and 3/9, so its nine runs were *not* identical: they were nine samples of
a varying process, averaged into one figure. Today's are one deterministic point. A
difference between an average over a mixture and a single point is not evidence of change
over time, which is exactly why this section does not claim drift.

### 22.5 What this means for what comes next

- **Quote §15–§20's slot numbers as of their measurement date, not as current.** Two of
  them do not reproduce today under unchanged code and an unchanged serving state.
- **A2's acceptance criterion is currently unfalsifiable — and inverted.** It asks for
  "nine identical runs on all axes at k=5"; k=1 already delivers that, so the only way A2
  could fail as written is by *introducing* variance. Restating it against the pre-pinned
  regime's variance does not help either, since nobody can currently reproduce that regime.
  A2 needs a bar the present conditions can fail before it is worth starting.
- **A1 does not block on any of this.** It changed no decision, its acceptance is a
  same-conditions comparison, and the control supplies it.

---

## 23. §21.3 closed: `recall()` under the bench. The gate is fine; the budget is not.

Measured 2026-08-20, `gemma4:26b` @32768 and `mxbai-embed-large` @512 both pinned,
`MEMORE_GRAPH=memore_mxbai`, `--no-reader` (so no SubEM here — these are retrieval and
latency numbers).

**One run per arm, except the shipped 32k arm, which was run three times** — because a p95
is a tail statistic and one sample of it was about to retire two claims elsewhere. The three
runs: **272.4 / 283.4 / 284.1ms**, with `retrieval_hit` 0.960, `retrieval_any` 0.980 and
`gate_open` 1.000 identical in all three. So the breach is robust and the precision is
~±5%; quote it as **≈270–285ms**, never as a single figure. Retrieval numbers elsewhere in
this section are n=1, and §22.4's caution applies to them.

### 23.1 The item, and what the mechanism needed

§21.3: the bench never routes a headline number through `recall()`, so `score_floor`, the
lookup timeout and the failure-safety path are in none of them. `--via-recall` already
existed; nothing had been run with it, and in that branch the raw `hybrid_search` was
computed and **thrown away**.

That waste is now the **control**. It is the same retrieval `recall()` performs — `embed_one`
normalizes, `blend(v, None, alpha)` is idempotent, and `RecallConfig.k` is `--k`, so the two
query vectors are identical — which makes it the right comparator for the number the item
exists to produce:

```
gate_shut_but_retrievable   the gate returned nothing on a question whose gold WAS in the
                            store's live top-k
```

`_live_first` on both sides: a gate shutting on a question answerable only by a superseded
fact cost nothing, and counting it would credit the gate with a loss it did not cause.

**The expectation was pre-registered in the module docstring before the first run** — that
the gate should open on nearly all of single-hop, and that if it did, this would *not* be a
validation of `score_floor` but §11's argument applied to the gate: a corpus that cannot
express the failure a threshold exists to prevent is not evidence about that threshold.

### 23.2 The gate costs nothing on single-hop, at both scales

| arm | sh_6k `retrieval_hit` | `_any` | A–D p95 | sh_32k `retrieval_hit` | `_any` | A–D p95 |
|---|---|---|---|---|---|---|
| raw top-k (no gate) | 0.900 | 1.000 | — | 0.920 | 1.000 | — |
| `recall()`, shipped | **0.970** | 1.000 | 151.1ms | **0.960** | 0.980 | **≈270–285ms** |
| `recall()`, `subject_check=False` | 0.910 | 1.000 | 103.2ms | 0.910 | 0.990 | 114.2ms |

`gate_open_rate` is **1.000** and `gate_shut_but_retrievable` **0.000** in every arm, at both
scales. Zero lookup timeouts, zero lookup failures.

So the expectation held, and the pre-registered reading applies: **this is weak evidence for
`score_floor`, not a validation of it.** FactConsolidation questions are short, third-person
and lexically near-identical to their answer facts; the corpus cannot produce the off-domain
turn 0.57 was chosen to keep out. What it *does* establish is that the shipped floor is not
silently destroying the benchmark the headline numbers come from, which was the open risk.

**The shipped read path is better than raw top-k on top-1** — +7.0 at 6k, +4.0 at 32k — and
the third row says the whole improvement is the **subject check**, not the gate. With the
check off, `recall()` and raw top-k are the same to within a point at both scales, which is
what a gate that opens 100% of the time should look like.

Two points of `retrieval_any` are lost at 32k, and the metric above cannot see where: the
gate never shut on an answerable question, so the loss is **inside an open gate**. The third
row separates it — one point to the subject check (1.000 → 0.990), one to the gate or
`inject_token_budget` (0.990 → 0.980).

### 23.3 The finding is latency, and §21.3 pointed at the wrong risk

**A–D p95 is ≈270–285ms at 32k (n=3), against a 200ms budget.** §21.3 predicted "a production
integration hits this first" and was right about that; it expected the gate to be the hazard,
and the gate is the part that works.

Component medians, same store, same session, 8 queries each:

| | sh_6k (455 facts) | sh_32k (2310 facts) |
|---|---|---|
| `embed_one` | 66.5ms | 66.5ms |
| `hybrid_search` | 20.4ms | 13.5ms |
| `subject_view` | 20.0ms | **102.6ms** |
| `recall()` full | 127.3ms | 211.3ms |
| `recall()` without `subject_check` | 94.2ms | **79.6ms** |

**`subject_view` is O(session) and it is the entire breach.** 5.1x the facts, 5.1x the time —
one full session fetch, per query, rebuilding the vocabulary each time. Everything else is
flat or better at 5x scale: the vector index does its job, and `recall()` with the check off
is ~80–115ms at *both* sizes.

That also explains why nothing caught this. §5's ~90ms was measured on calibration fixtures
holding tens of facts, where an O(session) scan is invisible. It is not that the figure was
wrong; it is that the quantity it measured does not vary with the thing that matters, and no
fixture was large enough to show it. Same shape as §13's fixture drift — **a fixture that
cannot express the failure cannot measure against it**, now for the third time in this file.

### 23.4 B6 gets a price, and the performance addendum gets two corrections

**B6 ("`subject_min_competitors` retires") now has a latency argument nobody had made.** §9
measured the check's accuracy trade conversationally — crowded-chat hard-negative FPR 0.846
→ 0.462, and 8% recall to close it further. Its cost on the bench, first measured here:

```
subject_check ON   +6.0 / +5.0 top-1 (6k / 32k)   -1 point retrieval_any at 32k
                   +48ms at 455 facts   +158ms at 2310 facts, and rising linearly
```

**Only the latency half of that transfers to B6, and the accuracy half must not be quoted
against B6's bar.** B6 is scored on *chat_crowded hard-negative FPR* — right relation, wrong
entity — and FactConsolidation is one-attribute-per-subject by construction, so it cannot
produce a hard negative at all. The +5.0/+6.0 is the check improving top-1 on a corpus where
wrong-subject confusion is structurally rare; it says nothing about the case the check exists
for. This is §23.2's caveat about `score_floor` again, one axis over: a corpus that cannot
express the failure is not evidence about the fix. **§9's 0.846 → 0.462 remains the only
measurement on B6's actual axis.** What §23 adds is the price, which transfers anywhere.

That is a check worth having and a check that cannot be paid for as written. It is the
strongest available argument for B6's "fold it in as a graded feature" over the current veto
— and, before any of that, for **C6** in `performance-addendum.md`: cache `subject_view`,
which is session state that changes only on write and is currently rebuilt on every read.
C6 is deliberately *not* folded into this section. It is a fix for what §23 found, it has a
correctness question of its own (`subject_view` returns live and superseded facts while the
vocabulary statistics are computed over live subjects only, so a wrong invalidation silently
changes `subjects.admits()` — the gate's precision path), and §10's arrival-order invariant
is a standing warning that cached vocabulary state has bitten this codebase before. It gets
its own pre-registered bar.

Two figures in `performance-addendum.md` do not survive this:

- **"~90ms p95 against a 200ms budget… roughly 110ms unused."** Measured at ≈270–285ms p95
  at 32k (n=3), i.e. ~70–85ms *over*. The headroom C2 spends on a reranker exists at conversational scale and
  does not exist at 32k. C2's own "A–D p95 stays under 200ms" acceptance is now the binding
  constraint rather than a formality.
- **"FalkorDB lookup 2.6–4.3ms"**, cited in the "do not optimise" list. Measured at 13.5–20.4ms.
  The *conclusion* survives — it is still not the dominant cost and still not worth optimising
  — but the number is wrong by 3–5x and should not be quoted.

### 23.5 What this does not cover

One surface of the three §21.3 names, and the counters exist to keep that honest rather than
implied:

- **`score_floor`** — genuinely exercised, 200 questions across two corpora. Weakly, per §23.2.
- **Lookup timeout** — *touched, not tested*. 0 timeouts, which is the expected result when
  the lookup answers in 13–20ms against a 180ms limit. Zero here is not evidence the timeout
  path works.
- **Failure safety** — **not exercised at all.** Still covered only by `FailingStore` /
  `SlowStore` in the unit suite. `recall()` never raises by contract, so a store that timed
  out and a store that answered are indistinguishable from the return value; the degradation
  counter exists so that a run where the store fell over on every question cannot be read as
  a gate that simply never opened.

Multi-hop is unchanged and already measured: §8's arm B ran `recall()` gate-only on mh_6k and
`retrieval_any` fell 0.400 → 0.050, for the structural reason the chain-walk invariant states.
Nothing here revisits that.
