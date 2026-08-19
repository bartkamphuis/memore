# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

**Starting a session? Read `RESULTS.md` §21, then §22.** §21 is the standing assessment —
the three-tier verdict on what here is and is not an advance, the convergence finding that
the bottleneck is subject identity rather than freshness, and the ranked list of what is
left. It exists so none of that gets re-derived from twenty sections, and it is the one
section to read before deciding what to work on. **§22 qualifies it**: A1 of
`specs/identity-and-gate-spec.md` is built and controlled, and the same measurement found
that two slot-harness numbers from §15–§20 no longer reproduce under unchanged code and an
unchanged serving state. Read §22.4 before quoting any harness figure or starting A2.

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
0.200 to **0.760** exact-match (0.800 SubEM). It is off by default (`expansion_hops = 0`)
— see the invariant below before turning it on, and note that "off by default" is part of
how the number must be quoted: it is not the shipped conversational config's result, and
§9 calls expansion a no-op in that regime. The comparison is **0.515** (arXiv:2606.01435,
gpt-4o), not the single-digit Graphiti figure this line used to cite — RESULTS.md §0.

Subject aliasing is built and measured (RESULTS.md §10): `memore/aliases.py` merges two
namings of one subject when they differ only by *generic relation words*, decided by
document frequency across the session's subjects. This is the gated form of the subset
merge §3 rejected, and it closes §3's last open item — sh_32k oracle 0.940 → **0.960**
with **zero** over-merges, reaching the score ungated subset-merging got while still
refusing every merge §3 objected to. On by default; inert below 200 subjects, so
conversational stores are untouched.

Subject co-reference is fixed at P1, not with a matcher (RESULTS.md §15). One entity was
being stored under several names — `Lisa` / `the user's sister Lisa` / `the user` — because
the prompt asked for the shortest phrasing *and* for reuse, which conflict exactly when the
stored label is long. Five ordered naming rules replace them; the slot harness grew nine
turns and an over-merge refuse-list first. Rules 1–4 took subject coherence from **39/51 to
51/51 with the refuse-list unmoved at 9/12**. Rule 5 then went through two versions, and on
the final 36-turn script the shipped one **dominates**: 18/18 collide, 33/33 coexist, 54/57
coherent, **15/15 distinct** against v1's 12/15 at identical coherence. The apparent
3-for-3 trade seen at 34 turns was an artifact of the script holding one fact about the
merged-in entity. Two shapes of split existed and only one was reachable by a surface merge
rule, which is why no merge rule was built — read §15 before proposing one.

Consolidation was corrected twice more in RESULTS.md §16, from the first trace of a real
typed session (a two-column gateway console run, 24 turns per column, identical input).
Both defects mislabelled a **still-true** fact SUPERSEDED — nothing was deleted — and
neither is expressible in FactConsolidation, which feeds one fact per turn: candidates from
one utterance were superseding each other on an ordinal that recorded array position, and a
shorter restatement was retiring the fuller fact it was contained in. The third fix that
looked obvious — canonicalising drifting attribute names the way §10 canonicalises subjects
— was measured against that run and **refused**; §16.5 has the numbers, and also corrects
the reading that attribute drift happens within a session at all.

§16 left one defect open and §17 closes it: the assistant's reply was inside the block P1
was told to extract from, so on question turns the model stored things the user never said
— four of the run's seven DUPLICATEs are the exact string recall injected on that same
turn, and one turn stored a *join* over two of them. The channel was isolated by a
three-arm control (the hint list is not one; prior turns cannot be, at
`extract_window_turns=3`) and closed by moving text rather than adding a rule.

RESULTS.md §18 found P1 naming a slot after the fact's *category* (`preference` — literally
a `FactType` value) instead of the property, so several simultaneously-true facts landed in
one slot and retired each other. It is **fixed**: `CandidateFact.single_valued` is a
required boolean in the P1 schema, consumed by `_classify` as a field lookup. Measured over
`--runs 9` (three is not enough at this defect's rate), all nine runs identical, and the
only two failures anywhere in them are the two pre-existing standing ones — `(24, 38)` and
`[3, 4]`. The fix also removed the *variance*: pre-fix, identical input scored 2/3 then
1/3; post-fix, nine identical runs. Confirmed live in §18.14 — the same 27-turn console
script through the real gateway path, both columns, **zero wrong supersedes** against the
original run's 13, with §16/§17/§18 exercised together for the first time. Still not
settled: the bench is argued inert (default True + a unit test) rather than re-run.

RESULTS.md §19 is **built** (§19.8–§19.10): temporal expiry, where a fact stops being
*upcoming* without anyone contradicting it and consolidation cannot see it, because nothing
arrived. `occurs_at` / `recurring` are P1 schema fields consumed mechanically, and PAST is a
read-time label in `assemble.render_hit`. Over 9 runs the field is the most stable thing in
the harness — `MUST_BE_PAST` 18/18, zero wrong dates, and turn 50's "2019 Subaru" trap never
once fired. §19.5's open question is answered **YES by construction** and recorded as a
decision in §19.9: §19.2's rule labels a completed durable event PAST, so shipping the rule
settles the question whether or not you say so. Still no example in `_SYSTEM` either way.

**§19 was controlled against `HEAD~1`, and that control corrected a claim in this file.**
`[9->10]` fails **5/9 pre-§19** and 3/9 with it, so "the only two failures anywhere are
`(24, 38)` and `[3, 4]`" was true when §18 measured it and is **not true now** — read a
`[9->10]` failure as a standing one, not as a regression in whatever ran last. It is also a
fixture that still depends on naming: it resolves only when P1 coins something single-valued,
and when it coins `likes`, `single_valued=false` is *correct* by `_SYSTEM`'s own example list.

**§22 overtakes that list again, and this time §18.12's fix is what stopped reproducing.**
On 2026-08-20, under unchanged code and with Ollama serving exactly what this file
documents, the standing failures are **four**, each 9/9: `(24, 38)`, `[3, 4]`, `[9->10]`
(up from 3/9), and **`[28, 29]` `miso | user`** — §18.12's owner-collection subject split,
back at 9/9 though `_SYSTEM` still carries the fix verbatim. **Do not read this as drift**
— that is a claim about time §22.4 deliberately does not make. All nine runs were
byte-identical in both arms where §19.10's were not, so today's figures are one
deterministic point and §19's were an average over a varying process: not the same kind of
number, and a difference between them is not evidence of change. §22.4 leaves the cause
open and **§20 still forbids reopening the example list to find out**. The harness bar is now pre-registered in `slots.py`; use it rather than
a figure quoted from a section, because two of those no longer describe HEAD.

**That deletion has now been tried, twice, and neither version shipped — RESULTS.md §20.**
Removing `likes`/`interests` from the `false` examples *does* fix `[9->10]` completely
(0/9 from 3/9) and lifts collide to 63/72, confirming the diagnosis. But the wholesale
replacement costs 4 points of subject coherence (`[32,34]` split 4/9), and the *minimal*
edit is worse still — it produces a `(15,30)` subject OVER-MERGE in 2/9, a destroyed fact,
which the asymmetry rule disqualifies outright. Do not re-run this expecting a different
answer; if `(9, 10)` is worth fixing, the lever is turn 9's own ambiguity, not the example
list. Both passes are on the record.

Still unbuilt, all deliberately deferred by `recall-poc-spec.md` §5: the async job
machinery, rolling-summary-vector key synthesis, the queryable audit log, and
cross-session recall. The §14 200ms P95 latency budget **is now met** (~96ms P95, ~146ms
with chain expansion on) after calibrating `score_floor` over a real query set and
switching the embedder — RESULTS.md §5. The open weakness is wrong-subject recall, not
latency: see the scalar-floor limit below, and **§21.4 for what that costs and what it
would take to fix** — subject identity is ranked first there by two independent
measurements, and this deferred list is ranked fourth.

**§5's calibration is superseded by §13.** The shipped pairing is now `gate_on="cosine"`
with floor **0.57**, not the fused score at 0.48. §5's numbers were measured against
calibration fixtures that had drifted from the write path and have since been regenerated
from the real extractor, so read §12 and §13 before quoting any gate figure from §5.

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

# Slot + subject fidelity (RESULTS.md §11, §14, §15, §17, §18). The instrument for any change to
# the P1 prompt -- the FactConsolidation bench cannot express these failures at all.
# Read the runs SEPARATELY: P1 varies at temperature 0 and the variance is the finding.
MEMORE_GRAPH=memore_slots uv run python -m memore.bench.slots --runs 3

# score_floor calibration -- re-run after ANY change to the embedder or the floor
uv run python -m memore.bench.gen_calib_fixtures --write   # regenerate fixtures FIRST
uv run python -m memore.bench.calibrate --configs all --reingest
# --analyze re-scores stored observations. Point it at a run from §13 or later: earlier
# JSON has no `top1_cosine`, which defaults to 0.0 and silently scores the cosine arm as
# if it were the fused one -- a wrong answer that looks like a right one.
uv run python -m memore.bench.calibrate --analyze data/results/calibration_gate.json

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

Source docstrings cite six design documents by bare filename:

- `recall-poc-spec.md` — scopes the standalone terminal PoC this repo builds. The entry point.
- `recall-stage-spec.md` — the read path (recall stage A–D), store choice, config, invariants (§13), definition of done (§14).
- `recall-writepath-spec.md` — the write path (P1 extract → P2 consolidate → P3 commit); expands stage-spec §8.
- `recall-stage-test-spec.md` — test suites and fixtures for the read path.
- `identity-and-gate-spec.md` — handoff, downstream of §21.2's finding. Part A (write lane:
  identity, A1–A6) and Part B (read gate: scalar floor → profile, B1–B6). **A1 is done —
  RESULTS.md §22.** Its acceptance criteria are pre-registered, and two have been corrected
  against measurement since: A1's stale `63/72`, and A2's bar, which was met at k=1 and could
  only fail by introducing variance.
- `performance-addendum.md` — serving, latency and ranking quality; explicitly no change to
  consolidation, identity, or the gate's decision logic. Its "do not" list is restated in the
  invariants above, because it is the most re-derivable content in the set.

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
- **A subject is a TOPIC; the slot that holds one value at a time is `attribute`.** Facts
  compete on `(subject_key, attribute)`, and only same-slot facts can supersede. Before
  RESULTS.md §11 the supersede loop ran over every live fact on the subject, which meant
  six independently-true facts about one topic knocked each other out: measured across
  three real sessions, **18 supersedes fired and 1 was correct**. Do not "simplify" the
  competing-set filter back to `live` — that single word silently reverts the whole fix.
  `attribute=""` means unspecified and collides with everything, which is what keeps old
  graphs and the bench inert; do not change it to match nothing.
- **The bench cannot validate slotting, and unchanged sh/mh numbers are not evidence
  either way.** FactConsolidation is one-attribute-per-subject *by construction*, so the
  §11 defect is structurally inexpressible in it. Those runs are a regression guard only;
  the real fixtures are the trace-derived tests at the end of `tests/test_consolidation.py`.
- **Subject naming is five ORDERED rules in the P1 prompt, and rule 2 is load-bearing.**
  A named entity takes its bare name (`Lisa`, not `the user's sister Lisa`) — but the
  *speaker* is always `the user`, never their name, or "My name is Bart" splits the
  store's most-loaded subject in two. Do not "simplify" the rules to just "use the
  proper name": that one deletion is the whole failure. RESULTS.md §15.
- **Subject splitting and subject OVER-merging are two axes and both must be reported.**
  `memore/bench/slots.py` scores four things, in two opposed pairs: slot split vs slot
  collision, and subject split vs subject over-merge (`MUST_DISTINGUISH`, the
  hand-written refuse-list). Every fix for splitting pushes toward merging. A rule that
  merges everything scores 100% on the co-reference axis alone, which is why the
  refuse-list exists and why it is written *before* the fix, as in §3 and §10. The two
  are not equally bad — a split costs recall and both facts survive; a merge destroys a
  fact — so trading one for one is a regression, not a wash. RESULTS.md §15.
- **A subject over-merge is CONTAINED by the attribute** — merged subjects only destroy
  a fact when the *attribute* collides too, which is why the `the memory system` /
  test-suite over-merge measured zero-cost with must-coexist untouched. It was predicted
  that a second fact about the merged-in entity would collect the bill; a turn was added
  to test it and **it did not happen** (33/33 under both rule-5 variants). Do not repeat
  the latent-supersede claim as measured — it is a plausible mechanism that has been
  looked for once and not found. RESULTS.md §15.
- **Value comparison in consolidation is normalized-string equality, and `use_embedding_comparison` defaults to False.** Do not "improve" this by turning embedding similarity back on: a false DUPLICATE discards an update permanently, while a false CONTRADICTION keeps both facts with the right one live. Measured at 32k, sentence embeddings put real value changes ("rugby union"→"rugby", 0.982) *above* the threshold and genuine paraphrases (0.877) *below* real contradictions (0.849–0.911), so no threshold separates the cases. RESULTS.md §6.
- **Freshness does not exist inside a batch, so CONTRADICTION is withheld there.** One
  `consolidate()` call is one extraction of one utterance; its candidates' ordinals differ
  only by position in P1's output array. Before the guard, "the user does not like milk in
  their tea" and "the user likes green tea" — one sentence, both true — retired each other
  in **both** columns of the 2026-08-17 console run. The guard sits *after* the two
  containment branches deliberately: REFINEMENT and the restatement branch are decided by
  what the strings say and need no ordering, and the exact-DUPLICATE scan must still fire
  or the fix trades a mislabelled fact for a second copy. RESULTS.md §16.
- **The assistant's reply is CONTEXT in the P1 prompt, never part of the turn to extract
  from.** It sat inside `THE TURN TO EXTRACT FROM:` with equal standing to the user's own
  words, so the reader — which holds the whole conversation, against the three messages P1
  is shown — could launder anything from anywhere into "what this turn asserts". Measured:
  question turns emitted 2 facts with the reply inside the block and 0 with it demoted,
  2/2 each way, while a genuine user assertion carrying a reply still extracted. The hint
  list was ruled out as a channel by the same experiment, and prior turns by
  `extract_window_turns=3`. Do not "fix" a recurrence by adding a rule to `_SYSTEM` —
  that was measured (arm E) and bought nothing, and every word there can perturb §15's
  five naming rules. Do not reach for a question-suppressor either: §12a. RESULTS.md §17.
- **`slots.py` scores FIVE axes now, and the fifth is meaningless alone.**
  `MUST_NOT_EXTRACT` (turns 36–37) is passed 2/2 by anything that mutes P1 — a salience
  change, a stricter confidence floor, a question-suppressor — while destroying the store.
  Turn 38 (a real assertion delivered with a leaky reply, paired with 24 on `MUST_COLLIDE`)
  and the four existing axes are the guard. `REPLIES` is deliberately empty for turns 0–35
  so those numbers stay comparable with §11/§14/§15; do not give them replies — nor turns
  39–45, which measure *naming* and would otherwise confound §18 with §17. The
  `wrote nothing [...]` line is a **diagnostic, not a sixth axis**: unscored on purpose,
  because every scored axis is blind to a turn that stored nothing (a coexist group with a
  missing member still reads OK, there being no dead ordinal to find). Do not promote it
  to an axis on one observation. RESULTS.md §17.4, §18.8. Turns 46-50 (`MUST_BE_PAST` /
  `MUST_NOT_BE_PAST`) are **scored now** that `occurs_at` exists — but only in that order:
  field, nine runs, read the real output, *then* the scorer. Scoring an interface before
  seeing its output is what made the turn-45 pair vacuous. A turn that stored nothing
  reports `NOTHING-STORED` rather than passing: it trivially satisfies "nothing is past",
  which is the §17.4 argument again. RESULTS.md §19.8.
- **Temporal expiry is a READ-time label, never a write-time state change, and it must
  never suppress recall.** A fact does not become false when its date passes; *now* is
  different on every read, so a stored `expired` flag bakes in the moment it was computed.
  It must not reuse `invalid_at`, which means *superseded* — a claim that a newer fact
  replaced this one, which is not what the calendar did. And PAST must not shut the gate,
  drop the fact or dock its score: expiry that suppresses recall is deletion wearing a
  different hat. `MUST_BE_PAST` exists because a one-sided refuse-list is scored perfectly
  by `occurs_at: null` on everything — the §18 hole again. RESULTS.md §19.
- **An attribute names a PROPERTY, never the fact's category, and P1 gets this wrong about
  two thirds of the time.** `preference` / `interests` / `likes` are `FactType` values, not
  slots; naming a slot that way puts every simultaneously-true fact of that type in one
  place, where they retire each other. The hint list amplifies it — `subject_slots` shows
  the category back to P1 and the prompt asks for exact reuse — so one bad first naming is
  self-reinforcing for the rest of the session. Naming cannot carry the judgment, so P1 is
  asked it directly: **`CandidateFact.single_valued` is a required boolean in the P1 schema
  and `_classify` reads it as a field lookup.** No LLM enters the decision — the field is
  read exactly as `attribute` and `subject_hint` are, `_classify` stays a pure function of
  its arguments, and `test_no_llm_in_the_consolidation_decision` still pins the
  collaborators. Four things must not be undone. It **defaults True**, which is the pre-§18
  code path, and that is what keeps old graphs and the bench inert — do not flip the default
  to False "for safety", it would silently stop every contradiction the bench measures. It
  sits **below the exact-DUPLICATE scan**, or a collection accumulates copies of one
  sentence. It replaces two dead ends: a rule in `_SYSTEM` is arm E (the rule is already
  there verbatim and P1 violated it four times in one column), and a `COLLECTION_TYPES`
  keyword list scores 3/3 on turns 39–41 while breaking `(42, 43)`, because a favourite is
  a preference holding one value at a time. And **judge any change here on `--runs 9`, not
  3** — pre-fix, identical input scored 2/3 then 1/3. RESULTS.md §18.
- **`single_valued` is recorded on the SLOT, and the record wins over what P1 emits.**
  A `:Slot` node keyed `(session_id, subject_key, attribute)` — the pair `_competing`
  filters on, because that pair *is* the slot. The first fact to land declares the arity
  via `ensure_slot_schema`, which is `ON CREATE SET` in Cypher: no later fact may revise
  it implicitly, and that is enforced by the query, not by a comment. `set_slot_schema` is
  the one deliberate correction path and **rewrites no fact** — arity is read at
  classification time and never stored on a `StoredFact`, so a correction lands on the
  next fact in the slot and leaves the ones already classified alone. Disagreements are
  logged at INFO and **not acted on**; acting on them is the newest answer winning, which
  is the variance the field exists to remove. `_classify` takes the arity as a plain
  argument, so it stays a pure function of its arguments and no LLM enters the decision.
  Inert on `attribute == ""` — nothing recorded, nothing looked up — which is what keeps
  old graphs and `bench/extract.py` on the pre-A1 path; a store lacking the three methods
  degrades there too. `clear_session` must keep deleting `:Slot` nodes: nothing else lists
  them, so orphans accumulate invisibly. Measured **0 disagreements in 108 slot reuses**,
  so it changed no decision on the 51-turn script — its value is permanence and the
  correction path, not a score. RESULTS.md §22.
- **The `single_valued` examples in `_SYSTEM` must not be collections held by an OWNER.**
  The first version listed "office locations" and "todo list items", and P1 generalised the
  shape: "I have a cat called Miso" filed as `the user :: pets` instead of subject `Miso`,
  splitting `[28, 29]` in 9 runs out of 9 while every other axis held. §15's rule 5 — a
  thing Y *contains*, with properties of its own, is its own subject — was being actively
  contradicted by an example in the same prompt. Keep the examples attributes of the
  subject itself ("languages spoken", "allergies"). This is the concrete form of the
  general warning above: every word in `_SYSTEM` can perturb §15, and an example does it
  faster than a rule. RESULTS.md §18.12.
- **A gateway console trace is evidence only after you check the installed revision.**
  `llm_gateway` pins `memore @ git+…`, so the console can run a memore days behind the
  working tree; the 2026-08-17 run was four commits stale and 7 of its 11 wrong supersedes
  were already fixed. `source_episode_id == ""` on every fact is the fingerprint of a
  pre-§16 install. Reinstall with an explicit `--python .venv/bin/python`: with
  `VIRTUAL_ENV` unset, `uv pip install` does not discover `./.venv` and silently targets
  the default interpreter while reporting the right git sha. Verify by grepping the
  installed file — `dist-info/direct_url.json` kept showing the old commit id after a
  successful reinstall. RESULTS.md §18.2.
- **`consolidate(session, candidates)` is ONE UTTERANCE, not a throughput batch.** Facts
  that arrived separately must go in separate calls or they silently stop resolving against
  each other. Both bench harnesses had been ingesting in chunks of 50 to batch the embedder,
  which after the guard above declared 50 independent turns simultaneous and lost 18
  supersedes at 6k and 16 at 32k — while oracle accuracy stayed 0.990/0.960 and
  `gold_fact_superseded` stayed 0, so the score hid it completely. Use
  `DeterministicConsolidator.prewarm(texts)` to batch the embedder instead. RESULTS.md §16.4.
- **A slot may now hold several live facts on purpose, so a contradiction no longer
  retires the whole slot.** It retires the incumbent plus any live fact from a *different*
  batch — the incumbent's own batch siblings were never ordered against it. Without this,
  the same-batch guard is only a one-turn delay: the next correction collects both facts.
  `source_episode_id` carries the batch id (it was written empty before); facts with `""`
  fall back to retiring the whole competing set, the same inertness argument
  `attribute == ""` makes in `_competing`. RESULTS.md §16.
- **A restatement that says LESS coexists — it must not be answered DUPLICATE.** Candidate
  strictly contained in the incumbent, same named slot, is not news and must not supersede
  on recency ("Bud sits in the red chair" retired "…in the Whangarei office"). But
  containment that way round does not imply "adds nothing": the one such pair in sh_32k's
  2310 facts is `"…the sport of rugby union"` → `"…of rugby"`, a real update, and no
  surface rule separates the two — both are strict prefixes differing only in whether the
  trailing phrase opens with a preposition. DUPLICATE there would discard an update
  permanently. Both facts stay live instead. Requires a real attribute on both sides, which
  is what keeps the bench inert. RESULTS.md §16.
- **Attribute-name snapping was measured and REFUSED; do not build it as "§10 for slots".**
  At conversational scale the df gate is degenerate — 30–34 distinct attributes means every
  token appearing once reads 0.029–0.033, above the 0.015 threshold — and the only subset
  pair it could act on in 48 turns was `location` ⊂ `work location` on one subject, whose
  merge reproduces the exact defect being fixed. Note also the premise it rested on is
  wrong: attribute drift is **between runs, not within a session**; `subject_slots` reuse
  works inside one store. RESULTS.md §16.5.
- `recall()` never raises. Store error or timeout → closed result, logged, turn proceeds.
- `enabled=False` on either path → full no-op.
- `inject_token_budget` is a hard cap, not a target.
- Session-scoped only; no cross-session recall in v1.
- No Graphiti types leak past the `MemoryStore` boundary.
- Supersede, never delete — superseded facts stay and surface to recall marked `SUPERSEDED`.
- **`score_floor`, `EmbedConfig.model` and `RecallConfig.gate_on` are ONE decision, not three.** Each combination puts the relevant/irrelevant boundary somewhere different, so a floor only means something next to the model *and* the quantity it was calibrated against. Current triple: `mxbai-embed-large`, no query prefix, `gate_on="cosine"`, floor `0.57`. Changing any one requires re-running `uv run python -m memore.bench.calibrate` and re-checking all three regimes. The spec's 0.35 was never calibrated for *any* model; 0.48 was calibrated against fixtures that had drifted from the write path and let off-domain false opens reach 0.077. RESULTS.md §5, §13.
- **The gate tests the un-fused cosine; ranking still uses the fused score.** Fusion's range is `[cos/(1+w), cos]`, so BM25 can only ever *deduct* — a fact sharing no term with the turn is docked ~23% before the floor sees it, which is why gate behaviour tracked wording rather than whether the memory existed. Gating on cosine is worth +16 points of conversational recall at matched false-open, and cannot inflate anything past the floor since `fused <= cos` always, so it does not reintroduce what the multiplicative-fusion invariant guards against. `MemoryHit.similarity` carries it; a store that leaves it 0.0 shuts the gate rather than silently falling back. RESULTS.md §12, §13.
- **Calibration fixtures are GENERATED from the real extractor, never hand-written.** `uv run python -m memore.bench.gen_calib_fixtures --write`, then re-run `calibrate` — the fixtures, the embedder, the gated quantity and the floor are one chain. Hand-authored facts drifted from what P1 actually stores (terse fragments vs third-person sentences, and one fact the write path refuses outright), and hand-authored positives all shared word stems with their facts, so the deduction that dominates real gate behaviour never fired during the run that chose 0.48. Adding paraphrase-only positives alone moved the shipped config's measured chat TPR 0.920 → 0.730. A fixture that cannot express the failure cannot calibrate against it. RESULTS.md §13.
- **A model-card prefix is a candidate, not a setting.** `config.KNOWN_PREFIXES` records what a card documents; `_PREFIXES` records what actually ships, and `from_env` consults only the latter. `mxbai-embed-large`'s query prefix is measured and *rejected* — it costs conversational recall at every operating point. Listing it in `_PREFIXES` would enable it silently everywhere. RESULTS.md §13.
- **A different embedder also means a different vector-index width**, and a graph indexed for the old width cannot be searched with the new one. `FalkorStore.connect()` raises on the mismatch; do not "fix" that by catching it. Use a fresh `MEMORE_GRAPH` or rebuild. It is guarded because the failure is silent otherwise: the write succeeds, only the query raises, and `recall()` swallows store errors by design — leaving a gate shut forever. RESULTS.md §5.
- **The gate keeps *off-topic* memory out, not *wrong-subject* memory.** Measured at the calibrated floor: off-domain false-open ~0.03, but **hard**-negative (right topic, wrong subject) false-open **0.68–0.88** — and that is the *best* case, bought with recall; at 0.35 it is 1.000. The distributions overlap; no threshold separates them. Do not try to tune it away — fixing it needs a subject-key check the score cannot carry. RESULTS.md §5.
- **Subject aliasing merges on document frequency, and the threshold is a RATIO, never a count.** `affiliated` is df 9 at 6k and df 60 at 32k — the same relation word — so an absolute cutoff tuned on one corpus is either inert or catastrophic on the other. `AliasConfig.df_ratio` is `df / n_subjects`, defaulting to `0.015`, chosen in the *wide* gap between the hand-labelled bands (2.1x margin over `kingdom` at 0.0071) rather than the narrow one at 0.0083. Measured: 0.015 and 0.008 score identically, so the margin costs nothing. Re-run `python -m memore.bench.oracle_run --alias-df-ratio ...` on **both** corpora before touching it, and judge it on the printed merge log, never the score — RESULTS.md §3 rejected the ungated rule *despite* a better score. RESULTS.md §10.
- **`AliasConfig.min_subjects` (200) is a safety guard, not a performance knob.** A relative threshold degrades worse than an absolute one at small scale: in a 12-subject session `location` in two subjects reads as ratio 0.17, eleven times the threshold, so an unguarded rule fires hardest where its evidence is weakest. Nothing below 303 subjects has been validated. Lowering it merges conversational stores on evidence that does not exist. RESULTS.md §10.
- **Aliasing reads document frequency in ARRIVAL order, not from the finished corpus.** The rule is conservative when cold — `affiliated` is not yet a relation word on its first appearance — and converges as the session fills. `bench.oracle.build_groups` replays the same growing vocabulary deliberately: grouping a completed corpus in one pass credits the store with merges ingest never made and scores a system that was never run. If you add a caller, pass it the same `AliasConfig` the ingest used. RESULTS.md §10.
- **Chain expansion runs AFTER the gate, never before, and never seeds it.** A multi-hop answer fact shares no entity with the question — that is what makes it multi-hop — so it cannot clear a similarity floor and must not be asked to. Judging relevance on the seeds and letting the chain ride along is what keeps `score_floor` meaning what it was calibrated to mean. Moving expansion ahead of the gate would silently re-open the gate on turns that have nothing relevant, and would invalidate §5's calibration. Chain facts carry `score=0.0` because they were never ranked against the turn — do not invent a similarity for them. RESULTS.md §8.
- **Hybrid fusion is multiplicative** (`cos · (1 + w·bm25)/(1 + w)`), not additive. Additive fusion mixes an absolute score with a set-relative one and lets an irrelevant query clear `score_floor`. Full-text queries must be `|`-joined term unions, or the index ANDs them and the BM25 arm silently returns nothing. RESULTS.md §5.
- **Performance: four expensive wrong turns, each closed by arithmetic. Do not reopen them.**
  Restated here from `specs/performance-addendum.md` because that file is not in the public
  repo and this list is the single most re-derivable thing in it — every item is a plausible
  instinct that the measurements rule out.
  - **No FPGA or GPU vector index.** The workload is a scan over 1559 vectors of 1024
    dimensions — 6.4MB. Custom silicon starts paying near 10^7 vectors, three orders of
    magnitude away.
  - **No rewrite in Rust or another language.** Wall time is in an HTTP call to a model
    server and in FalkorDB. Interpreter overhead is inside the ~18ms "everything else"
    figure of a budget that is half empty (§5: ~90ms p95 against 200ms).
  - **Do not optimise the FalkorDB lookup.** 2.6–4.3ms — 2% of a budget with ~110ms unused.
  - **No smaller embedder for speed.** Measured and rejected in §5: `nomic-embed-text` gets
    the speed and loses the ranking (`retrieval_hit` 0.93 → 0.76 across candidates at the
    same speed). Quality is the binding axis, not speed.
  - **No smaller model on the write lane.** §11 and §18 establish the residual errors are
    identity judgements. The one place a small model belongs is A4's pairwise matcher — a
    same-or-not decision over a short candidate list.

  The framing that generates all five: **the read path is latency-bound and its remaining
  errors need a *better* model; the write path is accuracy-bound and has no latency
  constraint.** Optimisation runs in opposite directions in the two lanes, and the instinct
  that is right in one is wrong in the other.

  What IS worth doing, and it is measured rather than proposed: **the embedder pays a
  per-call load charge on a model that is already resident.** Re-measured 2026-08-20 under
  the pinned serving state — 10 queries, wall 50.6–75.6ms, `load_duration` **36–56ms per
  call** while `/api/ps` reports `mxbai-embed-large` resident at 0.62GB, pinned. Pinning does
  not remove it; only an in-process embedder does. Actual compute is the remaining ~15–20ms.
  (§5's "`/api/ps` showing 0 GB VRAM" line is about `embeddinggemma` — do not cite it for
  `mxbai`.)
- **Do not wrap this in MCP.** MCP is permanently the wrong shape for pre-fetch recall (it would put recall back behind a model decision, which is the pattern this design leaves behind). It is only the right shape for the deliberate model-initiated fallback lookup — see `recall-poc-spec.md` §5a.

Deferred PoC scope is listed in `recall-poc-spec.md` §5 (cross-session recall, async job machinery, rolling-summary key synthesis, provider abstraction, audit-log query path). Mark these in code with comments pointing at the production spec section rather than omitting them silently.
