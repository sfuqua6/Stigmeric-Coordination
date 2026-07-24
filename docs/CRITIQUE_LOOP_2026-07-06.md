# Critique Loop — 2026-07-06 (17:11–19:11)

A two-hour continuous, adversarial engagement with this codebase: each iteration attacked one mechanism or assumption, verified claims **at the code level** (not from the repo's own prose), checked them against 2025–2026 literature, and then explicitly dissented against earlier iterations' conclusions — deliberately mimicking the swarm's own scout/critic/hater dynamic. Three code-audit subagents were used; every load-bearing claim below carries a file:line cite that was read this session.

**Epistemic status:** code citations were verified against the working tree at commit `6d6a884` (branch `cleanup/restructure`). Literature claims come from web search abstracts, not full paper reads. The self-dissent subsections are not rhetorical decoration — several reversed the iteration's own headline (most notably Iteration 1).

---

## 0. TL;DR — the seven load-bearing findings

1. **The core architectural premise has never been tested — in either direction.** The partition→diversity probe run of Jul 3 is invalid (wrong partitioner engaged, degenerate outputs scored as diversity, mislabeled corpus). Semantic partitioning itself was committed **2026-07-03** and has never run in any real run; every judged loss in the empirical record predates it.
2. **"Field pressure" is a labeling system, not a dynamic.** No code path anywhere reduces a signal's strength in response to CRITIQUE, OBJECTION, or failed VERIFICATION. Dissent structurally *helps* its target (rebuttal recruitment, credibility gate, self-deduplication of objections).
3. **The stigmergy port dropped stigmergy's immune system.** ACO's proportional evaporation (the negative feedback that prevents premature convergence) became rank-preserving, near-inert additive logit decay.
4. **The faithfulness audit audits the appendix, not the answer** — citation tags are rewritten to footnotes *before* the 4-gram check runs. The shipped reader answer has neither audit coverage nor citations. One-line fix.
5. **The failure mode has flipped and the docs haven't noticed:** 30/41 newest real runs halt on `quality` at ~70s through a gate that (for debate/analysis/problem_solving/creative) never asks about verification; only 1/41 hits `cap_time` — the problem `render_set_stable` was built to solve.
6. **The eval ladder is missing its decisive rung:** no condition ever receives the evidence the swarm retrieves. Condition F (single-call RAG over the same pack) is the baseline the whole thesis must beat, and the current prompt set (god_exists, ban_cars…) structurally cannot express the swarm's only theoretical advantage.
7. **The fitness function rewards the pathology the architecture is supposed to prevent:** monotone growth scores 1.0, contested clusters 0.2, consensus = 1/(1+dissent) — while the survival gate counts dissent as credibility, and the one "fully model-independent" fitness term is a stub returning a constant 0.5.

The strategic synthesis (Section 8): the project's honest path is not more mechanism work — it's (a) re-aim the eval at the regime where multi-agent systems demonstrably can win (evidence packs that degrade single-call context utilization), (b) make dissent and verification *exist dynamically*, (c) exploit the fact that the negative empirical record largely doesn't apply to the current code.

---

## 1. Iteration 1 — Partitioning-as-diversity: the probe measured the wrong experiment

### The data (recovered this session)
`eval/results/partition_probe_20260703_214551/` (corpus=retrieve) crashed before writing report.md (died on prompt 5; an earlier attempt hit `ZeroDivisionError` at `partition_probe.py:141` when median partition size = 1). Metrics computed from `rows.jsonl`:

| pid | bleu_P | bleu_U | bleu_H | cdist_P | cdist_U | cdist_H |
|---|---|---|---|---|---|---|
| ban_cars | 0.137 | 0.120 | 0.060 | 0.108 | 0.095 | 0.140 |
| god_exists | 0.083 | 0.153 | 0.094 | 0.184 | 0.132 | 0.290 |
| climate_action | 0.088 | 0.175 | 0.060 | 0.117 | 0.106 | 0.306 |
| ai_regulation | 0.118 | 0.148 | 0.180 | 0.094 | 0.116 | 0.204 |

Naive reading: P beats U only 3/4 (fails the probe's own pre-registered sign test), and H — identical evidence at temp 1.0 — beats P on centroid distance **4/4**. Temperature bought more measured diversity than partitioning.

### Why that reading is wrong (code audit)
1. **Semantic partitioning never engaged.** `core/intake.py:121` gates it on `len(chunks) >= num_scouts * 2`; probe default `n_agents=6` needed ≥12 chunks, runs had 7–9. Every row used `_partition_contiguous` — positional slices of a relevance-ranked list, the exact "quality gradient, not topical partitions" failure the semantic path was built to fix (intake.py:103–107). Partition sizes were (2,2,2,1): the last agent saw half the evidence of the others.
2. **Model confound.** The run used Qwen2.5-1.5B-Instruct, HF 4-bit, driven as a **raw completer with no chat template** (`core/llm.py:601–633`), right-truncated at 1400 tokens.
3. **H's diversity is garbage-diversity.** H outputs include a line of pure underscores, `[Your answer based on reading & analyzing excerpted content]`, and grader roleplay. Junk embeds far from real text (inflates cdist) and yields few `[a-z]+` n-grams (deflates self-BLEU). Both metrics score degeneracy as diversity; the biggest H "wins" are the rows with the most junk.
4. **Corpus mislabeled.** `CompositeRetriever` silently substituted the placeholder corpus for climate_action while the row is stamped `retrieve` (`core/retrieval.py:562–568`). At most 2/4 rows tested retrieved evidence.
5. n=4 prompts < the probe's own pre-registered n≥8 floor.

**Verdict: the load-bearing premise (CLAUDE.md principle #2) remains unmeasured in both directions.**

### Literature
- Mode collapse is alignment-induced, not decoding-induced; distribution-level prompting restores 1.6–2.1× diversity orthogonally to temperature — [Verbalized Sampling, arXiv:2510.01171](https://arxiv.org/abs/2510.01171). The repo's multi-claim scout (`SCOUT_CLAIMS_PER_CALL=4`, portfolio of angles + `select_novel_claim`) **independently converged on this** and is the repo's best-literature-supported diversity mechanism.
- Rigid templates suppress diversity even at high temperature — [The Price of Format, arXiv:2505.18949](https://arxiv.org/html/2505.18949v1). The swarm's CLAIM:-style templates may cap diversity below what any evidence manipulation can recover.
- The real ceiling on multi-agent gains is **correlated errors**: [arXiv:2506.07962](https://arxiv.org/pdf/2506.07962) (same-family models converge on the same mistakes), [arXiv:2602.03794](https://arxiv.org/html/2602.03794v1) (homogeneous agent scaling saturates via output correlation; heterogeneous personas outperform), and the "popularity trap" in ensemble voting (arXiv:2510.21513): consensus selection filters out minority-correct answers.

### Fixes
- Decisive probe re-run: retrieval target ≥ 2×n_agents chunks (or `--n-agents 4`), chat-templated backend, `--mini 8`, **plus a V condition (verbalized sampling, identical evidence)** — benchmark partitioning against the strongest cheap diversity lever, not just temperature. Record embedder path and per-output lengths; drop outputs < 30 alpha tokens before metrics; fix the k=1 ZeroDivision; write report.md in a `finally`.
- The honest competitive frame is **partitioning vs verbalized sampling vs model-family heterogeneity** (GroqRouter exists) — three diversity engines, one probe.

### Standing dissent
Disjoint evidence remains the only lever that can produce *grounded-claim coverage* (claims citing different facts) — which neither self-BLEU nor centroid distance measures. A claim-level coverage metric would test what partitioning is actually for.

---

## 2. Iteration 2a — The deployed diversity engine is not what the docs describe

1. **Semantic partitioning has never run in a real run.** `git log -S "_partition_semantic"` → commit `aba21ef`, **2026-07-03**. Every saved real-run log predates it (cell12.txt Jun 17, cell10.txt Jun 23, donotread.txt Jun 29, groqgroq.txt Jun 11). The logs capture other stdout prints, and `[intake] semantic partitioning:` appears in none of them.
2. **Therefore the entire negative empirical record (0 judged wins) tested contiguous rank-slicing, not the current design.** Symmetrically, nothing yet supports the current design.
3. **Production's actual diversity stack** (`run_swarm.py:399–474`, `NUM_SCOUTS=4`, `WEB_PARTITION_COUNT=2`): scouts 0–1 get **facet-based web partitions** (FacetPlanner query angles → per-facet live search, `partition_id="web_<slug>"`) that bypass `partition_for_scouts` entirely; scouts 2–3 partition the retrieved corpus (gate needs only ≥4 chunks in this config). Half the "information partitioning" story is really **query diversification** — the same move as STORM's perspective-guided research, and better literature-supported than corpus partitioning.
4. **Observability gap:** `_partition_semantic` returns `None` silently on any failure (intake.py:151–165) and the contiguous fallback prints nothing. A run cannot tell you which partitioner it used. Fix: log the fallback loudly + record `partitioner: semantic|contiguous` in `summary.json`.

**Dissent vs Iteration 1:** "the flagship mechanism may be silently disabled in production" was half-right but mis-aimed — the true statement is stronger (the mechanism is three days old and unexercised) and more charitable (production already runs a different, arguably better diversity mechanism that the docs barely name).

**Re-framed ledger:** the diversity stack is (i) facet query diversification — deployed, never ablated; (ii) semantic corpus partitioning — new, never run for real; (iii) multi-claim scout + novelty selection — deployed, best literature support. Ablating (i) and (iii) on real runs is worth more than any partition probe, because they are the levers actually firing.

---

## 3. Iteration 2b — "Field pressure" is a labeling system

Adversarial audit of `core/signal_store.py`, `core/worker_pool.py`, `core/projection.py`.

### The asymmetry (the central mechanical finding of the session)
The only strength mutations in the codebase are: deposit, dedup-amplify (+0.10 logit), trail-amplify (+0.03/cluster per SUPPORT), decay, prune. **No code path reduces any signal's strength in response to CRITIQUE, OBJECTION, or a failed VERIFICATION.** A failed verification merely withholds a boost. Anti-evidence acts only at projection time — and the verification survival requirement is disabled for debate/analysis/problem_solving/creative (`SURVIVAL_TASK_PROFILES`, config.py:513–531), i.e. for the flagship tasks.

### Dissent structurally helps its target
- REFINE consumes objections and deposits rebuttal SUPPORT into the attacked cluster (worker_pool.py:384–406).
- The survival credibility gate counts ≥1 dissent as *credibility* (config.py:493–496).
- OBJECT targets the argmax most-vulnerable rep (worker_pool.py:1557–1575), so successive objections are near-identical and **self-dedup** (signal_store.py:353–364): N objections collapse into one signal while support accrues via many distinct depositors. `rejected_by_field` needs dissent/support ≈ 3.48; realistic peak against a 20-member cluster is ~1–2. **Dissent can label a cluster contested; it cannot kill it.**

### Where rich-get-richer actually lives
Not primarily in dedup (narrow window: 0.95 string ratio, 3-most-recent, 5 minutes). The real loop: `sample_from_clusters` weights clusters by `Σ member_strength / log1p(n)` — monotone in size (a 20-member cluster ≈ 4.2× a fresh singleton per draw) → more SUPPORT → trail-amplify → higher Σ. Plus **peer-relative prune protection** (members of ≥3-member clusters prune at half the floor: 0.15 vs 0.30) — dust from popular clusters survives what kills isolated minority claims.

### The ACO divergence
Ant-colony evaporation is proportional (ρ·τ) — it erodes strong trails fastest and is the algorithm's load-bearing negative feedback. This port uses uniform **additive logit decay** (−0.01 logit/30s; ~0.3 logit over a full 15-minute run — less than three dedup hits), which **preserves rank exactly**. Stigmergy's own immune system was silently removed in the logit refactor. First-mover advantage is then mostly an opportunity clock: cold-start is 70% SCOUT for 30 iters, founders define centroids, late claims arrive after the scout gate demotes SCOUT and near the stability halt.

### Doc rot found (CLAUDE.md vs code)
- Dedup is **not** embedding-based; the embedding-rejection path was removed — paraphrases now join clusters (the absorption channel is clustering).
- Gap 3 local biases are **no longer dead** on the hot path (biases computed before the primary `choose_action` — CLAUDE.md stale in the pessimistic direction).
- `support_diversity` is now a union of discriminators, not `len(partitions) + len(strategies)`.
- `store.amplify()` / `DELTA_AMPLIFY=0.30` — the documented corroboration amplification — has **zero production callers**.

### Fixes (ranked)
1. **Signed trail amplification** — negative logit delta on the target cluster for CRITIQUE_NEGATIVE/OBJECTION, symmetric to `_amplify_cluster_trail`. Cheapest change that makes dissent exist dynamically.
2. **Fix dissent self-dedup** — count duplicate-rejected objections in `dissent_pressure` (weight by dedup count), or sample OBJECT targets ∝ vulnerability instead of argmax.
3. **Evaporation-style (strength-proportional) decay** so leads must be re-earned continuously.
4. **Mean-based or capped cluster sampling weight**; add the exploration bonus to the four raw-strength samplers (worker_pool.py:361–421).
5. Delete or wire `DELTA_AMPLIFY`; correct the three stale CLAUDE.md paragraphs.

### Self-dissent
The dampers are real (log1p size penalty, size-adjusted join bar, youth grace, gap-fill DEVELOP sampler, projection-time age discounting): this is not a runaway winner-take-all — it's a slow-consensus machine whose *read-out* does most of the discriminating. Strength dynamics are nearly decorative for selection **except** where strength gates sampling, which is exactly where the asymmetry compounds exposure. And per the [LLM Blackboard System](https://arxiv.org/pdf/2510.01285v1) results (13–57% gains with *no strength dynamics at all*), it is a live possibility that the shared-artifact medium, not the pheromone arithmetic, carries whatever value exists — an ablation worth running before investing in fancier dynamics ([CodeCRDT](https://arxiv.org/pdf/2510.18893) is the formal-guarantees version of that alternative).

---

## 4. Iteration 3 — Eval methodology: the ladder is missing its most important rung

### What the protocol does right
Both-orders judging, disagreement→tie (judge.py:23–24), ties at half-credit in a Wilson 95% CI with "real win" = lower bound > 0.5 (judge.py:195–275), blind scattered pack mode, condition E as attribution control. This matches 2025–26 best practice ([position-bias measurement](https://mbrenndoerfer.com/writing/position-bias-in-llm-judges), [fluency bias](https://arxiv.org/pdf/2601.13649), [judge non-robustness to artifacts](https://arxiv.org/pdf/2503.09347)) better than most published agent evals.

### The holes, ranked
1. **No condition receives the evidence.** B/C/D/E get only `{q}` (ab_harness.py:224–255) while A retrieves live web evidence mid-run. A-vs-B confounds orchestration with retrieval access; E controls for the synthesis prompt but not evidence. **The missing rung is condition F: one direct call + the same retrieved evidence pack (single-call RAG)** — the baseline any practitioner would deploy, and the only comparison that can isolate the orchestration's contribution.
2. **The prompt set cannot express the thesis.** ban_cars / god_exists / climate_action are answerable from parametric knowledge. Per the 2026 compute-matched results — [single-agent wins under equal token budgets via a Data-Processing-Inequality argument](https://arxiv.org/abs/2604.02460), [The Illusion of Multi-Agent Advantage](https://arxiv.org/pdf/2606.13003), with [Pareto-optimal multi-agent scaling](https://arxiv.org/pdf/2605.01566) as the counterpoint — multi-agent systems win **when the single agent's effective context utilization degrades**. That is precisely the swarm's compression theory (composer input stays O(K×brief) regardless of corpus size), and precisely what a no-evidence debate prompt can never show. Losing 0-for-8 on these prompts is *consistent with theory* and says little about the target regime.
3. **Length asymmetry, twice**: B–E share `args.max_tokens`, A is unbounded; then the judge truncates at 4,000 chars (judge_answers.py:66–67), historically cutting the swarm's 3–4× longer answers mid-argument. No anti-verbosity rubric language.
4. **Judge-family self-preference unaudited**: the judge is a Llama variant while B and D are Llama models; the 2026 literature puts self-preference at 10–25%.
5. **At n=8 with ties as half-credit, only ~7/8 both-order-agreed wins clears the bar.** Right bar for claiming victory — but the symmetric reading matters: the existing "0 wins" record is mostly *uninformative noise ties*, not proven loss (except cell11's 4 both-order-agreed losses, which are real).

### Fixes
1. Add condition F; build a task set with evidence packs at 1×/4×/16× the model context; make A-vs-F the headline metric.
2. Length-control A's judged text; add rubric language; report a length-matched subset.
3. Off-family judge (or 2-judge panel); record judge identity in scores.json.
4. Report "informative n" (both-order-agreed verdicts) alongside win-rate.

### Self-dissent
Mechanism fixes (Sections 3, 5) are invisible on the current eval — sequencing matters: **eval first, mechanisms second.** But the counter-risk is real too: A-vs-F on hard over-context packs is a brutal bar; a survivable milestone ladder is A>B (failed) → A≈F at 1× → A>F at 4–16×.

---

## 5. Iteration 4 — Convergence & synthesis: the safety story is stale in both directions

Audit of `core/convergence.py`, `agents/synthesizer.py`, `core/clean_answer.py`, cross-checked against the 41 real-run artifacts in `driveoutputs/`.

### Convergence: the disease moved and the docs didn't
1. **`cap_time` is no longer the failure mode**: 30/41 newest runs halt `quality` at ~67–77s (seconds after the 60s floor lifts); 10 halt `saturation` **with zero surviving clusters**; 1 cap_time. `render_set_stable` defends against last month's problem.
2. **The quality gate is a fork with a rubber stamp on the flagship side** (convergence.py:272–313, config.py:520–523): debate/analysis/problem_solving/creative need only `support_diversity ≥ 4 ∧ dissent < 0.5 ∧ support_depth ≥ 3` — nothing about grounding; coding/None demand 2 distinct validators at strength ≥ 0.7, empirically unreachable (verification 0.02–0.27 across all 41 runs; validator abstain plateau ~0.5). **One branch can't fail, the other can't succeed.**
3. **Counter bugs**: phantom +10 seeding of every windowed counter (`_last_body_iter = -tick_interval`, convergence.py:175 — sweep_small halted "saturation" at 50–55 iters against a 60 window); the saturation strength window is 60 *2-second polls* (~120s wall), not 60 iterations; saturation can fire with 0 survivors.
4. **`render_set_stable` resets on jitter the reader never sees**: signature keys on strength-argmax `representative_id` (a rank swap = reset); a `build_plan` exception resets accumulated stability; recluster period (25) < stability window (40). Deeper conflict: trail amplification and cluster-aware sampling **steer workers into the render-set clusters**, so the halt requires the pool to abandon its own attractors. The detector also stabilizes `build_plan(full projection)` while the synthesizer renders a *differently filtered* plan.
5. **`novelty_rate` measures cluster creation, not novelty**: the size-adjusted join bar (0.55 + 0.03·log2(n)) makes near-duplicates count as "novel" against big clusters exactly during convergence; SUPPORT/OBJECTION prose counts; nothing ever merges. Developer churn alone can hold novelty above the floor indefinitely.

### Synthesis: the compression invariant holds; the safety claims don't
6. **Compression invariant structurally verified**: `_compose_answer` (synthesizer.py:2559–2646) sees only briefs (hard-capped 200 words each) + scalar digest lines ≈ 2.5–3.2K tokens; all fallback paths equally clean. (K is 8 now, not "≤ ~6" — `RENDER_K`, config.py:716.)
7. **The faithfulness audit is neutralized by call ordering**: `resolve_inline_citations` rewrites inline `[INITIAL_xxxxx]` tags to `[1]`-footnotes at synthesizer.py:1317 **before** `_build_faithfulness_audit` runs at :1335 — the 4-gram check then audits the footnote appendix (where IDs sit beside verbatim excerpts and pass trivially), not the Section 1/2 prose. Also negation-blind ("It is false that {4 words of the rep}" passes) and purely advisory. **One-line fix: swap the order.**
8. **The shipped answer has no citations at all**: `clean_answer._SCAFFOLD` strips `[N]` tags from answer.txt; provenance survives only in diagnostics.md. Combined with 7: the reader artifact has neither audit coverage nor provenance.
9. **Quiet minorities die at the bottleneck**: Section 2 preserves only high-dissent clusters (top 6 + 8 one-liners); distinct-but-uncontested weak clusters — usually the majority of the field (13–45 of 30–75) — are routed to diagnostics.md by the `## 3.` cut. And the synthesizer ignores `plan.dissent_clusters`, recomputing its own dissent set — the planner's dissent selection is dead code on the render path.

### Self-dissent
Iteration 2b implied "slow consensus, never converges"; the newest artifacts show the opposite symptom — **premature convergence** through an ungrounded gate. Both share one cause: no halt or gate is tied to epistemic quality; counters measure churn, not knowledge. And against the audit's own alarm: quality-halt-at-70s at 5 iters/s on a 4-scout config may be genuine saturation — fast ≠ wrong; what makes it wrong is that the gate asked nothing about grounding.

### Fixes (ranked)
1. Swap audit/resolve ordering (synthesizer.py:1317↔1335). One line; restores the only hallucination defense.
2. Key the render signature on `cluster_id` with hold-on-exception; align recluster period with the stability window.
3. Give the flagship quality gate a grounding requirement (≥1 verified atom or N SEARCH-grounded supports) — the verification-dilution fixes of Jul 3 make this newly feasible.
4. Ship demoted-distinct one-liners in Section 2; make the renderer consume `plan.dissent_clusters`.
5. Restrict novelty accounting to INITIALs at the base join threshold; kill the phantom +10.

---

## 6. Iteration 5 — Cluster genome/fitness: selection pressure without an objective

From `core/fitness.py` directly:

1. **The "only model-independent term" is a stub**: `_wikidata_resolution_fraction` returns a constant 0.5 (fitness.py:183–188) even when enabled; in coding it carries 0.12 weight of pure dilution. The module and CLAUDE.md sell it as the Tier-3 anchor.
2. **Verification double-counts under two names**: absent `llm_judged`, `semantic_strength` falls back to the atom verification mean (fitness.py:222–224) while the field path adds a separate `verification` term from the same lineage (fitness.py:251).
3. **The celebrated LLM cap is decorative**: `min(llm_raw, 0.35)` caps a term whose *weight* is 0.06–0.12 — an uncapped 1.0 contributes ≤ ~0.12 anyway. The genuinely model-coupled Tier-2 terms (centroid_stability + novelty_density, computed with the same embedder that clusters the signals — a closed loop) carry ~0.21 uncapped.
4. **Scale instability**: with `field` the compositor normalizes over 10 terms, without it 7 (fitness.py:244–258) — and `trajectory.monotone_growth` compares fitness across refreshes that may differ in this.
5. **The fitness function is anti-aligned with the survival story**: `consensus = 1/(1+dissent_pressure)` rewards dissent *absence*; `_trajectory_score` gives contested (oscillating) clusters 0.2 and monotone growth 1.0 — i.e., **fitness rewards exactly the popularity compounding Iteration 2b diagnosed**, while the survival gate counts dissent as credibility. Two modules pull in opposite epistemic directions.
6. **Deepest: seven hand-set weight tables, no external objective.** In ACO, reinforcement couples to path length. Here every term proxies internal coherence/geometry — a Goodhart setup with no reward model. The one task with a real objective (coding: tests pass) has the right weight instincts, wired partly to the stubbed term.

**Self-dissent:** a learned fitness needs labels the project lacks (~14 judge verdicts, position-biased) — transparent hand weights are a defensible bootstrap. The indefensible parts are the constant term, the duplicate term, and the anti-aligned terms. And `verification` *is* an external objective in principle — it's just ≈0 in practice and dropped from flagship survival; the fix is coverage, not a new objective.

**Fixes:** delete `entity_resolution` until implemented; unify verification under one name; re-sign `consensus` to reward *survived contestation* (dissent present AND support grew after); ablate weight tables on fixed store snapshots via `synthesize.py` (cheap, no live swarm).

---

## 7. Doc-rot ledger (CLAUDE.md / docstrings vs code, accumulated)

| Claim | Reality | Where |
|---|---|---|
| Dedup is embedding-based with string fallback | Embedding rejection removed; only >0.95 string, 3-recent/5-min window | signal_store.py:343–370 |
| Gap 3 local biases dead on hot path | Fixed — computed before primary choose_action | worker_pool.py:997–1008 |
| `support_diversity = len(partitions)+len(strategies)` | Now a union of discriminators | projection.py:1785–1806 |
| Corroboration amplification (`AMPLIFY_FACTOR`) | Zero production callers — dead config | signal_store.py:906 |
| `MIN_INTER_CLUSTER_EDGES` default 1 | Default 0 (disabled) | convergence.py:68 |
| Composer K ≤ ~6 | `RENDER_K = 8` | config.py:716 |
| Quality gate = dual validators ≥ 0.7 | Only for coding/None; flagship tasks skip verification entirely | config.py:520–523 |
| "Substantive runs end on cap_time" | 30/41 quality @ ~70s, 10 saturation @ 0 survivors, 1 cap_time | driveoutputs/*/summary.json |
| Saturation window = 60 iterations | 60 × 2s polls ≈ 120s wall | convergence.py:125,196–198 |
| Faithfulness audit guards the answer | Runs after citation resolution → audits the appendix | synthesizer.py:1317 vs 1335 |
| entity_resolution = Tier-3 independent term | Stub returning 0.5 | fitness.py:183–188 |

---

## 8. Final synthesis — what this all adds up to

### Thesis 1: The architecture is unfalsified because it is untested, and untested because the eval points at the wrong regime.
The single most consequential combination of findings: (a) semantic partitioning never ran for real (Iter 2a); (b) no eval condition gets the evidence, and the prompt set needs none (Iter 3); (c) the 2026 compute-matched literature says multi-agent wins live where single-call context utilization degrades. The project has been grading its compression engine on tasks with nothing to compress, using baselines that don't control for retrieval, on a build that predates its flagship mechanism. **Every hour spent on internal mechanics before re-aiming the eval is unmeasurable.**

### Thesis 2: The system's selection dynamics are consensus-shaped in three independently-implemented places — and its epistemology says they shouldn't be.
Positive-only strength dynamics (Iter 2b), a fitness function that scores monotone growth 1.0 and contestation 0.2 (Iter 5), and a synthesis bottleneck that discards quiet minorities (Iter 4). The architecture's *story* is adversarial field pressure; its *arithmetic* is agreement accumulation, three times over. This coherence of error is actually good news: it means one design decision (make negative evidence a first-class force) fixes a family of pathologies, not one bug.

### Thesis 3: The port from ACO dropped the part of ACO that does the work.
Proportional evaporation and objective-coupled reinforcement are what make ant-colony optimization converge on *good* solutions rather than *early* ones. This system has rank-preserving decay and reinforcement coupled to internal proxies. That's not stigmergy failing; it's stigmergy not yet implemented.

### Thesis 4: The cheap fixes are disproportionately valuable because the expensive machinery is already built.
One-line: audit/resolve swap. Small: signed trail deltas, dissent dedup weighting, render-signature keying, phantom +10, partitioner provenance log, delete stub/dead terms. Medium: condition F + over-context task set, grounded quality gate, evaporation decay. The projection/genome/topology superstructure is elaborate and mostly sound *as read-out machinery* — it's the forces feeding it and the test bench around it that are broken.

### Thesis 5 (the hater's closing argument): consider that the blackboard result already refutes the pheromone layer.
The strongest external challenge to this codebase isn't "multi-agent doesn't work" — it's that [a blackboard with no strength dynamics beats orchestrated baselines by 13–57%](https://arxiv.org/pdf/2510.01285v1), and [compute-matched single agents beat message-passing systems](https://arxiv.org/abs/2604.02460) when context suffices. The minimal architecture consistent with all evidence in this session is: **facet-diversified retrieval → shared typed artifact store (no strength arithmetic) → deterministic projection → compression synthesis** — i.e., this repo with the pheromone layer deleted. The strength dynamics have to *earn* their complexity in an ablation against that null model, and right now nothing in the empirical record shows they do. The counter-position (Iter 2b dissent): the store's value may be economic (token compression at scale) rather than epistemic — which is testable on the same over-context benchmark as everything else.

### The ordered plan (if the next month had to be scheduled from this document)
1. **Eval**: condition F (single-call RAG, same pack) + evidence packs at 1×/4×/16× context; off-family judge; length control; informative-n reporting.
2. **One-liners**: audit/resolve swap; partitioner provenance; phantom +10; delete `entity_resolution` stub and `DELTA_AMPLIFY`.
3. **Make dissent exist**: signed trail amplification + dissent dedup-count weighting + re-signed consensus term.
4. **Make halting mean something**: grounded flagship quality gate; cluster-id render signature; INITIAL-only novelty at base threshold.
5. **Then and only then**: the corrected partition probe (semantic gate engaged, chat-templated model, V condition) and the null-model ablation (strength dynamics off vs on) on the over-context benchmark.
6. Update CLAUDE.md against the doc-rot ledger (Section 7) — eleven entries.

### Sources (session literature)
[Verbalized Sampling](https://arxiv.org/abs/2510.01171) · [The Price of Format](https://arxiv.org/html/2505.18949v1) · [Correlated Errors in LLMs](https://arxiv.org/pdf/2506.07962) · [Agent Scaling via Diversity](https://arxiv.org/html/2602.03794v1) · [More Agents Is All You Need](https://arxiv.org/pdf/2402.05120) · [LLM Blackboard System](https://arxiv.org/pdf/2510.01285v1) · [CodeCRDT](https://arxiv.org/pdf/2510.18893) · [Single-Agent > Multi-Agent at Equal Budgets](https://arxiv.org/abs/2604.02460) · [Illusion of Multi-Agent Advantage](https://arxiv.org/pdf/2606.13003) · [Pareto-Optimal Multi-Agent Test-Time Scaling](https://arxiv.org/pdf/2605.01566) · [Position Bias in LLM Judges](https://mbrenndoerfer.com/writing/position-bias-in-llm-judges) · [Fairness or Fluency (judge language bias)](https://arxiv.org/pdf/2601.13649) · [LLMs as Judges Not Robust to Artifacts](https://arxiv.org/pdf/2503.09347)

---
*Generated by a 2-hour self-dissenting critique loop (Claude Fable 5), 2026-07-06. All file:line cites read from the working tree this session; subagent audits: signal store, convergence+synthesis (completed), eval harness (completed inline after subagent hit session limit).*

---

## Post-loop correction (found during implementation, 2026-07-06 evening)

Iteration 2a understated the problem. While wiring `summary.json: partitioner`, a MOCK smoke run returned `partitioner: ""` — because **`partition_for_scouts` is not in the default pipeline at all**. Verified: the corpus-partition block (`run_swarm.py:396–475`, including the facet web partitions) belongs to `run_pipeline`, i.e. the `--legacy-rounds`/phase-isolated paths. `run_continuous_pipeline` (the default) builds no partitions; in the continuous pool, `partition_id` is the depositing worker's agent_id (`core/worker_pool.py:1302–1314`) and scouts fetch evidence by per-action live search.

Consequences:
1. "Scouts get disjoint corpus partitions" (CLAUDE.md principle #2) describes the legacy path only. On the default path there are no corpus partitions to be disjoint.
2. The July-3 semantic-partitioning commit improves a code path production doesn't execute.
3. hey fable's "support_diversity=62 = workers stamped as partitions" observation is not an inflation bug — it is the literal definition of partition on the default path.
4. The deployed diversity levers are: query-planner stance queries, multi-claim scout + novelty selection, model-family heterogeneity. Any partitioning experiment must either run `--legacy-rounds` or first wire corpus partitions into the continuous pool (next-tranche feature).

CLAUDE.md now states this under Rules of evidence. `summary.json: partitioner` distinguishes the paths going forward ("" = continuous/no corpus partitioning).
