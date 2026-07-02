# Pipeline Map — the quick key

One page: every mechanism, where it lives, its knob, and the **exact log line to grep**
in a Colab/run log to confirm it fired. Updated 2026-06-12.

## End-to-end flow

```
                        ┌─ EXPLORATION (parallel, context is cheap) ─────────────────┐
 corpus/search ─► SCOUT │ K-claim portfolio ─► novelty selection (vs field + KB) ─►  │
                        │ INITIAL deposit ─► dedup pre-screen ─► cluster join        │
                        │ DEVELOP/CHAIN ─► paraphrase gate ─► SUPPORT (+source_tags) │
                        │ CRITIQUE / OBJECT ─► dissent     VALIDATE ─► atoms/scores  │
                        └─ decay · prune · trail-amplify ────────────────────────────┘
                                              │
                        ┌─ READOUT (bounded, one deciding mind) ─────────────────────┐
 build_projection ─► classify clusters ─► genome/fitness ─► PLAN (python MMR;        │
   LLM planner optional, digest char-budgeted)                                       │
   ─► per-cluster briefs (Stage 1, ≤RENDER_K) ─► GLOBAL COMPOSITION (Stage 2,        │
       alias tags in / real tags out, thesis-first, conditions conclusion)           │
   ─► dissent: top-6 renders ─► DISSENT COMPOSITION ─► overflow one-liners           │
   ─► Section 3/5/6 (deterministic) ─► revision (Sections 1-2 ONLY) ─►               │
       PROCESS NOTES last ─► citations stamp ─► faithfulness audit                   │
                        └────────────────────────────────────────────────────────────┘
                                              │
                              KB save (consensus/contradictions) ─► next run's
                              novelty references + projection priors
```

**Invariant:** composer/planner input is O(K × brief), never O(field) or O(corpus).
The swarm explores wide; exactly one bounded call writes.

## Mechanism key

| # | Mechanism | Where | Knob / env | Log signature (grep this) | Status |
|---|---|---|---|---|---|
| 1 | Multi-claim scout (K-portfolio + novelty selection) | `core/actions.py` `split_scout_claims`/`select_novel_claim`; wired in `worker_pool.py` + `agents/scout.py` | `SCOUT_CLAIMS_PER_CALL` (4), `SWARM_SCOUT_CLAIMS`, `SCOUT_NOVELTY_RECENT_N` (30) | `[scout multiclaim] 4 candidates; selected least-similar (max_sim=…)` | ✅ working (43–73 fires/run) |
| 2 | KB-aware novelty (priors steer scouts away from known claims) | `SignalStore.set_novelty_references`; wired in `run_swarm.py` after `kb.load()` | needs `--use-kb` | `[kb] N prior-consensus claims registered as scout novelty references` | 🆕 untested in real run |
| 3 | Paraphrase-support gate (DEVELOP/CHAIN must add info) | `core/actions.py` `support_adds_information`; wired in `worker_pool.py` | `_SUPPORT_PARAPHRASE_RATIO` (0.72) | `REJECT DEVELOP: paraphrase of parent` | ✅ live; 0 fires so far (prompt compliance) — watch threshold |
| 4 | Particulars demand (numbers/names in supports) | `develop_prompt`/`chain_prompt` in `core/actions.py` | prompt-side | (see Section 1 prose: named cities/figures) | ✅ working (kb run3) — fabrication risk, see #15 |
| 5 | Source tags into briefs | `worker_pool.py` (meta) → `_render_cluster_position` | — | `(source: …)` in briefs | 🆕 untested in real run |
| 6 | Dedup pre-screen + cluster-join | `signal_store.py` `deposit()` / `ClusterRegistry` | `SWARM_CLUSTER_JOIN_*` | `[CLUSTER] JOIN … centroid_sim=…` | ✅ working when embedder alive |
| 7 | Embedder fallback chain (ST → transformers AutoModel) | `signal_store.py` `_try_load_embedder` | — | `[store] embedder loaded:` / `transformers AutoModel fallback` / `*** embedder UNAVAILABLE` | ✅; check `summary.json: embedder` every run |
| 8 | Cascade VRAM pre-check (skip doomed fp16 rungs) | `core/llm.py` `_should_skip_attempt` | — | `[llm-cascade] SKIP 'configured' … weights ≈N GB exceed` | 🆕 untested in real run |
| 9 | Projection + genome/fitness | `core/projection.py` `build_projection` | thresholds in `config.py` | `[CLUSTER AUDIT] rep=… members=…` | ✅; dead lattice fields (frames/propositions/cross_level_edges) **removed 2026-06-18** — `atoms` + `cluster_sensitivities` kept (feed `genome.atoms`/`genome.sensitivity`) |
| 10 | Planner — **deterministic `build_plan` only** (LLM planner retired to default-off after 3/3 context failures; no LLM call at planning) | `build_plan` (projection.py); `_plan_synthesis` opt-in | `RENDER_K` (5), `SWARM_RENDER_K`; ablation opt-in `SWARM_USE_LLM_PLANNER=1` | `[PLAN] render=… render_ids=[…]` | ✅ context wall at planning eliminated |
| 11 | Global composition (Stage 2; thesis-first; conditions conclusion; verification-calibrated figures) | `_compose_answer` in `synthesizer.py` | `_SYNTHESIZER_USE_GLOBAL_COMPOSITION`, `_GLOBAL_COMPOSE_MAX_TOKENS` | `[synthesizer] global composition: N briefs -> M chars` | ✅ working since groqgroq |
| 12 | Citation-tag aliasing ([S#] in, real tags out) | `_alias_citation_tags`/`_unalias_citation_tags` | `_COMPOSE_MIN_TAG_RETENTION` (0.5) | failure mode now rare: `dropped too many citation tags` | 🆕 fixes 3/3 dissent-compose tag drops |
| 13 | Dissent cap + composition + overflow one-liners | Section 2 block in `_render`; `_compose_dissent` | `_DISSENT_RENDER_CAP` (6), `_DISSENT_OVERFLOW_CAP` (8), `_DISSENT_COMPOSE_FRAGMENT_CHARS` (900) | `dissent composition: N notes -> M chars`; `Further contested positions` in answer | compose ✅ once (groqgroq), tag-drop now aliased |
| 14 | Contradiction pairs (deduped, capped, compact block) | `_contradictions_from_projection` | `_CONTRADICTION_CAP` (5) | `Cluster pairs in direct tension` in answer | ✅ |
| 15 | Verification / SAFE atoms (API path batches: decompose+plan, then score-all — 2 calls, full atom coverage instead of cap-to-1). **No-result abstain (2026-06-18):** atoms with no retrieved snippet abstain at 0.5 instead of being scored ~0.0 by the LLM on an empty snippet (false refutation); aggregate is now over evidence-backed atoms only, with `verification_coverage` reported separately | validators + `_build_atoms`; `_safe_decompose_and_plan`/`_safe_score_atoms_batch`/`_is_no_result_snippet` in `worker_pool.py` | `SWARM_SAFE_BATCH_ATOMS` (1=on, API only), `SWARM_SAFE_BATCH_MAX_ATOMS` (3) | `summary.json: max_verification_score`; signal metadata `verification_coverage`, `evidenced_atom_count`; brief `EVIDENCE COVERAGE: N%` | ⚠️ improving: no-result deflation fixed (the ~0.04 floor was empty-snippet false-refutation); next lever is retrieval recall on narrow atom queries + scoring rubric (now: 0.0 = explicit contradiction only) |
| 16 | Revision loop (Sections 1–2 only) | `_revision_loop` call site in `_render` | `_SYNTHESIZER_REVISION_ROUNDS` (1); skipped on Groq | `revision round 0:` … | scoped 2026-06-12 after it ate Section 3 + PROCESS NOTES |
| 17 | Extractive fallbacks (Section 1/2 never empty) | `_extractive_position`; dissent extract | — | `using extractive fallback for Section 1` | ✅ |
| 18 | Faithfulness audit (4-gram per citation) | `_build_faithfulness_audit` | `_AUDIT_WARNING_THRESHOLD` (20) | `faithfulness audit: N flag(s)`; `renderer_audit.json` | ✅; audits prose-vs-signals, NOT signals-vs-reality |
| 19 | Knowledge base (cross-run memory) | `core/knowledge_base.py` | **off by default — `--use-kb`**; `--reset-kb` | `[kb] loaded N priors`; `[kb] saved … (dedup: N merged, contradictions: N)`; `summary.json: kb_diff` | ✅ consolidating (new 25→11, matched 2→10 over 3 runs) |
| 20 | Abstention gates (structural + genome) | `_render` top | `_ABSTAIN_*` | `ABSTAINING:` / `GENOME ABSTAIN` | ✅ |
| 21 | Search backend chain (Tavily→[SE]→DDG→follow-up→**Wikipedia**→Cohere) | `core/search_tool.py` `search()` | `TAVILY_API_KEY` (unset = DDG primary) | `[search] tavily ok` / `ddg ok` / `wikipedia fallback` / `*** ALL BACKENDS FAILED ***` | Wikipedia fallback 🆕 — DDG is no longer a single point of failure |
| 22 | Stack Exchange backend (coding tasks; free, no key, ~300 req/day) | `_stackexchange_search`; merged before diversify when `task_type="coding"` | — | `[search] stackexchange ok: N results` | 🆕 untested in real run |
| 23 | Result quality: source cap → relevance gate → BM25+dense RRF → **fact-density + coding-domain priors** → MMR → **page enrichment of FINAL survivors** (2026-06-18: moved from raw DDG top-K inside `_ddg_search` to `_diversify(enrich_pages=True)`, so pages are fetched only for results that survive ranking — quality+latency) | `_diversify`, `_quality_rerank`, `_enrich_with_pages` | `SWARM_SEARCH_FACT_DENSITY_WEIGHT` (0.15), `SWARM_SEARCH_CODING_DOMAIN_BOOST` (0.25), `SWARM_SEARCH_FETCH_*`, `SWARM_SEARCH_RELEVANCE_MIN` | `[search] enriched N/K chunks`; `relevance gate: kept N/M` | priors 🆕 — feed the particulars gate fact-dense evidence |
| 24 | Query planning (task-aware stances; fragments mined from high-strength INITIALs — stigmergic; step-back + HyDE; dedup fingerprints; per-pool search budget) | `core/query_planner.py`; budget in `worker_pool.py` | stance tables `_STANCE_BY_TASK` (coding rows exist) | served-query dedup is silent; HyDE/step-back log via worker | ✅ |
| 25 | SEARCH signal traces (query + top URLs + **content excerpt**) | `summarize_for_signal` | — | deposits contain `TOP: …` line | excerpt 🆕 — traces now carry facts, not just URLs |
| 26 | **Number-grounding gate** (STORM-style): figures in SCOUT/DEVELOP/CHAIN/REFINE deposits must appear in shown evidence (chunks/parent/task prompt) — reject as fabrication when evidence was present; tag `numbers_grounded=false` when not; briefs mark `(UNSOURCED FIGURES — present as claimed)` | `ungrounded_numbers` (core/actions.py); gate in `worker_pool.py` | — | `REJECT … ungrounded figure(s) … (fabrication gate)` | 🆕 — answers the fabricated-Oslo/Copenhagen failure |
| 28 | **Adaptive relevance gate** (2026-06-18, default OFF): when `SWARM_SEARCH_RELEVANCE_REL_MARGIN > 0`, the relevance gate also drops chunks more than that cosine margin below the top result (effective threshold `max(SEARCH_RELEVANCE_MIN, top_sim − margin)`) — prunes clearly-inferior evidence, adapting to query difficulty. Floor + keep-≥2 still apply. Ship gated so default field behaviour is unchanged until A/B'd | `_relevance_filter` (search_tool.py) | `SWARM_SEARCH_RELEVANCE_REL_MARGIN` (0.0=off) | `relevance gate: kept N/M (>= X)` (threshold now adaptive) | 🆕 untested in real run — flag default-off |
| 29 | **Ingestion recall program** (2026-07-01, evidence-intake fix — the swarm's only edge over a flagship is retrieved particulars): (a) **relaxed-query rescue** — `relax_query()` (deterministic keyword reduction, numbers/proper-nouns prioritized) retried once when a planned atom query or a whole search returns zero hits, in both `worker_pool._retrieve_for_query` and the last rung of `_search_impl`; (b) **wider atom-evidence window** — `_snippet_from_hits`: top-2 sources × 350 chars (was 1 × 300), scorer snippet caps 350→750 to match; (c) **Semantic Scholar aux backend** (key-free) merged alongside web results for analysis/debate tasks — abstracts carry the effect sizes/sample counts DDG snippets don't | `relax_query` (query_planner.py); `_snippet_from_hits` (worker_pool.py); `_semanticscholar_search`/rescue rung (search_tool.py) | `SWARM_SEARCH_SCHOLAR` (1=on, 0=off) | `[safe] relaxed-query rescue:` / `[search] relaxed-query rescue:` / `[search] semanticscholar ok:` | 🆕 tested (tests/test_ingestion_recall.py, 12); watch `verification_coverage` + `avg_verification_score` on next real run |
| 30 | **Compute program pt 1** (2026-07-01, stop paying for post-saturation churn): (a) **render-set stability halt** — the detector tracks the top-`RENDER_K` set `build_plan()` would hand the composer, plus each selected cluster's evidence counts; unchanged for `RENDER_STABLE_ITERS` (40) iterations ⇒ halt `render_set_stable`. Robust to dust-cluster fragmentation, which resets every other counter (why real runs died on `cap_time`); (b) **pre-call scout gate** — `scout_gate_engaged()`: once field novelty < `SCOUT_GATE_NOVELTY_FLOOR` (0.10) over a full window, SCOUT draws are demoted to exploit actions BEFORE the LLM call (novelty filtering used to run after generation, when the call was already spent) | `_render_signature`/halt (convergence.py); `scout_gate_engaged` + `iterate()` wiring (worker_pool.py) | `SWARM_RENDER_STABLE_ITERS` (40, 0=off); `SWARM_SCOUT_GATE_NOVELTY_FLOOR` (0.10, 0=off), `SWARM_SCOUT_GATE_MIN_INITIALS` (8), `SWARM_SCOUT_GATE_WINDOW` (40) | `convergence_reason: render_set_stable`; `[gate] scout demoted`; `summary.json: scout_gate_skips` | 🆕 tested (tests/test_compute_gates.py, 8); quantify wall-clock + skip count on next real run |
| 27 | **Latency program** (2026-06-18, hold-quality/cut-wallclock): (a) **non-blocking search** — all `_search` via `asyncio.to_thread` so the blocking HTTP no longer freezes the single event loop (the dominant cost: 135 searches/545s in one run); (b) **enrich-after-rank** (#23); (c) **in-run page cache** (url→text, cross-worker); (d) **per-validator atom retrievals gathered**; (e) **synthesizer renders sized to the engine** — vLLM/Groq run all ≤6 briefs concurrently instead of `LLM_CONCURRENCY=1` serial; (f) **Tier-0 timing** in `summary.json` | `worker_pool.py` (to_thread, `_retrieve_for_query` async, gather); `search_tool.py` (`_search_impl`/stats/`_PAGE_CACHE`); `synthesizer.py` `_render_concurrency()` | `SWARM_SEARCH_FETCH_*`; `LLM_CONCURRENCY` no longer caps synth renders on batching engines | `summary.json: timing.search_fraction_of_wallclock`, `timing.search.{fetch_cache_hits,…}` | ✅ tested in MOCK; quantify on next real run via the timing block |

## Log grep cheat-sheet (paste into PowerShell/Select-String or grep)

```
scout multiclaim|REJECT (DEVELOP|CHAIN)|kb\] |embedder|CLUSTER\] JOIN|SKIP 'configured'
PLAN\] render|digest capped|char budget reached|plan call failed
global composition|dissent composition|extractive fallback|revision round
faithfulness audit|ABSTAIN|quality_met|degraded|kb_diff|centroid_cosine|self_bleu
search\] |stackexchange|semanticscholar|wikipedia fallback|enriched|relevance gate|relaxed-query rescue|ALL BACKENDS FAILED
```

## Health check, per run (summary.json)

`embedder` = model name (not UNAVAILABLE) · `degraded: false` · `quality_met: true` ·
`largest_cluster` > 1 · `kb_diff.matched` rising across same-topic runs ·
`max_verification_score` (want ≥0.3; currently ~0.2 — open problem) ·
`output_diversity.self_bleu` falling = more diverse prose.

**Latency (`summary.json: timing`, added 2026-06-18):** `timing.search_fraction_of_wallclock`
is the headline — retrieval was historically the dominant sink (135 searches /
545s in one real run). `timing.search.{calls,total_s,max_s,empty_results,fetch_calls,
fetch_total_s,avg_s}` break it down. Searches now run off the asyncio event loop
(`asyncio.to_thread` in `worker_pool.py`) so they overlap with each other and with
GPU inference — expect `search_fraction_of_wallclock` to fall sharply vs. pre-fix
runs even though `search.total_s` (cumulative across threads) stays similar.

## Known open issues

1. **Verification coverage** (#15) — specificity outran the validators. The
   empty-snippet false-refutation deflation is **fixed (2026-06-18)**: no-result
   atoms abstain at 0.5, score aggregates over evidence-backed atoms only, and
   `verification_coverage` is reported separately. Mitigated further by the
   number-grounding gate (#26) and rule-9 phrasing. Retrieval recall on
   narrow atom queries is now attacked by the ingestion recall program (#29):
   relaxed-query retry + wider two-source evidence window + Semantic Scholar
   for analysis/debate. Quantify on the next real run: `verification_coverage`
   up = the ladder is landing evidence; `avg_verification_score` should follow.
2. **Dead lattice** (#9) — **RESOLVED (2026-06-18):** `propositions`,
   `frames`, `cross_level_edges` and their builders removed; `atoms` and
   `cluster_sensitivities` retained (they feed `genome.atoms`/`genome.sensitivity`).
   Planner `merge_groups` is still computed-then-discarded — folds into issue 4.
3. **Within-cluster evidence selection** — `support_set[:3]` renders the *first-deposited*
   supports, not the strongest ("EvidenceCard" refactor proposed).
4. **LLM planner ablation** — retire `_plan_synthesis` if it shows no lift over `build_plan`.
5. **Baseline A/B** (DEFERRED P4.1) — still the missing measurement for the whole thesis.
