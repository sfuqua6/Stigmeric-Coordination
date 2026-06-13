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
| 9 | Projection + genome/fitness | `core/projection.py` `build_projection` | thresholds in `config.py` | `[CLUSTER AUDIT] rep=… members=…` | ✅; lattice fields (frames/propositions/cross-level) are **dead freight** — no consumers |
| 10 | Planner — **deterministic `build_plan` only** (LLM planner retired to default-off after 3/3 context failures; no LLM call at planning) | `build_plan` (projection.py); `_plan_synthesis` opt-in | `RENDER_K` (5), `SWARM_RENDER_K`; ablation opt-in `SWARM_USE_LLM_PLANNER=1` | `[PLAN] render=… render_ids=[…]` | ✅ context wall at planning eliminated |
| 11 | Global composition (Stage 2; thesis-first; conditions conclusion; verification-calibrated figures) | `_compose_answer` in `synthesizer.py` | `_SYNTHESIZER_USE_GLOBAL_COMPOSITION`, `_GLOBAL_COMPOSE_MAX_TOKENS` | `[synthesizer] global composition: N briefs -> M chars` | ✅ working since groqgroq |
| 12 | Citation-tag aliasing ([S#] in, real tags out) | `_alias_citation_tags`/`_unalias_citation_tags` | `_COMPOSE_MIN_TAG_RETENTION` (0.5) | failure mode now rare: `dropped too many citation tags` | 🆕 fixes 3/3 dissent-compose tag drops |
| 13 | Dissent cap + composition + overflow one-liners | Section 2 block in `_render`; `_compose_dissent` | `_DISSENT_RENDER_CAP` (6), `_DISSENT_OVERFLOW_CAP` (8), `_DISSENT_COMPOSE_FRAGMENT_CHARS` (900) | `dissent composition: N notes -> M chars`; `Further contested positions` in answer | compose ✅ once (groqgroq), tag-drop now aliased |
| 14 | Contradiction pairs (deduped, capped, compact block) | `_contradictions_from_projection` | `_CONTRADICTION_CAP` (5) | `Cluster pairs in direct tension` in answer | ✅ |
| 15 | Verification / SAFE atoms | validators + `_build_atoms` | — | `summary.json: max_verification_score`, `total_atoms` | ⚠️ **weakest link**: scores ~0.04 avg while answers now carry specific figures — rule 9 (claimed-vs-established phrasing) mitigates, real fix is validator coverage |
| 16 | Revision loop (Sections 1–2 only) | `_revision_loop` call site in `_render` | `_SYNTHESIZER_REVISION_ROUNDS` (1); skipped on Groq | `revision round 0:` … | scoped 2026-06-12 after it ate Section 3 + PROCESS NOTES |
| 17 | Extractive fallbacks (Section 1/2 never empty) | `_extractive_position`; dissent extract | — | `using extractive fallback for Section 1` | ✅ |
| 18 | Faithfulness audit (4-gram per citation) | `_build_faithfulness_audit` | `_AUDIT_WARNING_THRESHOLD` (20) | `faithfulness audit: N flag(s)`; `renderer_audit.json` | ✅; audits prose-vs-signals, NOT signals-vs-reality |
| 19 | Knowledge base (cross-run memory) | `core/knowledge_base.py` | **off by default — `--use-kb`**; `--reset-kb` | `[kb] loaded N priors`; `[kb] saved … (dedup: N merged, contradictions: N)`; `summary.json: kb_diff` | ✅ consolidating (new 25→11, matched 2→10 over 3 runs) |
| 20 | Abstention gates (structural + genome) | `_render` top | `_ABSTAIN_*` | `ABSTAINING:` / `GENOME ABSTAIN` | ✅ |
| 21 | Search backend chain (Tavily→[SE]→DDG→follow-up→**Wikipedia**→Cohere) | `core/search_tool.py` `search()` | `TAVILY_API_KEY` (unset = DDG primary) | `[search] tavily ok` / `ddg ok` / `wikipedia fallback` / `*** ALL BACKENDS FAILED ***` | Wikipedia fallback 🆕 — DDG is no longer a single point of failure |
| 22 | Stack Exchange backend (coding tasks; free, no key, ~300 req/day) | `_stackexchange_search`; merged before diversify when `task_type="coding"` | — | `[search] stackexchange ok: N results` | 🆕 untested in real run |
| 23 | Result quality: source cap → relevance gate → BM25+dense RRF → **fact-density + coding-domain priors** → MMR → page enrichment (DDG top-K) | `_diversify`, `_quality_rerank`, `_enrich_with_pages` | `SWARM_SEARCH_FACT_DENSITY_WEIGHT` (0.15), `SWARM_SEARCH_CODING_DOMAIN_BOOST` (0.25), `SWARM_SEARCH_FETCH_*`, `SWARM_SEARCH_RELEVANCE_MIN` | `[search] enriched N/K chunks`; `relevance gate: kept N/M` | priors 🆕 — feed the particulars gate fact-dense evidence |
| 24 | Query planning (task-aware stances; fragments mined from high-strength INITIALs — stigmergic; step-back + HyDE; dedup fingerprints; per-pool search budget) | `core/query_planner.py`; budget in `worker_pool.py` | stance tables `_STANCE_BY_TASK` (coding rows exist) | served-query dedup is silent; HyDE/step-back log via worker | ✅ |
| 25 | SEARCH signal traces (query + top URLs + **content excerpt**) | `summarize_for_signal` | — | deposits contain `TOP: …` line | excerpt 🆕 — traces now carry facts, not just URLs |
| 26 | **Number-grounding gate** (STORM-style): figures in SCOUT/DEVELOP/CHAIN/REFINE deposits must appear in shown evidence (chunks/parent/task prompt) — reject as fabrication when evidence was present; tag `numbers_grounded=false` when not; briefs mark `(UNSOURCED FIGURES — present as claimed)` | `ungrounded_numbers` (core/actions.py); gate in `worker_pool.py` | — | `REJECT … ungrounded figure(s) … (fabrication gate)` | 🆕 — answers the fabricated-Oslo/Copenhagen failure |

## Log grep cheat-sheet (paste into PowerShell/Select-String or grep)

```
scout multiclaim|REJECT (DEVELOP|CHAIN)|kb\] |embedder|CLUSTER\] JOIN|SKIP 'configured'
PLAN\] render|digest capped|char budget reached|plan call failed
global composition|dissent composition|extractive fallback|revision round
faithfulness audit|ABSTAIN|quality_met|degraded|kb_diff|centroid_cosine|self_bleu
search\] |stackexchange|wikipedia fallback|enriched|relevance gate|ALL BACKENDS FAILED
```

## Health check, per run (summary.json)

`embedder` = model name (not UNAVAILABLE) · `degraded: false` · `quality_met: true` ·
`largest_cluster` > 1 · `kb_diff.matched` rising across same-topic runs ·
`max_verification_score` (want ≥0.3; currently ~0.2 — open problem) ·
`output_diversity.self_bleu` falling = more diverse prose.

## Known open issues

1. **Verification coverage** (#15) — specificity now outruns the validators.
   Mitigated by the number-grounding gate (#26: figures must come from shown
   evidence) and rule-9 phrasing; full SAFE-atom coverage still the real fix.
2. **Dead lattice** (#9) — `propositions` (no builder/consumer), `frames`/`cross_level_edges`
   (no consumers), `cluster_sensitivities` (expensive build, one decorative read),
   planner `merge_groups` (asked for, then discarded). Deletion/gating decision pending.
3. **Within-cluster evidence selection** — `support_set[:3]` renders the *first-deposited*
   supports, not the strongest ("EvidenceCard" refactor proposed).
4. **LLM planner ablation** — retire `_plan_synthesis` if it shows no lift over `build_plan`.
5. **Baseline A/B** (DEFERRED P4.1) — still the missing measurement for the whole thesis.
