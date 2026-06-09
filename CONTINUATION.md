# CONTINUATION — pick up where we left off

This doc is **gitignored** (`.gitignore` in this dir excludes it). Hand it to the next session.

Last session pushed 6 commits to `main` on `github.com/sfuqua6/Stigmeric-Coordination`. The repo is now running the **continuous worker pool** by default — there are no longer "rounds" or "phases."

---

## 1. What the user must do for DDG / search to actually work

The new search path is **Tavily → DDG → Cohere (opt-in)**. To make it work on a fresh Colab/laptop session:

### Required (one of):

```bash
# Best option: Tavily — 1000 free queries/month, lower latency, better quality
pip install tavily-python
export TAVILY_API_KEY="tvly-..."   # get from https://tavily.com (free signup)

# OR fallback: ddgs (renamed from duckduckgo-search; the old package warns on import)
pip install ddgs
```

The pipeline auto-detects which one is available and prints which is primary at startup:
- `[search] tavily key found; primary=tavily`
- `[search] no TAVILY_API_KEY; primary=duckduckgo (fallback)`

If DDG is being rate-limited (it does this fairly aggressively), set `TAVILY_API_KEY` and re-run.

### Optional / disabled by default:

- **Cohere wiki-simple** (last-resort retrieval). Re-enable with `SWARM_SEARCH_USE_COHERE=1`. It was disabled because (a) the first call triggers a ~1 GB dataset download, and (b) it reintroduces the topic-skew the agentic-search rewrite was meant to escape.

### Mock mode:

`MOCK_LLM=1` short-circuits `search()` to deterministic placeholder chunks. No network calls. Used by every smoke test.

---

## 2. Do agents call for more info when they need it?

**Yes — three of the seven actions issue searches:**

| Action | Calls search? | When |
|---|---|---|
| `SCOUT` | **Always** | Every iteration: query phrasing rotated from user prompt → search → SEARCH signal deposited → INITIAL grounded in retrieved chunks |
| `DEVELOP` | **Conditionally** | When the sampled INITIAL has < 2 SUPPORT children (sparse-cluster trigger). Query is the first ~8 words of the INITIAL content |
| `VALIDATE` | **Always** | Every iteration: search snippet is injected as the "EXTERNAL SNIPPET" block in the validator prompt |
| `CHAIN` | No | Reasons from a sampled SUPPORT signal only |
| `CRITIQUE` | No | Reasons from a sampled INITIAL only |
| `OBJECT` | No | Reasons from a DBSCAN cluster's representatives |
| `REFINE` | No | Reasons from a surviving-but-unverified INITIAL only |

Each SEARCH deposit appears as its own typed signal in the store with metadata `{"query": "...", "n_results": N, "trigger": "sparse_support"|"scout"|...}`. Other agents see it via the store and could in principle avoid re-querying — that **dedup-on-SEARCH** logic is **not yet wired**; currently each agent issues its own query independently. See "Open work" below.

---

## 3. Where the work left off (last 6 commits)

```
8f974f0 fix(kb): restore [kb] loaded banner on the continuous pipeline path
d17b5d1 fix(validator): JSON-structured output, raw logging, robust parse
fdd6932 feat(convergence): ConvergenceDetector with quality gate + saturation + caps
45464d3 feat(pool): continuous Worker pool replaces round/phase scheduler
d7c3af6 feat(actions): action templates replace per-role classes; share-based balancing
49bf3ac chore(deps): ddgs rename, clean vLLM shutdown, JIT warmup
```

These implement two prior directive scopes:

- **"Continuous emergent swarm"** (Phase 0 + Phase 1–5)
- **"Scaffolding for the continuous swarm"** (6A action preconditions, 6B cold-start, 6D worker cooldown, 6E halt floors, 6G dual-validator quality gate, 6H per-tick logging)

Phase 4-of-the-A100-prompt — adaptive **Worker with field-state action selection** — was the listed-as-optional phase. It's been **implemented** as the default; this is the worker pool itself, not a separate item.

---

## 4. How to run

```bash
# Default: continuous worker pool, agentic search, KB off
python run_swarm.py debate "Was the Apollo program economically justified?"

# Smaller pool (default is 24; on laptop/GGUF reduce to ~4)
python run_swarm.py debate "..." --workers=4

# Old round/phase scheduler (A/B comparison)
python run_swarm.py debate "..." --legacy-rounds

# Mock smoke test (no LLM, no network)
MOCK_LLM=1 python run_swarm.py debate "Test"

# A100 / Colab — model is Qwen2.5-32B-Instruct, vLLM:
#   - max_model_len=4096
#   - enforce_eager=False
#   - JIT warmup fires automatically at engine init
```

`summary.json` in each run dir now includes:
```json
{
  "execution_mode": "continuous_pool",
  "total_iterations": ...,
  "wall_clock_s": ...,
  "quality_met": false,
  "convergence_reason": "quality" | "saturation" | "cap_iterations" | "cap_time",
  "action_shares": {"SCOUT": 0.22, "DEVELOP": 0.38, ...}
}
```

---

## 5. Test status

`pytest tests/` runs in ~4–5 min and reports **195 passed, 7 skipped** as of the last commit. The skipped tests are embedder-related (sentence-transformers offline) and were pre-existing.

`tests/test_phase_isolation.py` still passes — phase-isolated execution (`--isolated --phase=X`) is preserved as a legacy path, separate from the new continuous pool.

---

## 6. Known issues / open work

### Empirically untested
- **No real-LLM run has been done yet** with the continuous pool. Smoke tests use MockLLM (deterministic SHA-seeded phrases, no real semantics). The first vLLM run will reveal whether:
  - The convergence detector's `quality` halt path ever fires (or if everything halts via `saturation` / `cap`)
  - Action share floors/ceilings stay within their bands once dedup isn't dominating rejections
  - The dual-validator gate is actually reachable with real verification scores

### Known cosmetic issues
- In mock mode the worker pool races past `MAX_ITERATIONS=2000` and halts ~12× over the cap (saw 23779 in 62s). This is because the detector only ticks every 2s and 24 workers churn fast with MockLLM. On vLLM each iteration is ~hundreds of ms so this won't be visible. If you want a hard cap, change `worker_loop` to check `pool_state.iteration_counter >= max_iterations` itself.
- `[swarm t=Xs iter=N]` logs print every 2s; on a 15-minute run that's ~450 lines. Increase the sleep in the orchestrator loop if it's too noisy.

### Open work — search dedup
Currently each SCOUT/DEVELOP/VALIDATE worker calls `search_tool.search()` independently. The SEARCH signals deposited in the store are *visible* to other agents but **not consulted before issuing a new query**. Worth wiring: in `worker_pool._gather_target` for SCOUT/DEVELOP, check `store.by_type(SEARCH)` for an exact `query` match before calling `_search()`; reuse the prior result if found. ~30 lines of code, plus a similarity check on query strings (sequence matcher >0.9 ratio is fine).

### Open work — critic / object / refine search hooks
The CRITIQUE / OBJECT / REFINE actions don't search. Hooks that could help:
- **CRITIQUE**: search for counterevidence to the target INITIAL when its strength is very high (> 0.8) — gives the critic actual external grounds rather than first-principle objections.
- **OBJECT**: search for evidence about the cluster's *shared assumption* once identified. Currently the hater names the weakness but doesn't ground it.
- **REFINE**: search for specifics that could make the unverified claim testable.

None of these are wired. Each would be ~20 lines in `worker_pool._gather_target`.

### Open work — heterogeneous routing
`--heterogeneous` still goes through `run_pipeline` (the legacy round/phase scheduler) because the multi-model router is shaped around per-phase model loading. To bring it into the continuous pool: the router would need to expose `acquire(role)` from a worker iteration; workers would need to declare which model they want per action and acquire on demand. Substantial work.

---

## 7. Validator diagnostic recipe

If validator scores are systematically 0.0 (the Apollo-program failure mode the user mentioned):

1. Open `<output_dir>/validator_raw.log` — the first 5 VALIDATE raw outputs are saved verbatim.
2. Check whether the model emitted valid JSON or natural-language prose.
   - **Valid JSON, score=0.0:** prompt/snippet failure (search returned irrelevant content)
   - **Natural language, score=0.0:** prompt failure (model ignored the JSON format directive)
3. The validate_parse path is JSON-first, falls back to `SCORE: X` regex, then defaults to 0.5. Score==0.0 means "supports: false, confidence: ~1.0" came through — i.e. the model actively rejected the claim, not parse failure.

---

## 8. Files of interest

```
core/actions.py            — action templates (SCOUT/DEVELOP/CRITIQUE/OBJECT/VALIDATE/CHAIN/REFINE)
core/worker_pool.py        — Worker class + choose_action + run_pool + decay_loop
core/convergence.py        — ConvergenceDetector with floors/quality/saturation/cap
core/search_tool.py        — Tavily/DDG/Cohere cascade with MOCK_LLM short-circuit
core/signal_types.py       — INITIAL/SUPPORT/CRITIQUE_*/OBJECTION/VERIFICATION/SEARCH
core/projection.py         — survival filter (untouched; honors no-leak rule)
agents/base.py             — strip_reasoning, type_parent_instruction, parse_type_proposal
                             (legacy role classes still exist for backward compat / tests)
agents/scout.py            — legacy Scout; the continuous pool uses worker_pool.Worker instead
run_swarm.py:run_continuous_pipeline — the new default orchestrator
run_swarm.py:run_pipeline             — legacy --legacy-rounds path
```

---

## 9. If you have to back out

Every change is committed. To revert to the pre-pool state:

```bash
git revert d17b5d1 fdd6932 45464d3 d7c3af6 49bf3ac
```

That preserves `--legacy-rounds` behavior (which is unchanged from before; the new commits added the pool path *alongside* it, they didn't remove the rounds path).

To partially back out — keep agentic search but lose the continuous pool — revert only `d17b5d1 fdd6932 45464d3`.
