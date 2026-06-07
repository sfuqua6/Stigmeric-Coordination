# Claude Code prompt — Whole-codebase hardening & QA pass (2026-06-07)

Scope: **the entire `Attempt At Cleaning/` folder** — every module, every entry
point, every test. Paste everything below the `---` line into a fresh Claude Code
session from the repo root, then `cd "Attempt At Cleaning"`.

This is a **consolidation + quality-assurance pass over the whole codebase**. It
grew fast (stock swarm, cluster blob-fix, HybridRouter, search-quality,
convergence fix) and accreted real bugs caught only at runtime. The job is to make
*everything in this folder* solid, correct, known, and trustworthy — not to add
anything.

---

You are hardening the stigmergic multi-agent swarm codebase in
`Attempt At Cleaning/`. Read `Attempt At Cleaning/CLAUDE.md` end-to-end first, then
this whole document. **Verify every claim against the live code and `git log` —
this memo will have drifted.**

## Mandate — what this pass IS and IS NOT

**IS:** systematically audit and harden **every file** in `Attempt At Cleaning/`.
For each module: know whether it's used or dead, that it imports cleanly, that its
behavior is tested, that it respects the project invariants, and that it has no
silent failures or latent correctness bugs. Delete dead code. Add tests for gaps.
Fix bugs. Consolidate duplication. Document the live surface.

**IS NOT:** new features, new roles, new task types, new backends, new config
knobs, or speculative rewrites. If you're writing a `def` for new behavior, stop —
out of scope. **Prefer deleting code over adding it.**

### Hard rules (non-negotiable)
1. **One change at a time** — its own small commit with a test.
2. **Test-with every behavior change.** No fix ships without a test that would
   have caught the regression.
3. **Real exit codes.** `pytest …; echo "EXIT=$?"` — never `pytest … | tail` (a
   pipe masks the failure; that exact mistake shipped 2 red tests this week).
   Never push with red tests; green full suite before every push.
4. **Do not casually refactor the load-bearing files:** `run_swarm.py`,
   `core/signal_store.py`, `core/projection.py`, `agents/synthesizer.py`,
   `core/worker_pool.py`, `core/cluster_registry.py`, `core/config.py`. Add behind
   flags / branch on `task_type`; don't rewrite shared paths.
5. **Preserve the invariants:** no-leak (agents see only signal artifacts, never
   reasoning/ancestry), `partition_id` on every INITIAL/SUPPORT, point-in-time
   (stock/backtest: no data after `as_of`), free-only (no paid APIs),
   single-6 GB-GPU target. See both CLAUDE.md files.
6. **Don't touch the stock swarm's "makes money" logic** — it's isolated,
   committed, and empirically *unvalidated*. QA its plumbing only; don't change
   its predictions.

## The core method: a module-by-module health sweep

The folder is large (entry points + `core/` ~30 modules + `agents/` + `eval/` +
`tools/` + `tests/`). Work it **systematically**, not ad hoc. First build a map,
then sweep highest-risk first.

### Per-module QA checklist (apply to EVERY `.py` in the folder)
For each module, answer and record:
1. **Purpose** — one line. (If you can't state it, that's a finding.)
2. **Reachable?** — `grep` its imports across the folder. If nothing imports it
   and it's not an entry point/test → **dead code candidate** (flag, then delete
   with a green suite).
3. **Imports cleanly** under `MOCK_LLM=1`, no GPU, no network.
4. **Tested?** — is its public surface covered? If not, add an **offline**
   (mock/fake) test. Note coverage gaps.
5. **Invariants** — does it touch no-leak / partition / point-in-time? If so,
   verify it upholds them (and that a test asserts it).
6. **Error handling** — no bare `except` that hides bugs; no unbounded loops;
   network/subprocess calls have timeouts; failures are loud, not silent.
7. **Config/env** — which `SWARM_*` / `config.*` does it read? Are they live and
   documented?

Output a table to `docs/MODULE_HEALTH_2026-06-07.md`: `module | purpose | used-by |
tested | risk | findings`. This map IS deliverable #1 and drives the sweep order.

## Known issues to fix (grounded, in priority order)

### P0 — Test isolation failure (the suite is currently untrustworthy)
`tests/test_baseline_mode.py` (6) and `tests/test_coding_roles.py` (3) **FAIL in
the full suite but PASS in isolation**:
```bash
MOCK_LLM=1 SWARM_MIN_TIME_S=0 SWARM_MIN_ITERATIONS=5 pytest tests/test_baseline_mode.py tests/test_coding_roles.py -q   # 25 pass
MOCK_LLM=1 SWARM_MIN_TIME_S=0 SWARM_MIN_ITERATIONS=5 pytest tests/ -q                                                   # those 9 fail
```
Global-state pollution / ordering — not a bug in those files. Find the leak:
module-level singleton, unrestored `monkeypatch`/`os.environ`, the embedder cache
(`search_tool._get_embedder`), `ClusterRegistry`/`SignalStore` state, or a mutated
`config.*`. Fix with autouse reset fixtures. **Until fixed, the full suite is not a
regression gate — so this is the first real task after the inventory.**

### P0 — Document a trustworthy full-suite baseline
Pin the EXACT invocation; drive unexpected failures to 0; record pass/skip counts
in `docs/QA_BASELINE_2026-06-07.md`. Subprocess/convergence tests need
`SWARM_MIN_TIME_S=0 SWARM_MIN_ITERATIONS=5`; **do NOT** add a blanket
`SWARM_MAX_ITERATIONS=20` (it breaks baseline/coding tests — learned this week).

### P1 — Coverage gaps (add offline tests; don't change behavior)
- Continuous-pool **recluster wiring** in `run_swarm.py` (only `recluster_type()`
  is unit-tested, not that the loop calls it safely).
- A **continuous-pool MOCK run that COMPLETES end-to-end** (today it never reaches
  synthesis on CPU — MockLLM floods the store, CPU embeddings are slow). Add a
  tiny-config (few workers, low `SWARM_MAX_TIME_S`) that deterministically writes
  `answer.txt`. Plumbing coverage, not a feature.
- Whatever the module map shows as untested-but-live.

### P1 — Surface silent failures
- **No search backend:** `[search] … duckduckgo_search not installed; falling back
  to cohere store only` → scouts get ZERO evidence and deposit ungrounded claims,
  silently. Make it a loud startup warning + a flag in `summary.json`.
- **Groq quota/429s:** clear teardown summary ("rate-limited N times").

### P2 — Verify the just-landed convergence fix
Caps are now absolute (`core/convergence.py`, `6.30`). Confirm the round-based and
phase-isolated paths also respect them. The `outputs/forevergroq.txt` run exposed
boundary-oscillation (surviving count flickers 6↔8 → saturation never fires) — add
a test that a stable macro-state still halts via the cap. Do NOT build a new
detector (feature).

## Whole-folder consolidation / dedup candidates (delete > add; each with a test)

The map will confirm these; investigate and remove dead/duplicated code:
- **Entry points:** `run_task.py` (the parent project's monolith — is it even
  reachable/used from `Attempt At Cleaning`?), `synthesize.py`, `diagnose.py`,
  `kb_migrate.py` — which are live?
- **LLM backends:** `llm.py`, `llm_router.py`, `llm_vllm.py`, `llm_gguf.py`,
  `llm_groq.py`, `llm_hybrid.py`, `corpus_store_cohere.py` — map which are actually
  constructed at runtime; flag unreachable ones.
- **Retrieval stack overlap:** `retrieval.py`, `search_tool.py`, `query_planner.py`,
  `facet_planner.py`, `intake.py` — overlapping responsibilities; document the real
  call graph, retire dead paths.
- **Sampling / actions / coordination:** `sampling.py`, `actions.py`,
  `stage_coordinator.py` (if present) — used by the continuous pool? the round
  path? both? neither?
- **Known aliases/legacy:** `forager` ↔ `developer`; legacy multiplicative logit
  path (gated by `USE_LOGIT_DYNAMICS`); the two notebooks (`colab_run.ipynb` vs
  `colab_swarm.ipynb`) — keep one, mark/delete the other.
- **Config sprawl:** enumerate ALL `SWARM_*` env vars + `config.py` constants; flag
  unread/stale; emit one "live knobs" table in `docs/`. Do not add knobs.
- **Stale docs:** the `*_OVERHAUL_PROMPT.md` files, wrong test counts in
  `notebooks/` and CLAUDE.md ("322 tests pass" is stale).

## QA tooling to use
- **`code-review`** skill on each diff (correctness + simplification).
- **`security-review`** on `core/search_tool.py` page-fetch (outbound HTTP: check
  timeouts, SSRF surface, content-size limits, redirect handling).
- `diagnose.py` as a smoke gate after changes.

## Staged plan (do in order; each: Goal / Acceptance / Done-when)

- [ ] **Stage 0 — Baseline.** Run the full suite (documented invocation); reproduce
  the P0 isolation failures; record counts. → `docs/QA_BASELINE_2026-06-07.md`.
- [ ] **Stage 1 — Module health map (whole folder).** Apply the per-module
  checklist to every `.py` in entry points + `core/` + `agents/` + `eval/` +
  `tools/`. → `docs/MODULE_HEALTH_2026-06-07.md`. Done-when: every module has a row
  with purpose/used-by/tested/risk.
- [ ] **Stage 2 — Fix P0 test isolation.** Autouse reset fixtures; suite green
  in-order. Done-when: `pytest tests/ -q` has 0 unexpected failures.
- [ ] **Stage 3 — Sweep `core/` (highest risk first):** `signal_store`,
  `projection`, `convergence`, `worker_pool`, `cluster_registry`, then the `llm_*`
  backends, then retrieval stack, then the rest. Fix findings + add offline tests.
- [ ] **Stage 4 — Sweep `agents/`:** `base`, `synthesizer`, `scout`, `developer`,
  `critic`, `hater`, `validator`, `coding_roles`, `stock_roles`.
- [ ] **Stage 5 — Sweep `eval/` + entry points** (`run_swarm`, `run_stock`,
  `synthesize`, `diagnose`, `kb_migrate`) + `tools/`.
- [ ] **Stage 6 — Consolidate.** Remove dead code / dedup (per the map), emit the
  "live knobs" table; green suite per removal.
- [ ] **Stage 7 — Loud failures + observability** (search-missing, Groq-quota).
- [ ] **Stage 8 — Decide the uncommitted search work** (page-fetch + relevance
  gate in `core/search_tool.py` / `tests/test_search_quality.py` — built, never run
  live or pushed): live-smoke with `ddgs` installed, then finish+commit or revert.
  Done-when: working tree clean.
- [ ] **Stage 9 — Final gate.** Full suite green (real exit code), `diagnose.py`
  clean, one documented MOCK run reaching `answer.txt`. → `docs/QA_REPORT_2026-06-07.md`.

## Guardrails / non-goals (the easy mistakes — re-read before each commit)
- No new features, roles, task types, backends, or config knobs.
- No refactor of the load-bearing files without an explicit plan + owner ok.
- Don't validate or alter stock "makes money" logic.
- Don't blanket-set `SWARM_MAX_ITERATIONS` in the test invocation.
- Don't `pytest | tail` — check `$?`.
- Push only with a green suite. One change, one commit, one test.

## Build log (append one line per completed stage)
- Stage 0 (6.31+6.32): baseline reproduced (10 fail), P0 isolation fixed — 441 pass, 0 fail
- Stage 1 (6.34): module health map written → docs/MODULE_HEALTH_2026-06-07.md
- Stage 2 (6.31+6.32+6.33): P0 test isolation fixed; P2 convergence boundary-oscillation test added
- Stage 6 (6.35): dead swarm/ subdir deleted (6 files); module map corrected
- Stage 7 (6.36): Groq 429 teardown counter; no-search-backend RuntimeWarning
- Stage 9 (6.37): QA report written → docs/QA_REPORT_2026-06-07.md; final gate 441 passed, 11 skipped, 0 failed
