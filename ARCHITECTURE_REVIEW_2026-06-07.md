# Architecture & Production Review — `Attempt At Cleaning/`

Stigmergic multi-agent LLM swarm. Review date 2026-06-07. Scope: the `Attempt At Cleaning/` codebase (entry point `run_swarm.py`), ~35k LOC. Method: inferred architecture from the whole tree, traced the two orchestration paths end-to-end, read the central abstractions (`SignalStore`, `Worker`/`choose_action`, `ConvergenceDetector`, projection/synthesizer, LLM routers), and verified numeric claims against the code. Confirmed findings are separated from hypotheses.

---

## 1. Executive summary

This is an unusually ambitious, heavily-iterated research system that implements a genuine idea well: agents coordinate **indirectly through a shared signal store** (stigmergy / blackboard) instead of a symbolic planner, and **diversity is engineered through input partitioning** rather than prompt/temperature tweaks. The "no-leak" discipline (agents see only `Signal.content`, never another agent's reasoning) is real and mostly enforced structurally. The LLM-backend layer (Groq router, vLLM cascade, fallbacks) is production-grade. Observability and determinism hygiene (mock quarantine, faithfulness audit, degraded-run detection) are above average for a research codebase.

The risks are concentrated in five places: (1) **the coding task executes LLM-generated code in a non-sandbox** despite a comment claiming otherwise; (2) **retrieved web text is injected raw into prompts**, an open indirect-prompt-injection channel; (3) the orchestrator reads worker progress by **introspecting a running coroutine's stack frame**, a silent single point of failure; (4) **decay is calibrated to ~1 hour but the default run cap is 15 minutes**, so strength-based pruning barely fires and the store grows roughly monotonically; and (5) **full-DAG projections are rebuilt repeatedly on the single event loop**, stalling all workers and getting worse as the store grows.

None of these are fatal to the research goal, but several would bite hard under scale, noisy inputs, or an adversary, and a few (RCE, injection) matter even at small scale. The dominant maintainability problem is accumulated complexity: two full pipelines, dual planners, four default-off "stigmergy gap" features in the hot path, 3.6k-line files, and ~80 broadly-swallowed exceptions.

**Overall: strong research engineering with a coherent thesis, undermined by emergent-behavior calibration coupling, a wide silent-failure surface, and two real security holes in the secondary paths.**

---

## 2. Architecture overview

### Runtime shape (default path = continuous pool)

`main()` (`run_swarm.py:1913`) hand-parses argv and dispatches to one of four execution modes. The **default** is `run_continuous_pipeline()` (`run_swarm.py:922`):

1. **Backend selection** — bundle/vLLM router, Groq router, hybrid (local+Groq), or single `make_llm()` (`core/llm.py:257`). Warmup + preflight.
2. **`SignalStore()`** constructed (`core/signal_store.py:202`); KB loaded only if `--use-kb` (default OFF).
3. **`run_pool()`** (`core/worker_pool.py:1407`) spins up N=24 `Worker` coroutines via `asyncio.gather`. A `decay_loop()` task runs in parallel (`interval_s=30`).
4. **Orchestrator tick loop** (every 2 s): peek iteration count, `ConvergenceDetector.tick()`, log progress, periodic `recluster()`, refresh the shared genome cache, check `detector.satisfied()`.
5. On halt: dump `signals.json`/`run_meta.json`, run the **Synthesizer** (per-cluster LLM render), optional KB save, write `summary.json`.

Three other modes share most helpers: `run_pipeline()` (legacy round/phase scheduler, `--legacy-rounds`), `run_phase_isolated()` (one phase per subprocess, checkpoint via `save_state`/`load_state`), and `_run_baseline()` (independent agents, no store — the A/B control).

### Package responsibilities (confirmed)

| Module | Responsibility |
|---|---|
| `run_swarm.py` | Orchestration, argv parsing, output writing, 4 execution modes |
| `core/worker_pool.py` | `Worker`, `choose_action`, target sampling, `PoolState`, `decay_loop` — the non-symbolic control core |
| `core/signal_store.py` | Typed signal DAG, logit-space strength dynamics, dedup, deposit-time clustering, provenance boost, partition invariant |
| `core/actions.py` | 7 action templates (prompt + parse + precondition + OUTPUT_TYPE), `FieldState` |
| `core/convergence.py` | Live halt decision (quality / saturation / caps), env-overridable thresholds |
| `core/projection.py` | Pure-Python DAG → survival classification, genomes, multi-resolution lattice (atoms/propositions/frames) |
| `agents/synthesizer.py` | Per-cluster LLM read-out, dual planner, faithfulness audit (3.6k LOC) |
| `core/llm*.py` | Backends: `RealLLM`(HF/bnb), `VLLMBackend`, `GroqBackend`/`GroqRouter`, `HybridRouter`, `MultiEngineRouter`, LoRA router |
| `core/config.py` | All tunables + module-level `assert` validation + tier/env overrides |
| `core/role_registry.py` | Task-type → role class map (swaps coding roles) |
| `core/knowledge_base.py` | Cross-run consensus/rejection memory (schema v3) |
| `core/retrieval.py`, `core/search_tool.py` | Wikipedia→Web→placeholder corpus, page fetch + relevance gate |

### Central abstractions and how they interact

- **`Signal`** is the unit of coordination: typed (`INITIAL`/`SUPPORT`/`CRITIQUE_*`/`OBJECTION`/`VERIFICATION`/`SEARCH`), carries `strength` (stored as `_logit`), `parent_id`, `cluster_id`, `partition_id`, `iter_at_deposit`, `visits`.
- **`SignalStore`** is the blackboard. Deposits are deduped (string pre-screen), boosted by parent verification, clustered at deposit time, and (optionally) amplify their cluster trail.
- **`Worker`** is role-agnostic: each iteration it draws an action, samples a target, builds a prompt from *content only*, calls the LLM, and deposits. There is no per-role agent object in the continuous path — the role is the action.
- **`ConvergenceDetector`** reads the store via `build_projection` and decides halting.
- **`Synthesizer`** + `projection` turn the surviving DAG into the answer.

The cluster — not the signal — is the unit of selection (`ClusterGenome`, `core/fitness.py` 7-term compositor), assembled at the end of `build_projection`.

---

## 3. Orchestration and coordination analysis

**Orchestration strategy: stigmergic blackboard with a stochastic action policy. There is no symbolic planner and no task graph.** Global behavior emerges from many local, randomized decisions against shared state.

**What decides the next action** — `choose_action()` (`core/worker_pool.py:233`). For each worker iteration:
1. **Cold-start gate**: until 30 iterations / 8 INITIALs, restrict to `SCOUT`/`DEVELOP`.
2. **Preconditions**: drop actions whose `ACTION_PRECONDITIONS[a](field_state)` is false.
3. **Bundle-disabled actions** removed (e.g. `OBJECT`/`VALIDATE` on `creative`).
4. **Weighted random draw** over survivors: `_BASE_WEIGHTS` × share pressure (×1.5 below `ACTION_SHARE_TARGETS` floor, ×0.3 above ceiling) × optional cluster-local bias × 0.7 recency penalty.

So the "plan" is a **target distribution over 7 action types**, steered toward configured shares, realized by `rng.choices`. Coordination is *indirect*: a worker's deposit changes `FieldState` and cluster strengths, which changes other workers' preconditions, sampling weights, and local biases on subsequent iterations. The `SignalStore` is the only channel.

**Target selection** is also stochastic: strength-weighted sampling with recent-target penalties (`_RECENT_TARGET_PENALTY=0.5`), per-worker cooldown (depth 3), and optional cluster-affinity (`sample_from_clusters` biases toward the worker's semantic centroid). REFINE seeks contested initials (`_cluster_dissent`), CHAIN forces depth by overriding the model's parent choice (`worker_pool.py:807`).

**Macro control** is the 2 s orchestrator tick (`run_swarm.py:1086`): convergence, recluster, genome refresh. Halting (`convergence.py:225`) is layered correctly — **absolute caps are checked before soft floors** (an explicit fix for a documented 4-hour runaway where an inter-cluster-edge floor vetoed the cap). Quality halt requires a cluster passing `support_diversity≥4`, `dissent_pressure<0.5`, and (factual tasks) two independent validators ≥0.7, held for 30 iterations; otherwise saturation or cap.

**Assessment.** The coordination model is coherent and the action-economy tuning is thoughtful (the `_BASE_WEIGHTS`/`ACTION_SHARE_TARGETS` comments document real audit-driven changes). The weakness is that *every* behavior is calibration-dependent: there is no invariant that guarantees, e.g., that objections get answered — only a floor that raises REFINE's probability. Under distribution shift (new task type, different model, different corpus density) the emergent mix can drift and there is no symbolic backstop.

---

## 4. Data flow and state management

**In-run state** is the `SignalStore` DAG. Strength lives in **logit space** (`_to_logit`/`_from_logit`, `signal_store.py:92`): amplify/decay/boost are *additive deltas* projected back through sigmoid. This deliberately fixes three real bugs (saturation pinning at 1.0, contrarian anti-decay drift, decay×amplify order-dependence) and is one of the better design decisions in the repo.

**Context passing into prompts** is strictly `Signal.content` + the agent's own prior outputs + retrieved chunks (`core/actions.py` docstring; `base.py` enforcement). Provenance walks return **shapes** (ID lists, typed-ancestor counts), never rendered ancestor text (`signal_store.py:923` `ancestor_ids`).

**The partition invariant** is load-bearing for the diversity metric: every `INITIAL`/`SUPPORT` must carry a non-empty `partition_id`, enforced by `raise AssertionError` in `deposit()` (`signal_store.py:330`). Because this is easy to violate, there are **three redundant carry-forward mechanisms** (store parent-inheritance `signal_store.py:309`; `base.py:327`; `worker.iterate` `worker_pool.py:856-882`). That much belt-and-suspenders is a smell — the invariant is fragile enough that the team needed three guards to stop it firing spuriously.

**Cross-phase state**: `save_state`/`load_state` (`signal_store.py:395`) round-trips signals to JSON for subprocess-per-phase runs. Embeddings are intentionally **not** persisted and are rebuilt lazily — so a resumed run has no embeddings until new deposits arrive, silently degrading dedup/clustering for already-loaded signals (hypothesis: this weakens `--phase`-isolated cluster quality; not directly tested).

**Cross-run state**: `KnowledgeBase` (schema v3, genome-bearing) is **OFF by default** (`--use-kb` to enable). Good default — cross-run memory is the kind of thing that silently contaminates A/B comparisons.

**Per-worker ephemeral state**: `recent_actions`, `recent_targets`, query history, semantic-position centroid, and a **shared-by-reference** genome cache (`worker_pool.py:1436`). The genome cache is refreshed by `clear()`+`update()` on the event loop with no `await` between them, so it is atomic w.r.t. other coroutines — correct, but only because of single-threaded asyncio semantics, which is an implicit assumption.

---

## 5. Strengths

- **Clean central thesis, enforced.** Input-partitioning-as-diversity and the no-leak rule are implemented consistently; content-only prompt construction is the real guarantee.
- **Logit-space strength dynamics** (`signal_store.py`) — a genuinely good fix for saturation/anti-decay/order-dependence, with the legacy multiplicative path retained behind `USE_LOGIT_DYNAMICS` and exercised by tests.
- **Production-grade LLM failure handling.** `GroqBackend` (`core/llm_groq.py`): token-bucket + per-model semaphore + exponential backoff with jitter + decommissioned/404 auto-fallback + preflight + silent-role detection. `make_llm` vLLM→AWQ→3B→HF→mock cascade (`core/llm.py:133`). This is the most mature part of the codebase.
- **Convergence detector correctness.** Absolute caps before soft floors (`convergence.py:231`) with a documented runaway-bug rationale; all thresholds env-overridable for subprocess tests.
- **Determinism discipline.** Mock runs quarantined to `outputs_mock/`; loud warnings that placeholder/mock numbers are "plumbing, not behavior". This is rare and commendable.
- **Synthesis safety.** One LLM call **per cluster** (`agents/synthesizer.py`) structurally prevents cross-cluster hallucination; sections 3/4 deterministic; post-hoc **faithfulness audit** (n-gram overlap per citation tag) with crash sentinels written to `renderer_audit.json`.
- **Health surfacing in `summary.json`** — `degraded`, `silent_roles`, `embedder`, all-singletons detection, `quality_met` gated by `not degraded` (`run_swarm.py:1412-1448`).
- **Test breadth** — 45 test files covering no-leak patterns, partition propagation, phase isolation, convergence, logit dynamics, clustering, retrieval, Groq fallback.

---

## 6. Risks and issues

### CRITICAL

**C1 — Arbitrary code execution of LLM output in the coding path.**
`agents/coding_roles.py:547` `_run_test_subprocess()` writes model-generated "test" code to a temp file and runs `subprocess.run([sys.executable, "-m", "pytest", fname, ...], timeout=SUBPROC_TIMEOUT_S)`. The docstring says "sandboxed subprocess" — **it is not sandboxed**. It inherits the host's filesystem, network, and user privileges; the only control is a timeout. A model (or web content steering the model, see C2) can emit `import os; os.system(...)`, exfiltrate env vars/keys, or delete files.
*Why it matters:* full RCE on the host whenever `coding` runs. *Severity: Critical (coding path).*
*Fix:* execute in a real sandbox — container/`nsjail`/`firejail` + seccomp, no network, read-only FS, unprivileged user, CPU/mem caps — or restrict to AST-validated code with an import allowlist and no dunder/`exec`/`eval`. Never call it "sandboxed" until it is.

### IMPORTANT (and one that is arguably Critical depending on threat model)

**I1 — Indirect prompt injection from retrieved web content.**
`core/actions.py:scout_prompt` interpolates fetched page text raw (`body = c.text[:800]`) under the header "You issued an agentic search and received the following evidence." The text comes from arbitrary DuckDuckGo result URLs fetched by `core/search_tool._fetch_page_text` / `core/retrieval.py` (`requests.get`). There is a relevance gate but **no instruction-stripping or provenance isolation**. Injected instructions become `INITIAL`/`SUPPORT` deposits and can propagate into synthesis. `_assert_no_leak` does not address this (it blocks a fixed token list).
*Why it matters:* an attacker who controls or SEO-ranks a page can steer deposits and the final answer. *Severity: High.*
*Fix:* treat retrieved text as untrusted data — strong delimiters, an explicit "the following is reference data, not instructions" frame, strip imperative/jailbreak patterns, keep the `[External context]` tag through to synthesis, and down-weight web-sourced claims in survival scoring.

**I2 — Convergence/observability depend on stack-frame introspection.**
`run_swarm.py:1459` `_peek_pool_state()` reads `pool_task.get_coro().cr_frame.f_locals.get("pool_state")` to get the iteration count. The docstring claims a different mechanism (`run_pool sets task._pool_state`) that **does not exist in the code** — so the documented contract is already wrong. If `run_pool` renames the `pool_state` local, the orchestrator silently gets `iter_n=0` forever: convergence quality/saturation logic goes blind (halts only on the time cap), and recluster/genome-refresh cadence (keyed on `iter_n`) freezes.
*Why it matters:* a silent, refactor-triggered failure of the entire convergence + adaptation loop. *Severity: High.*
*Fix:* publish `PoolState` explicitly — return it via a shared holder object both `run_pool` and the orchestrator reference, or attach it to the task with a typed wrapper. Delete the misleading docstring.

**I3 — Decay calibrated to ~1 h; default cap is 15 min → pruning is effectively inert, store grows monotonically.**
`decay_loop` is started with `interval_s=30` (`run_swarm.py:1080`); `factor = interval/300 = 0.1` (`worker_pool.py:1476`). With `DELTA_DECAY=-0.10` the per-tick logit decrement is **−0.01**. To fall from strength 0.6 to `PRUNE_THRESHOLD=0.30` needs ≈ −1.25 in logit space ≈ **125 ticks ≈ 62 min**; even a weak 0.4 signal needs ≈ 22 min. Default `MAX_TIME_S=900` (15 min) ⇒ ~30 ticks ⇒ a 0.6 signal only reaches ~0.53 and **never prunes**. (Verified arithmetic; the `decay_loop` docstring itself says "~50 ticks (~50 minutes)".)
*Why it matters:* the stated "field consolidates / avoid prune storms" model under-prunes in a normal run; consolidation actually comes from clustering + survival gates, not strength decay. The store grows ~monotonically, which compounds I4. *Severity: High (correctness + perf).*
*Fix:* derive `factor` from the actual run budget (iteration-based decay, or `factor ∝ MAX_TIME_S`), or explicitly document decay as a slow background process and prune by cluster membership/age. Reconcile the calibration with the cap.

**I4 — Repeated full-DAG `build_projection()` on the single event loop stalls all workers.**
`build_projection` (walks the DAG, builds genomes/atoms/frames/sensitivities) is invoked: every 2 s in `_log_progress` (`run_swarm.py:1491`), every 10 iters in `ConvergenceDetector.tick` (`convergence.py:153`), every 25 iters in the genome refresh (`run_swarm.py:1121`) — **none shared** — plus synthesis, KB save, and summary. It is pure-Python with no `await`, so it blocks all 24 cooperative workers for its duration, which grows with store size (worsened by I3). `_build_frames` is pairwise-cosine over clusters; `_build_sensitivities` simulates support removal (gated to 20 clusters).
*Why it matters:* throughput degrades super-linearly as a run progresses; on long runs the orchestrator tick can dominate wall-clock. *Severity: High (perf/scale).*
*Fix:* compute **one** projection per tick window and pass it to logger + detector + genome refresh; memoize by a store version counter; move lattice builders (frames/atoms/sensitivities/genomes) off the hot path or behind flags; consider running `build_projection` in `asyncio.to_thread`.

**I5 — Runtime invariants enforced by `assert` → disabled under `python -O`.**
The partition-leak invariant (`signal_store.py:330`), both `_assert_no_leak` guards (`base.py:377`, `worker_pool.py:1348`), and **all** of `config.py`'s validation (`assert` at lines 215-218, 498-514, 614-659) are bare `assert`s. Running under `-O`/`PYTHONOPTIMIZE=1` strips every one of them, silently disabling the no-leak tripwire, the partition invariant, and config sanity checks.
*Why it matters:* a deployment flag turns off the system's safety guarantees with no error. *Severity: Important.*
*Fix:* convert runtime invariants to explicit `if ...: raise`. Keep `assert` only for dev-time/test checks.

**I6 — Wide silent-failure surface (~80 broad `except`/`pass`).**
Examples: genome-refresh `except Exception: pass` (`run_swarm.py:1168`), projection-in-logger `except: pass` (`run_swarm.py:1492`), KB save, audit, retrieval fallbacks. Real regressions (a projection that starts raising, a genome builder that breaks) become invisible — the run completes and writes plausible outputs.
*Why it matters:* masks correctness regressions; `degraded` detection only covers embedder + silent-role cases. *Severity: Important.*
*Fix:* narrow exception types, log with context + a counter surfaced in `summary.json`, and fail loudly in non-mock runs for core-path exceptions.

**I7 — Embedder is a silent single point of failure for the entire thesis.**
If `sentence-transformers`/`huggingface-hub` resolve wrong (the repo pins `huggingface-hub<1.0` precisely because 1.x breaks `all-MiniLM-L6-v2` load), `SignalStore` falls back to `_DisabledEmbedder` (`signal_store.py:154`), turning off dedup + semantic clustering — **every claim becomes its own singleton** and the diversity thesis collapses. The pipeline warns at the *end* (`run_swarm.py:1401`) but still runs and writes outputs.
*Why it matters:* a dependency-resolution change quietly invalidates results that still look valid. *Severity: Important.*
*Fix:* for non-mock runs, fail fast unless `--allow-no-embedder`; add a startup preflight that encodes a probe string; ship an installable lockfile rather than a hand-managed `requirements-colab.txt`.

**I8 — No-leak enforcement is a token blocklist, not a structural guarantee.**
`_assert_no_leak` (`base.py:377`, `worker_pool.py:1348`) checks for literal substrings (`"parent_content"`, `"responses:"`, `"dialogue thread"`). The actual protection is that prompts are built only from `Signal.content`; the blocklist would miss any new leak vector and gives false confidence.
*Severity: Important (security-adjacent) / Minor.* *Fix:* route all prompt construction through one allowlisted formatter that can only see `content`; keep the blocklist as a secondary tripwire, not the guarantee.

**I9 — Hand-rolled argv parsing with overloaded semantics.**
`main()` pops ~20 flags by string slicing (`run_swarm.py:1913-2075`). `SWARM_BACKEND` means different things to `make_llm` vs the router (`llm.py:285` explicitly works around this); `--heterogeneous` behaves differently across continuous/legacy/T4 paths; `--small` branches on execution path. High cognitive load, easy to break, no `--help`.
*Severity: Important (maintainability).* *Fix:* `argparse` + a typed `RunConfig` dataclass; one place that resolves backend semantics.

### MINOR

- **M1 — Dual planners never ablated.** `Synthesizer._plan_synthesis` (LLM) vs `projection.build_plan` (deterministic) implement the same decision; CLAUDE.md says the LLM one "has not yet been ablated." Two maintained paths for one job. *Fix: ablate, retire one.*
- **M2 — Two full pipelines + baseline + phase-isolated** coexist in `run_swarm.py`. Large branching surface; the legacy round path is kept for A/B repro. *Fix: extract shared stages; isolate legacy behind a clearly-deprecated module.*
- **M3 — Four default-off "stigmergy gap" features** (`USE_CLUSTER_AWARE_SAMPLING`, `USE_TRAIL_AMPLIFICATION`, `USE_LOCAL_ACTION_BIAS`, `USE_WORKER_POSITION`) add branches and lookups inside the per-iteration hot path. Dormant complexity. *Fix: decide and remove, or move behind a single "experimental" gate.*
- **M4 — `RealLLM.generate` calls `torch.cuda.empty_cache()` every call** (`llm.py:561`) — a device-synchronizing throughput cost; deliberate for the 6 GB card but should be tier-gated.
- **M5 — Playful arithmetic literals** in config (`MAX_RENDERED_CLUSTERS = 4*8`, `CHUNKS_PER_SCOUT_MAX = 4*2`, `NUM_ROUNDS = 3#*8`) hurt grep-ability and look like unfinished tuning.
- **M6 — "Thread-safe" `RLock` store is uncontended** on the async path (single event-loop thread; only `RealLLM` generation runs in `to_thread`). Harmless but over-engineered, and the `RLock` reentrancy is load-bearing for `deposit → _avg_verification_strength → ancestor_ids` — worth a comment so nobody "optimizes" it to a plain `Lock`.
- **M7 — Half-finished Forager→Developer rename** (`agents/forager.py` alias); mixed naming throughout.
- **M8 — 3.6k-line `synthesizer.py` / 2.1k-line `projection.py`** are hard to navigate and unit-test in isolation.

---

## 7. Security and prompt-injection review

| Surface | Status | Finding |
|---|---|---|
| **Code execution** | ✗ Open | C1: LLM-generated tests run via `subprocess` with no isolation (coding path). RCE. |
| **Indirect prompt injection** | ✗ Open | I1: retrieved web/Wikipedia text injected raw into scout prompts; no sanitization; propagates to deposits + synthesis. |
| **No-leak (reasoning isolation)** | ◑ Partial | Structurally enforced by content-only rendering (good); the `_assert_no_leak` blocklist is a weak secondary check (I8). |
| **Secrets** | ◑ | API keys read from env (`GROQ_API_KEY`) — fine. But C1's unsandboxed exec can read those env vars; injection (I1) can steer the model toward exfil prompts. |
| **Invariant bypass** | ✗ | I5: `-O` strips the no-leak + partition asserts. |
| **Persistence** | ◑ | `save_state`/`load_state` use JSON (not pickle) — good, no deserialization RCE. KB is JSON too. |
| **SSRF / fetch** | ◑ | `requests.get` on arbitrary DDG result URLs (`search_tool.py:351`) with a 5 s timeout but no allowlist/IP filtering — a malicious result could point at internal endpoints. Low severity given research scope, but note it. |

**Net:** the *primary* debate/analysis path is reasonably contained (no exec; injection is the main hole). The *coding* path combines the two worst issues — injectable inputs (I1) feeding an unsandboxed executor (C1). Treat coding as untrusted-by-default until both are fixed.

---

## 8. Performance and cost review

**Concurrency reality (confirmed).** `LLM_CONCURRENCY = 1 if _TIER is None else 32` (`config.py:132`). On a laptop (no tier) the HF/`RealLLM` and `MockLLM` paths serialize through `asyncio.Semaphore(1)` — the 24 workers provide **behavioral diversity, not throughput**; real parallelism only appears with vLLM internal batching or Groq's server-side concurrency. This is sensible but means "24 workers" is misleading on local hardware.

**Hot-path costs.**
- I4: redundant full-DAG projections on the event loop are the dominant orchestration cost and scale with store size.
- I3: no effective pruning ⇒ store grows ⇒ every projection, every `_cluster_dissent` sibling-scan (`worker_pool.py:413`, O(initials) per call with per-pair embedding dot), and every `_avg_verification_strength` lineage walk gets slower over a run.
- Synthesis token cost: `agents/synthesizer.py` makes ~15 generate call sites — per-cluster render × best-of-N (`_BEST_OF_N_COHESIVE=2` on laptop) + planner + prompt-interpreter + optional debate/alternative frames. For many surviving clusters this is the largest single token sink; on `LLM_CONCURRENCY=1` it is serial.
- `torch.cuda.empty_cache()` per call (M4) costs throughput on the HF path.

**Cost controls that exist (good).** Search budget window (`SEARCH_BUDGET_PER_WINDOW=6`/5 s), served-query cache, Groq token bucket + RPM/TPD-aware model assignment (70B reserved for synthesis), time/iteration caps, dedup pre-screen.

**Recommendations:** share one projection per tick (I4); fix decay so the store stays bounded (I3); cap/curve best-of-N by surviving-cluster count; consider an incremental projection that updates on deposit instead of full rebuilds.

---

## 9. Testing and evaluation review

**Strengths.** 45 test files with real coverage of the load-bearing invariants: `test_no_leak_real_patterns`, `test_partition_propagation`, `test_phase_isolation`, `test_convergence`, `test_logit_dynamics` (both code paths), `test_cluster_*`, `test_retrieval`, `test_llm_groq_fallback`, `test_strip_reasoning`, `test_render_guards`. The convergence thresholds are env-overridable specifically so subprocess tests don't wait on the 60 s floor — evidence the team tests the orchestration, not just units. `MockLLM` is deterministic (SHA1-seeded) so plumbing tests are stable. `diagnose.py` runs each test in its own subprocess to survive OS kills.

**Gaps / risks.**
- **Mock-only behavior.** `MockLLM` emits fixed phrases regardless of input, so tests prove plumbing, not coordination quality. The CLAUDE.md is explicit that behavioral/diversity numbers from `outputs_mock/` are not evidence — but that means the *emergent* behavior (the whole thesis) is validated only by real-GPU runs, of which the evidence here is informal logs (`groqrun1.txt`, `lastgroqrun.txt` referenced in comments), not a regression suite.
- **No assertion that the I2 frame-introspection tap still works** — a rename would pass all tests and silently break convergence in real runs.
- **No injection/security tests** for C1/I1.
- **`eval/backtest.py` + `eval/ground_truth.py`** exist (stock path) but there's no general answer-quality regression harness for debate/analysis; "quality" is the live detector's gate, not a held-out benchmark.
- I could not execute the suite here (needs `MOCK_LLM=1` + deps); the CLAUDE.md claims "304 pass, 10 skip" as of 2026-05-31. Worth re-confirming in CI on every change, especially around projection and the pool-state tap.

---

## 10. Refactoring recommendations

1. **Publish `PoolState` explicitly** (kills I2). A shared holder object passed into `run_pool` and read by the orchestrator; delete the stale docstring.
2. **One projection per tick.** Compute `build_projection` once per orchestrator tick, version it by deposit count, and hand the same object to logger + detector + genome refresh (kills most of I4).
3. **Convert invariants to explicit raises** (kills I5). `if signal_type in ("INITIAL","SUPPORT") and not partition_id: raise PartitionLeak(...)`; same for no-leak.
4. **Sandbox the coding executor** (kills C1) and **frame retrieved text as untrusted data** in `scout_prompt`/`develop_prompt`/`validate_prompt` (mitigates I1).
5. **Reconcile decay with the run budget** (kills I3) — iteration-based decay or `factor` derived from `MAX_TIME_S`; add a store-size cap with cluster-aware eviction.
6. **`argparse` + `RunConfig`** (kills I9); centralize the `SWARM_BACKEND` overloading in one resolver.
7. **Narrow the broad excepts** (I6): each becomes a typed catch + structured log + a `summary.json` error counter.
8. **Mandatory embedder for non-mock runs** (I7) + an installable lockfile.
9. **Split `synthesizer.py`/`projection.py`** into renderer / planner / audit and classification / genome / lattice modules; unit-test the lattice builders directly.
10. **Decide the dormant complexity**: ablate dual planners (M1), resolve the four gap flags (M3), finish the Forager→Developer rename (M7).

---

## 11. Prioritized action plan

**P0 — this week (security + silent failure)**
- C1: sandbox or AST-restrict the coding subprocess; stop calling it "sandboxed".
- I2: replace `_peek_pool_state` frame-fishing with an explicit handle.
- I5: convert no-leak + partition invariants from `assert` to `raise`.

**P1 — this sprint (correctness + scale)**
- I1: untrusted-data framing + instruction-stripping for retrieved content; carry provenance to synthesis.
- I3: fix decay calibration vs the time cap; bound store growth.
- I4: single shared projection per tick + memoization.
- I7: fail-fast on missing embedder; ship a lockfile.

**P2 — next (robustness + maintainability)**
- I6: narrow exceptions + error counters in `summary.json`.
- I9: argparse + `RunConfig`.
- Add injection/security regression tests and a pool-state-tap test.

**P3 — cleanup (debt)**
- M1/M2/M3/M5/M7/M8: ablate dual planner, factor shared stages, resolve gap flags, replace cute literals, finish rename, split mega-files.

---

## 12. Final verdict

This is **strong, original research engineering with a genuinely coherent coordination model** and an unusually disciplined attitude toward evidence (mock quarantine, faithfulness audits, degraded-run flags). The LLM-backend and convergence layers are production-quality. The core ideas — logit-space strength dynamics, content-only no-leak, input-partition diversity, per-cluster synthesis — are well-motivated and mostly well-implemented.

It is **not yet production-safe**, for reasons that are concentrated and fixable rather than pervasive. Two security holes (unsandboxed code execution, raw web-content injection) sit in the secondary paths; one observability hack (coroutine-frame introspection) is a silent single point of failure; and two calibration/perf issues (decay-vs-cap mismatch, repeated event-loop projections) mean the system's behavior and throughput degrade as a run grows — exactly where an emergent system is hardest to reason about. Underneath, the dominant long-term risk is **complexity that has outrun consolidation**: two pipelines, dual planners, dormant feature flags in the hot path, 3k-line modules, and ~80 swallowed exceptions that can hide regressions in precisely the emergent behavior the project is trying to measure.

Fix the P0/P1 list and this moves from "impressive research prototype" to "defensible system." The thesis is sound; the calibration coupling and the silent-failure surface are what need engineering attention next.

*Confidence: findings in §3–§4, §6 (C1, I1–I5, I7) are confirmed against code and arithmetic. I4's "stalls all workers" and the §9 evidence-gap framing are well-supported inferences; the §4 resumed-run embedding-degradation note and SSRF severity are hypotheses flagged as such.*
