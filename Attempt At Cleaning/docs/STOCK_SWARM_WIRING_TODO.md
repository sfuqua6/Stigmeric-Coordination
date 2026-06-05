# Stock Swarm — mechanical wiring checklist

The correctness-critical cores are built and tested (47 tests pass). What
remains is **mechanical glue** suitable for a cheaper/faster model (Sonnet /
Opus fast mode). Do these in order; confirm line numbers against the live code
(they drift). Full context: `STOCK_SWARM_POC_PROMPT.md`.

## Already done (do not redo)
- `core/stock_verify.py` — numeric claim extraction + closeness + `verify_claim` (REAL, tested)
- `core/stock_data.py` — `Snapshot`, `LENS_FIELDS`, `assert_no_lookahead`, `FrozenSnapshotProvider`, **and `YFinanceProvider` (`_fetch_raw` + `_raw_to_snapshot`, REAL)** with documented unit conversions and session+backoff. Offline unit tests lock the conversions (`tests/test_stock_yf_mapping.py`)
- `eval/ground_truth.py` — grader-only `realized_return` + import boundary (REAL, tested)
- `eval/backtest.py` — `score_case` / `aggregate` scorecard (REAL, tested); `_run_engine` is the stub
- `agents/stock_roles.py` — `LensScout`, `DataValidator`, `ValuationCritic`, `ThesisDeveloper`, `RiskHater`, `EquityBriefSynthesizer`, `build_stock_agents()` (REAL); LLM-prose rendering is the stub
- `core/stock_pipeline.py` — **standalone round-based orchestrator** + `make_llm_for()` (REAL, tested)
- `run_stock.py` — manual CLI (REAL, smoke-tested under MOCK)
- `eval/backtest.py::_run_engine` — runs the pipeline in-process vs the frozen DB (REAL, tested)
- `core/config.py` — `SURVIVAL_TASK_PROFILES["stock"]` added
- Tests: `tests/test_stock_verify.py`, `test_stock_pointintime.py`, `test_backtest_scoring.py`, `test_ground_truth_boundary.py`, `test_stock_roles.py`, `test_stock_yf_mapping.py`, `test_stock_pipeline.py`, `test_backtest_e2e.py` (64 pass, 1 gated-skip)

> **The POC is already runnable end-to-end via the standalone path**
> (`run_stock.py` + `core/stock_pipeline.py` + `eval/backtest.py`). Steps 1–7
> below wire the SAME roles into `run_swarm.py` so `python run_swarm.py stock …`
> works too — this is now **OPTIONAL parity work**, not a blocker. Skip to "Then:
> data + backtest" if you don't need run_swarm parity.

## 1. `core/config.py` — survival profile (1 line)
In `SURVIVAL_TASK_PROFILES` (~line 417) add:
```python
"stock": {"requires_verification": True, "credibility_chain_depth": 999,
          "support_diversity_min": 2},
```
Mirrors `coding`: numeric verification gates survival, so unverified number-claims don't reach the brief.

## 2. `core/search_tool.py` — follow-up modifiers (1 entry)
In `_FOLLOWUP_MODIFIERS_BY_TASK` (~line 47) add:
```python
"stock": ["earnings", "guidance", "analyst rating", "risks", "10-Q", "outlook"],
```

## 3. `core/topology.py` — topology template (1 entry)
In `_TOPOLOGY_TEMPLATES` (~line 69) add a `"stock"` template (axes: stance
{bullish|neutral|bearish} × horizon {1-week|1-month|3-month} × driver
{valuation|growth|technical|news|risk}; anchors: strongest bull, strongest
bear, neutral/range-bound; exclude anything needing post-as_of data). See
STOCK_SWARM_POC_PROMPT.md Part 3.4 for the exact wording.

## 4. `agents/synthesizer.py` — output strategy (1 line)
In `_OUTPUT_STRATEGY_BY_TASK` (~line 236) add:
```python
"stock": "sectioned",
```
(The `EquityBriefSynthesizer` overrides `synthesize()` anyway; this only affects
the fallback path.)

## 5. `core/role_registry.py` — register stock roles
Add a `_register_stock_roles()` mirroring `_register_coding_roles()` (lines
46–85) and a `_stock_registered` guard in `get_role_classes()`:
```python
def _register_stock_roles() -> None:
    try:
        from agents.stock_roles import (
            LensScout, ThesisDeveloper, ValuationCritic, RiskHater,
            DataValidator, EquityBriefSynthesizer,
        )
        _TASK_ROLE_OVERRIDES["stock"] = {
            "scout": LensScout, "developer": ThesisDeveloper,
            "critic": ValuationCritic, "hater": RiskHater,
            "validator": DataValidator, "synthesizer": EquityBriefSynthesizer,
        }
    except ImportError as exc:
        import warnings
        warnings.warn(f"[role_registry] stock roles not available: {exc}",
                      RuntimeWarning, stacklevel=2)
```
NOTE: the stock role constructors differ from the defaults (they need the
snapshot). Prefer constructing them via `agents.stock_roles.build_stock_agents()`
rather than the generic factory — see step 7.

## 6. `run_swarm.py` — task prompt + role activation (2 entries)
```python
# TASK_PROMPTS (~line 113)
"stock": "Assess whether to buy {prompt} over a 1-week-to-3-month horizon and project its return.",
# ROLES_FOR_TASK (~line 124)
"stock": {"critic", "hater", "validator"},
```

## 7. `run_swarm.py` — the real integration (NOT trivial; budget care here)
This is the one wiring step that needs thought (consider a strong model). The
stock pipeline does not use corpus partitions; it uses a `Snapshot` + lenses.

a. **Flags.** In `main()` flag parsing (~lines 1888–1908, alongside `--corpus=`/
   `--mode=`) parse `--symbol <T>` and `--as-of <YYYY-MM-DD>` and thread them
   into the pipeline call.

b. **Snapshot construction.** Add a `task_type == "stock"` branch where corpus
   partitions are built (`_build_corpus_partitions` ~line 1442 / its callers).
   Instead of chunks, construct the provider + snapshot:
   ```python
   from core.stock_data import FrozenSnapshotProvider, YFinanceProvider
   from datetime import date
   as_of = date.fromisoformat(as_of_str) if as_of_str else date.today()
   provider = (YFinanceProvider() if as_of == date.today()
               else FrozenSnapshotProvider(os.environ.get("STOCK_DB", "eval/datasets")))
   snapshot = provider.get_snapshot(symbol, as_of)   # runs assert_no_lookahead
   ```

c. **Agent construction.** Where agents are instantiated for the round-based
   pipeline, branch on stock and use the helper:
   ```python
   from agents.stock_roles import build_stock_agents
   llm_for = router.engine_for if router is not None else (lambda _r: llm)
   stock_agents = build_stock_agents(llm_for, snapshot, task_prompt,
                                     horizon_days=horizon)
   # slot stock_agents["scout"] etc. into the existing phase scheduling
   ```
   The synthesizer already carries `_snapshot`/`_ticker`/`_horizon_days`.

d. **Continuous-pool path caveat.** The Groq runs use the continuous-pool
   executor (`worker_pool.py`), which is action-driven, not the per-round
   `agent.run()` model `build_stock_agents` targets. EITHER (i) run stock via
   the round-based `run_pipeline` path first (simplest, gradable), OR (ii) trace
   `worker_pool.py` and adapt the stock roles to its action interface. Do (i)
   for the first gradable backtest; defer (ii).

## 8. Verify
```bash
MOCK_LLM=1 python run_swarm.py stock "NVDA" --symbol NVDA --as-of 2024-01-15
MOCK_LLM=1 SWARM_MIN_TIME_S=0 SWARM_MIN_ITERATIONS=5 pytest tests/ -q
```
Expect: a run dir with `answer.txt` + `prediction.json`; full suite green.

## Then: data + backtest (Stages 4/5/6 in the POC plan)
- **yfinance is rate-limited (Stage-0 finding):** `pip install curl_cffi`, then
  write a SLOW ingestion script that calls `YFinanceProvider.get_snapshot` once
  per (symbol, as_of) with backoff and freezes the result to
  `eval/datasets/<TICKER>/<as_of>.json`. Do NOT fetch live per backtest run.
  Verify the live path once with `RUN_YF_LIVE=1 pytest tests/test_stock_yf_mapping.py::test_live_fetch_smoke`.
- Build the historical DB under `eval/datasets/` (FrozenSnapshotProvider layout)
  + frozen price series for the grader (Stage 4).
- Implement `eval/backtest.py::_run_engine` to shell out to `run_swarm` and read
  `prediction.json` (Stage 5).
- News-driven `discover_symbols` (Stage 6).
