# QA Baseline — 2026-06-07

## Canonical full-suite invocation

```bash
cd "Attempt At Cleaning"
MOCK_LLM=1 SWARM_MIN_TIME_S=0 SWARM_MIN_ITERATIONS=5 python -m pytest tests/ -q
echo "EXIT=$?"
```

**Important:** do NOT add `SWARM_MAX_ITERATIONS` to this invocation — it breaks
`test_baseline_mode` and `test_coding_roles` (learned 2026-06-07 per hardening doc).

## Pre-fix baseline (commit 4bc0e82 / before 6.31)

```
10 failed, 430 passed, 11 skipped
```

Failing tests:
- `tests/test_baseline_mode.py` — 6 failures (TestBaselineCoordinator)
- `tests/test_coding_roles.py` — 3 failures (TestStaticCritic, TestFizzbuzzEndToEnd)
- `tests/test_search_quality.py` — 1 failure (test_enrich_replaces_snippet_with_page)

## Post-fix baseline (commits 6.31 + 6.32)

```
440 passed, 11 skipped, 0 failed   (run time ~4:15 min)
```

## Root causes fixed

### P0-A: asyncio event loop isolation (6.31)
`test_baseline_mode.py` and `test_coding_roles.py` used the legacy
`asyncio.get_event_loop().run_until_complete()` pattern.  In Python 3.10+,
`asyncio.run()` closes the event loop it creates; a subsequent
`get_event_loop()` call raises `RuntimeError: There is no current event loop`.
Fix: replaced all 9 occurrences with `asyncio.run()`.

### P0-B: sys.modules pollution from test_concurrency (6.32)
`test_concurrency.py::_purge_core_modules()` deleted every `core.*` entry from
`sys.modules` to force a re-import of `core.config` with different env vars.
This caused `_enrich_with_pages` (in `core/search_tool.py`) to reimport a
*fresh* `core.config` object via `from . import config as _cfg`.  Because
monkeypatch had patched attributes on the *old* module object,
`_cfg.SEARCH_FETCH_MIN_CHARS` reverted to its default of 400 instead of the
test's 50, and the 230-char page text failed the length gate.
Fix: replaced `_purge_core_modules()` with `importlib.reload()`, which updates
the module in-place (same object identity, refreshed attribute values).

## Skipped tests (11 — all intentional)

| test | reason |
|------|--------|
| test_backtest_e2e::test_backtest_end_to_end | requires yfinance + live network |
| test_contradiction_tracking::test_atom_contradiction | skip marker in code |
| test_phase_isolation::* (3) | subprocess tests; skipped under MOCK_LLM |
| test_heterogeneous_routing::* (2) | Groq API key not present |
| test_kb_default_off::* (3) | subprocess tests |

(Exact skip reasons visible with `pytest tests/ -v --tb=no` and looking at the `s` markers.)

## Notes for future runs
- Subprocess tests (test_phase_isolation, test_heterogeneous_routing,
  test_kb_default_off) need `SWARM_MIN_TIME_S=0 SWARM_MIN_ITERATIONS=5` in the
  subprocess env dict — they already set this internally.
- Mock mode and real-model runs land in `outputs_mock/` vs `outputs/` (kept
  separate by design — mock output is plumbing evidence only).
