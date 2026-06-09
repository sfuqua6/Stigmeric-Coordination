# Legacy pipeline (unmaintained)

The original stigmergic swarm implementation: `run_task.py` orchestrator + the `swarm/`
package (signal store, stage coordinator, colony biomimicry primitives), plus its tests,
the ablation harness, and the historical `archive/` of session docs and old entry points.

Superseded by the pipeline now at the repository root (formerly `Attempt At Cleaning/`),
which enforces the no-leak rule and information partitioning. Preserved for reference and
for the ablation/benchmark harnesses that target the `swarm.*` package.

**Running it:** imports use the `swarm.*` package, so run from this directory with
`PYTHONPATH=.` (e.g. `cd legacy && python run_task.py debate "..."` or
`PYTHONPATH=. python tests/test_pipeline_sanity.py` — verified 11/11 passing 2026-06-09).
On Windows consoles also set `PYTHONIOENCODING=utf-8` (the test banners use emoji).
