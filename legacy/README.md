# Legacy pipeline (unmaintained)

The original stigmergic swarm implementation: `run_task.py` orchestrator + the `swarm/`
package (signal store, stage coordinator, colony biomimicry primitives), plus its tests,
the ablation harness, and the historical `archive/` of session docs and old entry points.

Superseded by the pipeline now at the repository root (formerly `Attempt At Cleaning/`),
which enforces the no-leak rule and information partitioning. Preserved for reference and
for the ablation/benchmark harnesses that target the `swarm.*` package.

**Running it:** imports use the `swarm.*` package, so run from this directory
(e.g. `cd legacy && python run_task.py debate "..."` or `python tests/test_pipeline_sanity.py`).
