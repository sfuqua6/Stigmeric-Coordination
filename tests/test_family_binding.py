"""Stable worker->family binding on MultiEngineRouter (intake diversity).

List-form roles used to round-robin per CALL, so one scout alternated model
families across its own iterations — no partition was explored by a
consistent prior. With worker_id, the binding is stable for the whole run:
prior diversity x partition diversity, and the offset lists in
research_ensemble give cross-family development/verification.
Runs entirely on the mock load path (no models).
"""
import asyncio

from core.llm_router import MultiEngineRouter


def _router():
    r = MultiEngineRouter("research_ensemble")
    r._use_mock = True
    asyncio.run(r.load())
    return r


def test_scout_binding_is_stable_and_spread():
    r = _router()
    assert r.engine_for("scout", worker_id=0) is r.engines["primary"]
    assert r.engine_for("scout", worker_id=1) is r.engines["secondary"]
    assert r.engine_for("scout", worker_id=2) is r.engines["tertiary"]
    assert r.engine_for("scout", worker_id=3) is r.engines["primary"]  # wraps
    # Stability: repeated calls never rotate.
    for _ in range(5):
        assert r.engine_for("scout", worker_id=1) is r.engines["secondary"]


def test_cross_family_development():
    # Same worker: the family that scouts a partition is NOT the family
    # that develops/verifies it (offset lists).
    r = _router()
    assert r.engine_for("scout", worker_id=0) is r.engines["primary"]
    assert r.engine_for("developer", worker_id=0) is r.engines["secondary"]
    assert r.engine_for("validator", worker_id=0) is r.engines["tertiary"]


def test_no_worker_id_falls_back_to_round_robin():
    r = _router()
    seen = {id(r.engine_for("scout")) for _ in range(6)}
    assert len(seen) >= 2  # rotates across the list without a worker_id


def test_scalar_roles_unaffected_by_worker_id():
    r = _router()
    assert r.engine_for("synthesizer", worker_id=7) is r.engines["primary"]
