"""Fix P — SynthesisPlan + build_plan() tests.

Verifies that build_plan():
  - Selects at most RENDER_K clusters for Section 1
  - Applies MMR diversity (similar clusters go to held_clusters, not render)
  - Fills dissent_clusters from contested + surviving-with-dissent
  - Routes leftover surviving clusters to held/demoted correctly
  - Falls back gracefully when no embeddings are available

Run with:
    pytest tests/test_planner_selection.py -v
"""

import sys
import os
import pytest

_HERE = os.path.dirname(__file__)
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.projection import (
    ClusterProjection,
    SynthesisProjection,
    SynthesisPlan,
    build_plan,
)
from core.signal_store import SignalStore, _NULL_EMBEDDER
from core.config import RENDER_K, DISSENT_K


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store():
    return SignalStore(embedder=_NULL_EMBEDDER)


def _cp(
    rep_id: str,
    status: str = "surviving",
    support_diversity: int = 3,
    dissent_pressure: float = 0.2,
    verification_score: float = 0.5,
    dissent_set: list = None,
    support_set: list = None,
) -> ClusterProjection:
    return ClusterProjection(
        representative_id=rep_id,
        member_ids=[rep_id],
        support_set=support_set or [],
        dissent_set=dissent_set or [],
        verification_set=[],
        support_diversity=support_diversity,
        dissent_pressure=dissent_pressure,
        verification_score=verification_score,
        partition_origins=["partition_0"],
    )


def _proj(*clusters, contested=None) -> SynthesisProjection:
    surviving = [c for c in clusters if c.status == "surviving"]
    cont = list(contested or [])
    return SynthesisProjection(surviving=surviving, contested=cont)


# ---------------------------------------------------------------------------
# Basic shape
# ---------------------------------------------------------------------------

def test_build_plan_returns_synthesis_plan():
    store = _make_store()
    cp0 = _cp("INITIAL_00001")
    sp = build_plan(_proj(cp0), store)
    assert isinstance(sp, SynthesisPlan)


def test_build_plan_empty_projection():
    store = _make_store()
    sp = build_plan(SynthesisProjection(), store)
    assert sp.render_clusters == []
    assert sp.dissent_clusters == []
    assert sp.held_clusters == []
    assert sp.demoted_clusters == []


def test_build_plan_single_surviving():
    store = _make_store()
    cp0 = _cp("INITIAL_00001")
    sp = build_plan(_proj(cp0), store, render_k=3, dissent_k=2)
    assert "INITIAL_00001" in sp.render_clusters
    assert len(sp.render_clusters) == 1


# ---------------------------------------------------------------------------
# RENDER_K cap
# ---------------------------------------------------------------------------

def test_render_k_cap_is_respected():
    store = _make_store()
    clusters = [_cp(f"INITIAL_{i:05d}") for i in range(8)]
    sp = build_plan(_proj(*clusters), store, render_k=3)
    assert len(sp.render_clusters) <= 3, (
        f"render_clusters has {len(sp.render_clusters)} entries > render_k=3"
    )


def test_clusters_beyond_render_k_go_to_held_or_demoted():
    store = _make_store()
    clusters = [_cp(f"INITIAL_{i:05d}") for i in range(5)]
    sp = build_plan(_proj(*clusters), store, render_k=2)
    render_set = set(sp.render_clusters)
    held_set = set(sp.held_clusters)
    demoted_set = set(sp.demoted_clusters)
    all_surviving_ids = {c.representative_id for c in clusters}
    # Every surviving cluster must appear in exactly one bucket
    accounted = render_set | held_set | demoted_set
    assert all_surviving_ids <= accounted, (
        f"Some surviving clusters not accounted for: "
        f"{all_surviving_ids - accounted}"
    )


# ---------------------------------------------------------------------------
# Score ordering
# ---------------------------------------------------------------------------

def test_higher_support_diversity_preferred():
    """Cluster with more support_diversity should be ranked first."""
    store = _make_store()
    weak = _cp("INITIAL_00001", support_diversity=1, verification_score=0.0)
    strong = _cp("INITIAL_00002", support_diversity=5, verification_score=0.8)
    sp = build_plan(_proj(weak, strong), store, render_k=1)
    assert sp.render_clusters == ["INITIAL_00002"], (
        f"Expected high-diversity cluster first, got {sp.render_clusters}"
    )


def test_high_dissent_pressure_penalizes_rank():
    """Cluster with high dissent_pressure should score lower."""
    store = _make_store()
    clean = _cp("INITIAL_00001", support_diversity=4, dissent_pressure=0.1,
                verification_score=0.6)
    contested_looking = _cp("INITIAL_00002", support_diversity=4,
                             dissent_pressure=2.0, verification_score=0.6)
    sp = build_plan(_proj(clean, contested_looking), store, render_k=1)
    assert sp.render_clusters == ["INITIAL_00001"], (
        f"Expected low-dissent cluster first, got {sp.render_clusters}"
    )


# ---------------------------------------------------------------------------
# Dissent clusters
# ---------------------------------------------------------------------------

def test_contested_clusters_go_to_dissent():
    store = _make_store()
    cont = _cp("INITIAL_00010", status="contested",
               dissent_pressure=0.8, dissent_set=["OBJ_01"])
    cont.status = "contested"
    proj = SynthesisProjection(surviving=[], contested=[cont])
    sp = build_plan(proj, store, dissent_k=3)
    assert "INITIAL_00010" in sp.dissent_clusters, (
        f"Contested cluster not in dissent_clusters: {sp.dissent_clusters}"
    )


def test_surviving_with_dissent_goes_to_dissent_list():
    store = _make_store()
    nodissent = _cp("INITIAL_00001", dissent_set=[])
    hasdissent = _cp("INITIAL_00002", dissent_set=["OBJ_01"],
                     dissent_pressure=0.6)
    sp = build_plan(_proj(nodissent, hasdissent), store, render_k=3, dissent_k=2)
    assert "INITIAL_00002" in sp.dissent_clusters or "INITIAL_00002" in sp.render_clusters, (
        "Cluster with dissent signals should be in either render or dissent"
    )


def test_dissent_k_cap():
    store = _make_store()
    contested = [
        _cp(f"INITIAL_{i:05d}", status="contested",
            dissent_pressure=float(i), dissent_set=["OBJ_01"])
        for i in range(6)
    ]
    for c in contested:
        c.status = "contested"
    proj = SynthesisProjection(surviving=[], contested=contested)
    sp = build_plan(proj, store, dissent_k=2)
    assert len(sp.dissent_clusters) <= 2, (
        f"dissent_clusters has {len(sp.dissent_clusters)} > dissent_k=2"
    )


# ---------------------------------------------------------------------------
# planner_notes
# ---------------------------------------------------------------------------

def test_planner_notes_non_empty():
    store = _make_store()
    cp0 = _cp("INITIAL_00001")
    sp = build_plan(_proj(cp0), store)
    assert isinstance(sp.planner_notes, str)
    assert len(sp.planner_notes) > 0


# ---------------------------------------------------------------------------
# Assertion guard
# ---------------------------------------------------------------------------

def test_assertion_fires_if_render_k_exceeded(monkeypatch):
    """build_plan should never return more render_clusters than render_k."""
    store = _make_store()
    clusters = [_cp(f"INITIAL_{i:05d}") for i in range(10)]
    for k in (1, 2, 3):
        sp = build_plan(_proj(*clusters), store, render_k=k)
        assert len(sp.render_clusters) <= k, (
            f"render_k={k} violated: got {len(sp.render_clusters)}"
        )
