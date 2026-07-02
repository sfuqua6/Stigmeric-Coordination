"""Semantic pass in find_cached_query (embedding-cosine query dedup).

Real runs issued 57 searches with 6 cache hits because dedup was lexical
only — SequenceMatcher can't see that two rewordings are one fetch. The
semantic pass reuses a served query when cosine >= SEMANTIC_DUP_COS.
Vectors are monkeypatched so the tests need no embedding model.
"""
import math

import core.query_planner as qp


def _unit(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


def _patch_vecs(monkeypatch, table):
    monkeypatch.setattr(qp, "_query_vec", lambda q: table.get(q))


def test_lexical_exact_still_wins(monkeypatch):
    _patch_vecs(monkeypatch, {})  # semantic pass would find nothing
    served = {"apollo program cost": 5}
    assert qp.find_cached_query("apollo program cost", served) == "apollo program cost"


def test_semantic_match_reuses_served(monkeypatch):
    # Lexically distant, semantically identical (cos = 1.0).
    v = _unit([1.0, 2.0, 3.0])
    _patch_vecs(monkeypatch, {
        "how expensive was the apollo program": v,
        "apollo program cost overruns": v,
    })
    served = {"apollo program cost overruns": 5}
    got = qp.find_cached_query("how expensive was the apollo program", served)
    assert got == "apollo program cost overruns"


def test_below_cosine_threshold_no_reuse(monkeypatch):
    _patch_vecs(monkeypatch, {
        "solar subsidies in germany": _unit([1.0, 0.0, 0.0]),
        "apollo program cost overruns": _unit([0.0, 1.0, 0.0]),  # cos = 0
    })
    served = {"apollo program cost overruns": 5}
    assert qp.find_cached_query("solar subsidies in germany", served) is None


def test_poorly_served_queries_excluded(monkeypatch):
    # A served query below MIN_GOOD_RESULTS must not be reused even at cos 1.
    v = _unit([1.0, 1.0, 0.0])
    _patch_vecs(monkeypatch, {"q one": v, "q two": v})
    served = {"q two": 0}
    assert qp.find_cached_query("q one", served) is None


def test_embedder_unavailable_degrades_to_lexical(monkeypatch):
    _patch_vecs(monkeypatch, {})  # every _query_vec -> None
    served = {"apollo program cost overruns": 5}
    assert qp.find_cached_query("how expensive was the apollo program", served) is None
