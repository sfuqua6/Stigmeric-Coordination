"""Tests for core/relationships.py — dated peer/sector graph (point-in-time)."""

from datetime import date

import pytest

from core.relationships import RelationshipGraph


def _grocery_graph():
    g = RelationshipGraph()
    g.add("WMT", "DG", "same_sector", label="grocery", learned_on=date(2024, 1, 10))
    return g


def test_peers_undirected_symmetry():
    g = _grocery_graph()
    assert g.peers_of("WMT", as_of=date(2024, 1, 15)) == ["DG"]
    assert g.peers_of("DG", as_of=date(2024, 1, 15)) == ["WMT"]


def test_point_in_time_hides_future_edges():
    g = _grocery_graph()  # learned 2024-01-10
    assert g.peers_of("WMT", as_of=date(2024, 1, 5)) == []          # before learned
    assert g.relationship_between("WMT", "DG", as_of=date(2024, 1, 5)) is None
    assert g.relationship_between("WMT", "DG", as_of=date(2024, 1, 15)) == {
        "type": "same_sector", "label": "grocery"}


def test_relationship_between_either_ordering():
    g = _grocery_graph()
    asof = date(2024, 2, 1)
    assert g.relationship_between("WMT", "DG", as_of=asof)["label"] == "grocery"
    assert g.relationship_between("DG", "WMT", as_of=asof)["label"] == "grocery"


def test_sector_of():
    g = _grocery_graph()
    assert g.sector_of("DG", as_of=date(2024, 2, 1)) == "grocery"
    assert g.sector_of("DG", as_of=date(2024, 1, 1)) is None  # not yet learned


def test_supplier_of_is_directed():
    g = RelationshipGraph()
    g.add("TSM", "AAPL", "supplier_of", learned_on=date(2024, 1, 1))
    asof = date(2024, 2, 1)
    assert g.relationship_between("TSM", "AAPL", as_of=asof)["type"] == "supplier_of"
    assert g.relationship_between("AAPL", "TSM", as_of=asof) is None  # directed
    # peers_of surfaces the link in both directions (it's still a neighbour)
    assert "AAPL" in g.peers_of("TSM", as_of=asof)
    assert "TSM" in g.peers_of("AAPL", as_of=asof)


def test_types_filter():
    g = RelationshipGraph()
    g.add("A", "B", "competitor_of", learned_on=date(2024, 1, 1))
    g.add("A", "C", "same_sector", label="tech", learned_on=date(2024, 1, 1))
    asof = date(2024, 2, 1)
    assert set(g.peers_of("A", as_of=asof, types={"competitor_of"})) == {"B"}
    assert set(g.peers_of("A", as_of=asof)) == {"B", "C"}


def test_invalid_type_raises():
    g = RelationshipGraph()
    with pytest.raises(ValueError):
        g.add("A", "B", "frenemy")


def test_json_roundtrip(tmp_path):
    g = _grocery_graph()
    g.add("TSM", "AAPL", "supplier_of", learned_on=date(2024, 1, 2), source="manual")
    p = tmp_path / "rel.json"
    g.save(p)
    g2 = RelationshipGraph.load(p)
    asof = date(2024, 2, 1)
    assert g2.relationship_between("WMT", "DG", as_of=asof)["label"] == "grocery"
    assert g2.relationship_between("TSM", "AAPL", as_of=asof)["type"] == "supplier_of"


def test_load_missing_returns_empty(tmp_path):
    g = RelationshipGraph.load(tmp_path / "nope.json")
    assert g.peers_of("X") == []
