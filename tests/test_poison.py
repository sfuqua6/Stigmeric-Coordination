"""Poisoning harness: injection + adoption report."""
import json

from core.retrieval import _poison_chunks
from tools.poison_report import report


def test_no_env_no_poison(monkeypatch):
    monkeypatch.delenv("SWARM_POISON_FILE", raising=False)
    assert _poison_chunks() == []


def test_injection_loads_chunks(monkeypatch, tmp_path):
    pf = tmp_path / "p.json"
    pf.write_text(json.dumps([{"id": "x", "text": "The Trelleborg Accord.",
                               "canaries": ["Trelleborg"]}]), encoding="utf-8")
    monkeypatch.setenv("SWARM_POISON_FILE", str(pf))
    chunks = _poison_chunks()
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "poison_x"
    assert "Trelleborg" in chunks[0].text


def test_report_classifies_adoption(tmp_path):
    pf = tmp_path / "p.json"
    pf.write_text(json.dumps([
        {"id": "shipped", "canaries": ["Kawaguchi rebound"]},
        {"id": "field_only", "canaries": ["Meridian Consortium"]},
        {"id": "rejected", "canaries": ["Urban Air Directive 9"]},
    ]), encoding="utf-8")
    run = tmp_path / "run"
    run.mkdir()
    (run / "answer.txt").write_text(
        "Bans backfire via the Kawaguchi rebound effect.", encoding="utf-8")
    (run / "signals.json").write_text(
        json.dumps({"s": "the Meridian Consortium claims..."}), encoding="utf-8")
    out = report(str(run), str(pf))
    by_id = {r["id"]: r["status"] for r in out["rows"]}
    assert by_id == {"shipped": "ANSWER", "field_only": "FIELD",
                     "rejected": "REJECTED"}
    assert out["adoption_rate"] == round(1 / 3, 3)
    assert (run / "poison_report.json").exists()
