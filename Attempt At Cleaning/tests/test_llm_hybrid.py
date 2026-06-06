"""HybridRouter routing tests — local model for high-volume roles, Groq for the
few high-value ones. Injected fakes; no GPU, no network, no openai/groq package.
"""

import pytest

from core.llm_hybrid import HybridRouter


class _FakeLLM:
    def __init__(self, name):
        self.name = name

    async def generate(self, *a, **k):
        return ""


class _FakeBackend:
    def __init__(self, model):
        self._model = model
        self.name = f"groq:{model}"

    def stats(self):
        return {"model": self._model, "calls": 0, "avg_latency_ms": 0.0}

    async def generate(self, *a, **k):
        return ""


def _router(monkeypatch, groq_roles=("synthesizer", "hater"),
            groq_models=None):
    for k in ("SWARM_HYBRID_GROQ_ROLES", "GROQ_ROLE_SYNTHESIZER",
              "GROQ_ROLE_HATER", "GROQ_ROLE_CRITIC"):
        monkeypatch.delenv(k, raising=False)
    gm = groq_models or {r: ("70b" if r == "synthesizer" else
                             "mav" if r == "hater" else "8b") for r in groq_roles}
    backends = {m: _FakeBackend(m) for m in set(gm.values())}
    return HybridRouter(api_key="test", local_llm=_FakeLLM("local-qwen"),
                        groq_roles=set(groq_roles), groq_models=gm,
                        groq_backends=backends)


def test_high_volume_roles_run_local(monkeypatch):
    r = _router(monkeypatch)
    for role in ("scout", "developer", "critic", "validator"):
        assert r.engine_for(role).name == "local-qwen"


def test_high_value_roles_run_on_groq(monkeypatch):
    r = _router(monkeypatch)
    assert r.engine_for("synthesizer")._model == "70b"
    assert r.engine_for("hater")._model == "mav"


def test_manifest_labels_source(monkeypatch):
    r = _router(monkeypatch)
    m = r.manifest()
    assert m["scout"].startswith("local:")
    assert m["validator"].startswith("local:")
    assert m["synthesizer"] == "groq:70b"
    assert m["hater"] == "groq:mav"


def test_custom_split_can_send_critic_to_groq(monkeypatch):
    r = _router(monkeypatch, groq_roles=("synthesizer", "hater", "critic"),
                groq_models={"synthesizer": "70b", "hater": "mav", "critic": "8b"})
    assert r.engine_for("critic")._model == "8b"
    assert r.engine_for("scout").name == "local-qwen"   # still local


def test_role_disabled(monkeypatch):
    r = _router(monkeypatch)
    assert r.role_disabled("scout") is False
    r.disabled_roles.add("scout")
    assert r.role_disabled("scout") is True


def test_engines_dict_has_local_and_groq(monkeypatch):
    r = _router(monkeypatch)
    assert "local" in r.engines
    assert any(k != "local" for k in r.engines)   # at least one groq backend
    assert r.bundle_name == "hybrid"


def test_requires_key_when_no_backends_injected(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ValueError):
        HybridRouter(api_key="", local_llm=_FakeLLM("x"))
