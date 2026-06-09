"""GroqBackend availability handling — the fix for the lastgroqrun.txt failure
where the hater's model 404'd 34x and made 0 calls all run.

A 404 ("does not exist or you do not have access") must trigger the same
fallback swap as a decommissioned-400, and preflight must perform that swap
*before* the run. No network, no openai/groq package — the client is faked.
"""

import asyncio

import core.llm_groq as g
from core.llm_groq import GroqBackend, _FALLBACK_MODEL

_BAD = "meta-llama/llama-4-maverick-17b-128e-instruct"


class _Resp:
    def __init__(self, content):
        self.choices = [type("C", (), {"message": type("M", (), {"content": content})})]


class _FakeCompletions:
    def __init__(self, fail_models):
        self.fail_models = set(fail_models)
        self.calls = []

    async def create(self, model, messages, max_tokens, temperature):
        self.calls.append(model)
        if model in self.fail_models:
            raise Exception(
                f"Error code: 404 - {{'error': {{'message': 'The model `{model}` "
                f"does not exist or you do not have access to it.'}}}}"
            )
        return _Resp("ok")


class _FakeClient:
    def __init__(self, fail_models):
        self.chat = type("Chat", (), {"completions": _FakeCompletions(fail_models)})()


def _backend(monkeypatch, model, fail_models):
    g._DECOMMISSIONED.clear()
    client = _FakeClient(fail_models)
    monkeypatch.setattr(g, "_get_groq_client", lambda api_key: client)
    return GroqBackend(model=model, api_key="x"), client


def test_is_unavailable_matches_404_and_decommissioned(monkeypatch):
    be, _ = _backend(monkeypatch, _BAD, fail_models=set())
    assert be._is_unavailable_error("Error code: 404 - does not exist")
    assert be._is_unavailable_error("model has been decommissioned")
    assert be._is_unavailable_error("you do not have access to it")
    assert not be._is_unavailable_error("Error code: 429 - rate limit reached")
    assert not be._is_unavailable_error("connection timeout")


def test_404_swaps_to_fallback_and_succeeds(monkeypatch):
    be, client = _backend(monkeypatch, _BAD, fail_models={_BAD})
    out = asyncio.run(be.generate("hi", max_tokens=1))
    assert out == "ok"                               # fallback answered
    assert be._model == _FALLBACK_MODEL              # role now routes to fallback
    assert client.chat.completions.calls[0] == _BAD  # tried the bad model first
    assert client.chat.completions.calls[-1] == _FALLBACK_MODEL


def test_preflight_swaps_before_run(monkeypatch):
    be, _ = _backend(monkeypatch, _BAD, fail_models={_BAD})
    ok, model = asyncio.run(be.preflight())
    assert ok is True
    assert model == _FALLBACK_MODEL
    assert be._model == _FALLBACK_MODEL


def test_preflight_reports_dead_when_fallback_also_down(monkeypatch):
    # Both the requested model AND the fallback fail -> role is genuinely dead.
    be, _ = _backend(monkeypatch, _BAD, fail_models={_BAD, _FALLBACK_MODEL})
    ok, _model = asyncio.run(be.preflight())
    assert ok is False
    assert be._call_count == 0                       # silent_roles will flag this
