"""Task C: the synthesizer's relevance gate keeps grossly off-topic clusters
out of Sections 1-2 (defense in depth against contamination like an AAAI/ML
claim rendered into a theology answer)."""
from types import SimpleNamespace

from agents.synthesizer import _on_topic, _filter_on_topic


class _FakeStore:
    """No embedder (-> _on_topic uses the keyword-overlap fallback)."""

    def __init__(self, contents=None):
        self._c = contents or {}

    def _encode(self, _text):
        return None

    def get(self, sid):
        return SimpleNamespace(content=self._c[sid])


_PROMPT = "Should cities ban private cars to reduce traffic congestion?"


def test_on_topic_keyword_fallback():
    store = _FakeStore()
    assert _on_topic("Banning private cars reduces traffic in dense cities",
                     _PROMPT, store)
    assert not _on_topic("Artificial intelligence research organizations like AAAI",
                         _PROMPT, store)


def test_filter_drops_offtopic_cluster():
    store = _FakeStore({
        "INITIAL_1": "Banning private cars reduces traffic in dense cities",
        "INITIAL_2": "Artificial intelligence research organizations like AAAI",
    })
    cps = [SimpleNamespace(representative_id="INITIAL_1"),
           SimpleNamespace(representative_id="INITIAL_2")]
    ids = [cp.representative_id for cp in _filter_on_topic(cps, _PROMPT, store)]
    assert "INITIAL_1" in ids and "INITIAL_2" not in ids


def test_filter_never_returns_empty():
    # If the gate would drop everything, it must fall back to the full set
    # rather than starve the read-out.
    store = _FakeStore({"INITIAL_1": "Banning private cars"})
    cps = [SimpleNamespace(representative_id="INITIAL_1")]
    assert len(_filter_on_topic(cps, "totally unrelated xyzzy", store)) == 1
