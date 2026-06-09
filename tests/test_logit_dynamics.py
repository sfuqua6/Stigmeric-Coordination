"""Unit tests for logit-space strength dynamics (§4 of the directive).

# FUTURE-CLAUDE NOTE: These tests are the contract for USE_LOGIT_DYNAMICS.
# If you change DELTA_* constants in config.py, the saturation and decay
# expectations below may need adjusting, but the *qualitative* properties
# (no saturation crash, contrarians decay, order invariance) MUST hold.
# If they do not, the dynamics are broken — fix the dynamics, not the test.

Run with:
    python -m unittest tests.test_logit_dynamics -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import config
from core.signal_store import (
    SignalStore,
    _NULL_EMBEDDER,
    _to_logit,
    _from_logit,
)
from core.signal_types import INITIAL, CRITIQUE, OBJECTION


class TestLogitHelpers(unittest.TestCase):
    def test_round_trip(self):
        for s in (0.01, 0.1, 0.5, 0.7, 0.9, 0.99):
            self.assertAlmostEqual(_from_logit(_to_logit(s)), s, places=5)

    def test_endpoints_clamped(self):
        # No ±inf; values near 0/1 are clamped via eps
        x_lo = _to_logit(0.0)
        x_hi = _to_logit(1.0)
        self.assertTrue(x_lo < 0 and x_hi > 0)
        # sigmoid is monotone and finite
        self.assertGreater(_from_logit(x_hi), 0.999)
        self.assertLess(_from_logit(x_lo), 0.001)


@unittest.skipUnless(config.USE_LOGIT_DYNAMICS,
                     "Logit dynamics disabled via config.USE_LOGIT_DYNAMICS")
class TestNoSaturation(unittest.TestCase):
    def test_repeated_amplify_does_not_pin_at_one(self):
        store = SignalStore(embedder=_NULL_EMBEDDER)
        sid = store.deposit(
            signal_type=INITIAL, content="A claim.", strength=0.5,
            depositor="scout", metadata={"partition_id": "test_partition_0"},
        )
        for _ in range(10):
            store.amplify(sid)
        s = store.get(sid).strength
        # logit(0.5)=0; +10*0.3 = 3.0; sigmoid(3) ≈ 0.953
        self.assertLess(s, 0.999,
                        f"Strength should not saturate at 1.0; got {s}")
        self.assertGreater(s, 0.9)


@unittest.skipUnless(config.USE_LOGIT_DYNAMICS,
                     "Logit dynamics disabled via config.USE_LOGIT_DYNAMICS")
class TestContrarianDecay(unittest.TestCase):
    def test_contrarian_decays_without_amplify(self):
        """The bug fix: contrarian signals must decay (slowly) when not
        actively corroborated. Old code's exact anti-decay let
        dissent_pressure accumulate forever."""
        store = SignalStore(embedder=_NULL_EMBEDDER)
        # Need a parent INITIAL so OBJECTION has a target
        pid = store.deposit(INITIAL, "Target claim.", 0.6, "scout",
                            metadata={"partition_id": "test_partition_0"})
        oid = store.deposit(OBJECTION, "An objection.", 0.7, "hater",
                            parent_id=pid)
        start = store.get(oid).strength
        for _ in range(50):
            store.decay_all()
        end = store.get(oid).strength
        self.assertLess(end, start - 0.2,
                        f"Contrarian should decay over 50 rounds: "
                        f"start={start:.3f}, end={end:.3f}")


@unittest.skipUnless(config.USE_LOGIT_DYNAMICS,
                     "Logit dynamics disabled via config.USE_LOGIT_DYNAMICS")
class TestOrderInvariance(unittest.TestCase):
    def test_decay_then_amplify_equals_amplify_then_decay(self):
        """Logit-space updates are exact additions; order must not matter."""
        store_a = SignalStore(embedder=_NULL_EMBEDDER)
        store_b = SignalStore(embedder=_NULL_EMBEDDER)
        a = store_a.deposit(INITIAL, "Same claim.", 0.5, "scout",
                            metadata={"partition_id": "test_partition_0"})
        b = store_b.deposit(INITIAL, "Same claim.", 0.5, "scout",
                            metadata={"partition_id": "test_partition_0"})
        # A: decay then amplify
        store_a.decay_all()
        store_a.amplify(a)
        # B: amplify then decay
        store_b.amplify(b)
        store_b.decay_all()
        self.assertAlmostEqual(
            store_a.get(a).strength,
            store_b.get(b).strength,
            places=6,
        )


class TestLegacyEscapeHatch(unittest.TestCase):
    """Confirms USE_LOGIT_DYNAMICS=False still runs without exceptions.

    We don't assert specific values — the legacy path is on a deprecation
    track. The test only ensures we haven't accidentally broken it while
    refactoring; remove this test when the legacy branch is deleted.
    """
    def test_legacy_path_runs(self):
        # Toggle the flag via monkey-patch on the imported module
        from core import signal_store as ss
        original = ss.USE_LOGIT_DYNAMICS
        ss.USE_LOGIT_DYNAMICS = False
        try:
            store = SignalStore(embedder=_NULL_EMBEDDER)
            sid = store.deposit(INITIAL, "A claim.", 0.5, "scout",
                                metadata={"partition_id": "test_partition_0"})
            store.amplify(sid)
            store.decay_all()
            self.assertIsNotNone(store.get(sid))
            self.assertGreater(store.get(sid).strength, 0.0)
        finally:
            ss.USE_LOGIT_DYNAMICS = original


if __name__ == "__main__":
    unittest.main()
