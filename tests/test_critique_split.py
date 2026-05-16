"""Regression test for the CRITIQUE_POSITIVE / CRITIQUE_NEGATIVE split.

CRITIQUE_POSITIVE must be counted as support in build_projection (so it
contributes to support_diversity); CRITIQUE_NEGATIVE must be counted as
dissent (so it contributes to dissent_pressure). The pre-split code put
both under a single CRITIQUE bucket that downstream code treated as
dissent, which collapsed all clusters with positive critic feedback to
support_diversity=0 → weakly_supported. See core/signal_types.py for
the in-code rationale.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.signal_store import SignalStore, _NULL_EMBEDDER
from core.signal_types import (
    INITIAL, SUPPORT, CRITIQUE_POSITIVE, CRITIQUE_NEGATIVE, OBJECTION,
)
from core.projection import build_projection


class TestCritiqueSplit(unittest.TestCase):
    def test_projection_treats_critique_positive_as_support(self):
        """An INITIAL with two CRITIQUE_POSITIVE deposits from distinct critics
        should survive (support_diversity >= 2, no dissent)."""
        store = SignalStore(embedder=_NULL_EMBEDDER)
        init_id = store.deposit(
            INITIAL, "a substantive claim about renewable energy economics",
            0.6, "scout",
            metadata={"scout_agent_id": "scout_R1_0", "depositor_agent_id": "scout_R1_0"},
        )
        self.assertIsNotNone(init_id)
        # Two distinct critic strategy names → support_diversity=2
        store.deposit(
            CRITIQUE_POSITIVE, "well-grounded — corroborates multiple references",
            0.7, "critic", parent_id=init_id,
            metadata={"depositor_agent_id": "critic_R1_0_weighted_default"},
        )
        store.deposit(
            CRITIQUE_POSITIVE, "agrees with the broader evidence base",
            0.7, "critic", parent_id=init_id,
            metadata={"depositor_agent_id": "critic_R1_1_stratified_extremes"},
        )
        proj = build_projection(store, has_validators=False)
        self.assertEqual(len(proj.surviving), 1,
                         f"expected 1 surviving, got {len(proj.surviving)} "
                         f"(weakly_supported={len(proj.weakly_supported)})")
        self.assertGreaterEqual(proj.surviving[0].support_diversity, 2)

    def test_projection_treats_critique_negative_as_dissent(self):
        """An INITIAL with four CRITIQUE_NEGATIVE deposits and no support
        should land in rejected_by_field (dissent_pressure > 1.5)."""
        store = SignalStore(embedder=_NULL_EMBEDDER)
        init_id = store.deposit(
            INITIAL, "a thinly-supported speculative claim",
            0.5, "scout",
            metadata={"scout_agent_id": "scout_R1_0", "depositor_agent_id": "scout_R1_0"},
        )
        self.assertIsNotNone(init_id)
        for i in range(4):
            store.deposit(
                CRITIQUE_NEGATIVE,
                f"unsupported assertion number {i}; no source corroborates this",
                0.7, "critic", parent_id=init_id,
                metadata={"depositor_agent_id": f"critic_R1_{i}_strategy_{i}"},
            )
        proj = build_projection(store, has_validators=False)
        self.assertEqual(len(proj.rejected_by_field), 1,
                         f"expected 1 rejected_by_field, got "
                         f"{len(proj.rejected_by_field)} (surviving={len(proj.surviving)}, "
                         f"contested={len(proj.contested)})")

    def test_objection_also_counts_as_dissent(self):
        """OBJECTION (from Hater) is in the dissent bucket alongside
        CRITIQUE_NEGATIVE — make sure they accumulate together."""
        store = SignalStore(embedder=_NULL_EMBEDDER)
        init_id = store.deposit(
            INITIAL, "weak claim awaiting attack",
            0.5, "scout",
            metadata={"scout_agent_id": "scout_R1_0", "depositor_agent_id": "scout_R1_0"},
        )
        for i in range(3):
            store.deposit(
                OBJECTION, f"this cluster overgeneralizes case {i}",
                0.7, "hater", parent_id=init_id,
                metadata={"depositor_agent_id": f"hater_R1_{i}"},
            )
        proj = build_projection(store, has_validators=False)
        # All dissent, no support → rejected_by_field (dissent_pressure huge)
        self.assertEqual(len(proj.surviving), 0)
        self.assertEqual(len(proj.contested) + len(proj.rejected_by_field), 1)


class TestContrarianTypes(unittest.TestCase):
    """CONTRARIAN_TYPES drives the slower DELTA_DECAY_CONTRARIAN in
    SignalStore.decay_all. CRITIQUE_POSITIVE must NOT be contrarian —
    it should decay at the normal rate so positive evaluations don't
    accumulate indefinitely the way negative ones do.
    """
    def test_critique_positive_is_not_contrarian(self):
        from core.signal_types import CONTRARIAN_TYPES, CRITIQUE_POSITIVE
        self.assertNotIn(CRITIQUE_POSITIVE, CONTRARIAN_TYPES)

    def test_critique_negative_is_contrarian(self):
        from core.signal_types import CONTRARIAN_TYPES, CRITIQUE_NEGATIVE
        self.assertIn(CRITIQUE_NEGATIVE, CONTRARIAN_TYPES)

    def test_objection_is_contrarian(self):
        from core.signal_types import CONTRARIAN_TYPES, OBJECTION
        self.assertIn(OBJECTION, CONTRARIAN_TYPES)


if __name__ == "__main__":
    unittest.main()
