"""Fix A — Partition wiring propagation tests.

Verifies that partition_id flows from intake → scout deposit → signal store
→ projection, and that the hard assertion in deposit() fires when partition_id
is absent from an INITIAL or SUPPORT signal.

Run with:
    pytest tests/test_partition_propagation.py -v
"""

import sys
import os
import pytest

# Make the repo root importable whether tests run from tests/ or from the root.
_HERE = os.path.dirname(__file__)
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.intake import chunk_corpus, partition_for_scouts, CorpusChunk, ScoutPartition
from core.signal_store import SignalStore, Signal
from core.projection import _compute_initial_metrics


# ---------------------------------------------------------------------------
# 1. intake.py — partition_for_scouts stamps partition_id on every chunk
# ---------------------------------------------------------------------------

def test_chunks_get_partition_id_stamped():
    text = " ".join(["word"] * 200)
    chunks = chunk_corpus(text, source_tag="test")
    # All chunks start without a partition_id
    assert all(c.partition_id == "" for c in chunks), "chunks should start unpartitioned"

    partitions = partition_for_scouts(chunks, num_scouts=3)
    assert len(partitions) == 3

    for part in partitions:
        expected_pid = f"partition_{part.scout_index}"
        for ch in part.chunks:
            assert ch.partition_id == expected_pid, (
                f"chunk {ch.chunk_id} in scout {part.scout_index} has "
                f"partition_id={ch.partition_id!r}, expected {expected_pid!r}"
            )


def test_scout_partition_id_property():
    dummy_chunks = [CorpusChunk(chunk_id="c0", text="hello", source_tag="t")]
    p = ScoutPartition(scout_index=5, chunks=dummy_chunks)
    assert p.partition_id == "partition_5"


def test_empty_corpus_partitions_have_partition_id():
    partitions = partition_for_scouts([], num_scouts=2)
    assert len(partitions) == 2
    for part in partitions:
        assert part.partition_id == f"partition_{part.scout_index}"


# ---------------------------------------------------------------------------
# 2. signal_store.py — INITIAL deposit carries partition_id
# ---------------------------------------------------------------------------

def test_initial_deposit_stores_partition_id():
    store = SignalStore(embedder=None)
    sid = store.deposit(
        signal_type="INITIAL",
        content="The climate is warming due to greenhouse gases.",
        strength=0.6,
        depositor="scout",
        parent_id=None,
        metadata={"partition_id": "partition_0", "scout_agent_id": "scout_R1_0"},
    )
    assert sid is not None
    sig = store.get(sid)
    assert sig is not None
    assert sig.partition_id == "partition_0", (
        f"INITIAL signal should have partition_id='partition_0', got {sig.partition_id!r}"
    )


def test_support_inherits_partition_id_from_parent():
    store = SignalStore(embedder=None)
    init_id = store.deposit(
        signal_type="INITIAL",
        content="Renewable energy reduces carbon emissions significantly.",
        strength=0.6,
        depositor="scout",
        parent_id=None,
        metadata={"partition_id": "partition_1"},
    )
    assert init_id is not None

    sup_id = store.deposit(
        signal_type="SUPPORT",
        content="Solar capacity grew 30% in 2023 per IEA data.",
        strength=0.55,
        depositor="developer",
        parent_id=init_id,
        metadata={},  # no explicit partition_id — must inherit
    )
    assert sup_id is not None
    sup_sig = store.get(sup_id)
    assert sup_sig is not None
    assert sup_sig.partition_id == "partition_1", (
        f"SUPPORT should inherit partition_id from parent INITIAL, "
        f"got {sup_sig.partition_id!r}"
    )


def test_support_chain_inherits_partition_id():
    """SUPPORT -> SUPPORT chain: partition_id flows through two hops."""
    store = SignalStore(embedder=None)
    init_id = store.deposit(
        signal_type="INITIAL",
        content="Urban density correlates with transit use.",
        strength=0.6,
        depositor="scout",
        parent_id=None,
        metadata={"partition_id": "partition_2"},
    )
    sup1_id = store.deposit(
        signal_type="SUPPORT",
        content="Tokyo's density exceeds 6000 people/km².",
        strength=0.55,
        depositor="developer",
        parent_id=init_id,
        metadata={},
    )
    sup2_id = store.deposit(
        signal_type="SUPPORT",
        content="Tokyo's transit ridership is the world's highest.",
        strength=0.55,
        depositor="developer",
        parent_id=sup1_id,
        metadata={},
    )
    assert store.get(sup2_id).partition_id == "partition_2", (
        "partition_id must flow through SUPPORT chain"
    )


def test_deposit_assertion_fires_for_initial_without_partition_id():
    """The hard assertion must raise when a scout forgets partition_id."""
    store = SignalStore(embedder=None)
    with pytest.raises(AssertionError, match=r"\[PARTITION LEAK\]"):
        store.deposit(
            signal_type="INITIAL",
            content="Missing partition — this should raise.",
            strength=0.6,
            depositor="scout",
            parent_id=None,
            metadata={},  # no partition_id
        )


def test_deposit_assertion_fires_for_support_without_partition_id_and_no_parent():
    """SUPPORT with no parent and no metadata partition_id must raise."""
    store = SignalStore(embedder=None)
    with pytest.raises(AssertionError, match=r"\[PARTITION LEAK\]"):
        store.deposit(
            signal_type="SUPPORT",
            content="Orphan support — should raise.",
            strength=0.5,
            depositor="developer",
            parent_id=None,
            metadata={},
        )


def test_non_initial_non_support_skips_assertion():
    """OBJECTION and VERIFICATION do not need partition_id — no assertion."""
    store = SignalStore(embedder=None)
    # These should not raise even without partition_id
    oid = store.deposit(
        signal_type="OBJECTION",
        content="This claim ignores externalities.",
        strength=0.5,
        depositor="hater",
        parent_id=None,
        metadata={},
    )
    assert oid is not None

    vid = store.deposit(
        signal_type="VERIFICATION",
        content="Wikipedia confirms CO2 data.",
        strength=0.8,
        depositor="validator",
        parent_id=None,
        metadata={},
    )
    assert vid is not None


# ---------------------------------------------------------------------------
# 3. projection.py — _compute_initial_metrics reads partition_id from Signal
# ---------------------------------------------------------------------------

def test_projection_reads_partition_id_field():
    """After Fix A, projection prefers sig.partition_id over metadata parsing."""
    store = SignalStore(embedder=None)
    sid = store.deposit(
        signal_type="INITIAL",
        content="Deforestation accelerates climate feedback loops.",
        strength=0.7,
        depositor="scout",
        parent_id=None,
        metadata={
            "partition_id": "partition_3",
            "scout_agent_id": "scout_R1_3",
        },
    )
    sig = store.get(sid)
    metrics = _compute_initial_metrics(sig, store)
    assert metrics["partition_tag"] == "partition_3", (
        f"Expected partition_tag='partition_3', got {metrics['partition_tag']!r}"
    )


def test_projection_fallback_to_agent_id_parse():
    """Legacy signals with no partition_id field fall back to agent_id parse."""
    # Construct a Signal with no partition_id but with scout_agent_id in metadata.
    store = SignalStore(embedder=None)
    # Bypass deposit() assertion by depositing with partition_id then manually clearing.
    sid = store.deposit(
        signal_type="INITIAL",
        content="Legacy signal without explicit partition_id field.",
        strength=0.6,
        depositor="scout",
        parent_id=None,
        metadata={
            "scout_agent_id": "scout_R1_2",
            "partition_id": "partition_2",  # needed to pass assertion
        },
    )
    sig = store.get(sid)
    # Simulate a legacy signal by clearing partition_id
    sig.partition_id = ""

    metrics = _compute_initial_metrics(sig, store)
    # Should fall back to _parse_partition_tag("scout_R1_2") → "partition_2"
    assert metrics["partition_tag"] == "partition_2", (
        f"Fallback parse failed: got {metrics['partition_tag']!r}"
    )


# ---------------------------------------------------------------------------
# 4. Round-trip: partition_id survives save/load
# ---------------------------------------------------------------------------

def test_partition_id_round_trips_through_save_load(tmp_path):
    store1 = SignalStore(embedder=None)
    sid = store1.deposit(
        signal_type="INITIAL",
        content="Saved partition should survive serialization.",
        strength=0.6,
        depositor="scout",
        parent_id=None,
        metadata={"partition_id": "partition_7"},
    )
    path = tmp_path / "store.json"
    store1.save_state(path)

    store2 = SignalStore(embedder=None)
    store2.load_state(path)
    sig = store2.get(sid)
    assert sig is not None
    assert sig.partition_id == "partition_7", (
        f"partition_id lost during save/load: got {sig.partition_id!r}"
    )
