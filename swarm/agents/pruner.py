"""Pruner agent - maintains signal quality by removing weak/stale/duplicate signals."""

import asyncio
import time
from typing import List
from ..core.signal_store import SignalStore, Signal
from difflib import SequenceMatcher
from ..core.logging_config import get_logger

logger = get_logger(__name__)


class Pruner:
    """Pruner agent that actively manages signal quality.

    Distinct role from passive decay:
    - Decay is passive (automatic strength reduction over time)
    - Pruner is active (intelligent removal based on multiple criteria)

    Removes signals that are:
    1. Below strength threshold (weak)
    2. Stale (old with no recent engagement)
    3. Duplicates (too similar to stronger signals)
    4. Orphaned (parent was removed)
    """

    def __init__(self, agent_id: str, min_strength: float = 0.15,
                 staleness_threshold: float = 120.0,
                 similarity_threshold: float = 0.85):
        """Initialize pruner.

        Args:
            agent_id: Unique agent ID
            min_strength: Remove signals below this strength
            staleness_threshold: Remove signals older than this (seconds) with no engagement
            similarity_threshold: Remove duplicates above this similarity (0.0-1.0)
        """
        self.agent_id = agent_id
        self.min_strength = min_strength
        self.staleness_threshold = staleness_threshold
        self.similarity_threshold = similarity_threshold
        self.active = True
        self.actions_taken = 0
        self.total_pruned = 0

    async def run(self, signal_store: SignalStore, max_actions: int = None):
        """Run pruner behavior loop.

        Args:
            signal_store: Shared signal store
            max_actions: Optional maximum actions (None = run until stopped)
        """
        logger.info("%s starting (min_strength=%.2f, staleness_threshold=%.1fs)",
                   self.agent_id, self.min_strength, self.staleness_threshold)

        while self.active and (max_actions is None or self.actions_taken < max_actions):
            # Pruning pass
            pruned_count = await self.prune_pass(signal_store)

            if pruned_count > 0:
                self.total_pruned += pruned_count
                logger.info("%s pruned %d signals (total pruned: %d)",
                          self.agent_id, pruned_count, self.total_pruned)

            self.actions_taken += 1
            # Reduced delay for more responsive pruning (was 3.0s)
            await asyncio.sleep(1.0)

    async def prune_pass(self, signal_store: SignalStore) -> int:
        """Perform one pruning pass.

        Args:
            signal_store: Signal store

        Returns:
            Number of signals pruned
        """
        pruned = 0
        all_signals = signal_store.get_all_signals()

        if len(all_signals) == 0:
            return 0

        current_time = time.time()

        # Track signals to remove
        to_remove = []

        # 1. Remove weak signals (below threshold)
        weak_signals = [s for s in all_signals if s.strength < self.min_strength]
        to_remove.extend([s.id for s in weak_signals])
        if weak_signals:
            logger.debug("Found %d weak signals (strength < %.2f)",
                        len(weak_signals), self.min_strength)

        # 2. Remove stale signals (old + no engagement)
        stale_signals = []
        for signal in all_signals:
            age = current_time - signal.timestamp
            if age > self.staleness_threshold and signal.visits <= 1:
                # Only prune if not already marked for removal
                if signal.id not in to_remove:
                    stale_signals.append(signal)
                    to_remove.append(signal.id)

        if stale_signals:
            logger.debug("Found %d stale signals (age > %.1fs, visits <= 1)",
                        len(stale_signals), self.staleness_threshold)

        # 3. Remove duplicates (too similar to stronger signals)
        duplicates = self._find_duplicates(all_signals, signal_store)
        for dup_id in duplicates:
            if dup_id not in to_remove:
                to_remove.append(dup_id)

        if duplicates:
            logger.debug("Found %d duplicate signals (similarity > %.2f)",
                        len(duplicates), self.similarity_threshold)

        # 4. Remove orphaned signals (parent was removed)
        orphaned_signals = []
        for signal in all_signals:
            # BUGFIX: Use get_signal() with proper locking instead of direct access
            if signal.parent and signal_store.get_signal(signal.parent) is None:
                # Parent was removed, orphan this signal
                if signal.id not in to_remove:
                    orphaned_signals.append(signal)
                    to_remove.append(signal.id)

        if orphaned_signals:
            logger.debug("Found %d orphaned signals (parent removed)", len(orphaned_signals))

        # Actually remove signals using centralized delete_signal() method
        # This ensures consistent cleanup of embeddings, cache, etc.
        for signal_id in to_remove:
            if signal_store.delete_signal(signal_id):
                pruned += 1

        return pruned

    def _find_duplicates(self, all_signals: List[Signal],
                        signal_store: SignalStore) -> List[str]:
        """Find duplicate signals (too similar to stronger signals of same type).

        Args:
            all_signals: List of all signals
            signal_store: Signal store

        Returns:
            List of signal IDs to remove (duplicates)
        """
        duplicates = []

        # Group by type for comparison
        by_type = {}
        for signal in all_signals:
            if signal.type not in by_type:
                by_type[signal.type] = []
            by_type[signal.type].append(signal)

        # Within each type, find duplicates
        for signal_type, signals in by_type.items():
            if len(signals) < 2:
                continue  # Need at least 2 to find duplicates

            # Sort by strength (strongest first)
            signals.sort(key=lambda s: s.strength, reverse=True)

            # Check each signal against stronger ones
            for i, signal in enumerate(signals):
                if signal.id in duplicates:
                    continue  # Already marked

                # Compare with all stronger signals
                for j in range(i):
                    stronger_signal = signals[j]

                    if stronger_signal.id in duplicates:
                        continue  # Don't compare with duplicates

                    # Calculate similarity
                    similarity = self._calculate_similarity(signal.content, stronger_signal.content)

                    if similarity >= self.similarity_threshold:
                        # This is a duplicate of a stronger signal
                        duplicates.append(signal.id)
                        break  # Found duplicate, move to next signal

        return duplicates

    def _calculate_similarity(self, content1: str, content2: str) -> float:
        """Calculate similarity between two content strings.

        Args:
            content1: First content
            content2: Second content

        Returns:
            Similarity score 0.0-1.0
        """
        # Use SequenceMatcher for string similarity
        return SequenceMatcher(None, content1.lower(), content2.lower()).ratio()

    def get_stats(self) -> dict:
        """Get pruner statistics.

        Returns:
            Dictionary with stats
        """
        return {
            'agent_id': self.agent_id,
            'actions_taken': self.actions_taken,
            'total_pruned': self.total_pruned,
            'min_strength': self.min_strength,
            'staleness_threshold': self.staleness_threshold,
            'similarity_threshold': self.similarity_threshold
        }

    def stop(self) -> None:
        """Stop the agent."""
        self.active = False
