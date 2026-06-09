"""Stridulation protocol - quality-based vibration recruitment.

When leaf cutter ants cut leaves, they produce a high-frequency vibration called
stridulation using a specialized stridulatory organ (a file-and-scraper mechanism
on the gaster). This vibration serves two purposes:

1. MECHANICAL: The vibration stiffens soft leaf tissue, making it easier to cut
2. COMMUNICATION: The stridulation acts as a recruitment signal to nearby ants

Scientists found that significantly more ants stridulate when cutting TENDER leaves
than tough leaves, and when leaves are coated in sugar, almost ALL workers stridulate.
The vibrations communicate resource quality: "This is good stuff, come help!"

Interestingly, scientists believe the communication function evolved FIRST, and the
mechanical cutting benefit is an auxiliary adaptation. Stridulation is also used in
other contexts: buried ants stridulate to call for help, and vibrations coordinate
nest building and excavation.

In our system, stridulation models quality-based recruitment:
- When an agent finds high-quality material, it "stridulates" (emits a recruitment signal)
- The signal strength is proportional to the quality of the find
- Nearby agents detect the vibration and are recruited to the productive area
- This creates positive feedback: good resources attract more processing power
"""

import time
from threading import Lock
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class StridulationSignal:
    """A vibration signal emitted by an agent about resource quality.

    Like real ant stridulation, the signal encodes information about what was
    found and how good it is, attracting nearby workers to help process it.
    """
    id: str
    emitter: str                   # Agent who stridulated
    signal_id: str                 # What signal/resource they found
    frequency: float               # Intensity (maps to quality, 0.0-1.0)
    context: str                   # Topic/area where the resource was found
    timestamp: float
    decay_rate: float = 0.12       # Vibrations decay quickly (12% per tick)
    receivers: List[str] = field(default_factory=list)  # Agents who detected

    @property
    def is_expired(self) -> bool:
        """Check if vibration has decayed below threshold."""
        return self.frequency < 0.02

    def decay(self) -> float:
        """Apply one tick of vibration decay."""
        self.frequency *= (1.0 - self.decay_rate)
        return self.frequency


# Stridulation quality thresholds
# In real ants, almost ALL workers stridulate on sugar-coated leaves
# but significantly fewer on tough leaves
QUALITY_THRESHOLD_LOW = 0.3       # Below this: no stridulation
QUALITY_THRESHOLD_HIGH = 0.7      # Above this: strong stridulation
RECRUITMENT_RADIUS = 3            # How many "hops" the vibration travels
MAX_ACTIVE_SIGNALS = 100          # Limit active stridulation signals


class StridulationProtocol:
    """Quality-based recruitment through vibration signals.

    When agents find valuable resources, they stridulate to recruit helpers.
    The protocol manages active vibration signals and lets agents "listen"
    for nearby stridulations to decide where to focus their efforts.
    """

    def __init__(self, quality_threshold: float = QUALITY_THRESHOLD_LOW,
                 max_signals: int = MAX_ACTIVE_SIGNALS):
        """Initialize the stridulation protocol.

        Args:
            quality_threshold: Minimum quality to trigger stridulation
            max_signals: Maximum active stridulation signals
        """
        self._lock = Lock()
        self.active_signals: Dict[str, StridulationSignal] = {}
        self.quality_threshold = quality_threshold
        self.max_signals = max_signals
        self._next_id = 0

        # Indexes
        self._by_context: Dict[str, set] = {}  # context -> {signal_ids}
        self._by_emitter: Dict[str, set] = {}  # emitter -> {signal_ids}

        # Stats
        self.total_stridulations = 0
        self.total_recruitments = 0  # Times an agent responded to a stridulation

    def stridulate(self, emitter: str, signal_id: str,
                   quality: float, context: str) -> Optional[str]:
        """Emit a stridulation signal about a discovered resource.

        Only triggers if quality exceeds threshold (like real ants only
        stridulating on good leaves). The vibration frequency is proportional
        to the resource quality.

        Args:
            emitter: Agent emitting the stridulation
            signal_id: ID of the resource/signal found
            quality: Quality of the find (0.0-1.0)
            context: Topic/area context

        Returns:
            Stridulation signal ID, or None if quality too low
        """
        # Below quality threshold: no stridulation (like tough leaves)
        if quality < self.quality_threshold:
            return None

        with self._lock:
            # Enforce signal limit
            if len(self.active_signals) >= self.max_signals:
                self._remove_weakest_locked()

            strid_id = f"STRID_{self._next_id:04d}"
            self._next_id += 1

            # Frequency proportional to quality
            # Maps quality range [threshold, 1.0] to frequency [0.3, 1.0]
            quality_range = 1.0 - self.quality_threshold
            if quality_range > 0:
                normalized = (quality - self.quality_threshold) / quality_range
            else:
                normalized = 1.0
            frequency = 0.3 + normalized * 0.7

            signal = StridulationSignal(
                id=strid_id,
                emitter=emitter,
                signal_id=signal_id,
                frequency=frequency,
                context=context,
                timestamp=time.time(),
            )

            self.active_signals[strid_id] = signal

            # Update indexes
            if context not in self._by_context:
                self._by_context[context] = set()
            self._by_context[context].add(strid_id)

            if emitter not in self._by_emitter:
                self._by_emitter[emitter] = set()
            self._by_emitter[emitter].add(strid_id)

            self.total_stridulations += 1
            return strid_id

    def _remove_weakest_locked(self):
        """Remove weakest stridulation signal (lock must be held)."""
        if not self.active_signals:
            return
        weakest = min(self.active_signals.values(), key=lambda s: s.frequency)
        self._remove_signal_locked(weakest.id)

    def _remove_signal_locked(self, strid_id: str):
        """Remove a signal from all indexes (lock must be held)."""
        signal = self.active_signals.get(strid_id)
        if not signal:
            return

        # Remove from indexes
        ctx_set = self._by_context.get(signal.context)
        if ctx_set:
            ctx_set.discard(strid_id)
            if not ctx_set:
                del self._by_context[signal.context]

        emitter_set = self._by_emitter.get(signal.emitter)
        if emitter_set:
            emitter_set.discard(strid_id)
            if not emitter_set:
                del self._by_emitter[signal.emitter]

        del self.active_signals[strid_id]

    def listen(self, listener_id: str,
               context: Optional[str] = None,
               sensitivity: float = 0.7) -> List[StridulationSignal]:
        """Listen for nearby stridulations.

        Models the ant's ability to detect vibrations through the substrate.
        Sensitivity affects the minimum frequency detectable.

        Args:
            listener_id: Agent listening
            context: Optional context filter (only hear stridulations in this area)
            sensitivity: Detection sensitivity 0.0-1.0

        Returns:
            List of detectable stridulation signals, sorted by frequency
        """
        min_detectable = 0.05 / max(sensitivity, 0.01)

        with self._lock:
            if context:
                strid_ids = self._by_context.get(context, set())
            else:
                strid_ids = set(self.active_signals.keys())

            detectable = []
            for strid_id in strid_ids:
                signal = self.active_signals.get(strid_id)
                if not signal or signal.is_expired:
                    continue
                if signal.frequency < min_detectable:
                    continue
                # Don't hear your own stridulations
                if signal.emitter == listener_id:
                    continue
                detectable.append(signal)

            # Sort by frequency (loudest/highest quality first)
            detectable.sort(key=lambda s: s.frequency, reverse=True)
            return detectable

    def respond(self, listener_id: str, strid_id: str) -> bool:
        """Record that an agent responded to a stridulation.

        When an agent is recruited to help, we track it for the
        emitter and for colony statistics.

        Args:
            listener_id: Agent responding
            strid_id: Stridulation signal being responded to

        Returns:
            True if response recorded, False if signal not found
        """
        with self._lock:
            signal = self.active_signals.get(strid_id)
            if not signal:
                return False

            if listener_id not in signal.receivers:
                signal.receivers.append(listener_id)
                self.total_recruitments += 1
            return True

    def get_recruitment_targets(self, n: int = 5) -> List[str]:
        """Get the top contexts (topics/areas) that need recruitment.

        These are the areas with the strongest stridulation activity,
        indicating high-quality resources that need more processing power.

        Args:
            n: Maximum number of targets

        Returns:
            List of context strings, sorted by total vibration intensity
        """
        with self._lock:
            context_scores: Dict[str, float] = {}

            for signal in self.active_signals.values():
                if signal.is_expired:
                    continue
                context_scores[signal.context] = (
                    context_scores.get(signal.context, 0.0) + signal.frequency
                )

            sorted_contexts = sorted(
                context_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            return [ctx for ctx, _ in sorted_contexts[:n]]

    def decay(self) -> int:
        """Apply decay to all stridulation signals. Remove expired ones.

        Returns:
            Number of signals that expired
        """
        with self._lock:
            expired = []
            for strid_id, signal in self.active_signals.items():
                signal.decay()
                if signal.is_expired:
                    expired.append(strid_id)

            for strid_id in expired:
                self._remove_signal_locked(strid_id)

            return len(expired)

    def get_stats(self) -> dict:
        """Get stridulation protocol statistics.

        Returns:
            Dictionary with protocol metrics
        """
        with self._lock:
            frequencies = [s.frequency for s in self.active_signals.values()]

            return {
                "active_signals": len(self.active_signals),
                "total_stridulations": self.total_stridulations,
                "total_recruitments": self.total_recruitments,
                "active_contexts": len(self._by_context),
                "avg_frequency": sum(frequencies) / len(frequencies) if frequencies else 0.0,
                "max_frequency": max(frequencies) if frequencies else 0.0,
                "recruitment_rate": (
                    self.total_recruitments / max(self.total_stridulations, 1)
                ),
            }

    def clear(self):
        """Clear all active stridulation signals."""
        with self._lock:
            self.active_signals.clear()
            self._by_context.clear()
            self._by_emitter.clear()
