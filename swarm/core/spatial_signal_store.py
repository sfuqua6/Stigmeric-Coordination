"""Spatial signal store with locality constraints for true swarm behavior.

This replaces the global SignalStore with a spatial grid where:
- Signals have positions in 2D space
- Agents can only see nearby signals (radius-based)
- Agents can follow gradients toward stronger signals
- Emergent clustering and consensus formation are measurable
"""

import time
import random
import asyncio
import math
from threading import Lock
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, replace

# Import base Signal class
from .signal_store import Signal


@dataclass
class SpatialSignal:
    """Signal with position in 2D space.

    Separate from Signal to avoid dataclass inheritance issues.
    """
    id: str
    type: str
    content: str
    strength: float
    timestamp: float
    depositor: str
    position: Tuple[float, float] = (0.0, 0.0)
    parent: Optional[str] = None
    visits: int = 0
    metadata: dict = field(default_factory=dict)
    responses: List[str] = field(default_factory=list)
    is_response_to: Optional[str] = None

    def __post_init__(self):
        """Ensure fields are initialized correctly."""
        self.strength = max(0.0, min(1.0, self.strength))
        if self.metadata is None:
            self.metadata = {}
        if self.responses is None:
            self.responses = []


class SpatialSignalStore:
    """Spatial signal store with locality constraints.

    Key differences from SignalStore:
    - Signals have (x, y) positions in 2D space
    - Agents can only access nearby signals (get_nearby with radius)
    - No get_all_signals() - forces local information only
    - Gradient following for agent navigation
    - Spatial clustering detection
    """

    def __init__(self,
                 dimensions: int = 100,
                 decay_rate: float = 0.05,
                 prune_threshold: float = 0.1,
                 diversity_threshold: float = 0.85):
        """Initialize spatial signal store.

        Args:
            dimensions: Size of 2D grid (dimensions × dimensions)
            decay_rate: Per-iteration strength reduction (0.0-1.0)
            prune_threshold: Remove signals below this strength
            diversity_threshold: Similarity threshold for duplicate detection
        """
        self._lock = Lock()
        self.dimensions = dimensions
        self.decay_rate = decay_rate
        self.prune_threshold = prune_threshold
        self.diversity_threshold = diversity_threshold

        # Spatial grid: position → list of signals
        # Using discrete grid cells for efficient spatial queries
        self.grid: Dict[Tuple[int, int], List[SpatialSignal]] = {}

        # Signal lookup by ID
        self.signals: Dict[str, SpatialSignal] = {}

        self._next_id = 0

        # Event-driven coordination
        self._signal_events: Dict[str, asyncio.Event] = {}
        self._new_signal_event = asyncio.Event()

    def _grid_cell(self, position: Tuple[float, float]) -> Tuple[int, int]:
        """Convert continuous position to discrete grid cell.

        Args:
            position: (x, y) continuous coordinates

        Returns:
            (grid_x, grid_y) discrete cell coordinates
        """
        x, y = position
        # Clamp to grid boundaries
        x = max(0, min(self.dimensions - 1, x))
        y = max(0, min(self.dimensions - 1, y))
        return (int(x), int(y))

    def _distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two positions.

        Args:
            pos1: First position (x, y)
            pos2: Second position (x, y)

        Returns:
            Euclidean distance
        """
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        return math.sqrt(dx * dx + dy * dy)

    def deposit(self,
                signal_type: str,
                content: str,
                strength: float,
                depositor: str,
                position: Tuple[float, float],
                parent: Optional[str] = None,
                metadata: Optional[dict] = None) -> Optional[str]:
        """Deposit signal at specific position.

        Args:
            signal_type: Type of signal
            content: Signal content
            strength: Initial strength (0.0-1.0)
            depositor: Agent ID
            position: (x, y) position in space
            parent: Optional parent signal ID
            metadata: Optional metadata

        Returns:
            Signal ID if deposited, None if rejected
        """
        with self._lock:
            signal_id = f"{signal_type}_{self._next_id:04d}"
            self._next_id += 1

            signal = SpatialSignal(
                id=signal_id,
                type=signal_type,
                content=content,
                strength=strength,
                timestamp=time.time(),
                depositor=depositor,
                parent=parent,
                metadata=metadata or {},
                position=position
            )

            # Add to grid
            cell = self._grid_cell(position)
            if cell not in self.grid:
                self.grid[cell] = []
            self.grid[cell].append(signal)

            # Add to lookup
            self.signals[signal_id] = signal

            # EVENT-DRIVEN: Notify waiting agents
            if signal_type not in self._signal_events:
                self._signal_events[signal_type] = asyncio.Event()
            self._signal_events[signal_type].set()
            self._new_signal_event.set()

            return signal_id

    def get_nearby(self,
                   position: Tuple[float, float],
                   radius: float,
                   signal_type: Optional[str] = None) -> List[SpatialSignal]:
        """Get signals within radius of position (LOCAL ACCESS ONLY).

        This is the key method that enforces locality constraints.
        Agents can ONLY see nearby signals, not all signals.

        Args:
            position: Center position (x, y)
            radius: Search radius
            signal_type: Optional filter by signal type

        Returns:
            List of nearby signals sorted by distance (closest first)
        """
        with self._lock:
            nearby = []

            # Calculate bounding box of grid cells to check
            min_x = max(0, int(position[0] - radius))
            max_x = min(self.dimensions - 1, int(position[0] + radius))
            min_y = max(0, int(position[1] - radius))
            max_y = min(self.dimensions - 1, int(position[1] + radius))

            # Check all cells in bounding box
            for grid_x in range(min_x, max_x + 1):
                for grid_y in range(min_y, max_y + 1):
                    cell = (grid_x, grid_y)
                    if cell not in self.grid:
                        continue

                    for signal in self.grid[cell]:
                        # Check actual distance
                        dist = self._distance(position, signal.position)
                        if dist <= radius:
                            # Filter by type if specified
                            if signal_type is None or signal.type == signal_type:
                                nearby.append((dist, signal))

            # Sort by distance (closest first)
            nearby.sort(key=lambda x: x[0])
            return [signal for _, signal in nearby]

    def get_gradient(self,
                    position: Tuple[float, float],
                    radius: float,
                    signal_type: Optional[str] = None) -> Tuple[float, float]:
        """Calculate strength gradient at position for agent navigation.

        Agents can follow gradients to move toward stronger signals.

        Args:
            position: Current position (x, y)
            radius: Search radius for gradient calculation
            signal_type: Optional filter by signal type

        Returns:
            Gradient vector (dx, dy) pointing toward higher strength
        """
        nearby = self.get_nearby(position, radius, signal_type)

        if not nearby:
            return (0.0, 0.0)

        # Calculate weighted centroid of nearby signals
        total_strength = 0.0
        weighted_x = 0.0
        weighted_y = 0.0

        for signal in nearby:
            total_strength += signal.strength
            weighted_x += signal.position[0] * signal.strength
            weighted_y += signal.position[1] * signal.strength

        if total_strength == 0:
            return (0.0, 0.0)

        # Centroid position
        centroid_x = weighted_x / total_strength
        centroid_y = weighted_y / total_strength

        # Gradient points from current position toward centroid
        dx = centroid_x - position[0]
        dy = centroid_y - position[1]

        # Normalize to unit vector
        magnitude = math.sqrt(dx * dx + dy * dy)
        if magnitude > 0:
            dx /= magnitude
            dy /= magnitude

        return (dx, dy)

    def detect_spatial_clusters(self,
                                signal_type: Optional[str] = None,
                                cluster_radius: float = 10.0) -> List[List[SpatialSignal]]:
        """Detect spatial clusters of signals for emergence analysis.

        Args:
            signal_type: Optional filter by signal type
            cluster_radius: Radius for clustering

        Returns:
            List of clusters (each cluster is a list of signals)
        """
        with self._lock:
            # Get all signals of type
            signals = list(self.signals.values())
            if signal_type:
                signals = [s for s in signals if s.type == signal_type]

            if not signals:
                return []

            # Simple clustering: group signals within cluster_radius
            visited = set()
            clusters = []

            for signal in signals:
                if signal.id in visited:
                    continue

                # Start new cluster
                cluster = [signal]
                visited.add(signal.id)

                # Find nearby signals
                nearby = self.get_nearby(signal.position, cluster_radius, signal_type)
                for nearby_signal in nearby:
                    if nearby_signal.id not in visited:
                        cluster.append(nearby_signal)
                        visited.add(nearby_signal.id)

                clusters.append(cluster)

            return clusters

    def get_signal(self, signal_id: str) -> Optional[SpatialSignal]:
        """Get signal by ID.

        Args:
            signal_id: Signal ID

        Returns:
            Signal or None if not found
        """
        with self._lock:
            return self.signals.get(signal_id)

    def decay_all(self, contrarian_types: Optional[List[str]] = None, contrarian_boost: float = 1.10) -> int:
        """Apply decay to all signals with optional contrarian boost.

        Args:
            contrarian_types: Optional list of signal types to receive anti-decay boost
            contrarian_boost: Multiplier for contrarian signals (default: 1.10 = 10% boost)

        Returns:
            Number of signals remaining
        """
        with self._lock:
            for signal in self.signals.values():
                # Apply decay
                signal.strength *= (1.0 - self.decay_rate)

                # Apply contrarian boost to offset echo effects
                if contrarian_types and signal.type in contrarian_types:
                    signal.strength = min(1.0, signal.strength * contrarian_boost)

            return len(self.signals)

    def prune_weak(self) -> int:
        """Remove signals below pruning threshold.

        Returns:
            Number of signals pruned
        """
        with self._lock:
            to_remove = []

            for signal_id, signal in self.signals.items():
                if signal.strength < self.prune_threshold:
                    to_remove.append(signal_id)

            # Remove from both grid and lookup
            for signal_id in to_remove:
                signal = self.signals[signal_id]
                cell = self._grid_cell(signal.position)

                # Remove from grid
                if cell in self.grid:
                    self.grid[cell] = [s for s in self.grid[cell] if s.id != signal_id]
                    if not self.grid[cell]:
                        del self.grid[cell]

                # Remove from lookup
                del self.signals[signal_id]

            return len(to_remove)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about spatial signal store.

        Returns:
            Dictionary with stats including spatial metrics
        """
        with self._lock:
            by_type = {}
            for signal in self.signals.values():
                by_type[signal.type] = by_type.get(signal.type, 0) + 1

            strengths = [s.strength for s in self.signals.values()]
            avg_strength = sum(strengths) / len(strengths) if strengths else 0.0

            # Spatial metrics
            clusters = self.detect_spatial_clusters()
            avg_cluster_size = sum(len(c) for c in clusters) / len(clusters) if clusters else 0.0

            return {
                "total_signals": len(self.signals),
                "by_type": by_type,
                "avg_strength": avg_strength,
                "strongest": max(strengths) if strengths else 0.0,
                "num_clusters": len(clusters),
                "avg_cluster_size": avg_cluster_size,
                "grid_cells_used": len(self.grid),
            }

    def get_density_map(self, cell_size: int = 10) -> Dict[Tuple[int, int], int]:
        """Get density map for visualization.

        Args:
            cell_size: Size of density grid cells

        Returns:
            Dictionary mapping (grid_x, grid_y) to signal count
        """
        with self._lock:
            density = {}

            for signal in self.signals.values():
                # Map to coarse grid
                grid_x = int(signal.position[0] / cell_size)
                grid_y = int(signal.position[1] / cell_size)
                cell = (grid_x, grid_y)

                density[cell] = density.get(cell, 0) + 1

            return density

    async def wait_for_signal(self, signal_type: str, timeout: float = None) -> bool:
        """Wait for a signal of specified type to be deposited.

        Args:
            signal_type: Signal type to wait for
            timeout: Optional timeout in seconds

        Returns:
            True if signal was deposited, False if timeout
        """
        if signal_type not in self._signal_events:
            self._signal_events[signal_type] = asyncio.Event()

        event = self._signal_events[signal_type]

        try:
            if timeout:
                await asyncio.wait_for(event.wait(), timeout=timeout)
            else:
                await event.wait()
            return True
        except asyncio.TimeoutError:
            return False

    async def wait_for_signals(self, signal_types: List[str], timeout: float = None) -> Optional[str]:
        """Wait for ANY of the specified signal types to be deposited.

        Useful for event-driven agents that react to multiple signal types.

        Args:
            signal_types: List of signal types to wait for
            timeout: Optional timeout in seconds

        Returns:
            Signal type that was deposited, or None if timeout
        """
        # Create events for all types
        for signal_type in signal_types:
            if signal_type not in self._signal_events:
                self._signal_events[signal_type] = asyncio.Event()

        # Create tasks for each event wait
        tasks = {
            asyncio.create_task(self._signal_events[sig_type].wait()): sig_type
            for sig_type in signal_types
        }

        try:
            # Wait for first event to trigger
            if timeout:
                done, pending = await asyncio.wait(
                    tasks.keys(),
                    timeout=timeout,
                    return_when=asyncio.FIRST_COMPLETED
                )
            else:
                done, pending = await asyncio.wait(
                    tasks.keys(),
                    return_when=asyncio.FIRST_COMPLETED
                )

            # Cancel remaining tasks
            for task in pending:
                task.cancel()

            # Return which signal type triggered
            if done:
                triggered_task = list(done)[0]
                return tasks[triggered_task]
            else:
                # Timeout
                for task in tasks.keys():
                    task.cancel()
                return None

        except asyncio.TimeoutError:
            # Cancel all tasks
            for task in tasks.keys():
                task.cancel()
            return None

    def clear_signal_event(self, signal_type: str):
        """Clear event after agents have been notified.

        Args:
            signal_type: Signal type to clear event for
        """
        if signal_type in self._signal_events:
            self._signal_events[signal_type].clear()

    def has_signals(self, signal_type: str) -> bool:
        """Check if any signals of type exist.

        Args:
            signal_type: Signal type to check

        Returns:
            True if signals exist
        """
        with self._lock:
            return any(s.type == signal_type for s in self.signals.values())

    # Backward compatibility methods for existing agents
    def get_all_signals(self) -> List[SpatialSignal]:
        """Get all signals (backward compatibility with SignalStore).

        Returns:
            List of all signals
        """
        with self._lock:
            return list(self.signals.values())

    def sample_weighted(self, signal_type: str, n: int = 5) -> List[SpatialSignal]:
        """Sample signals weighted by strength (backward compatibility).

        Args:
            signal_type: Type of signals to sample
            n: Number of signals to sample

        Returns:
            List of sampled signals
        """
        with self._lock:
            type_signals = [s for s in self.signals.values() if s.type == signal_type]
            if not type_signals:
                return []

            # Weight by strength
            weights = [s.strength for s in type_signals]
            total_weight = sum(weights)
            if total_weight == 0:
                return type_signals[:n]

            # Sample with replacement
            import random
            sampled = random.choices(type_signals, weights=weights, k=min(n, len(type_signals)))
            return sampled

    def get_top_signals(self, signal_type: str, n: int = 5) -> List[SpatialSignal]:
        """Get top N signals by strength (backward compatibility).

        Args:
            signal_type: Type of signals
            n: Number to return

        Returns:
            Top N signals sorted by strength
        """
        with self._lock:
            type_signals = [s for s in self.signals.values() if s.type == signal_type]
            type_signals.sort(key=lambda s: s.strength, reverse=True)
            return type_signals[:n]
