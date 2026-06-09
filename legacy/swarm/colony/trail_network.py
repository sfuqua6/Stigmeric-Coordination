"""Trail network - emergent pathways connecting productive knowledge areas.

Of all animals that construct trails (elephants, cattle, wolves), Atta leaf cutter
ants have the most complex trail construction behavior. Workers remove vast amounts
of vegetation, cut passes through obstacles, and level soil to create smooth surfaces.
Colonies clear nearly 3 km of trail per year.

Trails serve two functions:
1. ORIENTATION: Persistent chemical marks guide ants to and from harvesting sites
2. RECRUITMENT: Volatile signals attract new foragers to productive trails

Trail dynamics:
- New trails start weak (single scout found something interesting)
- Successful foragers reinforce trails by depositing pheromones on return
- Well-used trails become "highways" (10 lanes wide in real Atta colonies)
- Unused trails evaporate as pheromones degrade
- Traffic creates positive feedback: busy trails attract more traffic

In our system, trails connect knowledge topics/clusters. When agents find productive
connections between topics, they reinforce the trail. Other agents follow strong
trails to find related productive areas. Trails that stop being useful evaporate.
"""

import time
import random
from threading import Lock
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field


@dataclass
class TrailSegment:
    """A single segment connecting two knowledge nodes.

    Each segment tracks pheromone strength and traffic, modeling how
    real ant trails gain strength through repeated use.
    """
    id: str
    source: str                    # Source node (topic, signal_id, cluster)
    destination: str               # Destination node
    strength: float                # Pheromone concentration (0.0-1.0)
    traffic: int                   # Number of agents that used this trail
    created_at: float
    last_reinforced: float         # Last time an agent deposited on this trail
    reinforcers: Set[str] = field(default_factory=set)  # Agents who reinforced

    @property
    def age(self) -> float:
        """Age in seconds since creation."""
        return time.time() - self.created_at

    @property
    def idle_time(self) -> float:
        """Time since last reinforcement."""
        return time.time() - self.last_reinforced

    @property
    def is_highway(self) -> bool:
        """Check if this trail qualifies as a 'highway' (heavily used).

        In real Atta colonies, main foraging highways can be 10 lanes wide
        and persist for months. We define a highway as having significant
        traffic and strength.
        """
        return self.traffic >= 5 and self.strength >= 0.5


# Trail evaporation rate
TRAIL_EVAPORATION_RATE = 0.04      # 4% per tick
TRAIL_MIN_STRENGTH = 0.02          # Below this, trail is removed
TRAIL_REINFORCEMENT_AMOUNT = 0.15  # Strength added per reinforcement
TRAIL_MAX_STRENGTH = 2.0           # Allow accumulation beyond 1.0 for highways


class TrailNetwork:
    """Network of trails connecting productive knowledge areas.

    Trails emerge from agent behavior - no central planning. When an agent
    successfully connects two knowledge areas, it reinforces the trail between
    them. Other agents sense these trails and follow them, further reinforcing
    productive connections while unused trails fade away.

    This creates efficient routing without any agent knowing the full network.
    """

    def __init__(self, evaporation_rate: float = TRAIL_EVAPORATION_RATE,
                 reinforcement_amount: float = TRAIL_REINFORCEMENT_AMOUNT):
        """Initialize trail network.

        Args:
            evaporation_rate: Strength reduction per tick (0.0-1.0)
            reinforcement_amount: Strength added per reinforcement
        """
        self._lock = Lock()
        self.segments: Dict[str, TrailSegment] = {}
        self.evaporation_rate = evaporation_rate
        self.reinforcement_amount = reinforcement_amount
        self._next_id = 0

        # Indexes for fast lookup
        self._outgoing: Dict[str, set] = {}    # node -> {segment_ids going out}
        self._incoming: Dict[str, set] = {}    # node -> {segment_ids coming in}
        self._edge_index: Dict[Tuple[str, str], str] = {}  # (src, dst) -> segment_id

    def _make_edge_key(self, source: str, destination: str) -> Tuple[str, str]:
        """Create canonical edge key (treats trails as bidirectional)."""
        return (min(source, destination), max(source, destination))

    def reinforce(self, source: str, destination: str,
                  depositor: str, amount: Optional[float] = None) -> str:
        """Reinforce (or create) a trail between two nodes.

        When an agent finds a productive connection between topics,
        it deposits pheromone on the trail. If the trail doesn't exist,
        a new one is created.

        Args:
            source: Source node
            destination: Destination node
            depositor: Agent reinforcing the trail
            amount: Reinforcement amount (default: self.reinforcement_amount)

        Returns:
            Trail segment ID
        """
        if source == destination:
            return ""  # No self-loops

        amount = amount or self.reinforcement_amount
        edge_key = self._make_edge_key(source, destination)

        with self._lock:
            # Check if trail already exists
            if edge_key in self._edge_index:
                seg_id = self._edge_index[edge_key]
                seg = self.segments[seg_id]
                seg.strength = min(TRAIL_MAX_STRENGTH, seg.strength + amount)
                seg.traffic += 1
                seg.last_reinforced = time.time()
                seg.reinforcers.add(depositor)
                return seg_id

            # Create new trail
            seg_id = f"TRAIL_{self._next_id:04d}"
            self._next_id += 1

            segment = TrailSegment(
                id=seg_id,
                source=source,
                destination=destination,
                strength=amount,
                traffic=1,
                created_at=time.time(),
                last_reinforced=time.time(),
                reinforcers={depositor},
            )

            self.segments[seg_id] = segment
            self._edge_index[edge_key] = seg_id

            # Update indexes
            for node in (source, destination):
                if node not in self._outgoing:
                    self._outgoing[node] = set()
                if node not in self._incoming:
                    self._incoming[node] = set()

            self._outgoing[source].add(seg_id)
            self._incoming[destination].add(seg_id)
            # Bidirectional
            self._outgoing[destination].add(seg_id)
            self._incoming[source].add(seg_id)

            return seg_id

    def follow(self, from_node: str,
               exploration_rate: float = 0.1) -> Optional[str]:
        """Follow a trail from a node, probabilistically choosing next destination.

        Models ant trail-following behavior: usually follows the strongest
        pheromone concentration, but occasionally explores weaker trails.

        Args:
            from_node: Current node
            exploration_rate: Probability of random exploration

        Returns:
            Destination node, or None if no trails from this node
        """
        with self._lock:
            outgoing_ids = self._outgoing.get(from_node, set())
            if not outgoing_ids:
                return None

            segments = [self.segments[sid] for sid in outgoing_ids
                       if sid in self.segments]
            if not segments:
                return None

            # Random exploration
            if random.random() < exploration_rate:
                seg = random.choice(segments)
                return seg.destination if seg.source == from_node else seg.source

            # Probabilistic selection weighted by strength
            total_strength = sum(s.strength for s in segments)
            if total_strength == 0:
                return None

            r = random.random() * total_strength
            cumulative = 0.0
            for seg in segments:
                cumulative += seg.strength
                if cumulative >= r:
                    return seg.destination if seg.source == from_node else seg.source

            # Fallback
            strongest = max(segments, key=lambda s: s.strength)
            return strongest.destination if strongest.source == from_node else strongest.source

    def get_neighbors(self, node: str) -> List[Tuple[str, float]]:
        """Get all connected nodes with trail strength.

        Args:
            node: Node to get neighbors for

        Returns:
            List of (neighbor_node, trail_strength) sorted by strength
        """
        with self._lock:
            outgoing_ids = self._outgoing.get(node, set())
            neighbors = []

            for sid in outgoing_ids:
                seg = self.segments.get(sid)
                if seg:
                    neighbor = seg.destination if seg.source == node else seg.source
                    neighbors.append((neighbor, seg.strength))

            neighbors.sort(key=lambda x: x[1], reverse=True)
            return neighbors

    def get_highways(self, min_traffic: int = 5,
                     min_strength: float = 0.5) -> List[TrailSegment]:
        """Get all 'highway' trails (heavily used, strong connections).

        In real Atta colonies, highways are major routes cleared of vegetation.
        In our system, highways represent well-established knowledge connections.

        Args:
            min_traffic: Minimum traffic count
            min_strength: Minimum pheromone strength

        Returns:
            List of highway segments sorted by traffic (busiest first)
        """
        with self._lock:
            highways = [
                seg for seg in self.segments.values()
                if seg.traffic >= min_traffic and seg.strength >= min_strength
            ]
            highways.sort(key=lambda s: s.traffic, reverse=True)
            return highways

    def evaporate(self) -> int:
        """Apply evaporation to all trails. Remove trails below threshold.

        Returns:
            Number of trails that evaporated completely
        """
        with self._lock:
            to_remove = []

            for seg in self.segments.values():
                seg.strength *= (1.0 - self.evaporation_rate)
                if seg.strength < TRAIL_MIN_STRENGTH:
                    to_remove.append(seg.id)

            for seg_id in to_remove:
                self._remove_locked(seg_id)

            return len(to_remove)

    def _remove_locked(self, seg_id: str):
        """Remove a trail segment from all indexes (lock must be held)."""
        seg = self.segments.get(seg_id)
        if not seg:
            return

        edge_key = self._make_edge_key(seg.source, seg.destination)
        self._edge_index.pop(edge_key, None)

        for node in (seg.source, seg.destination):
            if node in self._outgoing:
                self._outgoing[node].discard(seg_id)
            if node in self._incoming:
                self._incoming[node].discard(seg_id)

        del self.segments[seg_id]

    def get_path(self, start: str, end: str, max_hops: int = 10) -> Optional[List[str]]:
        """Find strongest path between two nodes (greedy best-first).

        Not guaranteed to find the optimal path, but models how ants
        navigate by following strongest local gradients.

        Args:
            start: Starting node
            end: Destination node
            max_hops: Maximum path length

        Returns:
            List of nodes in path, or None if no path found
        """
        with self._lock:
            visited = set()
            path = [start]
            current = start

            for _ in range(max_hops):
                if current == end:
                    return path

                visited.add(current)
                outgoing_ids = self._outgoing.get(current, set())

                # Find best unvisited neighbor
                best_seg = None
                best_strength = -1

                for sid in outgoing_ids:
                    seg = self.segments.get(sid)
                    if not seg:
                        continue
                    neighbor = seg.destination if seg.source == current else seg.source
                    if neighbor not in visited and seg.strength > best_strength:
                        best_seg = seg
                        best_strength = seg.strength

                if best_seg is None:
                    return None  # Dead end

                next_node = (best_seg.destination
                            if best_seg.source == current
                            else best_seg.source)
                path.append(next_node)
                current = next_node

            return None  # Exceeded max hops

    def _calculate_coverage(self) -> float:
        """Calculate coverage score (internal, no lock - caller must hold lock).

        Returns:
            Coverage score 0.0-1.0
        """
        if not self.segments:
            return 0.0

        total_nodes = len(set(self._outgoing.keys()) | set(self._incoming.keys()))
        total_segments = len(self.segments)
        avg_strength = (sum(s.strength for s in self.segments.values())
                      / total_segments if total_segments else 0)
        highway_count = sum(1 for s in self.segments.values() if s.is_highway)

        density = min(1.0, total_segments / max(total_nodes * 2, 1))
        strength_score = min(1.0, avg_strength)
        highway_ratio = highway_count / max(total_segments, 1)

        return 0.4 * density + 0.3 * strength_score + 0.3 * highway_ratio

    def get_coverage(self) -> float:
        """Get trail network coverage metric.

        Measures how well-connected the knowledge space is.

        Returns:
            Coverage score 0.0-1.0 based on trail density and strength
        """
        with self._lock:
            return self._calculate_coverage()

    def get_stats(self) -> dict:
        """Get trail network statistics.

        Returns:
            Dictionary with network metrics
        """
        with self._lock:
            if not self.segments:
                return {
                    "total_trails": 0,
                    "total_nodes": 0,
                    "highways": 0,
                    "coverage": 0.0,
                    "avg_strength": 0.0,
                    "avg_traffic": 0.0,
                }

            strengths = [s.strength for s in self.segments.values()]
            traffics = [s.traffic for s in self.segments.values()]
            total_nodes = len(set(self._outgoing.keys()) | set(self._incoming.keys()))

            return {
                "total_trails": len(self.segments),
                "total_nodes": total_nodes,
                "highways": sum(1 for s in self.segments.values() if s.is_highway),
                "coverage": self._calculate_coverage(),
                "avg_strength": sum(strengths) / len(strengths),
                "avg_traffic": sum(traffics) / len(traffics),
                "max_strength": max(strengths),
                "max_traffic": max(traffics),
            }

    def clear(self):
        """Clear all trails."""
        with self._lock:
            self.segments.clear()
            self._outgoing.clear()
            self._incoming.clear()
            self._edge_index.clear()
