"""Pheromone system for stigmergic coordination.

Leaf cutter ants use a dual-pheromone system:
1. TRAIL pheromones (persistent, low evaporation) - mark successful paths for
   long-term orientation. In Atta species, just 0.4 picograms per meter is enough
   to induce trail-following behavior. 1mg could lay a trail 60x around the planet.

2. RECRUITMENT pheromones (volatile, high evaporation) - attract nearby agents to
   productive areas. These travel farther but decay quickly, pulling workers from
   distant parts of the colony toward newly discovered resources.

3. ALARM pheromones (medium evaporation) - signal quality issues, contradictions,
   or threats (like phorid fly attacks). Trigger defensive responses from nearby agents.

The concentration gradient across the pheromone field guides agent behavior:
agents follow increasing concentration toward productive areas, and the natural
evaporation prevents stale trails from misleading the colony.
"""

import time
import random
from enum import Enum
from threading import Lock
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


class PheromoneType(Enum):
    """Types of pheromones, each with distinct evaporation dynamics.

    Modeled after Atta leaf cutter chemical communication:
    - TRAIL: Non-volatile, long-lasting orientation cues (from poison gland)
    - RECRUITMENT: Volatile, short-lived attraction signals
    - ALARM: Medium volatility, danger/quality issue signals
    """
    TRAIL = "trail"
    RECRUITMENT = "recruitment"
    ALARM = "alarm"


# Evaporation rates per tick (fraction of concentration lost)
# Higher = more volatile = shorter persistence
DEFAULT_EVAPORATION_RATES = {
    PheromoneType.TRAIL: 0.02,        # Persistent: 2% per tick (~50 ticks half-life)
    PheromoneType.RECRUITMENT: 0.15,  # Volatile: 15% per tick (~4 ticks half-life)
    PheromoneType.ALARM: 0.08,        # Medium: 8% per tick (~8 ticks half-life)
}

# Minimum concentration before pheromone is removed
EVAPORATION_THRESHOLD = 0.01


@dataclass
class Pheromone:
    """A single pheromone deposit in the environment.

    Like real ant pheromones, each deposit has a type (determines evaporation rate),
    a concentration (signal strength), and a context (what it marks).
    """
    id: str
    pheromone_type: PheromoneType
    context: str                     # What this pheromone marks (signal_id, topic, path)
    concentration: float             # 0.0-1.0 (current strength)
    depositor: str                   # Agent ID who deposited
    timestamp: float                 # When deposited
    evaporation_rate: float          # Per-tick concentration reduction
    target: Optional[str] = None     # Target signal/topic (for directed pheromones)
    metadata: dict = field(default_factory=dict)

    def evaporate(self) -> float:
        """Apply one tick of evaporation. Returns new concentration."""
        self.concentration *= (1.0 - self.evaporation_rate)
        return self.concentration

    @property
    def is_expired(self) -> bool:
        """Check if pheromone has evaporated below threshold."""
        return self.concentration < EVAPORATION_THRESHOLD


class PheromoneField:
    """The shared chemical landscape where agents deposit and sense pheromones.

    This is the core stigmergic medium: agents modify the environment (deposit
    pheromones) and other agents respond to those modifications (follow gradients).
    No direct communication between agents is needed.

    In Atta colonies, the olfactory pathway begins with antennae where receptors
    detect chemicals. Atta ants have up to 459 glomeruli in each antennal lobe
    (vs 43 in fruit flies), giving them extraordinary chemical sensitivity.
    """

    def __init__(self, evaporation_rates: Optional[Dict[PheromoneType, float]] = None):
        """Initialize pheromone field.

        Args:
            evaporation_rates: Custom evaporation rates per type. If None, uses defaults.
        """
        self._lock = Lock()
        self.pheromones: Dict[str, Pheromone] = {}
        self._next_id = 0
        self.evaporation_rates = evaporation_rates or dict(DEFAULT_EVAPORATION_RATES)

        # Indexes for fast lookup
        self._by_type: Dict[PheromoneType, set] = {t: set() for t in PheromoneType}
        self._by_context: Dict[str, set] = {}  # context -> {pheromone_ids}

        # Statistics
        self.total_deposited = 0
        self.total_evaporated = 0

    def deposit(self, pheromone_type: PheromoneType, context: str,
                concentration: float, depositor: str,
                target: Optional[str] = None,
                metadata: Optional[dict] = None) -> str:
        """Deposit a pheromone in the environment.

        If a pheromone of the same type and context already exists from the same
        depositor, reinforce it (add concentration) instead of creating a new one.
        This models how ants reinforce trails by repeatedly depositing on the same path.

        Args:
            pheromone_type: Type of pheromone (TRAIL, RECRUITMENT, ALARM)
            context: What this pheromone marks (signal_id, topic key, path)
            concentration: Initial concentration (0.0-1.0)
            depositor: Agent ID depositing
            target: Optional target for directed pheromones
            metadata: Optional additional data

        Returns:
            Pheromone ID (new or reinforced)
        """
        concentration = max(0.0, min(1.0, concentration))

        with self._lock:
            # Check for existing pheromone to reinforce
            existing = self._find_existing(pheromone_type, context, depositor)
            if existing:
                # Reinforce: add concentration (capped at 1.0)
                existing.concentration = min(1.0, existing.concentration + concentration * 0.5)
                existing.timestamp = time.time()
                return existing.id

            # Create new pheromone
            phero_id = f"P_{pheromone_type.value}_{self._next_id:04d}"
            self._next_id += 1

            evap_rate = self.evaporation_rates.get(
                pheromone_type, DEFAULT_EVAPORATION_RATES[pheromone_type]
            )

            pheromone = Pheromone(
                id=phero_id,
                pheromone_type=pheromone_type,
                context=context,
                concentration=concentration,
                depositor=depositor,
                timestamp=time.time(),
                evaporation_rate=evap_rate,
                target=target,
                metadata=metadata or {}
            )

            self.pheromones[phero_id] = pheromone

            # Update indexes
            self._by_type[pheromone_type].add(phero_id)
            if context not in self._by_context:
                self._by_context[context] = set()
            self._by_context[context].add(phero_id)

            self.total_deposited += 1
            return phero_id

    def _find_existing(self, pheromone_type: PheromoneType,
                       context: str, depositor: str) -> Optional[Pheromone]:
        """Find existing pheromone for reinforcement (lock must be held).

        Args:
            pheromone_type: Type to match
            context: Context to match
            depositor: Depositor to match

        Returns:
            Existing pheromone if found, None otherwise
        """
        context_pheros = self._by_context.get(context, set())
        type_pheros = self._by_type.get(pheromone_type, set())
        candidates = context_pheros & type_pheros

        for pid in candidates:
            p = self.pheromones.get(pid)
            if p and p.depositor == depositor:
                return p
        return None

    def sense(self, context: str,
              pheromone_type: Optional[PheromoneType] = None,
              sensitivity: float = 1.0) -> List[Pheromone]:
        """Sense pheromones at a given context location.

        Models the ant's antennal detection: sensitivity affects the minimum
        concentration detectable (lower sensitivity = higher threshold).

        Args:
            context: Context to sense at (signal_id, topic)
            pheromone_type: Optional filter by type
            sensitivity: Detection sensitivity 0.0-1.0 (1.0 = detect everything)

        Returns:
            List of detectable pheromones, sorted by concentration (strongest first)
        """
        min_detectable = EVAPORATION_THRESHOLD / max(sensitivity, 0.01)

        with self._lock:
            context_pheros = self._by_context.get(context, set())
            results = []

            for pid in context_pheros:
                p = self.pheromones.get(pid)
                if not p or p.is_expired:
                    continue
                if p.concentration < min_detectable:
                    continue
                if pheromone_type and p.pheromone_type != pheromone_type:
                    continue
                results.append(p)

            results.sort(key=lambda p: p.concentration, reverse=True)
            return results

    def get_concentration(self, context: str,
                         pheromone_type: PheromoneType) -> float:
        """Get total pheromone concentration at a context for a given type.

        Multiple deposits from different agents accumulate, modeling how
        busy trails have stronger pheromone concentrations.

        Args:
            context: Context to measure
            pheromone_type: Type to measure

        Returns:
            Total concentration (can exceed 1.0 with multiple deposits)
        """
        with self._lock:
            context_pheros = self._by_context.get(context, set())
            type_pheros = self._by_type.get(pheromone_type, set())
            candidates = context_pheros & type_pheros

            total = 0.0
            for pid in candidates:
                p = self.pheromones.get(pid)
                if p and not p.is_expired:
                    total += p.concentration
            return total

    def sample_gradient(self, contexts: List[str],
                       pheromone_type: PheromoneType,
                       sensitivity: float = 1.0) -> List[Tuple[str, float]]:
        """Sample pheromone gradient across multiple contexts.

        Returns contexts ranked by pheromone concentration, modeling how
        ants follow concentration gradients to find trails and resources.

        Args:
            contexts: List of contexts to sample
            pheromone_type: Type of pheromone to follow
            sensitivity: Agent's detection sensitivity

        Returns:
            List of (context, concentration) tuples, sorted by concentration descending
        """
        gradients = []
        min_detectable = EVAPORATION_THRESHOLD / max(sensitivity, 0.01)

        with self._lock:
            for ctx in contexts:
                ctx_pheros = self._by_context.get(ctx, set())
                type_pheros = self._by_type.get(pheromone_type, set())
                candidates = ctx_pheros & type_pheros

                total = 0.0
                for pid in candidates:
                    p = self.pheromones.get(pid)
                    if p and p.concentration >= min_detectable:
                        total += p.concentration

                if total > 0:
                    gradients.append((ctx, total))

        gradients.sort(key=lambda x: x[1], reverse=True)
        return gradients

    def follow_gradient(self, contexts: List[str],
                       pheromone_type: PheromoneType,
                       sensitivity: float = 1.0,
                       exploration_rate: float = 0.1) -> Optional[str]:
        """Follow pheromone gradient with stochastic exploration.

        Models ant trail-following: usually follows strongest concentration,
        but with some probability explores weaker trails. This prevents the
        colony from getting stuck on suboptimal paths.

        Args:
            contexts: Available contexts to choose from
            pheromone_type: Type to follow
            sensitivity: Detection sensitivity
            exploration_rate: Probability of random exploration (0.0-1.0)

        Returns:
            Selected context, or None if no pheromones detected
        """
        # Random exploration
        if random.random() < exploration_rate and contexts:
            return random.choice(contexts)

        gradients = self.sample_gradient(contexts, pheromone_type, sensitivity)
        if not gradients:
            return None

        # Probabilistic selection weighted by concentration
        total = sum(c for _, c in gradients)
        if total == 0:
            return None

        r = random.random() * total
        cumulative = 0.0
        for ctx, concentration in gradients:
            cumulative += concentration
            if cumulative >= r:
                return ctx

        return gradients[0][0]  # Fallback to strongest

    def evaporate(self) -> int:
        """Apply one tick of evaporation to all pheromones.

        Removes pheromones that fall below the threshold.
        Models natural chemical degradation that ensures stale signals
        don't mislead the colony.

        Returns:
            Number of pheromones that evaporated completely
        """
        with self._lock:
            expired = []
            for pid, p in self.pheromones.items():
                p.evaporate()
                if p.is_expired:
                    expired.append(pid)

            for pid in expired:
                self._remove_locked(pid)

            self.total_evaporated += len(expired)
            return len(expired)

    def _remove_locked(self, phero_id: str):
        """Remove pheromone from all indexes (lock must be held)."""
        p = self.pheromones.get(phero_id)
        if not p:
            return

        # Remove from indexes
        self._by_type[p.pheromone_type].discard(phero_id)
        ctx_set = self._by_context.get(p.context)
        if ctx_set:
            ctx_set.discard(phero_id)
            if not ctx_set:
                del self._by_context[p.context]

        del self.pheromones[phero_id]

    def get_all_of_type(self, pheromone_type: PheromoneType) -> List[Pheromone]:
        """Get all active pheromones of a given type.

        Args:
            pheromone_type: Type to retrieve

        Returns:
            List of pheromones sorted by concentration (strongest first)
        """
        with self._lock:
            phero_ids = self._by_type.get(pheromone_type, set())
            result = [self.pheromones[pid] for pid in phero_ids
                      if pid in self.pheromones and not self.pheromones[pid].is_expired]
            result.sort(key=lambda p: p.concentration, reverse=True)
            return result

    def get_stats(self) -> dict:
        """Get statistics about the pheromone field.

        Returns:
            Dictionary with field statistics
        """
        with self._lock:
            by_type = {}
            for ptype in PheromoneType:
                active = [self.pheromones[pid] for pid in self._by_type[ptype]
                          if pid in self.pheromones]
                if active:
                    concentrations = [p.concentration for p in active]
                    by_type[ptype.value] = {
                        "count": len(active),
                        "avg_concentration": sum(concentrations) / len(concentrations),
                        "max_concentration": max(concentrations),
                    }
                else:
                    by_type[ptype.value] = {"count": 0, "avg_concentration": 0.0, "max_concentration": 0.0}

            return {
                "total_active": len(self.pheromones),
                "total_deposited": self.total_deposited,
                "total_evaporated": self.total_evaporated,
                "by_type": by_type,
                "unique_contexts": len(self._by_context),
            }

    def clear(self):
        """Clear all pheromones from the field."""
        with self._lock:
            self.pheromones.clear()
            for ptype in PheromoneType:
                self._by_type[ptype].clear()
            self._by_context.clear()
