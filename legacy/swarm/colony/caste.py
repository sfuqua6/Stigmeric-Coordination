"""Caste system for leaf cutter ant colony agents.

Atta leaf cutter ant colonies exhibit extreme polymorphism - workers range from
tiny minims (1mm head width) to massive super majors (5mm head width), a 300x
weight difference. This is like one human being 2ft tall and another 18ft tall.

Each caste is optimized for specific tasks through morphological specialization:

- QUEEN: The sole reproductive individual. Lives 10-20 years (longest-lived insects).
  Mates once during nuptial flight with 3-8 males (obligate multiple mating) to
  ensure genetic diversity. Lays eggs with varying nutritional content to control
  which caste develops. In our system: task decomposer, colony lifecycle manager.

- MINIM: Smallest workers. Tend the fungus garden with their tiny bodies that can
  maneuver in delicate fungal structures. Also serve as sentinels riding on leaves
  carried by mediae, defending against phorid flies. In our system: knowledge
  maintenance, contamination detection, quality guard on other agents' outputs.

- MEDIA: Medium-sized workers optimized for leaf cutting and transport. Their
  mandibles are coated in zinc-protein biomaterial (hard as stainless steel).
  They stridulate (vibrate) when cutting, which both stiffens leaves and
  communicates leaf quality to nearby ants. In our system: primary foragers,
  resource gatherers, the workhorses of knowledge acquisition.

- MAJOR: Largest workers with massive heads and mandibles. Bite force is 80x
  stronger than their body size would predict (not 30x as expected) due to
  disproportionate muscle volume. Defend colony against army ants and predators.
  In our system: adversarial validators, quality gatekeepers, heavy-duty analysis.
"""

from enum import Enum
from typing import Dict, List, Optional
from dataclasses import dataclass, field


class Caste(Enum):
    """Ant castes with distinct morphological and behavioral specializations."""
    QUEEN = "queen"
    MINIM = "minim"
    MEDIA = "media"
    MAJOR = "major"


@dataclass
class CasteProfile:
    """Configuration profile for a caste, defining its capabilities.

    Each field maps a biological trait to a computational parameter:
    - size: Relative body size (affects carrying capacity, LLM token budget)
    - signal_sensitivity: Olfactory acuity (pheromone detection threshold)
    - carrying_capacity: How much information an agent can process per action
    - temperature: LLM temperature (exploration vs exploitation tradeoff)
    - energy_cost: Computational cost per action (larger = more expensive)
    - specializations: Set of tasks this caste is optimized for
    """
    caste: Caste
    size: float                                  # Relative size (0.1-5.0)
    signal_sensitivity: float                     # Pheromone detection (0.0-1.0)
    carrying_capacity: int                        # Max tokens per action
    temperature: float                            # LLM temperature
    energy_cost: float                            # Resource cost per action
    specializations: List[str] = field(default_factory=list)
    description: str = ""

    @property
    def bite_force_multiplier(self) -> float:
        """Calculate relative bite force (analysis power).

        In Atta majors, bite force scales disproportionately with size:
        a 30x heavier ant has 80x the bite force due to increased muscle volume.
        We model this as a super-linear scaling: force = size^1.5
        """
        return self.size ** 1.5


# Default caste profiles modeled on Atta leaf cutter ant morphology
QUEEN_PROFILE = CasteProfile(
    caste=Caste.QUEEN,
    size=5.0,
    signal_sensitivity=0.9,
    carrying_capacity=500,        # High: manages entire colony
    temperature=0.5,              # Conservative: strategic decisions
    energy_cost=0.1,              # Low: mostly delegates
    specializations=["decompose", "coordinate", "lifecycle"],
    description="Colony coordinator. Decomposes tasks, manages lifecycle, controls caste production."
)

MINIM_PROFILE = CasteProfile(
    caste=Caste.MINIM,
    size=0.3,
    signal_sensitivity=0.95,      # Highest: detect subtle quality issues
    carrying_capacity=80,         # Low: small but precise
    temperature=0.4,              # Low: careful, methodical
    energy_cost=0.05,             # Cheapest: tiny computational footprint
    specializations=["tend_garden", "sentinel", "decontaminate", "brood_care"],
    description="Garden tender and sentinel. Maintains knowledge quality, detects contamination."
)

MEDIA_PROFILE = CasteProfile(
    caste=Caste.MEDIA,
    size=1.0,
    signal_sensitivity=0.7,       # Good: follow trails and recruitment signals
    carrying_capacity=200,        # Medium: balanced information processing
    temperature=0.75,             # Moderate: balanced exploration
    energy_cost=0.15,             # Moderate: standard workload
    specializations=["forage", "cut", "transport", "stridulate"],
    description="Primary forager. Cuts and transports leaf material (knowledge resources)."
)

MAJOR_PROFILE = CasteProfile(
    caste=Caste.MAJOR,
    size=3.0,
    signal_sensitivity=0.5,       # Lower: focused on defense, less trail-following
    carrying_capacity=350,        # High: deep analysis capacity
    temperature=0.85,             # High: creative adversarial thinking
    energy_cost=0.3,              # Expensive: heavy computational load
    specializations=["defend", "challenge", "heavy_analysis", "guard"],
    description="Soldier and defender. Adversarial validation, deep critical analysis."
)

# Registry of all default profiles
DEFAULT_PROFILES = {
    Caste.QUEEN: QUEEN_PROFILE,
    Caste.MINIM: MINIM_PROFILE,
    Caste.MEDIA: MEDIA_PROFILE,
    Caste.MAJOR: MAJOR_PROFILE,
}


class CasteRegistry:
    """Registry managing caste profiles and agent-caste assignments.

    The queen ant lays eggs with varying nutritional content to control caste
    development. Similarly, the registry controls what kind of agents are spawned
    based on colony needs.
    """

    def __init__(self, profiles: Optional[Dict[Caste, CasteProfile]] = None):
        """Initialize caste registry.

        Args:
            profiles: Custom caste profiles. If None, uses defaults.
        """
        self.profiles = profiles or dict(DEFAULT_PROFILES)
        self.assignments: Dict[str, Caste] = {}  # agent_id -> Caste
        self.population: Dict[Caste, int] = {c: 0 for c in Caste}

    def get_profile(self, caste: Caste) -> CasteProfile:
        """Get the profile for a caste.

        Args:
            caste: Caste to look up

        Returns:
            CasteProfile for the caste
        """
        return self.profiles[caste]

    def assign(self, agent_id: str, caste: Caste) -> CasteProfile:
        """Assign an agent to a caste.

        Args:
            agent_id: Agent to assign
            caste: Caste to assign to

        Returns:
            CasteProfile for the assigned caste
        """
        # Remove from previous caste if reassigned
        if agent_id in self.assignments:
            old_caste = self.assignments[agent_id]
            self.population[old_caste] = max(0, self.population[old_caste] - 1)

        self.assignments[agent_id] = caste
        self.population[caste] = self.population.get(caste, 0) + 1
        return self.profiles[caste]

    def get_caste(self, agent_id: str) -> Optional[Caste]:
        """Get the caste of an agent.

        Args:
            agent_id: Agent to look up

        Returns:
            Caste or None if not assigned
        """
        return self.assignments.get(agent_id)

    def get_agents_by_caste(self, caste: Caste) -> List[str]:
        """Get all agent IDs assigned to a caste.

        Args:
            caste: Caste to filter by

        Returns:
            List of agent IDs
        """
        return [aid for aid, c in self.assignments.items() if c == caste]

    def get_population(self) -> Dict[Caste, int]:
        """Get current population counts by caste.

        Returns:
            Dictionary of caste -> count
        """
        return dict(self.population)

    def get_population_ratio(self) -> Dict[Caste, float]:
        """Get population ratios (fraction of total).

        In real Atta colonies, mediae (foragers) make up the majority,
        with minims and majors as smaller proportions.

        Returns:
            Dictionary of caste -> ratio (0.0-1.0)
        """
        total = sum(self.population.values())
        if total == 0:
            return {c: 0.0 for c in Caste}
        return {c: count / total for c, count in self.population.items()}

    def assess_needs(self, fungus_garden_health: float = 1.0,
                     trail_coverage: float = 1.0,
                     threat_level: float = 0.0) -> Dict[Caste, int]:
        """Assess how many of each caste the colony needs.

        Models the queen's ability to adjust egg-laying based on colony needs:
        - Low garden health → more minims (tenders)
        - Low trail coverage → more mediae (foragers)
        - High threat level → more majors (defenders)

        Args:
            fungus_garden_health: Garden health 0.0-1.0 (1.0 = healthy)
            trail_coverage: Trail network coverage 0.0-1.0 (1.0 = well-covered)
            threat_level: External threat level 0.0-1.0 (0.0 = safe)

        Returns:
            Dictionary of caste -> recommended count delta (+/- from current)
        """
        needs = {}

        # Queen: always exactly 1
        needs[Caste.QUEEN] = 1 - self.population.get(Caste.QUEEN, 0)

        # Minims: need more when garden is unhealthy
        minim_need = max(0, round(3 * (1.0 - fungus_garden_health)))
        needs[Caste.MINIM] = minim_need

        # Mediae: need more when trails are sparse
        media_need = max(0, round(4 * (1.0 - trail_coverage)))
        needs[Caste.MEDIA] = media_need

        # Majors: need more when threat level is high
        major_need = max(0, round(3 * threat_level))
        needs[Caste.MAJOR] = major_need

        return needs

    def unassign(self, agent_id: str) -> Optional[Caste]:
        """Remove an agent from its caste assignment.

        Args:
            agent_id: Agent to remove

        Returns:
            Previous caste or None if not assigned
        """
        if agent_id not in self.assignments:
            return None

        caste = self.assignments.pop(agent_id)
        self.population[caste] = max(0, self.population[caste] - 1)
        return caste

    def get_stats(self) -> dict:
        """Get registry statistics.

        Returns:
            Dictionary with population stats
        """
        return {
            "total_agents": sum(self.population.values()),
            "population": {c.value: count for c, count in self.population.items()},
            "ratios": {c.value: r for c, r in self.get_population_ratio().items()},
        }
