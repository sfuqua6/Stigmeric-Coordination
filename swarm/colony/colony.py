"""Colony coordinator - the superorganism lifecycle manager.

An Atta leaf cutter ant colony is a superorganism: individual ants are like cells
in a body, each performing specialized functions that contribute to the whole.
The colony's "brain" is distributed across millions of workers (250,000 neurons each),
and together they rival a human brain in total neuron count.

Colony lifecycle (modeled on Atta):
1. FOUNDING: Queen lands after nuptial flight, excavates nest, starts fungus garden
   with a fragment carried in her mouth. First minims hatch to tend garden.
2. GROWTH: Slightly larger ants hatch, begin foraging nearby. Colony expands.
3. MATURATION: Media foragers foray far for leaves. Majors hatch for defense.
   Trail network expands. Fungus garden grows into thousands of chambers.
4. REPRODUCTION: Colony produces winged males and new queens for nuptial flights.

Key coordination mechanisms (all stigmergic - no central control):
- Chemical trails guide foragers to productive areas
- Stridulation recruits workers to high-quality resources
- Fungus garden health drives caste production adjustments
- Pheromone gradients create emergent traffic patterns

The colony coordinates the existing signal store system with the new
leaf cutter ant stigmergic primitives (pheromones, trails, garden, stridulation).
"""

import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from .pheromone import PheromoneField, PheromoneType
from .caste import Caste, CasteRegistry, CasteProfile
from .fungus_garden import FungusGarden
from .trail_network import TrailNetwork
from .stridulation import StridulationProtocol


class ColonyPhase(Enum):
    """Colony lifecycle phases, modeled on Atta development."""
    FOUNDING = "founding"          # Initial setup, queen + first minims
    GROWTH = "growth"              # Expanding foraging, building trails
    MATURATION = "maturation"      # Full operation, all castes active
    REPRODUCTION = "reproduction"  # Producing outputs (synthesis)


@dataclass
class ColonyCycle:
    """Record of a single colony cycle (one tick of the simulation)."""
    cycle_number: int
    phase: ColonyPhase
    pheromones_deposited: int = 0
    pheromones_evaporated: int = 0
    trails_reinforced: int = 0
    trails_evaporated: int = 0
    leaves_deposited: int = 0
    gongylidia_produced: int = 0
    stridulations: int = 0
    recruitments: int = 0
    garden_health: float = 1.0
    timestamp: float = field(default_factory=time.time)


class Colony:
    """The superorganism - coordinates all leaf cutter ant stigmergic mechanics.

    Integrates:
    - PheromoneField: Chemical communication landscape
    - CasteRegistry: Agent specialization management
    - FungusGarden: Shared knowledge processing substrate
    - TrailNetwork: Emergent pathway formation
    - StridulationProtocol: Quality-based recruitment

    The Colony doesn't control agents directly - it provides the stigmergic
    environment that agents interact with. Coordination emerges from agents
    depositing and sensing pheromones, reinforcing trails, tending the garden,
    and stridulating about resources.
    """

    def __init__(self, task_prompt: str = "",
                 max_leaves: int = 500,
                 max_gongylidia: int = 200):
        """Initialize the colony.

        Args:
            task_prompt: The task/question the colony is working on
            max_leaves: Maximum leaf fragments in fungus garden
            max_gongylidia: Maximum gongylidia in garden
        """
        self.task_prompt = task_prompt

        # Core stigmergic subsystems
        self.pheromone_field = PheromoneField()
        self.caste_registry = CasteRegistry()
        self.fungus_garden = FungusGarden(
            max_leaves=max_leaves,
            max_gongylidia=max_gongylidia
        )
        self.trail_network = TrailNetwork()
        self.stridulation = StridulationProtocol()

        # Colony state
        self.phase = ColonyPhase.FOUNDING
        self.cycle_count = 0
        self.history: List[ColonyCycle] = []
        self.founded_at = time.time()

        # Threat tracking (for adaptive caste production)
        self.threat_level: float = 0.0  # 0.0 = safe, 1.0 = under attack

    def tick(self) -> ColonyCycle:
        """Execute one colony cycle.

        Each tick:
        1. Evaporate pheromones (chemical degradation)
        2. Evaporate trails (unused paths fade)
        3. Decay fungus garden freshness
        4. Decay stridulation signals
        5. Update colony phase
        6. Record cycle metrics

        Returns:
            ColonyCycle with metrics from this tick
        """
        self.cycle_count += 1

        cycle = ColonyCycle(
            cycle_number=self.cycle_count,
            phase=self.phase,
        )

        # 1. Pheromone evaporation
        cycle.pheromones_evaporated = self.pheromone_field.evaporate()

        # 2. Trail evaporation
        cycle.trails_evaporated = self.trail_network.evaporate()

        # 3. Garden freshness decay
        self.fungus_garden.decay()
        cycle.garden_health = self.fungus_garden.health

        # 4. Stridulation decay
        self.stridulation.decay()

        # 5. Update colony phase
        self._update_phase()

        # 6. Update threat level (decays naturally)
        self.threat_level = max(0.0, self.threat_level * 0.95)

        self.history.append(cycle)
        return cycle

    def _update_phase(self):
        """Update colony phase based on current state.

        Phase transitions modeled on Atta colony development:
        - FOUNDING → GROWTH: When garden has some gongylidia and agents exist
        - GROWTH → MATURATION: When all castes are represented and trails established
        - MATURATION → REPRODUCTION: When enough knowledge accumulated for synthesis
        """
        population = self.caste_registry.get_population()
        total_agents = sum(population.values())
        garden_stats = self.fungus_garden.get_stats()
        trail_stats = self.trail_network.get_stats()

        if self.phase == ColonyPhase.FOUNDING:
            # Transition when we have some processed knowledge and a few agents
            if (garden_stats["gongylidia"]["total"] >= 1 and total_agents >= 2):
                self.phase = ColonyPhase.GROWTH

        elif self.phase == ColonyPhase.GROWTH:
            # Transition when all castes represented and trail network exists
            has_all_castes = all(
                population.get(c, 0) > 0
                for c in [Caste.MINIM, Caste.MEDIA, Caste.MAJOR]
            )
            has_trails = trail_stats["total_trails"] >= 3

            if has_all_castes and has_trails:
                self.phase = ColonyPhase.MATURATION

        elif self.phase == ColonyPhase.MATURATION:
            # Transition to reproduction when enough knowledge for synthesis
            if garden_stats["gongylidia"]["total"] >= 10:
                self.phase = ColonyPhase.REPRODUCTION

    # ========================================================================
    # AGENT INTERACTION METHODS - How agents interact with the colony
    # ========================================================================

    def forage_deposit(self, agent_id: str, content: str, source: str,
                       keywords: Optional[List[str]] = None,
                       quality: float = 0.5) -> Optional[str]:
        """Media forager deposits a leaf fragment in the fungus garden.

        Also deposits trail pheromone marking the path from source to garden.

        Args:
            agent_id: Forager agent ID
            content: Raw content (the "leaf")
            source: Where the content came from
            keywords: Associated keywords
            quality: Content quality assessment

        Returns:
            Leaf fragment ID, or None if rejected
        """
        # Deposit leaf in garden
        leaf_id = self.fungus_garden.deposit_leaf(
            content=content,
            source=source,
            deposited_by=agent_id,
            keywords=keywords,
            quality_score=quality,
        )

        if leaf_id:
            # Deposit trail pheromone (forager leaving scent on return trip)
            self.pheromone_field.deposit(
                pheromone_type=PheromoneType.TRAIL,
                context=source,
                concentration=0.3 + quality * 0.4,  # Better quality = stronger trail
                depositor=agent_id,
                target="garden",
            )

            # Reinforce trail from source to garden
            self.trail_network.reinforce(source, "garden", agent_id)

            # Stridulate if quality is good (like ants vibrating on sweet leaves)
            if quality > 0.5:
                self.stridulation.stridulate(
                    emitter=agent_id,
                    signal_id=leaf_id,
                    quality=quality,
                    context=source,
                )

        return leaf_id

    def process_knowledge(self, agent_id: str, leaf_id: str,
                         distilled_content: str,
                         nutrient_value: float = 0.6,
                         cross_ref_ids: Optional[List[str]] = None) -> Optional[str]:
        """Minim processes a leaf fragment into a gongylidium.

        The minim takes raw input and transforms it into usable knowledge,
        like fungal mycelium breaking down cellulose into glucose.

        Args:
            agent_id: Minim agent processing
            leaf_id: Leaf to process
            distilled_content: The distilled knowledge output
            nutrient_value: Quality of processing output
            cross_ref_ids: Related gongylidia for cross-referencing

        Returns:
            Gongylidium ID, or None if processing failed
        """
        gong_id = self.fungus_garden.process_leaf(
            leaf_id=leaf_id,
            processor_id=agent_id,
            distilled_content=distilled_content,
            nutrient_value=nutrient_value,
            cross_ref_ids=cross_ref_ids,
        )

        if gong_id:
            # Deposit recruitment pheromone (fresh knowledge available!)
            self.pheromone_field.deposit(
                pheromone_type=PheromoneType.RECRUITMENT,
                context="garden",
                concentration=nutrient_value,
                depositor=agent_id,
            )

        return gong_id

    def consume_knowledge(self, agent_id: str, n: int = 3,
                         keywords: Optional[List[str]] = None) -> List[Any]:
        """Agent consumes gongylidia (processed knowledge) from the garden.

        Args:
            agent_id: Agent consuming
            n: Number of gongylidia to harvest
            keywords: Optional keyword filter

        Returns:
            List of Gongylidium objects
        """
        if keywords:
            return self.fungus_garden.harvest_by_keywords(
                keywords=keywords, n=n, consumer_id=agent_id
            )
        return self.fungus_garden.harvest(n=n, consumer_id=agent_id)

    def signal_threat(self, agent_id: str, context: str,
                      severity: float = 0.5):
        """Major signals a threat (quality issue, contradiction, attack).

        Deposits alarm pheromone and raises colony threat level.

        Args:
            agent_id: Agent detecting threat
            context: What/where the threat is
            severity: Threat severity (0.0-1.0)
        """
        self.pheromone_field.deposit(
            pheromone_type=PheromoneType.ALARM,
            context=context,
            concentration=severity,
            depositor=agent_id,
        )

        # Introduce contamination to garden if severe
        if severity > 0.7:
            self.fungus_garden.introduce_contamination(
                amount=severity * 0.2,
                source=context,
            )

        # Raise threat level
        self.threat_level = min(1.0, self.threat_level + severity * 0.3)

    def tend_garden(self, agent_id: str) -> dict:
        """Minim tends the fungus garden.

        Args:
            agent_id: Minim performing maintenance

        Returns:
            Maintenance report
        """
        return self.fungus_garden.tend(tender_id=agent_id)

    def scout_trail(self, agent_id: str, from_topic: str,
                    to_topic: str, quality: float = 0.5) -> str:
        """Scout agent discovers a connection between topics.

        Creates/reinforces a trail and deposits pheromone.

        Args:
            agent_id: Scout agent
            from_topic: Source topic
            to_topic: Destination topic
            quality: Connection quality

        Returns:
            Trail segment ID
        """
        seg_id = self.trail_network.reinforce(from_topic, to_topic, agent_id)

        # Deposit trail pheromone at both endpoints
        self.pheromone_field.deposit(
            pheromone_type=PheromoneType.TRAIL,
            context=from_topic,
            concentration=quality * 0.5,
            depositor=agent_id,
            target=to_topic,
        )

        return seg_id

    def follow_trail(self, agent_id: str, from_topic: str,
                     exploration_rate: float = 0.1) -> Optional[str]:
        """Agent follows trail network to find next productive area.

        Args:
            agent_id: Agent following trail
            from_topic: Current topic/position
            exploration_rate: Chance of random exploration

        Returns:
            Next topic to explore, or None if no trails
        """
        return self.trail_network.follow(from_topic, exploration_rate)

    def listen_for_recruitment(self, agent_id: str,
                              context: Optional[str] = None) -> List[Any]:
        """Agent listens for stridulation signals (recruitment).

        Args:
            agent_id: Listening agent
            context: Optional context filter

        Returns:
            List of stridulation signals heard
        """
        caste = self.caste_registry.get_caste(agent_id)
        if caste:
            profile = self.caste_registry.get_profile(caste)
            sensitivity = profile.signal_sensitivity
        else:
            sensitivity = 0.7

        return self.stridulation.listen(
            listener_id=agent_id,
            context=context,
            sensitivity=sensitivity,
        )

    # ========================================================================
    # COLONY ASSESSMENT AND MANAGEMENT
    # ========================================================================

    def assess_needs(self) -> Dict[Caste, int]:
        """Assess what castes the colony needs more of.

        Based on colony health metrics, determine staffing needs.
        Models the queen's ability to adjust caste production via
        egg nutritional content.

        Returns:
            Dictionary of caste -> recommended count adjustment
        """
        garden_health = self.fungus_garden.health
        trail_coverage = self.trail_network.get_coverage()

        return self.caste_registry.assess_needs(
            fungus_garden_health=garden_health,
            trail_coverage=trail_coverage,
            threat_level=self.threat_level,
        )

    def get_health(self) -> dict:
        """Get comprehensive colony health report.

        Returns:
            Dictionary with all subsystem health metrics
        """
        return {
            "phase": self.phase.value,
            "cycle_count": self.cycle_count,
            "age_seconds": time.time() - self.founded_at,
            "threat_level": self.threat_level,
            "pheromone_field": self.pheromone_field.get_stats(),
            "caste_registry": self.caste_registry.get_stats(),
            "fungus_garden": self.fungus_garden.get_stats(),
            "trail_network": self.trail_network.get_stats(),
            "stridulation": self.stridulation.get_stats(),
        }

    def get_agent_context(self, agent_id: str) -> dict:
        """Get the full environmental context for an agent.

        Combines pheromone sensing, trail information, garden status,
        and stridulation signals into a unified context that the agent
        can use to decide its next action.

        Args:
            agent_id: Agent requesting context

        Returns:
            Dictionary with environmental context
        """
        caste = self.caste_registry.get_caste(agent_id)
        profile = self.caste_registry.get_profile(caste) if caste else None

        context = {
            "agent_id": agent_id,
            "caste": caste.value if caste else None,
            "colony_phase": self.phase.value,
            "garden_health": self.fungus_garden.health,
            "threat_level": self.threat_level,
        }

        # Stridulation signals (recruitment opportunities)
        stridulations = self.listen_for_recruitment(agent_id)
        if stridulations:
            context["recruitment_targets"] = [
                {"context": s.context, "frequency": s.frequency, "signal_id": s.signal_id}
                for s in stridulations[:5]
            ]

        # Available gongylidia (processed knowledge)
        available_knowledge = self.fungus_garden.harvest(n=3, consumer_id=None)
        if available_knowledge:
            context["available_knowledge"] = [
                {"content": g.content[:100], "nutrient_value": g.nutrient_value}
                for g in available_knowledge
            ]

        # Trail network info
        if profile:
            trail_neighbors = self.trail_network.get_neighbors(
                self.task_prompt[:50] if self.task_prompt else "root"
            )
            if trail_neighbors:
                context["trail_options"] = [
                    {"destination": dest, "strength": strength}
                    for dest, strength in trail_neighbors[:5]
                ]

        # Alarm pheromones (threats)
        alarms = self.pheromone_field.get_all_of_type(PheromoneType.ALARM)
        if alarms:
            context["active_threats"] = [
                {"context": a.context, "concentration": a.concentration}
                for a in alarms[:3]
            ]

        return context

    def get_summary(self) -> str:
        """Get a human-readable colony summary.

        Returns:
            Formatted summary string
        """
        health = self.get_health()
        pop = health["caste_registry"]["population"]
        garden = health["fungus_garden"]
        trails = health["trail_network"]
        phero = health["pheromone_field"]

        lines = [
            f"=== Leaf Cutter Ant Colony ===",
            f"Phase: {self.phase.value.upper()} | Cycle: {self.cycle_count}",
            f"Threat Level: {self.threat_level:.1%}",
            f"",
            f"Population: {health['caste_registry']['total_agents']} agents",
            f"  Queen: {pop.get('queen', 0)} | Minims: {pop.get('minim', 0)} | "
            f"Mediae: {pop.get('media', 0)} | Majors: {pop.get('major', 0)}",
            f"",
            f"Fungus Garden: {garden['health']:.1%} health",
            f"  Leaves: {garden['leaf_pile']['unprocessed']} unprocessed / "
            f"{garden['leaf_pile']['total']} total",
            f"  Gongylidia: {garden['gongylidia']['total']} available",
            f"  Contamination: {garden['contamination_level']:.1%}",
            f"",
            f"Trail Network: {trails['total_trails']} trails, "
            f"{trails.get('highways', 0)} highways",
            f"  Coverage: {trails.get('coverage', 0):.1%}",
            f"",
            f"Pheromone Field: {phero['total_active']} active deposits",
            f"  Deposited: {phero['total_deposited']} | "
            f"Evaporated: {phero['total_evaporated']}",
        ]

        strid = health["stridulation"]
        if strid["total_stridulations"] > 0:
            lines.extend([
                f"",
                f"Stridulation: {strid['active_signals']} active, "
                f"{strid['total_recruitments']} recruitments",
            ])

        return "\n".join(lines)

    def clear(self):
        """Reset the entire colony."""
        self.pheromone_field.clear()
        self.fungus_garden.clear()
        self.trail_network.clear()
        self.stridulation.clear()
        self.phase = ColonyPhase.FOUNDING
        self.cycle_count = 0
        self.history.clear()
        self.threat_level = 0.0
