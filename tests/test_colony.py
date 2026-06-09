"""Tests for the leaf cutter ant colony coordination system.

Tests all colony subsystems without requiring LLM or GPU.
These are fast unit tests validating the stigmergic coordination primitives.
"""

import unittest
import time


class TestPheromoneField(unittest.TestCase):
    """Test the pheromone system (volatile/persistent chemical signals)."""

    def setUp(self):
        from swarm.colony.pheromone import PheromoneField, PheromoneType
        self.field = PheromoneField()
        self.PheromoneType = PheromoneType

    def test_deposit_basic(self):
        """Test depositing a pheromone."""
        pid = self.field.deposit(
            self.PheromoneType.TRAIL, "topic_A", 0.5, "agent_1"
        )
        self.assertIsNotNone(pid)
        self.assertTrue(pid.startswith("P_trail_"))

    def test_deposit_types(self):
        """Test all pheromone types deposit correctly."""
        for ptype in self.PheromoneType:
            pid = self.field.deposit(ptype, f"ctx_{ptype.value}", 0.5, "agent_1")
            self.assertIsNotNone(pid)

        stats = self.field.get_stats()
        self.assertEqual(stats["total_active"], 3)

    def test_reinforcement(self):
        """Test that same-type, same-context deposits reinforce instead of duplicate."""
        pid1 = self.field.deposit(
            self.PheromoneType.TRAIL, "topic_A", 0.4, "agent_1"
        )
        pid2 = self.field.deposit(
            self.PheromoneType.TRAIL, "topic_A", 0.3, "agent_1"
        )
        # Same depositor, same context, same type = reinforcement, not new deposit
        self.assertEqual(pid1, pid2)
        self.assertEqual(self.field.get_stats()["total_active"], 1)

        # Concentration should have increased
        p = self.field.pheromones[pid1]
        self.assertGreater(p.concentration, 0.4)

    def test_different_depositors_no_reinforce(self):
        """Test that different depositors create separate pheromones."""
        pid1 = self.field.deposit(
            self.PheromoneType.TRAIL, "topic_A", 0.4, "agent_1"
        )
        pid2 = self.field.deposit(
            self.PheromoneType.TRAIL, "topic_A", 0.3, "agent_2"
        )
        self.assertNotEqual(pid1, pid2)
        self.assertEqual(self.field.get_stats()["total_active"], 2)

    def test_sense(self):
        """Test sensing pheromones at a context."""
        self.field.deposit(self.PheromoneType.TRAIL, "topic_A", 0.8, "agent_1")
        self.field.deposit(self.PheromoneType.RECRUITMENT, "topic_A", 0.6, "agent_2")
        self.field.deposit(self.PheromoneType.TRAIL, "topic_B", 0.9, "agent_3")

        # Sense all at topic_A
        sensed = self.field.sense("topic_A")
        self.assertEqual(len(sensed), 2)

        # Sense only TRAIL at topic_A
        trail_sensed = self.field.sense("topic_A", self.PheromoneType.TRAIL)
        self.assertEqual(len(trail_sensed), 1)

    def test_get_concentration(self):
        """Test concentration measurement (accumulation from multiple depositors)."""
        self.field.deposit(self.PheromoneType.TRAIL, "topic_A", 0.3, "agent_1")
        self.field.deposit(self.PheromoneType.TRAIL, "topic_A", 0.4, "agent_2")

        conc = self.field.get_concentration("topic_A", self.PheromoneType.TRAIL)
        self.assertAlmostEqual(conc, 0.7, places=1)

    def test_evaporation(self):
        """Test pheromone evaporation over time."""
        self.field.deposit(self.PheromoneType.RECRUITMENT, "topic_A", 0.1, "agent_1")

        # Recruitment is volatile (15% per tick). After several ticks should evaporate.
        for _ in range(30):
            self.field.evaporate()

        stats = self.field.get_stats()
        # Volatile pheromone should have evaporated
        self.assertEqual(stats["total_active"], 0)
        self.assertGreater(stats["total_evaporated"], 0)

    def test_trail_persistence(self):
        """Test that trail pheromones persist longer than recruitment."""
        self.field.deposit(self.PheromoneType.TRAIL, "topic_A", 0.5, "agent_1")
        self.field.deposit(self.PheromoneType.RECRUITMENT, "topic_B", 0.5, "agent_2")

        # After 10 ticks
        for _ in range(10):
            self.field.evaporate()

        trail_conc = self.field.get_concentration("topic_A", self.PheromoneType.TRAIL)
        recruit_conc = self.field.get_concentration("topic_B", self.PheromoneType.RECRUITMENT)

        # Trail should be stronger than recruitment after same time
        self.assertGreater(trail_conc, recruit_conc)

    def test_sample_gradient(self):
        """Test gradient sampling across contexts."""
        self.field.deposit(self.PheromoneType.TRAIL, "topic_A", 0.3, "agent_1")
        self.field.deposit(self.PheromoneType.TRAIL, "topic_B", 0.8, "agent_2")
        self.field.deposit(self.PheromoneType.TRAIL, "topic_C", 0.1, "agent_3")

        gradients = self.field.sample_gradient(
            ["topic_A", "topic_B", "topic_C", "topic_D"],
            self.PheromoneType.TRAIL
        )

        # Should be sorted by concentration (B > A > C)
        self.assertEqual(len(gradients), 3)  # D has no pheromone
        self.assertEqual(gradients[0][0], "topic_B")

    def test_follow_gradient(self):
        """Test following pheromone gradient."""
        self.field.deposit(self.PheromoneType.TRAIL, "topic_A", 0.1, "agent_1")
        self.field.deposit(self.PheromoneType.TRAIL, "topic_B", 0.9, "agent_2")

        # With 0% exploration, should always follow strongest
        results = set()
        for _ in range(20):
            result = self.field.follow_gradient(
                ["topic_A", "topic_B"],
                self.PheromoneType.TRAIL,
                exploration_rate=0.0
            )
            if result:
                results.add(result)

        # topic_B should be heavily favored
        self.assertIn("topic_B", results)


class TestCasteSystem(unittest.TestCase):
    """Test the caste system (agent specialization)."""

    def setUp(self):
        from swarm.colony.caste import CasteRegistry, Caste
        self.registry = CasteRegistry()
        self.Caste = Caste

    def test_assign_caste(self):
        """Test assigning agents to castes."""
        profile = self.registry.assign("agent_1", self.Caste.MEDIA)
        self.assertEqual(profile.caste, self.Caste.MEDIA)
        self.assertEqual(self.registry.get_caste("agent_1"), self.Caste.MEDIA)

    def test_population_tracking(self):
        """Test population counts."""
        self.registry.assign("queen_1", self.Caste.QUEEN)
        self.registry.assign("minim_1", self.Caste.MINIM)
        self.registry.assign("minim_2", self.Caste.MINIM)
        self.registry.assign("media_1", self.Caste.MEDIA)
        self.registry.assign("major_1", self.Caste.MAJOR)

        pop = self.registry.get_population()
        self.assertEqual(pop[self.Caste.QUEEN], 1)
        self.assertEqual(pop[self.Caste.MINIM], 2)
        self.assertEqual(pop[self.Caste.MEDIA], 1)
        self.assertEqual(pop[self.Caste.MAJOR], 1)

    def test_caste_profiles(self):
        """Test that caste profiles have proper specializations."""
        from swarm.colony.caste import MINIM_PROFILE, MEDIA_PROFILE, MAJOR_PROFILE

        # Minims should be most sensitive to signals
        self.assertGreater(MINIM_PROFILE.signal_sensitivity, MEDIA_PROFILE.signal_sensitivity)
        self.assertGreater(MINIM_PROFILE.signal_sensitivity, MAJOR_PROFILE.signal_sensitivity)

        # Majors should have highest carrying capacity (among workers)
        self.assertGreater(MAJOR_PROFILE.carrying_capacity, MEDIA_PROFILE.carrying_capacity)
        self.assertGreater(MAJOR_PROFILE.carrying_capacity, MINIM_PROFILE.carrying_capacity)

        # Minims should be cheapest
        self.assertLess(MINIM_PROFILE.energy_cost, MEDIA_PROFILE.energy_cost)
        self.assertLess(MINIM_PROFILE.energy_cost, MAJOR_PROFILE.energy_cost)

    def test_bite_force_scaling(self):
        """Test super-linear bite force scaling (like real Atta majors)."""
        from swarm.colony.caste import MINIM_PROFILE, MAJOR_PROFILE

        # Major is 10x size of minim, but bite force should be >10x
        size_ratio = MAJOR_PROFILE.size / MINIM_PROFILE.size
        force_ratio = MAJOR_PROFILE.bite_force_multiplier / MINIM_PROFILE.bite_force_multiplier

        self.assertGreater(force_ratio, size_ratio)

    def test_assess_needs(self):
        """Test adaptive caste needs assessment."""
        # Colony with low garden health should need more minims
        needs = self.registry.assess_needs(
            fungus_garden_health=0.2,
            trail_coverage=0.8,
            threat_level=0.1
        )
        self.assertGreater(needs[self.Caste.MINIM], 0)

        # Colony under threat should need more majors
        needs_threat = self.registry.assess_needs(
            fungus_garden_health=0.9,
            trail_coverage=0.9,
            threat_level=0.9
        )
        self.assertGreater(needs_threat[self.Caste.MAJOR], 0)

    def test_unassign(self):
        """Test removing agent from caste."""
        self.registry.assign("agent_1", self.Caste.MEDIA)
        self.assertEqual(self.registry.get_population()[self.Caste.MEDIA], 1)

        old_caste = self.registry.unassign("agent_1")
        self.assertEqual(old_caste, self.Caste.MEDIA)
        self.assertEqual(self.registry.get_population()[self.Caste.MEDIA], 0)
        self.assertIsNone(self.registry.get_caste("agent_1"))

    def test_get_agents_by_caste(self):
        """Test filtering agents by caste."""
        self.registry.assign("minim_1", self.Caste.MINIM)
        self.registry.assign("minim_2", self.Caste.MINIM)
        self.registry.assign("media_1", self.Caste.MEDIA)

        minims = self.registry.get_agents_by_caste(self.Caste.MINIM)
        self.assertEqual(len(minims), 2)
        self.assertIn("minim_1", minims)
        self.assertIn("minim_2", minims)


class TestFungusGarden(unittest.TestCase):
    """Test the fungus garden (knowledge processing substrate)."""

    def setUp(self):
        from swarm.colony.fungus_garden import FungusGarden
        self.garden = FungusGarden(max_leaves=50, max_gongylidia=20)

    def test_deposit_leaf(self):
        """Test depositing a leaf fragment."""
        leaf_id = self.garden.deposit_leaf(
            content="The Earth's average temperature has risen 1.1C since pre-industrial times.",
            source="wikipedia",
            deposited_by="media_1",
            keywords=["temperature", "climate"],
        )
        self.assertIsNotNone(leaf_id)
        self.assertTrue(leaf_id.startswith("LEAF_"))

    def test_reject_empty_leaf(self):
        """Test that empty content is rejected."""
        leaf_id = self.garden.deposit_leaf(
            content="", source="test", deposited_by="agent_1"
        )
        self.assertIsNone(leaf_id)

    def test_process_leaf(self):
        """Test processing a leaf into a gongylidium."""
        leaf_id = self.garden.deposit_leaf(
            content="CO2 levels have reached 421 ppm, highest in 800,000 years.",
            source="nasa",
            deposited_by="media_1",
        )

        gong_id = self.garden.process_leaf(
            leaf_id=leaf_id,
            processor_id="minim_1",
            distilled_content="CO2 at record 421 ppm - unprecedented in 800K years",
            nutrient_value=0.8,
        )

        self.assertIsNotNone(gong_id)
        self.assertTrue(gong_id.startswith("GONG_"))

        # Leaf should be marked as processed
        leaf = self.garden.leaf_pile[leaf_id]
        self.assertTrue(leaf.processed)
        self.assertEqual(leaf.processing_agent, "minim_1")

    def test_harvest(self):
        """Test harvesting gongylidia."""
        # Create and process a leaf
        leaf_id = self.garden.deposit_leaf(
            content="Renewable energy costs have dropped 90% in a decade.",
            source="irena",
            deposited_by="media_1",
        )
        self.garden.process_leaf(
            leaf_id, "minim_1",
            "Renewable energy: 90% cost reduction over 10 years",
            nutrient_value=0.9,
        )

        harvested = self.garden.harvest(n=5, consumer_id="scout_1")
        self.assertEqual(len(harvested), 1)
        self.assertEqual(harvested[0].consumed_by, ["scout_1"])

    def test_harvest_by_keywords(self):
        """Test keyword-based harvesting."""
        # Deposit and process two leaves with different keywords
        leaf1 = self.garden.deposit_leaf(
            content="Solar panel efficiency has doubled.", source="test",
            deposited_by="media_1", keywords=["solar", "energy"],
        )
        self.garden.process_leaf(leaf1, "minim_1", "Solar efficiency doubled", 0.8)

        leaf2 = self.garden.deposit_leaf(
            content="Arctic ice is melting rapidly.", source="test",
            deposited_by="media_2", keywords=["arctic", "ice"],
        )
        self.garden.process_leaf(leaf2, "minim_2", "Arctic ice melting rapidly", 0.7)

        # Search for solar-related knowledge
        solar = self.garden.harvest_by_keywords(["solar", "energy"])
        self.assertGreater(len(solar), 0)
        self.assertIn("Solar", solar[0].content)

    def test_garden_health(self):
        """Test garden health calculation."""
        # Fresh garden should be moderately healthy
        health = self.garden.health
        self.assertGreater(health, 0.0)

        # Contamination should reduce health
        self.garden.introduce_contamination(0.5)
        health_after = self.garden.health
        self.assertLess(health_after, health)

    def test_tend_garden(self):
        """Test minim garden tending."""
        # Add low-quality leaf
        self.garden.deposit_leaf(
            content="This is unreliable low quality content for testing",
            source="test", deposited_by="media_1", quality_score=0.1,
        )
        # Add contamination
        self.garden.introduce_contamination(0.3)

        report = self.garden.tend("minim_1")
        self.assertGreater(report["leaves_composted"], 0)
        self.assertGreater(report["contamination_removed"], 0)

    def test_capacity_management(self):
        """Test that garden respects capacity limits."""
        for i in range(60):  # Exceed max_leaves=50
            self.garden.deposit_leaf(
                content=f"Leaf content number {i} with enough text to pass.",
                source="test", deposited_by="media_1",
                quality_score=0.5 + (i / 120),  # Varying quality
            )

        # Should not exceed capacity
        self.assertLessEqual(len(self.garden.leaf_pile), 50)

    def test_freshness_decay(self):
        """Test gongylidium freshness decay."""
        leaf_id = self.garden.deposit_leaf(
            content="Test content that needs to be long enough.",
            source="test", deposited_by="media_1",
        )
        gong_id = self.garden.process_leaf(
            leaf_id, "minim_1", "Distilled knowledge", 0.8
        )

        initial_freshness = self.garden.gongylidia[gong_id].freshness
        self.assertEqual(initial_freshness, 1.0)

        # Decay several times
        for _ in range(10):
            self.garden.decay()

        final_freshness = self.garden.gongylidia[gong_id].freshness
        self.assertLess(final_freshness, initial_freshness)


class TestTrailNetwork(unittest.TestCase):
    """Test the trail network (emergent pathway formation)."""

    def setUp(self):
        from swarm.colony.trail_network import TrailNetwork
        self.network = TrailNetwork()

    def test_reinforce_creates_trail(self):
        """Test that reinforcement creates a new trail."""
        seg_id = self.network.reinforce("topic_A", "topic_B", "agent_1")
        self.assertIsNotNone(seg_id)
        self.assertTrue(seg_id.startswith("TRAIL_"))

    def test_reinforce_strengthens_existing(self):
        """Test that repeated reinforcement increases trail strength."""
        seg_id = self.network.reinforce("topic_A", "topic_B", "agent_1")
        initial_strength = self.network.segments[seg_id].strength

        # Reinforce again from different agent
        seg_id2 = self.network.reinforce("topic_A", "topic_B", "agent_2")
        self.assertEqual(seg_id, seg_id2)  # Same trail

        final_strength = self.network.segments[seg_id].strength
        self.assertGreater(final_strength, initial_strength)

    def test_bidirectional_trails(self):
        """Test that trails are bidirectional."""
        self.network.reinforce("topic_A", "topic_B", "agent_1")

        # B->A should hit the same trail
        seg_id = self.network.reinforce("topic_B", "topic_A", "agent_2")
        self.assertEqual(len(self.network.segments), 1)

    def test_follow_trail(self):
        """Test following trails probabilistically."""
        self.network.reinforce("topic_A", "topic_B", "agent_1", amount=0.9)
        self.network.reinforce("topic_A", "topic_C", "agent_2", amount=0.1)

        # Follow from topic_A multiple times
        destinations = set()
        for _ in range(50):
            dest = self.network.follow("topic_A", exploration_rate=0.0)
            if dest:
                destinations.add(dest)

        # topic_B should be reached (it's stronger)
        self.assertIn("topic_B", destinations)

    def test_get_neighbors(self):
        """Test getting connected nodes."""
        self.network.reinforce("topic_A", "topic_B", "agent_1")
        self.network.reinforce("topic_A", "topic_C", "agent_2")

        neighbors = self.network.get_neighbors("topic_A")
        self.assertEqual(len(neighbors), 2)

    def test_trail_evaporation(self):
        """Test trail evaporation."""
        self.network.reinforce("topic_A", "topic_B", "agent_1", amount=0.05)

        # Evaporate many times
        for _ in range(100):
            self.network.evaporate()

        # Weak trail should have evaporated
        self.assertEqual(len(self.network.segments), 0)

    def test_highway_detection(self):
        """Test highway (heavily used trail) detection."""
        # Create a heavily used trail
        for i in range(10):
            self.network.reinforce("topic_A", "topic_B", f"agent_{i}")

        highways = self.network.get_highways()
        self.assertGreater(len(highways), 0)
        self.assertTrue(highways[0].is_highway)

    def test_get_path(self):
        """Test pathfinding through trail network."""
        self.network.reinforce("A", "B", "agent_1", amount=0.8)
        self.network.reinforce("B", "C", "agent_2", amount=0.7)
        self.network.reinforce("C", "D", "agent_3", amount=0.6)

        path = self.network.get_path("A", "D")
        self.assertIsNotNone(path)
        self.assertEqual(path[0], "A")
        self.assertEqual(path[-1], "D")

    def test_no_self_loops(self):
        """Test that self-loops are prevented."""
        result = self.network.reinforce("topic_A", "topic_A", "agent_1")
        self.assertEqual(result, "")
        self.assertEqual(len(self.network.segments), 0)

    def test_coverage_metric(self):
        """Test network coverage calculation."""
        # Empty network should have 0 coverage
        self.assertEqual(self.network.get_coverage(), 0.0)

        # Add some trails
        self.network.reinforce("A", "B", "agent_1", amount=0.8)
        self.network.reinforce("B", "C", "agent_2", amount=0.7)

        coverage = self.network.get_coverage()
        self.assertGreater(coverage, 0.0)
        self.assertLessEqual(coverage, 1.0)


class TestStridulationProtocol(unittest.TestCase):
    """Test the stridulation protocol (quality-based recruitment)."""

    def setUp(self):
        from swarm.colony.stridulation import StridulationProtocol
        self.protocol = StridulationProtocol()

    def test_stridulate_high_quality(self):
        """Test that high quality triggers stridulation."""
        strid_id = self.protocol.stridulate(
            emitter="media_1",
            signal_id="sig_001",
            quality=0.8,
            context="climate_data",
        )
        self.assertIsNotNone(strid_id)

    def test_no_stridulation_low_quality(self):
        """Test that low quality doesn't trigger stridulation (like tough leaves)."""
        strid_id = self.protocol.stridulate(
            emitter="media_1",
            signal_id="sig_001",
            quality=0.1,  # Below threshold
            context="bad_data",
        )
        self.assertIsNone(strid_id)

    def test_frequency_proportional_to_quality(self):
        """Test that vibration frequency reflects resource quality."""
        self.protocol.stridulate("media_1", "sig_001", 0.5, "ctx_low")
        self.protocol.stridulate("media_2", "sig_002", 0.95, "ctx_high")

        signals = list(self.protocol.active_signals.values())
        low_freq = [s for s in signals if s.context == "ctx_low"][0].frequency
        high_freq = [s for s in signals if s.context == "ctx_high"][0].frequency

        self.assertGreater(high_freq, low_freq)

    def test_listen(self):
        """Test listening for stridulation signals."""
        self.protocol.stridulate("media_1", "sig_001", 0.8, "topic_A")
        self.protocol.stridulate("media_2", "sig_002", 0.6, "topic_B")

        # Listen for all
        heard = self.protocol.listen("scout_1")
        self.assertEqual(len(heard), 2)
        # Should be sorted by frequency (strongest first)
        self.assertGreaterEqual(heard[0].frequency, heard[1].frequency)

    def test_dont_hear_own_stridulation(self):
        """Test that agents don't hear their own stridulations."""
        self.protocol.stridulate("media_1", "sig_001", 0.8, "topic_A")

        heard_self = self.protocol.listen("media_1")
        heard_other = self.protocol.listen("scout_1")

        self.assertEqual(len(heard_self), 0)
        self.assertEqual(len(heard_other), 1)

    def test_listen_by_context(self):
        """Test context-filtered listening."""
        self.protocol.stridulate("media_1", "sig_001", 0.8, "topic_A")
        self.protocol.stridulate("media_2", "sig_002", 0.7, "topic_B")

        heard = self.protocol.listen("scout_1", context="topic_A")
        self.assertEqual(len(heard), 1)
        self.assertEqual(heard[0].context, "topic_A")

    def test_respond(self):
        """Test recording recruitment response."""
        strid_id = self.protocol.stridulate("media_1", "sig_001", 0.8, "topic_A")
        success = self.protocol.respond("scout_1", strid_id)
        self.assertTrue(success)

        signal = self.protocol.active_signals[strid_id]
        self.assertIn("scout_1", signal.receivers)

    def test_recruitment_targets(self):
        """Test getting top recruitment target areas."""
        self.protocol.stridulate("media_1", "sig_001", 0.9, "hot_topic")
        self.protocol.stridulate("media_2", "sig_002", 0.8, "hot_topic")
        self.protocol.stridulate("media_3", "sig_003", 0.5, "cold_topic")

        targets = self.protocol.get_recruitment_targets(n=2)
        self.assertEqual(len(targets), 2)
        # hot_topic should rank first (more intense stridulation)
        self.assertEqual(targets[0], "hot_topic")

    def test_vibration_decay(self):
        """Test that stridulation signals decay quickly."""
        self.protocol.stridulate("media_1", "sig_001", 0.4, "topic_A")

        for _ in range(30):
            self.protocol.decay()

        # Signal should have expired
        stats = self.protocol.get_stats()
        self.assertEqual(stats["active_signals"], 0)


class TestColony(unittest.TestCase):
    """Test the colony coordinator (superorganism lifecycle)."""

    def setUp(self):
        from swarm.colony.colony import Colony
        self.colony = Colony(task_prompt="How can we address climate change?")

    def test_initialization(self):
        """Test colony starts in founding phase."""
        from swarm.colony.colony import ColonyPhase
        self.assertEqual(self.colony.phase, ColonyPhase.FOUNDING)
        self.assertEqual(self.colony.cycle_count, 0)

    def test_tick(self):
        """Test colony tick execution."""
        cycle = self.colony.tick()
        self.assertEqual(cycle.cycle_number, 1)
        self.assertEqual(self.colony.cycle_count, 1)

    def test_forage_deposit(self):
        """Test full foraging workflow: deposit leaf + trail + stridulation."""
        from swarm.colony.caste import Caste
        self.colony.caste_registry.assign("media_1", Caste.MEDIA)

        leaf_id = self.colony.forage_deposit(
            agent_id="media_1",
            content="Global temperatures rose 1.1C above pre-industrial levels in 2023.",
            source="ipcc_report",
            keywords=["temperature", "climate", "global"],
            quality=0.8,
        )

        self.assertIsNotNone(leaf_id)

        # Should have deposited trail pheromone
        from swarm.colony.pheromone import PheromoneType
        trail_conc = self.colony.pheromone_field.get_concentration(
            "ipcc_report", PheromoneType.TRAIL
        )
        self.assertGreater(trail_conc, 0)

        # Should have created a trail
        trails = self.colony.trail_network.get_stats()
        self.assertGreater(trails["total_trails"], 0)

        # Should have stridulated (quality > 0.5)
        strid_stats = self.colony.stridulation.get_stats()
        self.assertGreater(strid_stats["total_stridulations"], 0)

    def test_process_knowledge(self):
        """Test minim knowledge processing workflow."""
        from swarm.colony.caste import Caste
        self.colony.caste_registry.assign("media_1", Caste.MEDIA)
        self.colony.caste_registry.assign("minim_1", Caste.MINIM)

        # Forager deposits leaf
        leaf_id = self.colony.forage_deposit(
            "media_1", "Raw climate data with many details to process.",
            "nasa", ["climate"], quality=0.7,
        )

        # Minim processes it
        gong_id = self.colony.process_knowledge(
            agent_id="minim_1",
            leaf_id=leaf_id,
            distilled_content="Climate data summary: key finding",
            nutrient_value=0.8,
        )

        self.assertIsNotNone(gong_id)

        # Should have deposited recruitment pheromone
        from swarm.colony.pheromone import PheromoneType
        recruit_conc = self.colony.pheromone_field.get_concentration(
            "garden", PheromoneType.RECRUITMENT
        )
        self.assertGreater(recruit_conc, 0)

    def test_consume_knowledge(self):
        """Test agent consuming processed knowledge."""
        # Set up: deposit and process
        leaf_id = self.colony.fungus_garden.deposit_leaf(
            "Test content with enough length for processing.",
            "test_source", "media_1",
        )
        self.colony.fungus_garden.process_leaf(
            leaf_id, "minim_1", "Processed knowledge nugget", 0.9,
        )

        # Consume
        knowledge = self.colony.consume_knowledge("scout_1", n=5)
        self.assertEqual(len(knowledge), 1)
        self.assertIn("scout_1", knowledge[0].consumed_by)

    def test_signal_threat(self):
        """Test major signaling a threat."""
        from swarm.colony.caste import Caste
        self.colony.caste_registry.assign("major_1", Caste.MAJOR)

        initial_threat = self.colony.threat_level
        self.colony.signal_threat("major_1", "bad_source", severity=0.8)

        # Threat level should have increased
        self.assertGreater(self.colony.threat_level, initial_threat)

        # Alarm pheromone should be deposited
        from swarm.colony.pheromone import PheromoneType
        alarm_conc = self.colony.pheromone_field.get_concentration(
            "bad_source", PheromoneType.ALARM
        )
        self.assertGreater(alarm_conc, 0)

    def test_tend_garden(self):
        """Test minim garden tending."""
        # Add contamination
        self.colony.fungus_garden.introduce_contamination(0.3)

        report = self.colony.tend_garden("minim_1")
        self.assertIn("contamination_removed", report)
        self.assertGreater(report["contamination_removed"], 0)

    def test_scout_trail(self):
        """Test scout discovering topic connections."""
        seg_id = self.colony.scout_trail(
            "scout_1", "climate_change", "renewable_energy", quality=0.7
        )
        self.assertIsNotNone(seg_id)

        # Trail should exist
        neighbors = self.colony.trail_network.get_neighbors("climate_change")
        self.assertGreater(len(neighbors), 0)

    def test_follow_trail(self):
        """Test agent following trail network."""
        # Create some trails
        self.colony.trail_network.reinforce("start", "topic_A", "agent_1", amount=0.8)
        self.colony.trail_network.reinforce("start", "topic_B", "agent_2", amount=0.2)

        # Follow trail
        destination = self.colony.follow_trail("agent_3", "start")
        self.assertIsNotNone(destination)
        self.assertIn(destination, ["topic_A", "topic_B"])

    def test_listen_for_recruitment(self):
        """Test listening for recruitment signals with caste sensitivity."""
        from swarm.colony.caste import Caste

        self.colony.caste_registry.assign("minim_1", Caste.MINIM)
        self.colony.stridulation.stridulate("media_1", "sig_001", 0.8, "good_topic")

        signals = self.colony.listen_for_recruitment("minim_1")
        self.assertGreater(len(signals), 0)

    def test_assess_needs(self):
        """Test colony needs assessment."""
        from swarm.colony.caste import Caste

        # Sick garden should need more minims
        self.colony.fungus_garden.introduce_contamination(0.5)
        needs = self.colony.assess_needs()
        self.assertIn(Caste.MINIM, needs)

    def test_colony_lifecycle_phase_transition(self):
        """Test that colony phases transition correctly."""
        from swarm.colony.colony import ColonyPhase
        from swarm.colony.caste import Caste

        # Initially founding
        self.assertEqual(self.colony.phase, ColonyPhase.FOUNDING)

        # Add agents and some knowledge
        self.colony.caste_registry.assign("queen_1", Caste.QUEEN)
        self.colony.caste_registry.assign("minim_1", Caste.MINIM)

        # Process a leaf to get gongylidia
        leaf_id = self.colony.fungus_garden.deposit_leaf(
            "Test content with good information.", "test", "media_1"
        )
        self.colony.fungus_garden.process_leaf(leaf_id, "minim_1", "Knowledge", 0.8)

        # Tick to trigger phase check
        self.colony.tick()
        self.assertEqual(self.colony.phase, ColonyPhase.GROWTH)

    def test_get_health(self):
        """Test comprehensive health report."""
        health = self.colony.get_health()

        self.assertIn("phase", health)
        self.assertIn("pheromone_field", health)
        self.assertIn("caste_registry", health)
        self.assertIn("fungus_garden", health)
        self.assertIn("trail_network", health)
        self.assertIn("stridulation", health)

    def test_get_agent_context(self):
        """Test getting environmental context for an agent."""
        from swarm.colony.caste import Caste
        self.colony.caste_registry.assign("media_1", Caste.MEDIA)

        context = self.colony.get_agent_context("media_1")
        self.assertEqual(context["agent_id"], "media_1")
        self.assertEqual(context["caste"], "media")
        self.assertIn("colony_phase", context)
        self.assertIn("garden_health", context)

    def test_get_summary(self):
        """Test human-readable summary."""
        summary = self.colony.get_summary()
        self.assertIn("Leaf Cutter Ant Colony", summary)
        self.assertIn("Phase:", summary)
        self.assertIn("Fungus Garden:", summary)

    def test_threat_decay(self):
        """Test that threat level decays naturally over ticks."""
        self.colony.threat_level = 0.8
        for _ in range(20):
            self.colony.tick()
        self.assertLess(self.colony.threat_level, 0.8)

    def test_full_workflow(self):
        """Test complete colony workflow: forage -> process -> consume -> recruit."""
        from swarm.colony.caste import Caste

        # Setup colony
        self.colony.caste_registry.assign("queen_1", Caste.QUEEN)
        self.colony.caste_registry.assign("minim_1", Caste.MINIM)
        self.colony.caste_registry.assign("media_1", Caste.MEDIA)
        self.colony.caste_registry.assign("major_1", Caste.MAJOR)

        # 1. Media forages (deposits leaf)
        leaf_id = self.colony.forage_deposit(
            "media_1",
            "Arctic sea ice extent has declined 13% per decade since 1979.",
            "nsidc",
            keywords=["arctic", "ice", "decline"],
            quality=0.85,
        )
        self.assertIsNotNone(leaf_id)

        # 2. Minim processes leaf into knowledge
        gong_id = self.colony.process_knowledge(
            "minim_1", leaf_id,
            "Arctic ice declining 13%/decade - accelerating loss trend",
            nutrient_value=0.8,
        )
        self.assertIsNotNone(gong_id)

        # 3. Scout consumes knowledge
        knowledge = self.colony.consume_knowledge("scout_1", n=5)
        self.assertEqual(len(knowledge), 1)

        # 4. Scout creates trail connection
        self.colony.scout_trail("scout_1", "arctic", "climate_change", quality=0.7)

        # 5. Major detects quality issue (threat)
        self.colony.signal_threat("major_1", "arctic", severity=0.3)

        # 6. Colony tick (evaporation, phase updates)
        cycle = self.colony.tick()

        # Verify final state
        health = self.colony.get_health()
        self.assertGreater(health["trail_network"]["total_trails"], 0)
        self.assertGreater(health["fungus_garden"]["gongylidia"]["total"], 0)
        self.assertGreater(health["pheromone_field"]["total_active"], 0)


class TestColonyIntegration(unittest.TestCase):
    """Integration tests verifying subsystem interactions."""

    def test_pheromone_trail_correlation(self):
        """Test that foraging deposits both pheromone and trail."""
        from swarm.colony.colony import Colony
        from swarm.colony.pheromone import PheromoneType

        colony = Colony(task_prompt="test")

        colony.forage_deposit(
            "media_1", "Content with enough length for deposit.",
            "source_A", ["topic"], quality=0.7,
        )

        # Both trail pheromone and trail segment should exist
        trail_phero = colony.pheromone_field.get_concentration(
            "source_A", PheromoneType.TRAIL
        )
        trail_segments = colony.trail_network.get_stats()["total_trails"]

        self.assertGreater(trail_phero, 0)
        self.assertGreater(trail_segments, 0)

    def test_stridulation_recruitment_flow(self):
        """Test that high-quality foraging triggers stridulation and recruitment."""
        from swarm.colony.colony import Colony
        from swarm.colony.caste import Caste

        colony = Colony(task_prompt="test")
        colony.caste_registry.assign("media_1", Caste.MEDIA)
        colony.caste_registry.assign("minim_1", Caste.MINIM)

        # High quality deposit triggers stridulation
        colony.forage_deposit(
            "media_1", "Very high quality research finding.",
            "nature_journal", ["research"], quality=0.9,
        )

        # Minim should hear the stridulation
        signals = colony.listen_for_recruitment("minim_1", context="nature_journal")
        self.assertGreater(len(signals), 0)

    def test_contamination_threat_cycle(self):
        """Test contamination -> threat -> garden health cycle."""
        from swarm.colony.colony import Colony

        colony = Colony(task_prompt="test")

        # Signal severe threat
        colony.signal_threat("major_1", "bad_source", severity=0.9)

        # Garden should be contaminated
        self.assertGreater(colony.fungus_garden.contamination_level, 0)

        # Tending should help
        colony.tend_garden("minim_1")
        initial_contamination = colony.fungus_garden.contamination_level

        colony.tend_garden("minim_2")
        self.assertLessEqual(
            colony.fungus_garden.contamination_level, initial_contamination
        )

    def test_multi_tick_evaporation(self):
        """Test that all subsystems evaporate correctly over multiple ticks."""
        from swarm.colony.colony import Colony
        from swarm.colony.pheromone import PheromoneType

        colony = Colony(task_prompt="test")

        # Deposit various signals
        colony.pheromone_field.deposit(
            PheromoneType.RECRUITMENT, "ctx", 0.3, "agent_1"
        )
        colony.trail_network.reinforce("A", "B", "agent_1", amount=0.05)
        colony.stridulation.stridulate("media_1", "sig_001", 0.5, "ctx")

        # Many ticks should clean everything up
        for _ in range(100):
            colony.tick()

        stats = colony.get_health()
        self.assertEqual(stats["pheromone_field"]["total_active"], 0)
        self.assertEqual(stats["trail_network"]["total_trails"], 0)
        self.assertEqual(stats["stridulation"]["active_signals"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
