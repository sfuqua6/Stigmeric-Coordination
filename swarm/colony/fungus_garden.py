"""Fungus garden - the colony's shared knowledge processing substrate.

Leaf cutter ants are one of only a few animals that practice deliberate agriculture,
predating human farming by 60 million years. The fungus they cultivate (Leucoagaricus
gongylaphorus) exists nowhere else on Earth - it is entirely dependent on the ants,
and the ants entirely dependent on it. A perfect obligate symbiosis.

The process:
1. Media foragers cut and carry leaf fragments back to the nest
2. Smaller workers (minims) chew the leaves into a mulch
3. The mulch is added to the fungus garden as substrate
4. The fungus breaks down cellulose into digestible nutrients (glucose, amino acids)
5. The fungus produces gongylidia - swollen hyphal tips packed with metabolites
6. Ants harvest and eat the gongylidia as their primary food source
7. Minims tend the garden: removing contamination, secreting antimicrobial fluids

In our system:
- Leaf fragments = raw research inputs (unprocessed text, data, observations)
- Fungus mycelium = the processing substrate (cross-referencing, synthesis)
- Gongylidia = distilled knowledge nuggets ready for consumption by agents
- Garden health = overall knowledge quality (contamination = misinformation)
- Tending = quality maintenance (removing bad data, refreshing stale knowledge)
"""

import time
import random
from threading import Lock
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class LeafFragment:
    """Raw input material deposited by foragers.

    In the real colony, leaves aren't food - they're substrate for the fungus.
    Similarly, raw research fragments aren't directly usable knowledge - they
    need to be processed through the fungus garden's pipeline.
    """
    id: str
    content: str                  # Raw text/data
    source: str                   # Where it came from (URL, document, agent)
    deposited_by: str             # Agent who brought it
    timestamp: float
    keywords: List[str] = field(default_factory=list)
    quality_score: float = 0.5    # Initial quality assessment
    processed: bool = False       # Whether mycelium has processed this
    processing_agent: Optional[str] = None  # Which minim processed it


@dataclass
class Gongylidium:
    """Distilled knowledge nugget - the output of fungal processing.

    Gongylidia are swollen hyphal cells that package metabolites for ant
    consumption. In our system, they represent processed, cross-referenced,
    validated knowledge that agents can directly use.
    """
    id: str
    content: str                  # Distilled knowledge
    source_fragments: List[str]   # Which leaf fragments it was derived from
    nutrient_value: float         # How valuable (0.0-1.0)
    freshness: float              # Decays over time (1.0 = fresh, 0.0 = stale)
    produced_at: float            # Timestamp
    consumed_by: List[str] = field(default_factory=list)  # Agents that used this
    cross_references: List[str] = field(default_factory=list)  # Related gongylidia


# Garden health thresholds
CONTAMINATION_THRESHOLD = 0.3     # Above this, garden is "sick"
STALE_THRESHOLD = 0.2             # Freshness below this = stale
FRESHNESS_DECAY_RATE = 0.03       # Per-tick freshness reduction


class FungusGarden:
    """The colony's shared knowledge substrate.

    The garden maintains three areas:
    1. Leaf pile: Unprocessed inputs waiting for minim processing
    2. Mycelium (active garden): Currently being processed/cross-referenced
    3. Gongylidia store: Processed knowledge nuggets ready for consumption

    Garden health is affected by:
    - Contamination (low-quality or contradictory information)
    - Staleness (old information not refreshed)
    - Overloading (too many unprocessed leaves)
    """

    def __init__(self, max_leaves: int = 500, max_gongylidia: int = 200):
        """Initialize the fungus garden.

        Args:
            max_leaves: Maximum unprocessed leaf fragments
            max_gongylidia: Maximum stored gongylidia
        """
        self._lock = Lock()
        self.leaf_pile: Dict[str, LeafFragment] = {}
        self.gongylidia: Dict[str, Gongylidium] = {}
        self.max_leaves = max_leaves
        self.max_gongylidia = max_gongylidia
        self._next_leaf_id = 0
        self._next_gong_id = 0

        # Garden health metrics
        self.contamination_level: float = 0.0   # 0.0 = clean, 1.0 = heavily contaminated
        self.total_processed: int = 0
        self.total_consumed: int = 0

    def deposit_leaf(self, content: str, source: str, deposited_by: str,
                     keywords: Optional[List[str]] = None,
                     quality_score: float = 0.5) -> Optional[str]:
        """Deposit a leaf fragment (raw input) into the garden.

        Media foragers bring leaves to the nest. If the garden is full,
        the lowest quality leaves are composted (removed) first.

        Args:
            content: Raw text/data content
            source: Origin of the material
            deposited_by: Agent ID of the forager
            keywords: Associated keywords
            quality_score: Initial quality assessment (0.0-1.0)

        Returns:
            Leaf ID if deposited, None if rejected
        """
        with self._lock:
            # Reject empty or very short content
            if not content or len(content.strip()) < 10:
                return None

            # If at capacity, compost lowest quality unprocessed leaf
            if len(self.leaf_pile) >= self.max_leaves:
                self._compost_weakest_locked()

            leaf_id = f"LEAF_{self._next_leaf_id:04d}"
            self._next_leaf_id += 1

            leaf = LeafFragment(
                id=leaf_id,
                content=content.strip(),
                source=source,
                deposited_by=deposited_by,
                timestamp=time.time(),
                keywords=keywords or [],
                quality_score=max(0.0, min(1.0, quality_score)),
            )

            self.leaf_pile[leaf_id] = leaf
            return leaf_id

    def _compost_weakest_locked(self):
        """Remove the lowest quality unprocessed leaf (lock must be held)."""
        unprocessed = [l for l in self.leaf_pile.values() if not l.processed]
        if unprocessed:
            weakest = min(unprocessed, key=lambda l: l.quality_score)
            del self.leaf_pile[weakest.id]

    def get_unprocessed_leaves(self, n: int = 5) -> List[LeafFragment]:
        """Get unprocessed leaf fragments for minim processing.

        Prioritizes higher quality leaves for processing first, similar to
        how ants select better leaf material for the garden.

        Args:
            n: Maximum number of leaves to return

        Returns:
            List of unprocessed leaves, sorted by quality (best first)
        """
        with self._lock:
            unprocessed = [l for l in self.leaf_pile.values() if not l.processed]
            unprocessed.sort(key=lambda l: l.quality_score, reverse=True)
            return unprocessed[:n]

    def process_leaf(self, leaf_id: str, processor_id: str,
                     distilled_content: str,
                     nutrient_value: float = 0.6,
                     cross_ref_ids: Optional[List[str]] = None) -> Optional[str]:
        """Process a leaf fragment into a gongylidium (knowledge nugget).

        This is the core "fungal digestion" step where raw input is transformed
        into usable knowledge. In real ants, the fungus breaks down cellulose
        (which ants can't digest) into glucose and amino acids (which they can).

        Args:
            leaf_id: Leaf to process
            processor_id: Minim agent doing the processing
            distilled_content: The processed/distilled knowledge
            nutrient_value: Quality of the output (0.0-1.0)
            cross_ref_ids: IDs of related gongylidia for cross-referencing

        Returns:
            Gongylidium ID if successful, None if leaf not found
        """
        with self._lock:
            leaf = self.leaf_pile.get(leaf_id)
            if not leaf or leaf.processed:
                return None

            # Mark leaf as processed
            leaf.processed = True
            leaf.processing_agent = processor_id

            # Produce gongylidium
            gong_id = f"GONG_{self._next_gong_id:04d}"
            self._next_gong_id += 1

            # If at capacity, remove stalest gongylidium
            if len(self.gongylidia) >= self.max_gongylidia:
                self._remove_stalest_locked()

            gong = Gongylidium(
                id=gong_id,
                content=distilled_content,
                source_fragments=[leaf_id],
                nutrient_value=max(0.0, min(1.0, nutrient_value)),
                freshness=1.0,  # Brand new = maximum freshness
                produced_at=time.time(),
                cross_references=cross_ref_ids or [],
            )

            self.gongylidia[gong_id] = gong
            self.total_processed += 1
            return gong_id

    def _remove_stalest_locked(self):
        """Remove the stalest gongylidium (lock must be held)."""
        if self.gongylidia:
            stalest = min(self.gongylidia.values(), key=lambda g: g.freshness)
            del self.gongylidia[stalest.id]

    def harvest(self, n: int = 5, consumer_id: Optional[str] = None) -> List[Gongylidium]:
        """Harvest gongylidia for consumption by agents.

        Agents eat gongylidia to acquire knowledge. Harvesting tracks which
        agents consumed which knowledge for provenance.

        Args:
            n: Maximum number of gongylidia to harvest
            consumer_id: Agent consuming (for tracking)

        Returns:
            List of gongylidia, sorted by nutrient_value * freshness
        """
        with self._lock:
            available = list(self.gongylidia.values())
            # Sort by combined value: nutrient_value weighted by freshness
            available.sort(
                key=lambda g: g.nutrient_value * g.freshness,
                reverse=True
            )

            harvested = available[:n]

            if consumer_id:
                for g in harvested:
                    if consumer_id not in g.consumed_by:
                        g.consumed_by.append(consumer_id)
                        self.total_consumed += 1

            return harvested

    def harvest_by_keywords(self, keywords: List[str], n: int = 5,
                           consumer_id: Optional[str] = None) -> List[Gongylidium]:
        """Harvest gongylidia matching specific keywords.

        Args:
            keywords: Keywords to match against source fragment keywords
            n: Maximum results
            consumer_id: Agent consuming

        Returns:
            Matching gongylidia sorted by relevance
        """
        with self._lock:
            scored = []
            keyword_set = set(k.lower() for k in keywords)

            for gong in self.gongylidia.values():
                # Check source fragment keywords
                score = 0.0
                for leaf_id in gong.source_fragments:
                    leaf = self.leaf_pile.get(leaf_id)
                    if leaf:
                        leaf_keywords = set(k.lower() for k in leaf.keywords)
                        overlap = len(keyword_set & leaf_keywords)
                        score += overlap

                # Also check content for keyword presence
                content_lower = gong.content.lower()
                for kw in keyword_set:
                    if kw in content_lower:
                        score += 0.5

                if score > 0:
                    scored.append((gong, score * gong.freshness * gong.nutrient_value))

            scored.sort(key=lambda x: x[1], reverse=True)
            harvested = [g for g, _ in scored[:n]]

            if consumer_id:
                for g in harvested:
                    if consumer_id not in g.consumed_by:
                        g.consumed_by.append(consumer_id)
                        self.total_consumed += 1

            return harvested

    def tend(self, tender_id: str) -> dict:
        """Tend the garden - maintenance actions by minim agents.

        Minims perform several maintenance tasks:
        1. Remove contaminated material (low quality fragments)
        2. Refresh freshness of recently validated gongylidia
        3. Cross-reference related gongylidia
        4. Report garden health status

        In real Atta colonies, minims secrete antimicrobial compounds from
        their metapleural glands to keep the garden free of parasitic fungi.

        Args:
            tender_id: Minim agent performing maintenance

        Returns:
            Maintenance report with actions taken
        """
        with self._lock:
            report = {
                "tender": tender_id,
                "leaves_composted": 0,
                "gongylidia_refreshed": 0,
                "contamination_removed": 0.0,
            }

            # 1. Compost low-quality unprocessed leaves
            low_quality = [l for l in self.leaf_pile.values()
                          if not l.processed and l.quality_score < 0.2]
            for leaf in low_quality:
                del self.leaf_pile[leaf.id]
                report["leaves_composted"] += 1

            # 2. Remove stale gongylidia (freshness below threshold)
            stale = [g for g in self.gongylidia.values()
                    if g.freshness < STALE_THRESHOLD]
            for gong in stale:
                del self.gongylidia[gong.id]

            # 3. Reduce contamination level (antimicrobial secretion)
            if self.contamination_level > 0:
                reduction = min(0.1, self.contamination_level)
                self.contamination_level -= reduction
                report["contamination_removed"] = reduction

            # 4. Refresh high-value gongylidia
            for gong in self.gongylidia.values():
                if gong.nutrient_value > 0.7 and gong.freshness < 0.8:
                    gong.freshness = min(1.0, gong.freshness + 0.1)
                    report["gongylidia_refreshed"] += 1

            return report

    def decay(self) -> int:
        """Apply freshness decay to all gongylidia.

        Knowledge becomes stale over time if not refreshed through
        validation or new evidence.

        Returns:
            Number of gongylidia that became stale and were removed
        """
        with self._lock:
            removed = 0
            to_remove = []

            for gong in self.gongylidia.values():
                gong.freshness *= (1.0 - FRESHNESS_DECAY_RATE)
                if gong.freshness < STALE_THRESHOLD:
                    to_remove.append(gong.id)

            for gong_id in to_remove:
                del self.gongylidia[gong_id]
                removed += 1

            return removed

    def introduce_contamination(self, amount: float, source: str = "unknown"):
        """Introduce contamination to the garden.

        In real Atta colonies, parasitic fungi (especially Escovopsis) can
        devastate the garden. Minims must constantly patrol and remove it.

        Contamination in our system represents misinformation, contradictions,
        or low-quality knowledge that degrades the garden's output.

        Args:
            amount: Contamination amount (0.0-1.0)
            source: Source of contamination
        """
        with self._lock:
            self.contamination_level = min(1.0, self.contamination_level + amount)

    def _calculate_health(self) -> float:
        """Calculate health score (internal, no lock - caller must hold lock).

        Returns:
            Health score 0.0-1.0
        """
        # Contamination penalty
        contamination_penalty = self.contamination_level * 0.4

        # Backlog penalty (too many unprocessed leaves)
        unprocessed_count = sum(1 for l in self.leaf_pile.values() if not l.processed)
        backlog_ratio = unprocessed_count / max(self.max_leaves, 1)
        backlog_penalty = backlog_ratio * 0.2

        # Knowledge quality bonus
        if self.gongylidia:
            avg_quality = sum(
                g.nutrient_value * g.freshness
                for g in self.gongylidia.values()
            ) / len(self.gongylidia)
        else:
            avg_quality = 0.0

        health = 0.5 + avg_quality * 0.5 - contamination_penalty - backlog_penalty
        return max(0.0, min(1.0, health))

    @property
    def health(self) -> float:
        """Calculate overall garden health.

        Factors:
        - Contamination level (negative)
        - Processing backlog (negative if too many unprocessed)
        - Average gongylidium quality and freshness (positive)

        Returns:
            Health score 0.0-1.0
        """
        with self._lock:
            return self._calculate_health()

    def get_stats(self) -> dict:
        """Get garden statistics.

        Returns:
            Dictionary with garden metrics
        """
        with self._lock:
            unprocessed = sum(1 for l in self.leaf_pile.values() if not l.processed)
            processed = sum(1 for l in self.leaf_pile.values() if l.processed)

            gong_freshness = [g.freshness for g in self.gongylidia.values()]
            gong_values = [g.nutrient_value for g in self.gongylidia.values()]

            return {
                "health": self._calculate_health(),
                "contamination_level": self.contamination_level,
                "leaf_pile": {
                    "total": len(self.leaf_pile),
                    "unprocessed": unprocessed,
                    "processed": processed,
                },
                "gongylidia": {
                    "total": len(self.gongylidia),
                    "avg_freshness": sum(gong_freshness) / len(gong_freshness) if gong_freshness else 0.0,
                    "avg_nutrient_value": sum(gong_values) / len(gong_values) if gong_values else 0.0,
                },
                "totals": {
                    "total_processed": self.total_processed,
                    "total_consumed": self.total_consumed,
                },
            }

    def clear(self):
        """Clear the entire garden."""
        with self._lock:
            self.leaf_pile.clear()
            self.gongylidia.clear()
            self.contamination_level = 0.0
            self._next_leaf_id = 0
            self._next_gong_id = 0
