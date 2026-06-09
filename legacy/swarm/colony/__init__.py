"""Leaf cutter ant colony coordination system.

Implements stigmergic coordination patterns inspired by Atta leaf cutter ants:

- Pheromone system: Dual volatile/persistent chemical communication
- Caste system: Specialized agent roles (Queen, Minim, Media, Major)
- Fungus garden: Shared knowledge processing substrate
- Trail network: Emergent pathways reinforced through use
- Stridulation: Quality-based vibration recruitment
- Colony coordinator: Lifecycle management and adaptive caste production

Reference: Atta leaf cutter ants use 55+ species of fungus-growing ants
with colonies numbering up to 8 million workers, demonstrating the most
complex trail construction behavior of any animal and one of only a few
species to have mastered deliberate agriculture.
"""

from .pheromone import Pheromone, PheromoneType, PheromoneField
from .caste import Caste, CasteProfile, CasteRegistry
from .fungus_garden import LeafFragment, Gongylidium, FungusGarden
from .trail_network import TrailSegment, TrailNetwork
from .stridulation import StridulationSignal, StridulationProtocol
from .colony import Colony

__all__ = [
    'Pheromone', 'PheromoneType', 'PheromoneField',
    'Caste', 'CasteProfile', 'CasteRegistry',
    'LeafFragment', 'Gongylidium', 'FungusGarden',
    'TrailSegment', 'TrailNetwork',
    'StridulationSignal', 'StridulationProtocol',
    'Colony',
]
