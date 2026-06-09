"""Agent modules for swarm system."""

from .scout import Scout
from .forager import Forager
from .critic import Critic
from .hater import Hater
from .synthesizer import Synthesizer

__all__ = ['Scout', 'Forager', 'Critic', 'Hater', 'Synthesizer']
