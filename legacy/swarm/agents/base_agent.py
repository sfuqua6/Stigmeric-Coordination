"""Base agent class for unified architecture.

All stigmergic swarm agents inherit from this base class to ensure:
1. Consistent constructor signatures
2. Unified run() method interface
3. Shared information retrieval capability
4. Simplified architecture
"""

import asyncio
import random
from typing import Optional, Any
from ..core.signal_store import SignalStore
from ..llm.simple_llm import SimpleLLM

# DocumentRetriever is deprecated and moved to archive
# Use DynamicRetriever for active retrieval needs
try:
    from ..retrieval.dynamic_retriever import DynamicRetriever
    RetrieverType = DynamicRetriever
except ImportError:
    # Fallback to Any if DynamicRetriever not available
    RetrieverType = Any


class BaseAgent:
    """Base class for all stigmergic swarm agents.

    Provides unified interface and shared functionality:
    - Consistent constructor: (agent_id, task_config, retriever)
    - Unified run() method: (signal_store, llm, max_actions)
    - Information retrieval capability
    - Standard lifecycle management
    """

    def __init__(self, agent_id: str, task_config=None, retriever: Optional[RetrieverType] = None):
        """Initialize agent with unified interface.

        Args:
            agent_id: Unique agent identifier
            task_config: Task configuration (signal types, prompts)
            retriever: Optional retriever for information search (DynamicRetriever)
        """
        self.agent_id = agent_id
        self.task_config = task_config
        self.retriever = retriever
        self.active = True
        self.actions_taken = 0

        # Extract task-specific configuration
        if task_config:
            self.task_prompt = task_config.task_prompt
            self.signal_types = task_config.signal_types
        else:
            self.task_prompt = None
            self.signal_types = {}

    async def run(self, signal_store: SignalStore, llm: SimpleLLM, max_actions: int = 100):
        """Run agent behavior loop.

        Args:
            signal_store: Shared signal store
            llm: Language model for generation
            max_actions: Maximum actions before stopping
        """
        raise NotImplementedError("Subclasses must implement run()")

    def retrieve_information(self, keywords: list[str], max_results: int = 3) -> list[str]:
        """Retrieve information using keyword search.

        Args:
            keywords: Keywords to search for
            max_results: Maximum number of results

        Returns:
            List of relevant text snippets
        """
        if not self.retriever:
            return []

        try:
            results = self.retriever.search(keywords, max_results)
            if results and self.actions_taken < 2:
                print(f"[{self.agent_id}] Retrieved {len(results)} information snippets")
            return results
        except Exception as e:
            print(f"[{self.agent_id}] Retrieval error: {e}")
            return []
    def stop(self):
        """Stop the agent."""
        self.active = False

    def __repr__(self):
        return f"{self.__class__.__name__}('{self.agent_id}')"
