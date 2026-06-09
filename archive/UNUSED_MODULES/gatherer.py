"""Gatherer agents - validate insights with external knowledge sources."""

import asyncio
import random
from typing import Optional, List
from ..core.signal_store import SignalStore, Signal
from ..llm.simple_llm import SimpleLLM


class Gatherer:
    """Gatherer agent - validates insights using external knowledge sources.

    Gatherers bridge internal pattern-finding and external validation.
    They search for insights lacking evidence, query external sources
    (web search, Wikipedia, etc.), and deposit EVIDENCE signals.
    """

    def __init__(self, agent_id: str, mcp_client=None):
        """Initialize gatherer.

        Args:
            agent_id: Unique agent ID
            mcp_client: MCP client for external knowledge access (optional)
        """
        self.agent_id = agent_id
        self.mcp_client = mcp_client
        self.active = True
        self.actions_taken = 0

    async def run(self, signal_store: SignalStore, llm: SimpleLLM,
                  batch_size: int = 3, max_actions: int = 100):
        """Run gatherer behavior loop.

        Args:
            signal_store: Shared signal store
            llm: Language model
            batch_size: Number of insights to validate per cycle
            max_actions: Maximum actions before stopping
        """
        while self.active and self.actions_taken < max_actions:
            await self.validate_insights(signal_store, llm, batch_size)
            self.actions_taken += 1
            await asyncio.sleep(random.uniform(0.5, 1.5))

    async def validate_insights(self, signal_store: SignalStore, llm: SimpleLLM,
                               batch_size: int):
        """Find unvalidated insights and gather evidence.

        Args:
            signal_store: Signal store
            llm: Language model
            batch_size: How many to process this cycle
        """
        # Get insights lacking evidence
        unvalidated = signal_store.get_unvalidated_signals(
            signal_type="INSIGHT",
            min_evidence=2
        )

        if not unvalidated:
            return

        # Process a small batch to avoid rate limiting
        batch = unvalidated[:batch_size]

        print(f"[GATHERER] {self.agent_id} validating {len(batch)} insights")

        for insight in batch:
            await self.gather_evidence(insight, signal_store, llm)

    async def gather_evidence(self, insight: Signal, signal_store: SignalStore,
                             llm: SimpleLLM):
        """Gather external evidence for an insight.

        Args:
            insight: Insight signal to validate
            signal_store: Signal store
            llm: Language model
        """
        # Extract searchable claim from insight
        query = await self.extract_query(insight, llm)

        if not query:
            print(f"[GATHERER] {self.agent_id} couldn't extract query from {insight.id}")
            return

        print(f"[GATHERER] {self.agent_id} searching for: '{query[:60]}...'")

        # Search external sources
        results = await self.search_external(query)

        if not results:
            print(f"[GATHERER] {self.agent_id} no evidence found for {insight.id}")
            return

        # Deposit evidence signals
        for i, result in enumerate(results[:3]):  # Limit to top 3 results
            evidence_content = f"{result['summary']}\n\nSource: {result['source']}"

            signal_id = signal_store.deposit(
                signal_type="EVIDENCE",
                content=evidence_content,
                strength=0.6,  # External evidence starts with moderate strength
                depositor=self.agent_id,
                parent=insight.id,
                metadata={
                    'source_url': result.get('url', ''),
                    'source_type': result.get('type', 'web'),
                    'relevance': result.get('relevance', 0.5),
                    'query': query
                }
            )

            if signal_id:
                print(f"[GATHERER] {self.agent_id} found evidence: {signal_id}")
                # Amplify the insight now that it has evidence
                signal_store.amplify(insight.id, factor=1.2)

    async def extract_query(self, insight: Signal, llm: SimpleLLM) -> Optional[str]:
        """Extract a searchable query from an insight.

        Args:
            insight: Insight signal
            llm: Language model

        Returns:
            Search query string or None
        """
        prompt = f"""Extract a concise search query from this insight that could be used to find supporting evidence.

Insight:
{insight.content}

Task: Create a 5-10 word search query that captures the key factual claim.

Examples:
- Insight: "The studies show that urban green spaces reduce stress levels by 15-20%"
  Query: "urban green spaces reduce stress research studies"

- Insight: "Multiple observations indicate climate models underestimate Arctic warming rates"
  Query: "Arctic warming rates climate models underestimate"

Your search query (5-10 words):"""

        try:
            result = await llm.generate(prompt, max_tokens=30, temperature=0.5)
            if result:
                query = result.strip().strip('"\'')
                if len(query.split()) >= 3:
                    return query
        except Exception as e:
            print(f"[GATHERER] {self.agent_id} query extraction error: {e}")

        return None

    async def search_external(self, query: str) -> List[dict]:
        """Search external knowledge sources.

        Args:
            query: Search query

        Returns:
            List of search results with summary, source, url, relevance
        """
        results = []

        if self.mcp_client:
            # Use MCP client for actual external search
            try:
                mcp_results = await self.mcp_client.search(query, max_results=3)
                results.extend(mcp_results)
            except Exception as e:
                print(f"[GATHERER] {self.agent_id} MCP search error: {e}")
        else:
            # Fallback: Simulated search (for testing without MCP)
            print(f"[GATHERER] {self.agent_id} WARNING: No MCP client, using simulated search")
            results = [
                {
                    'summary': f"Simulated search result for: {query}",
                    'source': "Simulated Source",
                    'url': "http://example.com",
                    'type': 'simulated',
                    'relevance': 0.5
                }
            ]

        return results

    def stop(self):
        """Stop the agent."""
        self.active = False
