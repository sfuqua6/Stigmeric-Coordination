"""Synthesizer agent - consolidates swarm insights into coherent final answer."""

import asyncio
from swarm.core.signal_store import SignalStore
from swarm.llm.simple_llm import SimpleLLM
from swarm.core.signal_types import SignalType, CRITIQUE, OBJECTION


class Synthesizer:
    """Synthesizer agent that consolidates swarm signals into final answer.

    Runs at the end of swarm execution to produce a direct, coherent response
    that addresses the original question by synthesizing insights from all
    signal types.
    """

    def __init__(self, agent_id: str, task_prompt: str, task_config=None):
        """Initialize synthesizer.

        Args:
            agent_id: Unique agent identifier
            task_prompt: Original task/question to answer
            task_config: Optional task configuration for display names
        """
        self.agent_id = agent_id
        self.task_prompt = task_prompt
        self.task_config = task_config

    def _make_synthesis_prompt(self, signal_context: dict, signal_store: SignalStore) -> str:
        """Create synthesis prompt from top signals with full argument graph.

        Args:
            signal_context: Dictionary with top signals by type
            signal_store: Signal store to get full discourse threads

        Returns:
            Synthesis prompt
        """
        prompt = f"""You are a synthesis agent tasked with providing a direct, coherent answer to this question:

"{self.task_prompt}"

The swarm has generated and debated the following ideas. For each idea, you can see:
- The initial proposal
- Supporting evidence/implementation details
- Critical evaluations
- Objections and counter-arguments

This represents the full discourse of the swarm:

"""

        # Build full argument graph for each top signal
        # Use display names from task_config if available for prettier output
        for signal_type, signals in signal_context.items():
            if signals:
                # Get display name (e.g., "Solution" instead of "INITIAL")
                display_name = signal_type
                if hasattr(self, 'task_config') and self.task_config and hasattr(self.task_config, 'display_names'):
                    display_name = self.task_config.display_names.get(signal_type, signal_type)

                prompt += f"\n=== {display_name} SIGNALS (with full discourse) ===\n"
                for i, signal in enumerate(signals[:2], 1):  # Top 2 with full context
                    prompt += f"\n{i}. {display_name}: {signal.content}\n"
                    prompt += f"   [Strength: {signal.strength:.2f}]\n"

                    # Get children (evidence, implementations, etc.)
                    children = signal_store.get_children(signal.id)
                    if children:
                        prompt += f"   Support ({len(children)}): "
                        # OPTIMIZATION: Show fewer children with shorter content (reduce prompt size)
                        for child in children[:2]:  # Top 2 (reduced from 3)
                            prompt += f"{child.content[:60]}... "  # 60 chars (reduced from 100)
                        prompt += "\n"

                    # Get critiques/objections targeting this signal (use universal types)
                    # PERFORMANCE: Use indexed retrieval O(k) instead of O(n) scan
                    children_of_signal = signal_store.get_signals_by_parent(signal.id)
                    critiques = [s for s in children_of_signal
                                if SignalType.is_critique_type(s.type) or SignalType.is_objection_type(s.type)]
                    if critiques:
                        prompt += f"   Critiques ({len(critiques)}): "
                        # OPTIMIZATION: Shorter content for faster processing
                        for crit in critiques[:2]:
                            prompt += f"{crit.content[:60]}... "  # 60 chars (reduced from 100)
                        prompt += "\n"

        # Closing instruction — shaped by planner directive if present
        frame = getattr(self.task_config, 'task_frame', None) if hasattr(self, 'task_config') else None
        synth_directive = getattr(frame, 'synthesizer_directive', '') if frame else ''
        decomposition = getattr(frame, 'decomposition', []) if frame else []

        if synth_directive:
            closing = f"\n\n{synth_directive}"
        else:
            closing = (
                "\n\nSynthesize these insights into a clear answer. Address the question directly, "
                "integrate multiple perspectives, acknowledge complexity and tradeoffs, support "
                "claims with evidence, balance strengths and criticisms, and provide actionable takeaways."
            )

        if decomposition:
            sub_list = "\n".join(f"  - {s}" for s in decomposition)
            closing += f"\n\nEnsure the answer addresses each of these sub-problems:\n{sub_list}"

        output_format = getattr(frame, 'output_format', 'prose') if frame else 'prose'
        if output_format == 'code':
            closing += "\n\nYour synthesis (produce a complete, runnable code block):"
        elif output_format == 'list':
            closing += "\n\nYour synthesis (produce a numbered list):"
        else:
            closing += "\n\nYour synthesis (4-8 sentences):"

        prompt += closing
        return prompt

    async def synthesize(self, signal_store: SignalStore, llm: SimpleLLM,
                        signal_types: dict, task_config=None, temperature: float = 0.6) -> str:
        """Generate synthesis of swarm insights.

        Args:
            signal_store: Signal store with swarm outputs
            llm: Language model
            signal_types: Dictionary mapping roles to signal type names (universal types)
            task_config: Optional task configuration with display_names for pretty output
            temperature: Generation temperature (lower = more focused)

        Returns:
            Synthesized answer or None if generation fails
        """
        # Gather top signals from each type
        # NOTE: Using cluster sampling instead of top-N for better semantic diversity
        signal_context = {}
        total_signals = 0
        for role, signal_type in signal_types.items():
            # Try cluster sampling first (finds semantically related signals)
            cluster_signals = signal_store.sample_cluster(signal_type, size=3, similarity_threshold=0.6)

            # Fallback to top-N if cluster sampling returns nothing
            if not cluster_signals:
                cluster_signals = signal_store.get_top_signals(signal_type, n=3)

            if cluster_signals:
                signal_context[signal_type] = cluster_signals
                total_signals += len(cluster_signals)

        # If no signals at all, return early with error message
        if total_signals == 0:
            print(f"[{self.agent_id}] No signals available for synthesis")
            return None

        # Create synthesis prompt with full argument graph
        prompt = self._make_synthesis_prompt(signal_context, signal_store)

        # Generate synthesis with retries
        print(f"\n[{self.agent_id}] Synthesizing final answer...")

        # Determine max_tokens from intake profile or use high-quality default
        max_tokens = 600  # High-quality default
        if self.task_config and hasattr(self.task_config, 'intake_profile'):
            max_tokens = self.task_config.intake_profile.synthesizer_tokens

        print(f"[{self.agent_id}] Using {max_tokens} token limit for synthesis")

        # Try 1: Normal temperature with increased tokens
        try:
            result = await llm.generate(prompt, max_tokens=max_tokens, temperature=temperature)

            if result and len(result.strip()) > 15:  # Slightly relaxed from 20
                synthesis = result.strip()
                print(f"[{self.agent_id}] Generated synthesis ({len(synthesis)} chars)")
                return synthesis
            else:
                print(f"[{self.agent_id}] Synthesis too short, retrying with higher temperature...")

        except Exception as e:
            print(f"[{self.agent_id}] Error during synthesis (attempt 1): {e}")

        # Try 2: Higher temperature for more creativity
        try:
            result = await llm.generate(prompt, max_tokens=max_tokens, temperature=min(0.9, temperature + 0.3))

            if result and len(result.strip()) > 15:
                synthesis = result.strip()
                print(f"[{self.agent_id}] Generated synthesis on retry ({len(synthesis)} chars)")
                return synthesis
            else:
                print(f"[{self.agent_id}] Synthesis retry failed, trying fallback approach...")

        except Exception as e:
            print(f"[{self.agent_id}] Error during synthesis (attempt 2): {e}")

        # Try 3: Simplified prompt - just concatenate top signals
        try:
            simplified_prompt = f"""Provide a brief answer to: {self.task_prompt}

Based on these key points:
"""
            for signal_type, signals in signal_context.items():
                for signal in signals[:2]:  # Top 2 per type
                    simplified_prompt += f"\n- {signal.content[:150]}"

            simplified_prompt += "\n\nBrief answer:"

            # Use 70% of max_tokens for fallback
            fallback_tokens = int(max_tokens * 0.7)
            result = await llm.generate(simplified_prompt, max_tokens=fallback_tokens, temperature=0.7)

            if result and len(result.strip()) > 15:
                synthesis = result.strip()
                print(f"[{self.agent_id}] Generated synthesis with fallback approach ({len(synthesis)} chars)")
                return synthesis

        except Exception as e:
            print(f"[{self.agent_id}] Error during synthesis (attempt 3): {e}")

        # All attempts failed
        print(f"[{self.agent_id}] All synthesis attempts failed")
        return None
