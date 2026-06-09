"""Scout agents - read document sections and extract observations.

PROPER RAG INTEGRATION:
1. AdvancedRetriever performs deep research (100K+ words) BEFORE scouts run
2. Research fragments are divided among scouts (each scout gets ~N/num_scouts fragments)
3. Each scout processes its assigned fragments sequentially
4. Scouts present evidence-based observations from their research territory

This creates proper "division of labor" - scouts don't all search the same space.
"""

import asyncio
import random
from typing import Optional, List, Tuple
from ..core.signal_store import SignalStore, deposit_with_context
from ..llm.simple_llm import SimpleLLM
from ..core.logging_config import get_logger
from ..core.stage_coordinator import StagedAgent

logger = get_logger(__name__)


class Scout(StagedAgent):
    """Scout agent - explores and generates initial ideas/observations.

    Scouts generate diverse initial signals (claims, drafts, observations, solutions)
    that other agents can develop, critique, or challenge. Uses creative exploration
    with optional web search for dynamic knowledge retrieval.
    """

    def __init__(self, agent_id: str, signal_type: str = "DRAFT", task_prompt: Optional[str] = None,
                 dynamic_retriever=None, assigned_fragments=None, task_config=None):
        """Initialize scout.

        Args:
            agent_id: Unique agent ID
            signal_type: Signal type to deposit (e.g., "DRAFT", "INITIAL", "OBSERVATION")
            task_prompt: Task prompt for context (deprecated - use task_config)
            dynamic_retriever: Optional AdvancedRetriever for deep research
            assigned_fragments: Pre-assigned research fragments for this scout to intake
            task_config: Optional TaskConfig for prompt templates (composition pattern)
        """
        self.agent_id = agent_id
        self.signal_type = signal_type
        self.task_prompt = task_prompt
        self.dynamic_retriever = dynamic_retriever
        self.assigned_fragments = assigned_fragments or []  # ResearchFragments assigned to this scout
        self.fragment_index = 0  # Current position in assigned fragments
        self.cumulative_insights = []  # NEW: Track insights across fragments for stepwise synthesis
        self.task_config = task_config  # NEW: Config injection instead of monkey patching
        self.active = True
        self.actions_taken = 0

    async def run(self, signal_store: SignalStore, llm: SimpleLLM,
                  min_strength: float = 0.5, max_actions: int = 1, web_search_fn=None):
        """Run scout behavior - explore and generate initial signals.

        Args:
            signal_store: Shared signal store
            llm: Language model
            min_strength: Minimum strength to deposit signal
            max_actions: Maximum exploration actions
            web_search_fn: Optional async web search function for dynamic retrieval
        """
        # Log configuration on first run
        if self.actions_taken == 0:
            logger.info(f"{self.agent_id} exploring: signal_type={self.signal_type}, max_actions={max_actions}, min_strength={min_strength}")

        while self.active and self.actions_taken < max_actions:
            # Debug first few iterations
            if self.actions_taken < 2:
                logger.debug(f"{self.agent_id} iteration {self.actions_taken}: exploring...")

            idea = await self.explore_creative(llm, web_search_fn=web_search_fn)
            if idea:
                strength = self.assess_strength_creative(idea)
                if self.actions_taken < 2:
                    logger.debug(f"{self.agent_id} generated idea (strength={strength:.2f}, threshold={min_strength:.2f}): {idea[:60]}...")

                if strength >= min_strength:
                    signal_id = signal_store.deposit(
                        signal_type=self.signal_type,
                        content=idea,
                        strength=strength,
                        depositor=self.agent_id
                    )
                    logger.info(f"{self.agent_id} deposited {signal_id} (strength={strength:.2f})")
                else:
                    if self.actions_taken < 2:
                        logger.debug(f"{self.agent_id} idea rejected (strength {strength:.2f} < {min_strength:.2f})")
            else:
                if self.actions_taken < 2:
                    logger.debug(f"{self.agent_id} explore_creative returned None")

            self.actions_taken += 1
            # No sleep - pure stigmergic event-driven behavior

    async def explore_creative(self, llm: SimpleLLM, web_search_fn=None) -> Optional[str]:
        """Explore and generate ideas from assigned research or web search.

        **Priority 1**: If assigned research fragments, process those (proper RAG integration)
        **Priority 2**: Fall back to keyword extraction + web search (legacy behavior)

        This enables proper "division of labor" - scouts each get assigned portions
        of deep research (100K+ words) to intake and present as observations.

        Args:
            llm: Language model
            web_search_fn: Optional async web search function

        Returns:
            Generated idea or None
        """
        search_context = None

        # PRIORITY 1: Use assigned research fragments (proper RAG integration)
        if self.assigned_fragments and self.fragment_index < len(self.assigned_fragments):
            fragment = self.assigned_fragments[self.fragment_index]
            self.fragment_index += 1

            if self.actions_taken < 2:
                logger.debug(f"{self.agent_id} processing fragment {self.fragment_index}/{len(self.assigned_fragments)}")
                logger.debug(f"{self.agent_id} fragment source: {fragment.source} (rarity={fragment.rarity:.2f}, importance={fragment.importance:.2f})")

            # STEPWISE SYNTHESIS: Include previous insights for cumulative understanding
            # This allows scouts to build composite insights across fragments instead of isolated observations
            previous_context = ""
            if self.cumulative_insights:
                # Include last 2 insights for context (balance between continuity and prompt length)
                recent_insights = self.cumulative_insights[-2:]
                previous_context = (
                    f"\n\nYour previous insights from earlier research:\n" +
                    "\n".join(f"- {insight}" for insight in recent_insights) +
                    "\n\nBuild on these insights with the new research below."
                )

            # Use fragment content as context with metadata + cumulative insights
            search_context = (
                f"{previous_context}\n\n"
                f"New Research:\n"
                f"Source: {fragment.source}\n"
                f"Content: {fragment.content}\n"
                f"Keywords: {', '.join(fragment.keywords)}"
            )

        # PRIORITY 2: Fall back to keyword extraction (legacy behavior)
        elif self.dynamic_retriever and self.task_prompt:
            try:
                # Extract keywords from task
                keywords = self.dynamic_retriever.extract_keywords(self.task_prompt, max_keywords=5)

                # Search and append to temp file (only if not already searched)
                if web_search_fn and keywords:
                    if self.actions_taken < 2:
                        logger.debug(f"{self.agent_id} keywords: {keywords}")

                    # Search and append (retriever handles deduplication)
                    await self.dynamic_retriever.search_and_append(keywords, web_search_fn)

                # Read relevant context from temp file
                search_context = self.dynamic_retriever.get_context_snippet(keywords, max_chars=450)

                if search_context and self.actions_taken < 2:
                    preview = search_context[:150] if len(search_context) > 150 else search_context
                    logger.debug(f"{self.agent_id} context snippet: {preview}...")

            except Exception as e:
                logger.error(f"{self.agent_id} retrieval error: {e}", exc_info=True)

        # Generate prompt with search context
        prompt = self._make_prompt(search_context)

        # Use per-agent temperature from config
        from ..core.config import TEMP_SCOUT

        # TASK-ADAPTIVE TOKEN ALLOCATION: Read from intake_profile if available
        # Different tasks need different token budgets:
        # - Creative: 150 (standard)
        # - Technical/Analysis: 200 (preserve complex details)
        # - Debate: 180 (argument depth)
        # - Problem Solving: 160 (solution detail)
        max_tokens = 150  # Default fallback
        if self.task_config and hasattr(self.task_config, 'intake_profile'):
            max_tokens = self.task_config.intake_profile.scout_tokens

        try:
            # Disable cache for scouts to ensure diverse exploration
            # Token allocation is now task-adaptive (read from intake_profile)
            # Rationale: Scouts work with densest information (raw research 200-500 words)
            # Better to invest tokens at intake stage than lose information permanently
            if self.actions_taken < 2:
                logger.debug(f"{self.agent_id} starting LLM generation (max_tokens={max_tokens}, temp={TEMP_SCOUT})")

            result = await llm.generate(prompt, max_tokens=max_tokens, temperature=TEMP_SCOUT, use_cache=False)

            if self.actions_taken < 2:
                result_preview = result[:60] if result else "None"
                logger.debug(f"{self.agent_id} LLM returned: {result_preview}...")

            if result and len(result.strip()) > 10:
                clean_result = result.strip()

                # Relaxed quality check: Only reject extreme repetition
                words = clean_result.split()
                if len(words) > 5:
                    unique_ratio = len(set(words)) / len(words)
                    # Only reject if <30% unique (very extreme repetition)
                    if unique_ratio < 0.3:
                        if self.actions_taken < 2:
                            logger.debug(f"{self.agent_id} rejected excessive repetition (unique ratio: {unique_ratio:.2f})")
                        return None

                # Basic validation: at least some text
                if len(words) >= 2 and len(clean_result) >= 15:
                    # STEPWISE SYNTHESIS: Store insight for next fragment processing
                    # This builds cumulative understanding across fragments
                    self.cumulative_insights.append(clean_result)
                    return clean_result
                else:
                    if self.actions_taken < 2:
                        logger.debug(f"{self.agent_id} output too short (words={len(words)}, chars={len(clean_result)})")
                    return None
        except Exception as e:
            logger.error(f"{self.agent_id} exploration error: {e}", exc_info=True)

        return None

    def assess_strength_creative(self, idea: str) -> float:
        """Assess idea strength — dispatches to code or prose scoring based on TaskFrame."""
        frame = getattr(self.task_config, 'task_frame', None) if self.task_config else None
        if frame and getattr(frame, 'strength_mode', 'prose') == 'code':
            return self._assess_strength_code(idea)
        return self._assess_strength_prose(idea)

    def _assess_strength_code(self, text: str) -> float:
        """Score code signals by structural code indicators."""
        t = text
        score = 0.40
        score += 0.20 if ('def ' in t or 'class ' in t) else 0.0
        score += 0.10 if ('return ' in t or 'yield ' in t) else 0.0
        score += 0.08 if '```' in t else 0.0
        score += 0.05 if ('import ' in t[:150]) else 0.0
        score += min(0.15, len(t) / 400.0)
        score += random.uniform(-0.05, 0.05)
        return max(0.0, min(1.0, score))

    def _assess_strength_prose(self, idea: str) -> float:
        """Assess prose idea strength with heuristics (original logic)."""
        length = len(idea)
        has_numbers = any(c.isdigit() for c in idea)
        has_specifics = any(word in idea.lower() for word in
                           ['study', 'data', 'research', 'because', 'therefore',
                            'according to', 'evidence', 'shows', 'indicates'])
        has_citations = any(word in idea.lower() for word in
                           ['ipcc', 'nasa', 'report', 'journal', 'university',
                            'research', 'study', 'analysis'])
        has_quantifiers = any(word in idea.lower() for word in
                             ['significant', 'substantial', 'major', 'critical',
                              'important', 'key'])

        score = 0.35 + min(0.25, length / 250.0)
        if has_numbers:
            score += 0.12
        if has_specifics:
            score += 0.15
        if has_citations:
            score += 0.10
        if has_quantifiers:
            score += 0.08
        score += random.uniform(-0.08, 0.08)
        return max(0.0, min(1.0, score))

    def _make_prompt(self, search_context: Optional[str] = None) -> str:
        """Generate exploration prompt with research context.

        Priority order:
        1. task_frame.scout_directive (planner-set direction)
        2. task_config.scout_prompt_template (static template)
        3. Legacy inline prompts

        Args:
            search_context: Research context (either fragment or web search)

        Returns:
            Prompt string
        """
        frame = getattr(self.task_config, 'task_frame', None) if self.task_config else None
        task_prompt = self.task_config.task_prompt if self.task_config else (self.task_prompt or "")

        # Sub-problem focus assigned by run_task.py (for complex decomposed tasks)
        sub_problem = getattr(self, 'sub_problem', None)
        focus_line = f"\nFocus specifically on: {sub_problem}" if sub_problem else ""

        # PRIORITY 1: Planner-set directive
        if frame and getattr(frame, 'scout_directive', ''):
            directive = frame.scout_directive
            base = (
                f"Task: {task_prompt}{focus_line}\n\n"
                f"Your role: {directive}\n\n"
            )
            if search_context:
                return (
                    f"{base}"
                    f"Context from research:\n{search_context}\n\n"
                    f"Based on this context, produce your output:"
                )
            return base + "Produce your output:"

        # PRIORITY 2: Static template
        if self.task_config and self.task_config.scout_prompt_template:
            base_prompt = self.task_config.scout_prompt_template.format(
                task_prompt=task_prompt
            )
            if search_context:
                return f"""{base_prompt}

Context from research:
{search_context}

Based on this information, provide a specific, evidence-based response:"""
            return base_prompt

        # LEGACY: Fallback to inline prompts for backward compatibility
        if not self.task_prompt:
            return "Generate a creative idea:\n"

        # Check if this is a research fragment (has Source: prefix)
        is_fragment = search_context and search_context.startswith("Source:")

        if search_context:
            if is_fragment:
                # Rich prompt for research fragments (from deep RAG)
                return (f"You are a scout agent analyzing research findings.\n\n"
                        f"Task: {self.task_prompt}\n\n"
                        f"Research Fragment:\n{search_context}\n\n"
                        f"IMPORTANT: Extract and present ONE specific, evidence-based observation "
                        f"from this research (1-2 sentences). Include key details like numbers, "
                        f"sources, or specific findings:\n")
            else:
                # Generic prompt for web search context (legacy)
                return (f"You are a creative scout agent.\n\n"
                        f"Task: {self.task_prompt}\n\n"
                        f"Context from research:\n{search_context}\n\n"
                        f"Based on this information, generate ONE specific, evidence-based solution (1-2 sentences):\n")
        else:
            # No context - pure creative exploration
            return (f"You are a creative scout agent.\n\n"
                    f"Task: {self.task_prompt}\n\n"
                    f"Generate ONE creative idea (1-2 sentences):\n")

    def stop(self):
        """Stop the agent."""
        self.active = False

    # ========================================================================
    # STAGED EXECUTION SUPPORT (for parallel processing with LLM pool)
    # ========================================================================

    async def prepare_prompt(self, signal_store: SignalStore) -> Optional[Tuple[str, str, int, float]]:
        """Prepare prompt for staged execution (no LLM call).

        Collects context from assigned fragments or web search, builds prompt,
        and returns prompt data for batch processing.

        Returns:
            (agent_id, prompt, max_tokens, temperature) or None if nothing to do
        """
        # Check if we should generate
        if not self.active or self.actions_taken >= getattr(self, 'max_actions', 1):
            return None

        # Collect search context
        search_context = None

        # PRIORITY 1: Use assigned research fragments
        if self.assigned_fragments and self.fragment_index < len(self.assigned_fragments):
            fragment = self.assigned_fragments[self.fragment_index]
            self.fragment_index += 1

            # Include previous insights for cumulative understanding
            previous_context = ""
            if self.cumulative_insights:
                recent_insights = self.cumulative_insights[-2:]
                previous_context = (
                    f"\n\nYour previous insights from earlier research:\n" +
                    "\n".join(f"- {insight}" for insight in recent_insights) +
                    "\n\nBuild on these insights with the new research below."
                )

            # Use fragment content as context
            search_context = (
                f"{previous_context}\n\n"
                f"New Research:\n"
                f"Source: {fragment.source}\n"
                f"Content: {fragment.content}\n"
                f"Keywords: {', '.join(fragment.keywords)}"
            )

        # PRIORITY 2: Fall back to dynamic retriever (legacy)
        elif self.dynamic_retriever and self.task_prompt:
            try:
                keywords = self.dynamic_retriever.extract_keywords(self.task_prompt, max_keywords=5)
                if keywords:
                    search_context = self.dynamic_retriever.get_context_snippet(keywords, max_chars=450)
            except Exception as e:
                logger.error(f"{self.agent_id} retrieval error: {e}")

        # Build prompt
        prompt = self._make_prompt(search_context)

        # Get token allocation from config
        from ..core.config import TEMP_SCOUT
        max_tokens = 150  # Default
        if self.task_config and hasattr(self.task_config, 'intake_profile'):
            max_tokens = self.task_config.intake_profile.scout_tokens

        return (self.agent_id, prompt, max_tokens, TEMP_SCOUT)

    async def process_result(self, result: str, signal_store: SignalStore) -> bool:
        """Process LLM result and deposit signal (staged execution).

        Args:
            result: Generated text from LLM
            signal_store: Where to deposit

        Returns:
            True if signal deposited, False otherwise
        """
        if not result or len(result.strip()) <= 10:
            return False

        clean_result = result.strip()

        # Quality check - reject extreme repetition
        words = clean_result.split()
        if len(words) > 5:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:  # Too repetitive
                return False

        # Assess strength
        strength = self.assess_strength_creative(clean_result)
        min_strength = getattr(self, 'min_strength', 0.5)

        if strength < min_strength:
            return False

        # Deposit with enhanced context (no parent for scouts)
        signal_id = deposit_with_context(
            signal_store,
            signal_type=self.signal_type,
            content=clean_result,
            strength=strength,
            depositor=self.agent_id,
            parent_signal=None  # Scouts generate root signals
        )

        if signal_id:
            # Track cumulative insights for stepwise synthesis
            self.cumulative_insights.append(clean_result)
            logger.info(f"{self.agent_id} deposited {signal_id} (strength={strength:.2f})")
            self.actions_taken += 1
            return True

        return False


def assign_research_to_scouts(scouts: List[Scout], research_fragments: List) -> None:
    """Divide research fragments among scouts for parallel processing.

    This creates proper "division of labor" - each scout gets assigned a portion
    of the deep research to intake and present. Like biological scouts dividing
    territory to explore.

    Strategy:
    - Prioritize high-importance and high-rarity fragments
    - Distribute evenly to balance workload
    - Ensure each scout gets diverse keywords

    Args:
        scouts: List of Scout agents
        research_fragments: List of ResearchFragment objects from deep research

    Example:
        ```python
        # After deep research round
        round_knowledge = await retriever.deep_research_round(keywords, round_num=1)

        # Divide fragments among scouts
        assign_research_to_scouts(scouts, round_knowledge.fragments)

        # Scouts will now process their assigned fragments
        await asyncio.gather(*[scout.run(signal_store, llm) for scout in scouts])
        ```
    """
    if not scouts or not research_fragments:
        return

    # Sort fragments by importance * rarity (prioritize rare, important findings)
    sorted_fragments = sorted(
        research_fragments,
        key=lambda f: f.importance * (1 + f.rarity),
        reverse=True
    )

    # Round-robin assignment for even distribution
    for i, fragment in enumerate(sorted_fragments):
        scout_idx = i % len(scouts)
        scouts[scout_idx].assigned_fragments.append(fragment)

    # Log assignments
    for scout in scouts:
        if scout.assigned_fragments:
            total_importance = sum(f.importance for f in scout.assigned_fragments)
            avg_rarity = sum(f.rarity for f in scout.assigned_fragments) / len(scout.assigned_fragments)
            logger.info(f"[RAG] {scout.agent_id} assigned {len(scout.assigned_fragments)} fragments "
                        f"(importance={total_importance:.1f}, avg_rarity={avg_rarity:.2f})")
        else:
            logger.info(f"[RAG] {scout.agent_id} no fragments assigned (will use fallback behavior)")
