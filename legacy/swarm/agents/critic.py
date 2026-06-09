"""Critic agents - evaluate validation status and adjust signal strength."""

import asyncio
import random
from typing import Optional, Dict, Tuple
from ..core.signal_store import SignalStore, Signal, deposit_with_context
from ..llm.simple_llm import SimpleLLM
from ..core.signal_types import SignalType, SUPPORT
from ..core.logging_config import get_logger
from ..core.stage_coordinator import StagedAgent

logger = get_logger(__name__)


class Critic(StagedAgent):
    """Critic agent - evaluates validation status and adjusts signal strength.

    Critics sample signals and evaluate their quality, adjusting strength based
    on validation metrics or generating reasoned critiques with quality scores.
    """

    def __init__(self, agent_id: str, mode: str = "creative", thesis: Optional[str] = None,
                 task_config=None):
        """Initialize critic.

        Args:
            agent_id: Unique agent ID
            mode: "creative" for task-based evaluation (default)
            thesis: Task prompt or thesis for context
            task_config: Optional TaskConfig for prompt templates (composition pattern)
        """
        self.agent_id = agent_id
        self.mode = mode
        self.active = True
        self.actions_taken = 0
        self.thesis = thesis
        self.task_config = task_config  # NEW: Config injection instead of monkey patching

    async def run(self, signal_store: SignalStore, llm: SimpleLLM,
                  max_actions: int = None):
        """Run critic behavior loop.

        Args:
            signal_store: Shared signal store
            llm: Language model for generating reasoned critiques
            max_actions: Optional maximum actions (None = run until stopped)
        """
        # Event-driven evaluation: wait for and evaluate signals
        evaluate_type = getattr(self, 'evaluate_type', 'DRAFT')
        if self.actions_taken == 0:
            logger.info(f"{self.agent_id} event-driven mode: reacts to {evaluate_type} signals")

        while self.active and (max_actions is None or self.actions_taken < max_actions):
            # PURE EVENT-DRIVEN: Check if signals exist, if not, wait for event
            if not signal_store.has_signals(evaluate_type):
                # Wait for signal event with short timeout for responsiveness
                await signal_store.wait_for_signal(evaluate_type, timeout=1.0)
                signal_store.clear_signal_event(evaluate_type)
                continue

            # Signals exist - evaluate them
            await self.run_creative(signal_store, llm)
            self.actions_taken += 1
            # No sleep - pure stigmergic event-driven behavior

    async def run_creative(self, signal_store: SignalStore, llm: SimpleLLM):
        """Run creative mode: sample and evaluate signals, depositing CRITIQUE signals.

        Args:
            signal_store: Signal store
            llm: Language model
        """
        # Get signal type to evaluate (with defensive fallback to DRAFT)
        evaluate_type = getattr(self, 'evaluate_type', 'DRAFT')

        # IMPROVED: Use stratified sampling for balanced evaluation
        # Sample 1 weak, 1 medium, 1 strong signal for diverse quality assessment
        signals = signal_store.sample_stratified(
            evaluate_type,
            weak=1,    # Evaluate weak signals to help them improve or prune
            medium=1,  # Evaluate medium signals for balanced perspective
            strong=1   # Evaluate strong signals to ensure they deserve high strength
        )

        # Fallback to weighted sampling if stratified returns nothing
        if not signals:
            signals = signal_store.sample_weighted(evaluate_type, n=3)

        if not signals:
            # No signals to evaluate - just return, will retry
            return

        for signal in signals:
            # Use monkey-patched generate_critique if available
            if hasattr(self, 'generate_critique_with_context'):
                try:
                    # NEW: Get full provenance context for comprehensive evaluation
                    context = self._build_evaluation_context(signal, signal_store)

                    # Generate critique with full context
                    critique_text = await self.generate_critique_with_context(
                        signal, context, llm, 0.7
                    )

                    if not critique_text or len(critique_text.strip()) < 20:
                        logger.debug(f"{self.agent_id} generate_critique returned insufficient content for {signal.id}")
                        continue

                    # Calculate quality score from critique
                    multiplier = self.calculate_multiplier_from_content(critique_text)
                    quality_score = self._extract_quality_score(critique_text, multiplier)

                    # NEW: Deposit CRITIQUE signal (this is the key improvement!)
                    critique_id = signal_store.deposit(
                        signal_type="CRITIQUE",
                        content=critique_text,
                        depositor=self.agent_id,
                        parent=signal.id,  # Link to evaluated signal
                        strength=quality_score  # Critique strength reflects confidence
                    )

                    if critique_id is None:
                        logger.warning(f"{self.agent_id} failed to deposit CRITIQUE for {signal.id} (rejected as duplicate)")
                        continue

                    # Adjust parent signal strength based on critique
                    old_strength = signal.strength
                    signal.strength *= multiplier

                    logger.info(f"{self.agent_id} deposited CRITIQUE {critique_id} for {signal.id} "
                                f"(quality: {quality_score:.2f}, adjusted: {old_strength:.2f} → {signal.strength:.2f})")

                except Exception as e:
                    logger.error(f"{self.agent_id} error evaluating {signal.id}: {e}", exc_info=True)
            elif hasattr(self, 'generate_critique'):
                # Legacy path: old generate_critique without context
                try:
                    critique = await self.generate_critique(signal, llm, 0.7)

                    if not critique:
                        logger.debug(f"{self.agent_id} generate_critique returned None/empty for {signal.id}")
                        continue

                    # Calculate multiplier and deposit critique signal
                    multiplier = self.calculate_multiplier_from_content(critique)
                    quality_score = self._extract_quality_score(critique, multiplier)

                    # Deposit CRITIQUE signal (improved behavior)
                    critique_id = signal_store.deposit(
                        signal_type="CRITIQUE",
                        content=critique,
                        depositor=self.agent_id,
                        parent=signal.id,
                        strength=quality_score
                    )

                    if critique_id is None:
                        logger.warning(f"{self.agent_id} failed to deposit CRITIQUE for {signal.id} (rejected as duplicate)")
                        continue

                    # Adjust signal strength based on critique quality
                    old_strength = signal.strength
                    signal.strength *= multiplier

                    logger.info(f"{self.agent_id} deposited CRITIQUE {critique_id} for {signal.id} "
                                f"(multiplier: {multiplier:.2f}, {old_strength:.2f} → {signal.strength:.2f})")
                except Exception as e:
                    logger.error(f"{self.agent_id} error evaluating {signal.id}: {e}", exc_info=True)
            else:
                # Fallback: simple validation check (when LLM methods unavailable)
                quality_score = 0.6 if len(signal.content) > 50 else 0.4
                critique_text = (
                    f"Basic length evaluation: "
                    f"{'adequate' if len(signal.content) > 50 else 'too brief'} "
                    f"({len(signal.content)} chars)"
                )

                # Deposit CRITIQUE signal even in fallback mode (for consistency)
                critique_id = signal_store.deposit(
                    signal_type="CRITIQUE",
                    content=critique_text,
                    depositor=self.agent_id,
                    parent=signal.id,
                    strength=quality_score
                )

                if critique_id is None:
                    logger.warning(f"{self.agent_id} failed to deposit CRITIQUE for {signal.id} (rejected as duplicate)")
                    continue

                # Adjust parent signal strength based on quality
                multiplier = 1.1 if quality_score > 0.5 else 0.9
                old_strength = signal.strength
                signal.strength *= multiplier

                logger.debug(f"{self.agent_id} evaluated {signal.id} (fallback mode, "
                           f"deposited {critique_id}, multiplier: {multiplier:.2f})")

    def calculate_multiplier_from_content(self, critique: str) -> float:
        """Calculate strength multiplier from critique content.

        Args:
            critique: Critique text (should contain <score>X.X</score> tag)

        Returns:
            Multiplier value (0.6-1.5 for strong differentiation)
        """
        import re

        # Try to parse explicit quality score from critique
        score_match = re.search(r'<score>([\d.]+)</score>', critique)

        if score_match:
            try:
                quality_score = float(score_match.group(1))
                # Map 0.0-1.0 quality score to 0.6-1.5 multiplier
                # High quality (>0.7) → amplify (1.2-1.5)
                # Medium quality (0.4-0.7) → maintain (0.9-1.2)
                # Low quality (<0.4) → decay (0.6-0.9)
                if quality_score >= 0.7:
                    multiplier = 1.2 + (quality_score - 0.7) * 1.0  # 1.2-1.5
                elif quality_score >= 0.4:
                    multiplier = 0.9 + (quality_score - 0.4) * 1.0  # 0.9-1.2
                else:
                    multiplier = 0.6 + quality_score * 0.75  # 0.6-0.9

                return max(0.6, min(1.5, multiplier))
            except (ValueError, IndexError):
                pass  # Fall through to keyword-based heuristic

        # Fallback: keyword-based heuristic (if score not found)
        # Positive indicators (high quality)
        positive_words = ['good', 'strong', 'well', 'clear', 'excellent', 'solid',
                         'compelling', 'thorough', 'accurate', 'insightful', 'robust']
        # Negative indicators (low quality)
        negative_words = ['weak', 'poor', 'unclear', 'lacks', 'missing', 'vague',
                         'incomplete', 'superficial', 'questionable', 'flawed']

        critique_lower = critique.lower()
        positive_count = sum(1 for word in positive_words if word in critique_lower)
        negative_count = sum(1 for word in negative_words if word in critique_lower)

        # More aggressive differentiation: 0.6-1.5 range
        if positive_count > negative_count + 1:  # Clearly positive
            multiplier = 1.5  # Strong amplification for high quality
        elif positive_count > negative_count:
            multiplier = 1.25  # Moderate amplification
        elif negative_count > positive_count + 1:  # Clearly negative
            multiplier = 0.6  # Strong decay for low quality
        elif negative_count > positive_count:
            multiplier = 0.75  # Moderate decay
        else:
            multiplier = 1.0  # Neutral

        return multiplier

    def _build_evaluation_context(self, signal: Signal, signal_store: SignalStore) -> Dict:
        """Build full provenance context for comprehensive signal evaluation.

        Args:
            signal: Signal to evaluate
            signal_store: Signal store for traversing provenance

        Returns:
            Context dict with supporting evidence, objections, and related signals
        """
        context = {
            'signal': signal,
            'supporting_evidence': [],
            'objections': [],
            'critiques': [],
            'has_support': False,
            'has_objections': False,
            'support_count': 0,
            'objection_count': 0
        }

        # Get all descendants of this signal
        try:
            # Get supporting evidence (SUPPORT signals)
            support_signals = signal_store.get_descendants(signal.id, "SUPPORT")
            context['supporting_evidence'] = support_signals
            context['support_count'] = len(support_signals)
            context['has_support'] = len(support_signals) > 0

            # Get objections (OBJECTION signals)
            objection_signals = signal_store.get_descendants(signal.id, "OBJECTION")
            context['objections'] = objection_signals
            context['objection_count'] = len(objection_signals)
            context['has_objections'] = len(objection_signals) > 0

            # Get existing critiques (to avoid duplication)
            critique_signals = signal_store.get_descendants(signal.id, "CRITIQUE")
            context['critiques'] = critique_signals

        except Exception as e:
            logger.warning(f"Could not build full context for {signal.id}: {e}")

        return context

    def _extract_quality_score(self, critique_text: str, multiplier: float) -> float:
        """Extract quality score for the critique signal itself.

        The critique signal's strength reflects how confident the critic is in
        the evaluation, not the quality of the evaluated signal.

        Args:
            critique_text: Generated critique text
            multiplier: Calculated multiplier for parent signal

        Returns:
            Quality score for the critique signal (0.5-0.95)
        """
        # Base score from length (longer critique = more detailed = higher confidence)
        length_score = min(0.3, len(critique_text) / 500)  # Up to 0.3 for 500+ chars

        # Score from explicit score tag presence (structured = higher confidence)
        has_score_tag = '<score>' in critique_text
        structure_score = 0.2 if has_score_tag else 0.0

        # Base confidence score
        base_score = 0.5  # Minimum confidence for any critique

        # Total quality score
        quality_score = base_score + length_score + structure_score

        # Cap at 0.95 (critics should be somewhat humble)
        return min(0.95, quality_score)

    async def generate_critique(self, claim: Signal, llm: SimpleLLM,
                                temperature: float) -> Optional[str]:
        """Generate critique for a claim.

        Uses task_config.critic_prompt_template if available (composition pattern),
        otherwise falls back to legacy inline prompts (for backward compatibility).

        Args:
            claim: Claim signal to critique
            llm: Language model
            temperature: Sampling temperature

        Returns:
            Critique content or None
        """
        # NEW: Use task_config template if available (composition - no monkey patching!)
        if self.task_config and self.task_config.critic_prompt_template:
            base_prompt = self.task_config.critic_prompt_template.format(
                task_prompt=self.task_config.task_prompt,
                parent_content=claim.content
            )

            # Add quality score request (like run_task.py does)
            prompt = f"""{base_prompt}

Provide your critique and a quality score (0.0-1.0):
- 0.7-1.0: High quality (strong, well-supported, clear)
- 0.4-0.7: Medium quality (acceptable but has gaps)
- 0.0-0.4: Low quality (weak, unsupported, unclear)

Format: <critique>Your analysis</critique><score>0.X</score>

Evaluation:"""
        else:
            # LEGACY: Fallback to inline prompts for backward compatibility
            # Safeguard: check if legacy attribute is set
            if self.thesis is None:
                raise RuntimeError(
                    f"Critic.generate_critique() called but legacy attribute 'thesis' not set. "
                    f"This method is only for creative/debate mode."
                )

            prompt = (f"You are a critic agent analyzing arguments about this thesis:\n"
                     f"\"{self.thesis}\"\n\n"
                     f"Analyzing this specific claim:\n"
                     f"\"{claim.content}\"\n\n"
                     f"Task: Provide a focused, analytical critique OF THIS CLAIM in the "
                     f"context of the thesis above. Identify specific weaknesses, logical gaps, "
                     f"missing evidence, or potential counterarguments. Be constructive and "
                     f"specific about what needs improvement or further support.\n\n"
                     f"Critique:")

        # Read max_tokens from intake profile if available
        max_tokens = 280  # High-quality default
        if self.task_config and hasattr(self.task_config, 'intake_profile'):
            max_tokens = self.task_config.intake_profile.critic_tokens

        try:
            result = await llm.generate(prompt, max_tokens=max_tokens, temperature=temperature)
            if result and len(result.strip()) > 20:
                return result.strip()
        except Exception as e:
            logger.error(f"{self.agent_id} generation error: {e}", exc_info=True)

        return None

    def assess_strength(self, content: str) -> float:
        """Assess critique strength.

        Args:
            content: Generated critique

        Returns:
            Strength value 0.0-1.0
        """
        length = len(content)
        has_analysis = any(word in content.lower() for word in
                          ['however', 'although', 'while', 'despite', 'lacks',
                           'assumes', 'overlooks', 'ignores', 'fails to'])
        has_specifics = any(word in content.lower() for word in
                           ['specific', 'evidence', 'data', 'support', 'research',
                            'assumes', 'requires', 'needs'])
        has_constructive = any(word in content.lower() for word in
                              ['could', 'should', 'might', 'would benefit',
                               'needs more', 'requires', 'important to'])

        # Base score
        score = 0.40 + min(0.20, length / 300.0)

        # Bonuses for critical quality
        if has_analysis:
            score += 0.15
        if has_specifics:
            score += 0.15
        if has_constructive:
            score += 0.10

        # Small randomness
        score += random.uniform(-0.03, 0.03)

        return max(0.0, min(1.0, score))

    def stop(self) -> None:
        """Stop the agent."""
        self.active = False

    # ========================================================================
    # STAGED EXECUTION METHODS (for parallel execution via StageCoordinator)
    # ========================================================================

    async def prepare_prompt(self, signal_store: SignalStore) -> Optional[Tuple[str, str, int, float]]:
        """Prepare prompt for staged execution (no LLM call).

        Returns:
            (agent_id, prompt, max_tokens, temperature) or None if nothing to do
        """
        # Check if should generate
        max_actions = getattr(self, 'max_actions', None)
        if not self.active or (max_actions is not None and self.actions_taken >= max_actions):
            return None

        # Get signal type to evaluate
        evaluate_type = getattr(self, 'evaluate_type', 'DRAFT')

        # Check if signals exist
        if not signal_store.has_signals(evaluate_type):
            return None

        # Sample one signal to evaluate (stratified sampling preferred)
        signals = signal_store.sample_stratified(evaluate_type, weak=1, medium=0, strong=0)
        if not signals:
            # Fallback to weighted sampling
            signals = signal_store.sample_weighted(evaluate_type, n=1)

        if not signals:
            return None

        signal_to_evaluate = signals[0]

        # Build evaluation context
        context = self._build_evaluation_context(signal_to_evaluate, signal_store)

        # Store for process_result
        self._current_signal = signal_to_evaluate
        self._current_context = context

        # Build prompt using existing method
        if hasattr(self, 'generate_critique_with_context'):
            # Use advanced method if available (requires custom prompt building)
            prompt = self._build_critique_prompt_with_context(signal_to_evaluate, context)
        elif self.task_config and self.task_config.critic_prompt_template:
            # Use task_config template
            base_prompt = self.task_config.critic_prompt_template.format(
                task_prompt=self.task_config.task_prompt,
                parent_content=signal_to_evaluate.content
            )
            prompt = f"""{base_prompt}

Provide your critique and a quality score (0.0-1.0):
- 0.7-1.0: High quality (strong, well-supported, clear)
- 0.4-0.7: Medium quality (acceptable but has gaps)
- 0.0-0.4: Low quality (weak, unsupported, unclear)

Format: <critique>Your analysis</critique><score>0.X</score>

Evaluation:"""
        else:
            # Fallback to legacy inline prompt
            prompt = (f"You are a critic agent analyzing arguments about this thesis:\n"
                     f"\"{self.thesis}\"\n\n"
                     f"Analyzing this specific claim:\n"
                     f"\"{signal_to_evaluate.content}\"\n\n"
                     f"Task: Provide a focused, analytical critique OF THIS CLAIM in the "
                     f"context of the thesis above. Identify specific weaknesses, logical gaps, "
                     f"missing evidence, or potential counterarguments. Be constructive and "
                     f"specific about what needs improvement or further support.\n\n"
                     f"Critique:")

        # Get token allocation from config
        max_tokens = 280  # High-quality default
        if self.task_config and hasattr(self.task_config, 'intake_profile'):
            max_tokens = self.task_config.intake_profile.critic_tokens

        return (self.agent_id, prompt, max_tokens, 0.7)

    def _build_critique_prompt_with_context(self, signal: Signal, context: Dict) -> str:
        """Build critique prompt with full provenance context."""
        base_prompt = f"""You are a critic evaluating this claim:
"{signal.content}"

Context:
- Supporting evidence: {context['support_count']} signals
- Objections raised: {context['objection_count']} signals"""

        # Add supporting evidence if available
        if context['has_support']:
            base_prompt += "\n\nSupporting Evidence:\n"
            for i, evidence in enumerate(context['supporting_evidence'][:3], 1):
                base_prompt += f"{i}. {evidence.content[:100]}...\n"

        # Add objections if available
        if context['has_objections']:
            base_prompt += "\n\nObjections Raised:\n"
            for i, objection in enumerate(context['objections'][:3], 1):
                base_prompt += f"{i}. {objection.content[:100]}...\n"

        base_prompt += """

Task: Evaluate the quality of this claim considering its supporting evidence and objections.
- Is it well-supported by evidence?
- Do the objections raise valid concerns?
- Is the claim clear and specific?

Provide critique and quality score (0.0-1.0):
Format: <critique>Your analysis</critique><score>0.X</score>

Evaluation:"""

        return base_prompt

    async def process_result(self, result: str, signal_store: SignalStore) -> bool:
        """Process LLM result and deposit signal (staged execution).

        Args:
            result: Generated critique from LLM
            signal_store: Signal store

        Returns:
            True if signal deposited, False otherwise
        """
        # Quality checks
        if not result or len(result.strip()) < 20:
            return False

        clean_result = result.strip()

        # Get evaluated signal from prepare_prompt
        evaluated_signal = getattr(self, '_current_signal', None)
        if not evaluated_signal:
            logger.warning(f"{self.agent_id} no evaluated signal in process_result")
            return False

        # Calculate multiplier for parent signal strength adjustment
        multiplier = self.calculate_multiplier_from_content(clean_result)
        quality_score = self._extract_quality_score(clean_result, multiplier)

        # Deposit CRITIQUE signal with enhanced context
        critique_id = deposit_with_context(
            signal_store,
            signal_type="CRITIQUE",
            content=clean_result,
            strength=quality_score,
            depositor=self.agent_id,
            parent_signal=evaluated_signal
        )

        if critique_id:
            # Adjust parent signal strength based on critique
            old_strength = evaluated_signal.strength
            evaluated_signal.strength *= multiplier

            logger.info(f"{self.agent_id} deposited CRITIQUE {critique_id} for {evaluated_signal.id} "
                       f"(quality: {quality_score:.2f}, adjusted: {old_strength:.2f} → {evaluated_signal.strength:.2f})")

            self.actions_taken += 1
            return True

        else:
            logger.warning(f"{self.agent_id} failed to deposit CRITIQUE for {evaluated_signal.id} (rejected as duplicate)")
            return False
