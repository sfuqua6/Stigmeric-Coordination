"""Hater agents - adversarial agents that generate contradictory evidence and challenges."""

import asyncio
import random
import time
from typing import Optional, List, Tuple
from ..core.signal_store import SignalStore, Signal, deposit_with_context
from ..llm.simple_llm import SimpleLLM
from ..core.verification import SignalVerifier
from ..core.logging_config import get_logger
from ..core.stage_coordinator import StagedAgent

logger = get_logger(__name__)


class Hater(StagedAgent):
    """Adversarial agent that generates contradictory evidence and challenges strong signals."""

    def __init__(self, agent_id: str, task_prompt: str = "Challenge insights",
                 enable_verification: bool = True,
                 input_types: Optional[List[str]] = None,
                 output_type: str = "OBJECTION",
                 task_config=None):
        """Initialize hater.

        Args:
            agent_id: Unique agent ID
            task_prompt: Task description (for document mode) or thesis (for legacy mode)
            enable_verification: Enable signal quality verification
            input_types: Signal types to target (e.g., ["SOLUTION", "CLAIM"]). Defaults to legacy types.
            output_type: Signal type to deposit (e.g., "OBJECTION", "COUNTER_EVIDENCE")
            task_config: Optional TaskConfig for prompt templates (composition pattern)
        """
        self.agent_id = agent_id
        self.task_prompt = task_prompt
        # For backward compatibility with legacy code that expects 'thesis'
        self.thesis = task_prompt
        self.task_config = task_config  # NEW: Config injection instead of monkey patching
        self.active = True
        self.actions_taken = 0
        self.enable_verification = enable_verification

        # Signal type configuration (dynamic based on task mode)
        self.input_types = input_types or ["INSIGHT", "CLAIM", "EVIDENCE"]  # Legacy default
        self.output_type = output_type

        # Quality verifier (only used if enable_verification=True)
        self.verifier = SignalVerifier(min_quality_score=0.4) if enable_verification else None

    async def run(self, signal_store: SignalStore, llm: SimpleLLM,
                  min_strength: float = 0.3, max_actions: int = 200,
                  temperature: float = 0.85, target_consensus: bool = True):
        """Run hater behavior loop.

        Args:
            signal_store: Shared signal store
            llm: Language model
            min_strength: Minimum strength to deposit signal
            max_actions: Maximum actions before stopping (default: 200 to match foragers)
            temperature: Sampling temperature (high for adversarial creativity)
            target_consensus: If True, target consensus clusters; if False, target strongest
        """
        while self.active and self.actions_taken < max_actions:
            # Choose targeting strategy
            if target_consensus and self.actions_taken % 3 == 0:  # Every 3rd action, try consensus
                # Try to find and challenge consensus clusters
                target = self.find_consensus_target(signal_store)

                if target:
                    logger.debug(f"{self.agent_id} targeting consensus cluster around {target.id}")
            else:
                # Fallback to sampling strong signals
                target = None

            # If no consensus target found, use traditional sampling
            if not target:
                # Sample signals of configured input types (mode-aware)
                targets = []
                for signal_type in self.input_types:
                    sampled = signal_store.sample_weighted(signal_type, n=3)
                    targets.extend(sampled)

                if not targets:
                    self.actions_taken += 1
                    # Event-driven: wait for new signals
                    await signal_store.wait_for_signal(self.input_types[0], timeout=1.0)
                    signal_store.clear_signal_event(self.input_types[0])
                    continue

                # Pick highest strength signal (but prefer under-challenged ones)
                # Prioritize signals with few objections
                targets_with_scores = []
                for t in targets:
                    # Count existing objections
                    # PERFORMANCE: Use indexed retrieval O(k) instead of O(n) scan
                    children = signal_store.get_signals_by_parent(t.id)
                    objection_count = len([s for s in children
                                         if s.type in ["OBJECTION", "COUNTER_EVIDENCE"]])
                    # Score: high strength + low objections = priority
                    score = t.strength * 0.7 + (1.0 / (objection_count + 1)) * 0.3
                    targets_with_scores.append((score, t))

                targets_with_scores.sort(key=lambda x: x[0], reverse=True)
                target = targets_with_scores[0][1]

            # Generate contradictory evidence
            contradiction = await self.generate_contradiction(target, llm, temperature)

            if contradiction:
                # Verify objection quality using centralized verifier
                if self.enable_verification and self.verifier:
                    # Create temporary signal for verification
                    temp_objection = Signal(
                        id="temp_objection",
                        type=self.output_type,  # Use configured output type
                        content=contradiction,
                        strength=0.5,
                        timestamp=time.time(),
                        depositor=self.agent_id,
                        parent=target.id
                    )

                    quality_check = self.verifier.verify_objection_substantiveness(
                        temp_objection, target, signal_store
                    )

                    if not quality_check['valid']:
                        reasons_str = ', '.join(quality_check['reasons'])
                        logger.debug(f"{self.agent_id} rejected low-quality objection: {reasons_str}")
                        # Try to generate a better one
                        contradiction = await self.generate_stronger_contradiction(target, llm, temperature)
                        if contradiction:
                            temp_objection.content = contradiction
                            quality_check = self.verifier.verify_objection_substantiveness(
                                temp_objection, target, signal_store
                            )
                else:
                    # Fallback to internal verification if verifier disabled
                    quality_check = self.verify_objection_quality(contradiction)

                # Use quality score as strength if available
                if quality_check['valid']:
                    strength = quality_check['score']

                    # Deposit if strong enough
                    if strength >= min_strength:
                        # Use configured output type (mode-aware)
                        signal_type = self.output_type

                        signal_id = signal_store.deposit(
                            signal_type=signal_type,
                            content=contradiction,
                            strength=strength,
                            depositor=self.agent_id,
                            parent=target.id
                        )
                        logger.info(f"{self.agent_id} deposited {signal_id} "
                                    f"(contradicts {target.id}, strength={strength:.2f}, quality={quality_check['score']:.2f})")

            self.actions_taken += 1
            # No sleep - pure stigmergic event-driven behavior

    async def generate_contradiction(self, target: Signal, llm: SimpleLLM,
                                    temperature: float) -> Optional[str]:
        """Generate contradictory evidence for a signal.

        Args:
            target: Signal to contradict
            llm: Language model
            temperature: Sampling temperature

        Returns:
            Contradictory content or None
        """
        prompt = self._make_prompt(target)

        try:
            result = await llm.generate(prompt, max_tokens=150, temperature=temperature)
            if result and len(result.strip()) > 20:
                return result.strip()
        except Exception as e:
            logger.error(f"{self.agent_id} generation error: {e}", exc_info=True)

        return None

    def assess_strength(self, content: str) -> float:
        """Assess contradiction strength.

        Args:
            content: Generated contradiction

        Returns:
            Strength value 0.0-1.0
        """
        length = len(content)
        has_numbers = any(c.isdigit() for c in content)
        has_specifics = any(word in content.lower() for word in
                           ['however', 'but', 'actually', 'contrary', 'instead',
                            'research', 'study', 'data', 'shows', 'indicates'])
        has_references = any(word in content.lower() for word in
                            ['study', 'research', 'report', 'journal', 'analysis',
                             'university', 'published'])
        has_negation = any(word in content.lower() for word in
                          ['not', 'no', 'false', 'incorrect', 'misleading',
                           'flawed', 'questionable', 'disputed'])

        # Base score
        score = 0.40 + min(0.20, length / 300.0)

        # Bonuses for adversarial quality
        if has_numbers:
            score += 0.12
        if has_specifics:
            score += 0.15
        if has_references:
            score += 0.10
        if has_negation:
            score += 0.10

        # Small randomness
        score += random.uniform(-0.05, 0.05)

        return max(0.0, min(1.0, score))

    def _make_prompt(self, target: Signal) -> str:
        """Generate adversarial prompt.

        Uses task_config.hater_prompt_template if available (composition pattern),
        otherwise falls back to legacy inline prompts (for backward compatibility).

        Args:
            target: Signal to contradict

        Returns:
            Prompt string
        """
        # PRIORITY 1: Planner-set directive via TaskFrame
        frame = getattr(self.task_config, 'task_frame', None) if self.task_config else None
        if frame:
            directive = getattr(frame, 'hater_directive', '')
            if directive:
                task_prompt = self.task_config.task_prompt if self.task_config else self.task_prompt
                return (
                    f"Task: {task_prompt}\n\n"
                    f"Signal to challenge:\n\"{target.content}\"\n\n"
                    f"Your role: {directive}\n\n"
                    f"Your challenge (2-3 sentences):"
                )

        # PRIORITY 2: Static template
        if self.task_config and self.task_config.hater_prompt_template:
            return self.task_config.hater_prompt_template.format(
                task_prompt=self.task_config.task_prompt,
                parent_content=target.content,
                parent_type=target.type.lower()
            )

        # LEGACY: Fallback to inline prompts for backward compatibility
        if target.type == "INSIGHT":
            # Document mode - challenge insights
            return (f"You are an adversarial critic analyzing this insight:\n"
                   f"\"{target.content}\"\n\n"
                   f"Task: Generate a well-reasoned objection or alternative interpretation. "
                   f"Consider:\n"
                   f"- Could the data support a different conclusion?\n"
                   f"- What assumptions might be questionable?\n"
                   f"- Are there contradictory findings or alternative explanations?\n"
                   f"- What limitations or caveats should be noted?\n\n"
                   f"Be specific and constructive. Don't just say it's wrong - explain "
                   f"WHY and provide an alternative perspective.\n\n"
                   f"Objection:")
        elif SignalType.is_initial_type(target.type):
            # Universal INITIAL type (was hardcoded "CLAIM")
            return (f"You are an adversarial agent analyzing arguments about this thesis:\n"
                   f"\"{self.thesis}\"\n\n"
                   f"Challenging this specific claim:\n"
                   f"\"{target.content}\"\n\n"
                   f"Task: Generate a specific counterargument or contradictory "
                   f"evidence that challenges THIS CLAIM in the context of the thesis above. "
                   f"Be concrete and cite alternative perspectives or data. Don't just say "
                   f"it's wrong - explain WHY and provide an alternative view.\n\n"
                   f"Counterargument:")
        else:  # SUPPORT type (was hardcoded "EVIDENCE")
            return (f"You are an adversarial agent analyzing arguments about this thesis:\n"
                   f"\"{self.thesis}\"\n\n"
                   f"Challenging this specific evidence:\n"
                   f"\"{target.content}\"\n\n"
                   f"Task: Identify limitations, contradictory findings, or "
                   f"alternative interpretations of THIS EVIDENCE in the context of the "
                   f"thesis above. Be specific about what's questionable or incomplete.\n\n"
                   f"Challenge:")

    def find_consensus_target(self, signal_store: SignalStore) -> Optional[Signal]:
        """Find a signal that's part of a consensus cluster (potential groupthink).

        Args:
            signal_store: Signal store to search

        Returns:
            Target signal from consensus cluster, or None
        """
        # Get all signals of configured input types
        # PERFORMANCE: Use indexed retrieval O(k*m) instead of O(n) scan
        # where k=signals per type, m=number of types
        insights = []
        for signal_type in self.input_types:
            insights.extend(signal_store.get_signals_by_type(signal_type))

        if len(insights) < 3:
            return None  # Need at least 3 insights to detect consensus

        # Find clusters of similar insights (potential consensus)
        for insight in insights:
            # Find similar insights using signal store's similarity checking
            similar = signal_store.find_related_signals(
                insight,
                type=insight.type,  # Use the signal's actual type
                similarity_threshold=0.7,  # High similarity = consensus
                n=5
            )

            if len(similar) >= 2:  # Found a cluster (insight + 2+ similar)
                # Check if this cluster has weak diversity (potential groupthink)
                cluster = [insight] + similar
                weakness = self.analyze_consensus_weakness(cluster, signal_store)

                if weakness['score'] < 0.5:  # Weak consensus
                    logger.debug(f"Found weak consensus cluster: {weakness['weakness']}")
                    # Return the strongest signal in the cluster to target
                    return max(cluster, key=lambda s: s.strength)

        return None

    def analyze_consensus_weakness(self, cluster: List[Signal],
                                   signal_store: SignalStore) -> dict:
        """Analyze if consensus cluster is weak (low diversity = groupthink).

        Args:
            cluster: List of similar insight signals
            signal_store: Signal store

        Returns:
            Dictionary with weakness type and score
        """
        # Get all evidence for the cluster
        all_evidence = []
        for insight in cluster:
            # Use universal SUPPORT type (was hardcoded "EVIDENCE")
            # PERFORMANCE: Use indexed retrieval O(k) instead of O(n) scan
            children = signal_store.get_signals_by_parent(insight.id)
            evidence = [s for s in children if SignalType.is_support_type(s.type)]
            all_evidence.extend(evidence)

        if len(all_evidence) == 0:
            return {'weakness': 'no_evidence', 'score': 0.0}

        # Check evidence diversity (are they all saying the same thing?)
        evidence_similarity_scores = []
        for i, ev1 in enumerate(all_evidence):
            for ev2 in all_evidence[i+1:]:
                # Simple similarity check
                from difflib import SequenceMatcher
                sim = SequenceMatcher(None, ev1.content.lower(), ev2.content.lower()).ratio()
                evidence_similarity_scores.append(sim)

        evidence_diversity = 1.0 - (sum(evidence_similarity_scores) / len(evidence_similarity_scores)
                                   if evidence_similarity_scores else 0.5)

        # Check source diversity (do insights come from same observations?)
        source_docs = set()
        for insight in cluster:
            obs_ids = insight.metadata.get('observation_ids', [])
            for obs_id in obs_ids:
                obs = signal_store.get_signal(obs_id)
                if obs:
                    source_doc = obs.metadata.get('source_document', obs_id)
                    source_docs.add(source_doc)

        source_diversity = len(source_docs) / max(len(cluster), 1)

        # Overall weakness score (low score = weak consensus)
        weakness_score = (evidence_diversity * 0.5 + source_diversity * 0.5)

        if evidence_diversity < 0.3:
            weakness_type = 'low_evidence_diversity'
        elif source_diversity < 0.3:
            weakness_type = 'low_source_diversity'
        elif weakness_score < 0.5:
            weakness_type = 'groupthink'
        else:
            weakness_type = 'none'

        return {
            'weakness': weakness_type,
            'score': weakness_score,
            'evidence_diversity': evidence_diversity,
            'source_diversity': source_diversity
        }

    def verify_objection_quality(self, objection: str) -> dict:
        """Verify objection is substantive, not generic.

        Args:
            objection: Objection content to verify

        Returns:
            Dictionary with validation result, reason, and quality score
        """
        import re

        # Check length
        if len(objection) < 80:
            return {'valid': False, 'reason': 'too_short', 'score': 0.2}

        # Check for generic phrases only
        generic_phrases = [
            r'\bbut what about\b',
            r'\bhowever,?\b',
            r'\bon the other hand\b',
            r'\bit could be argued\b',
            r'\balternatively,?\b'
        ]

        generic_count = sum(1 for phrase in generic_phrases
                           if re.search(phrase, objection, re.IGNORECASE))

        # Check for specific details
        has_numbers = bool(re.search(r'\d+', objection))
        has_proper_nouns = bool(re.search(r'\b[A-Z][a-z]+\b', objection))
        has_technical_terms = len([w for w in objection.split() if len(w) > 10]) > 2

        specificity_score = sum([has_numbers, has_proper_nouns, has_technical_terms]) / 3.0

        # Check for alternative explanation
        has_alternative = any(phrase in objection.lower() for phrase in [
            'instead', 'alternative', 'rather', 'actually', 'in fact', 'different'
        ])

        # Check for reasoning words
        has_reasoning = any(phrase in objection.lower() for phrase in [
            'because', 'since', 'therefore', 'thus', 'hence', 'due to', 'as a result'
        ])

        # Overall quality
        if generic_count > 2 and not has_alternative:
            return {'valid': False, 'reason': 'too_generic', 'score': 0.3}

        quality_score = (
            specificity_score * 0.4 +
            (1.0 if has_alternative else 0.0) * 0.3 +
            (1.0 if has_reasoning else 0.0) * 0.3
        )

        return {
            'valid': quality_score >= 0.4,
            'reason': 'verified' if quality_score >= 0.4 else 'low_specificity',
            'score': max(quality_score, 0.4),  # Minimum 0.4 if valid
            'details': {
                'has_numbers': has_numbers,
                'has_proper_nouns': has_proper_nouns,
                'has_technical_terms': has_technical_terms,
                'has_alternative': has_alternative,
                'has_reasoning': has_reasoning
            }
        }

    async def generate_stronger_contradiction(self, target: Signal, llm: SimpleLLM,
                                              temperature: float) -> Optional[str]:
        """Generate a stronger, more specific contradiction.

        Args:
            target: Signal to contradict
            llm: Language model
            temperature: Sampling temperature

        Returns:
            Stronger contradictory content or None
        """
        # Enhanced prompt that requires more specificity
        enhanced_prompt = f"""You are an expert adversarial critic. Your job is to find substantive flaws.

Target insight to challenge:
\"{target.content}\"

Task: Generate a SPECIFIC, WELL-REASONED objection that:
1. Identifies a concrete weakness or alternative interpretation
2. Explains WHY it's problematic (use reasoning words like 'because', 'since')
3. Provides a specific alternative perspective (use 'instead', 'rather', 'actually')
4. Includes concrete details (numbers, names, specific examples)

Be constructive but rigorous. Don't use generic phrases like "but what about..." or "it could be argued..."

Strong objection:"""

        try:
            # Reduced from 200 to 120 tokens for faster generation
            result = await llm.generate(enhanced_prompt, max_tokens=120, temperature=temperature * 0.9)
            if result and len(result.strip()) > 50:
                return result.strip()
        except Exception as e:
            logger.error(f"{self.agent_id} stronger generation error: {e}", exc_info=True)

        return None

    def stop(self):
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
        max_actions = getattr(self, 'max_actions', 200)
        if not self.active or self.actions_taken >= max_actions:
            return None

        # Choose targeting strategy - try consensus first (every 3rd action)
        target_consensus = getattr(self, 'target_consensus', True)
        target = None

        if target_consensus and self.actions_taken % 3 == 0:
            # Try to find and challenge consensus clusters
            target = self.find_consensus_target(signal_store)
            if target:
                logger.debug(f"{self.agent_id} targeting consensus cluster around {target.id}")

        # Fallback to sampling strong signals if no consensus found
        if not target:
            # Sample signals of configured input types
            targets = []
            for signal_type in self.input_types:
                sampled = signal_store.sample_weighted(signal_type, n=3)
                targets.extend(sampled)

            if not targets:
                return None  # No signals to target

            # Pick signal with high strength but few objections (priority target)
            targets_with_scores = []
            for t in targets:
                # Count existing objections
                children = signal_store.get_signals_by_parent(t.id)
                objection_count = len([s for s in children
                                     if s.type in ["OBJECTION", "COUNTER_EVIDENCE"]])
                # Score: high strength + low objections = priority
                score = t.strength * 0.7 + (1.0 / (objection_count + 1)) * 0.3
                targets_with_scores.append((score, t))

            targets_with_scores.sort(key=lambda x: x[0], reverse=True)
            target = targets_with_scores[0][1]

        # Store target for process_result
        self._current_target = target

        # Build prompt using existing _make_prompt method
        prompt = self._make_prompt(target)

        # Get token allocation and temperature
        max_tokens = 150  # Standard default for objections
        if self.task_config and hasattr(self.task_config, 'intake_profile'):
            # Hater uses same tokens as forager (both develop parent signals)
            max_tokens = getattr(self.task_config.intake_profile, 'hater_tokens',
                                getattr(self.task_config.intake_profile, 'forager_tokens', 150))

        temperature = 0.85  # High for adversarial creativity

        return (self.agent_id, prompt, max_tokens, temperature)

    async def process_result(self, result: str, signal_store: SignalStore) -> bool:
        """Process LLM result and deposit signal (staged execution).

        Args:
            result: Generated objection from LLM
            signal_store: Signal store

        Returns:
            True if signal deposited, False otherwise
        """
        # Quality checks
        if not result or len(result.strip()) <= 20:
            return False

        contradiction = result.strip()

        # Get target signal from prepare_prompt
        target = getattr(self, '_current_target', None)
        if not target:
            logger.warning(f"{self.agent_id} no target signal in process_result")
            return False

        # Verify objection quality
        if self.enable_verification and self.verifier:
            # Create temporary signal for verification
            temp_objection = Signal(
                id="temp_objection",
                type=self.output_type,
                content=contradiction,
                strength=0.5,
                timestamp=time.time(),
                depositor=self.agent_id,
                parent=target.id
            )

            quality_check = self.verifier.verify_objection_substantiveness(
                temp_objection, target, signal_store
            )
        else:
            # Fallback to internal verification if verifier disabled
            quality_check = self.verify_objection_quality(contradiction)

        # Deposit if quality check passed
        if not quality_check['valid']:
            reasons = quality_check.get('reasons', [quality_check.get('reason', 'unknown')])
            if isinstance(reasons, list):
                reasons_str = ', '.join(reasons)
            else:
                reasons_str = str(reasons)
            logger.debug(f"{self.agent_id} rejected low-quality objection: {reasons_str}")
            return False

        # Use quality score as strength
        strength = quality_check['score']

        # Check minimum strength
        min_strength = getattr(self, 'min_strength', 0.3)
        if strength < min_strength:
            logger.debug(f"{self.agent_id} objection strength {strength:.2f} below minimum {min_strength:.2f}")
            return False

        # Deposit OBJECTION signal with enhanced context
        signal_id = deposit_with_context(
            signal_store,
            signal_type=self.output_type,
            content=contradiction,
            strength=strength,
            depositor=self.agent_id,
            parent_signal=target
        )

        if signal_id:
            logger.info(f"{self.agent_id} deposited {signal_id} "
                       f"(contradicts {target.id}, strength={strength:.2f}, quality={quality_check['score']:.2f})")
            self.actions_taken += 1

            # Validate reference quality (optional logging)
            from ..core.signal_store import validate_reference_quality
            deposited_signal = signal_store.get_signal(signal_id)
            if deposited_signal:
                ref_quality = validate_reference_quality(deposited_signal, target)
                if ref_quality < 0.3:
                    logger.warning(f"{self.agent_id} weak parent reference: quality={ref_quality:.2f}")

            return True

        else:
            logger.warning(f"{self.agent_id} failed to deposit OBJECTION for {target.id} (rejected as duplicate)")
            return False
