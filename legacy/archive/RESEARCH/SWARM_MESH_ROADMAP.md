# SWARM MESH ROADMAP: Achieving the Dream System

**Date:** 2025-11-13
**Objective:** Transform the swarm from "first pattern amplification" to "rigorous adversarial refinement that finds truth"

---

## 🎯 THE DREAM VISION

A self-healing, never-failing swarm mesh where:
1. **Critics and Haters are POWERFUL** - They have enhanced context, run longer, and actually challenge groupthink
2. **Agents interact smoothly** - No failures, programmatic health verification
3. **Disagreement refines ideas** - Adversarial agents persist across iterations with memory
4. **Roles are clearly specialized** - Each agent type excels at its specific job
5. **The mesh self-heals** - Detects and fixes issues automatically

---

## 🔴 CRITICAL FINDINGS FROM CURRENT STATE

### **The Fundamental Problem: BLIND SWARM**
- No agent sees >10% of the signal graph
- Critics count evidence but DON'T READ IT (accountants, not critics)
- Haters are outnumbered **75:1** by other agents (100 actions vs 7,500)
- Haters attack WRONG TARGETS (strongest signals, not consensus/groupthink)
- NO agent-to-agent dialogue (only one-way signal deposition)
- NO programmatic verification of insight quality
- Echo chambers form via sampling bias toward strong signals

### **Power Imbalance**
```
Agent Type    | Agents | Iterations | Total Actions | % of Total
--------------|--------|------------|---------------|----------
Foragers      | 50     | 150        | 7,500         | 68.2%
Gatherers     | 20     | 150        | 3,000         | 27.3%
Critics       | 20     | 100        | 2,000         | 18.2%
Haters        | 10     | 10         | 100           | 0.9%
```

**Haters have <1% of total agent actions, yet they're supposed to prevent groupthink!**

---

## 📋 IMPLEMENTATION ROADMAP

### **PHASE 1: EMPOWER CRITICS & HATERS (CRITICAL)**
*Priority: IMMEDIATE - These agents are what enable finding "the right path"*

#### Task 1.1: Give Critics Enhanced Context and Reasoning
**File:** `swarm/agents/critic.py`

**Current Problem:**
```python
# Critics just count evidence, don't read it
validation = get_validation_status(insight.id)  # Returns COUNTS only
if validation['validation_score'] >= 0.7:
    amplify(insight, 1.3)  # Blind amplification
```

**Required Changes:**
- [ ] Add `enhanced_context: bool = True` parameter to `__init__`
- [ ] When `enhanced_context=True`, fetch full provenance chains:
  ```python
  evidence_signals = signal_store.get_descendants(insight.id, "EVIDENCE")
  observation_signals = signal_store.get_ancestors(insight.id, "OBSERVATION")
  related_insights = signal_store.find_related_signals(insight, type="INSIGHT", n=5)
  ```
- [ ] Generate REASONED critiques (not just audits):
  ```python
  critique_prompt = f"""Evaluate this insight rigorously:

  INSIGHT: {insight.content}

  SUPPORTING EVIDENCE:
  {format_evidence_content(evidence_signals)}

  SOURCE OBSERVATIONS:
  {format_observation_content(observation_signals)}

  RELATED INSIGHTS:
  {format_insights(related_insights)}

  Critical analysis:
  1. Does the evidence actually support the insight?
  2. Are there logical gaps or leaps?
  3. Does this contradict or duplicate other insights?
  4. What's the strength of the argument (0.0-1.0)?
  5. What are the limitations or caveats?

  Provide a rigorous critique and final strength assessment."""
  ```
- [ ] Add consistency checking across insights:
  ```python
  def check_consistency(self, insight: Signal, related: List[Signal]) -> float:
      """Check if insight contradicts other insights."""
      for other in related:
          if semantic_contradiction(insight.content, other.content):
              return 0.0  # Flag contradiction
      return 1.0
  ```
- [ ] Increase iterations from 100 to **200** (match or exceed foragers)
- [ ] Add stratified sampling (sample across ALL strength levels):
  ```python
  # Sample from weak (0.0-0.4), medium (0.4-0.7), strong (0.7-1.0)
  weak = sample_by_strength_range("INSIGHT", 0.0, 0.4, n=1)
  medium = sample_by_strength_range("INSIGHT", 0.4, 0.7, n=1)
  strong = sample_by_strength_range("INSIGHT", 0.7, 1.0, n=1)
  insights = weak + medium + strong
  ```

**Acceptance Criteria:**
- [ ] Critics read actual content, not just counts
- [ ] Critics generate reasoned critiques with specific issues identified
- [ ] Critics check cross-insight consistency
- [ ] Critics sample from all strength levels (not biased toward strong)
- [ ] Critics run for 200+ iterations

---

#### Task 1.2: Transform Haters into Consensus Challengers
**File:** `swarm/agents/hater.py`

**Current Problem:**
```python
# Attack strongest signal (WRONG TARGET)
target = max(targets, key=lambda s: s.strength)

# Only 10 iterations (OUTNUMBERED 75:1)
```

**Required Changes:**
- [ ] Increase iterations from 10 to **200** (CRITICAL - match foragers)
- [ ] Change targeting from "strongest signal" to "consensus patterns":
  ```python
  def find_consensus_clusters(self, signal_store: SignalStore,
                              signal_type: str = "INSIGHT") -> List[List[Signal]]:
      """Find clusters of similar insights (potential groupthink)."""
      insights = signal_store.get_all_signals(signal_type)

      # Use semantic clustering to find consensus
      clusters = []
      for insight in insights:
          similar = signal_store.find_related_signals(
              insight, type=signal_type,
              similarity_threshold=0.7,  # High threshold = consensus
              n=5
          )
          if len(similar) >= 3:  # 3+ similar insights = consensus cluster
              clusters.append([insight] + similar)

      return clusters
  ```
- [ ] Add consensus weakness detection:
  ```python
  def analyze_consensus_weakness(self, cluster: List[Signal],
                                  signal_store: SignalStore) -> dict:
      """Check if consensus is justified or groupthink."""
      # Get all evidence for the cluster
      all_evidence = []
      for insight in cluster:
          evidence = signal_store.get_descendants(insight.id, "EVIDENCE")
          all_evidence.extend(evidence)

      # Check evidence diversity
      if len(all_evidence) == 0:
          return {'weakness': 'no_evidence', 'score': 0.0}

      evidence_diversity = calculate_diversity(all_evidence)

      # Check source diversity
      source_docs = set()
      for insight in cluster:
          obs_ids = insight.metadata.get('observation_ids', [])
          for obs_id in obs_ids:
              obs = signal_store.get_signal(obs_id)
              if obs:
                  source_docs.add(obs.metadata.get('source_document', ''))

      source_diversity = len(source_docs) / max(len(cluster), 1)

      if evidence_diversity < 0.3 or source_diversity < 0.3:
          return {'weakness': 'low_diversity', 'score': 0.3}

      return {'weakness': 'none', 'score': 1.0}
  ```
- [ ] Generate SUBSTANTIVE objections (not generic "but what about X?"):
  ```python
  objection_prompt = f"""You are an adversarial critic analyzing potential groupthink.

  CONSENSUS PATTERN: {len(cluster)} insights converge on:
  {summarize_cluster(cluster)}

  EVIDENCE BASE:
  Evidence diversity: {evidence_diversity:.2f} (0.0-1.0)
  Source diversity: {source_diversity:.2f}
  Evidence samples: {format_evidence(all_evidence[:5])}

  WEAKNESS ANALYSIS:
  {weakness_analysis}

  Task: Generate a well-reasoned objection that:
  1. Identifies the specific weakness (low diversity, circular reasoning, etc.)
  2. Proposes an alternative interpretation
  3. Suggests what evidence would be needed to support the alternative
  4. Is SPECIFIC (not "but what about X?")

  Objection:"""
  ```
- [ ] Add objection quality verification:
  ```python
  def verify_objection_quality(self, objection: str) -> bool:
      """Verify objection is substantive, not generic."""
      # Check length
      if len(objection) < 100:
          return False

      # Check for generic phrases
      generic_phrases = [
          "but what about",
          "however,",
          "on the other hand",
          "it could be argued"
      ]
      has_only_generic = all(phrase in objection.lower()
                             for phrase in generic_phrases[:2])
      if has_only_generic:
          return False

      # Check for specific details (numbers, names, etc.)
      has_specifics = (
          any(c.isdigit() for c in objection) or
          any(word.istitle() for word in objection.split())
      )

      return has_specifics
  ```
- [ ] Add objection impact monitoring:
  ```python
  def monitor_objection_impact(self, objection_id: str,
                               signal_store: SignalStore,
                               timeout: int = 50) -> bool:
      """Check if objection influenced subsequent signals."""
      # Wait for other agents to potentially respond
      await asyncio.sleep(timeout * 0.5)

      # Check if objection has descendants
      responses = signal_store.get_descendants(objection_id)

      # Check if targeted insights' strength decreased
      objection = signal_store.get_signal(objection_id)
      target = signal_store.get_signal(objection.parent)

      if target:
          strength_before = target.metadata.get('strength_history', [target.strength])[-2] if 'strength_history' in target.metadata else target.strength
          strength_after = target.strength

          if strength_after < strength_before * 0.9:  # 10% decrease
              return True  # Impact detected

      return len(responses) > 0  # Someone responded
  ```

**Acceptance Criteria:**
- [ ] Haters run for 200+ iterations (match foragers in power)
- [ ] Haters target consensus clusters, not strongest individual signals
- [ ] Haters detect consensus weakness (low evidence/source diversity)
- [ ] Haters generate substantive objections with specific alternatives
- [ ] Haters verify objection quality before depositing
- [ ] Haters monitor objection impact and follow up if ignored

---

#### Task 1.3: Add Hater Persistence Across Iterations
**File:** `swarm/agents/hater.py`

**Current Problem:**
- Haters deposit objections and move on
- No memory of what they challenged
- No sustained adversarial presence

**Required Changes:**
- [ ] Add memory system to Hater:
  ```python
  class Hater:
      def __init__(self, agent_id: str, task_prompt: str = "Challenge insights"):
          # ... existing init ...
          self.challenge_history: List[dict] = []  # Track what we challenged
          self.active_challenges: Dict[str, dict] = {}  # objection_id -> context
  ```
- [ ] Track challenge outcomes:
  ```python
  def record_challenge(self, target_id: str, objection_id: str,
                       weakness_type: str, consensus_size: int):
      """Record a challenge for follow-up."""
      self.challenge_history.append({
          'target_id': target_id,
          'objection_id': objection_id,
          'weakness_type': weakness_type,
          'consensus_size': consensus_size,
          'timestamp': time.time()
      })

      self.active_challenges[objection_id] = {
          'target_id': target_id,
          'needs_followup': True,
          'followup_count': 0
      }
  ```
- [ ] Follow up on ignored objections:
  ```python
  async def followup_on_challenges(self, signal_store: SignalStore,
                                   llm: SimpleLLM):
      """Check on active challenges and follow up if needed."""
      for objection_id, context in list(self.active_challenges.items()):
          if context['followup_count'] >= 3:
              # Stop after 3 follow-ups
              del self.active_challenges[objection_id]
              continue

          # Check if anyone responded
          had_impact = self.monitor_objection_impact(objection_id, signal_store)

          if not had_impact:
              # No response - escalate
              objection = signal_store.get_signal(objection_id)
              target = signal_store.get_signal(context['target_id'])

              # Generate stronger follow-up
              followup = await self.generate_followup(objection, target, llm)

              if followup:
                  followup_id = signal_store.deposit(
                      signal_type="OBJECTION",
                      content=followup,
                      strength=0.7,  # Stronger than original
                      depositor=self.agent_id,
                      parent=objection_id  # Chain from original objection
                  )

                  context['followup_count'] += 1
                  print(f"[HATER] {self.agent_id} follow-up {context['followup_count']} on {objection_id}")
  ```

**Acceptance Criteria:**
- [ ] Haters remember what they challenged
- [ ] Haters check if objections had impact
- [ ] Haters follow up on ignored objections (up to 3 times)
- [ ] Follow-ups are progressively stronger
- [ ] Challenge history persists across iterations

---

### **PHASE 2: PROGRAMMATIC SWARM HEALTH VERIFICATION**
*Priority: HIGH - Required for "never fail" guarantee*

#### Task 2.1: Create Swarm Health Monitor Agent
**File:** `swarm/agents/monitor.py` (NEW)

**Purpose:**
A meta-agent that watches the swarm and detects problems.

**Implementation:**
```python
"""Meta-agent for swarm health monitoring and self-healing."""

import asyncio
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from ..core.signal_store import SignalStore, Signal


@dataclass
class SwarmHealthMetrics:
    """Comprehensive swarm health metrics."""
    # Signal statistics
    total_signals: int
    signals_by_type: Dict[str, int]
    avg_strength: float
    strength_gini: float

    # Diversity metrics
    insight_diversity: float  # 0.0-1.0, how diverse are insights?
    evidence_diversity: float
    source_diversity: float

    # Interaction metrics
    objection_rate: float  # objections per insight
    validation_rate: float  # evidence per insight
    response_rate: float  # descendants per signal

    # Convergence metrics
    convergence_score: float  # 0.0-1.0, based on strength stability
    trajectory: str  # "converging", "oscillating", "stuck", "diverging"

    # Echo chamber detection
    echo_chamber_risk: float  # 0.0-1.0, based on consensus clustering
    groupthink_clusters: List[List[str]]  # Lists of similar insight IDs

    # Agent effectiveness
    agent_effectiveness: Dict[str, float]  # agent_id -> effectiveness score

    # Health flags
    is_healthy: bool
    warnings: List[str]
    critical_issues: List[str]


class SwarmMonitor:
    """Meta-agent that monitors swarm health and triggers corrections."""

    def __init__(self, agent_id: str = "SwarmMonitor"):
        self.agent_id = agent_id
        self.health_history: List[SwarmHealthMetrics] = []
        self.monitoring_interval: float = 10.0  # Check every 10 seconds
        self.active = True

    async def run(self, signal_store: SignalStore,
                  max_iterations: int = 100) -> None:
        """Monitor swarm health continuously."""
        for iteration in range(max_iterations):
            if not self.active:
                break

            # Collect metrics
            metrics = self.collect_metrics(signal_store)
            self.health_history.append(metrics)

            # Analyze health
            if not metrics.is_healthy:
                print(f"\n[MONITOR] ⚠️  SWARM HEALTH WARNING")
                for warning in metrics.warnings:
                    print(f"[MONITOR]   - {warning}")
                for issue in metrics.critical_issues:
                    print(f"[MONITOR]   ❌ CRITICAL: {issue}")

                # Trigger self-healing (if implemented)
                # await self.trigger_self_healing(signal_store, metrics)
            else:
                if iteration % 5 == 0:  # Report every 5 checks
                    print(f"\n[MONITOR] ✓ Swarm healthy")
                    print(f"[MONITOR]   Diversity: {metrics.insight_diversity:.2f}")
                    print(f"[MONITOR]   Objection rate: {metrics.objection_rate:.2f}")
                    print(f"[MONITOR]   Echo chamber risk: {metrics.echo_chamber_risk:.2f}")

            await asyncio.sleep(self.monitoring_interval)

    def collect_metrics(self, signal_store: SignalStore) -> SwarmHealthMetrics:
        """Collect comprehensive swarm health metrics."""
        stats = signal_store.get_stats()
        all_signals = signal_store.get_all_signals()

        # Calculate diversity metrics
        insights = [s for s in all_signals if s.type == "INSIGHT"]
        insight_diversity = self.calculate_diversity(insights)

        evidence = [s for s in all_signals if s.type == "EVIDENCE"]
        evidence_diversity = self.calculate_diversity(evidence)

        source_diversity = self.calculate_source_diversity(insights)

        # Calculate interaction metrics
        objection_count = len([s for s in all_signals if s.type in ["OBJECTION", "COUNTER_EVIDENCE"]])
        objection_rate = objection_count / max(len(insights), 1)

        evidence_count = len(evidence)
        validation_rate = evidence_count / max(len(insights), 1)

        # Calculate response rate (avg descendants per signal)
        response_rate = sum(len(signal_store.get_descendants(s.id))
                           for s in all_signals) / max(len(all_signals), 1)

        # Detect echo chambers
        groupthink_clusters = self.detect_echo_chambers(insights, signal_store)
        echo_chamber_risk = len(groupthink_clusters) / max(len(insights) / 5, 1)
        echo_chamber_risk = min(1.0, echo_chamber_risk)

        # Analyze convergence trajectory
        trajectory = self.analyze_trajectory()
        convergence_score = self.calculate_convergence_score()

        # Check health
        warnings = []
        critical_issues = []

        if insight_diversity < 0.3:
            warnings.append(f"Low insight diversity ({insight_diversity:.2f})")

        if objection_rate < 0.1:
            critical_issues.append(f"Very low objection rate ({objection_rate:.2f}) - no adversarial challenge")

        if echo_chamber_risk > 0.5:
            critical_issues.append(f"High echo chamber risk ({echo_chamber_risk:.2f})")

        if validation_rate < 0.5:
            warnings.append(f"Low validation rate ({validation_rate:.2f})")

        if trajectory == "stuck":
            critical_issues.append("Swarm convergence stuck - no progress")

        is_healthy = len(critical_issues) == 0

        return SwarmHealthMetrics(
            total_signals=stats['total_signals'],
            signals_by_type=stats['by_type'],
            avg_strength=stats['avg_strength'],
            strength_gini=stats.get('gini', 0.0),
            insight_diversity=insight_diversity,
            evidence_diversity=evidence_diversity,
            source_diversity=source_diversity,
            objection_rate=objection_rate,
            validation_rate=validation_rate,
            response_rate=response_rate,
            convergence_score=convergence_score,
            trajectory=trajectory,
            echo_chamber_risk=echo_chamber_risk,
            groupthink_clusters=groupthink_clusters,
            agent_effectiveness={},  # TODO: Implement
            is_healthy=is_healthy,
            warnings=warnings,
            critical_issues=critical_issues
        )

    def calculate_diversity(self, signals: List[Signal]) -> float:
        """Calculate semantic diversity of signal set."""
        if len(signals) < 2:
            return 1.0

        # Calculate pairwise similarity
        similarities = []
        for i, sig1 in enumerate(signals):
            for sig2 in signals[i+1:]:
                sim = self._similarity(sig1.content, sig2.content)
                similarities.append(sim)

        if not similarities:
            return 1.0

        # Diversity = 1 - avg_similarity
        avg_similarity = sum(similarities) / len(similarities)
        return 1.0 - avg_similarity

    def calculate_source_diversity(self, insights: List[Signal]) -> float:
        """Calculate diversity of source documents."""
        source_docs = set()
        for insight in insights:
            obs_ids = insight.metadata.get('observation_ids', [])
            for obs_id in obs_ids:
                # Extract source document from observation ID or metadata
                # This is a placeholder - actual implementation depends on metadata structure
                source_docs.add(obs_id.split('_')[0] if '_' in obs_id else obs_id)

        # Diversity = unique sources / total insights
        return len(source_docs) / max(len(insights), 1)

    def detect_echo_chambers(self, insights: List[Signal],
                            signal_store: SignalStore) -> List[List[str]]:
        """Detect clusters of similar insights (potential groupthink)."""
        clusters = []
        used = set()

        for insight in insights:
            if insight.id in used:
                continue

            # Find similar insights
            similar = []
            for other in insights:
                if other.id == insight.id or other.id in used:
                    continue

                sim = self._similarity(insight.content, other.content)
                if sim > 0.7:  # High similarity = potential echo chamber
                    similar.append(other.id)
                    used.add(other.id)

            if len(similar) >= 2:  # 3+ similar insights
                cluster = [insight.id] + similar
                clusters.append(cluster)
                used.add(insight.id)

        return clusters

    def analyze_trajectory(self) -> str:
        """Analyze convergence trajectory from health history."""
        if len(self.health_history) < 5:
            return "initializing"

        # Look at last 5 measurements
        recent = self.health_history[-5:]

        # Check if Gini is increasing (converging)
        gini_trend = [m.strength_gini for m in recent]
        if all(gini_trend[i] < gini_trend[i+1] for i in range(len(gini_trend)-1)):
            return "converging"

        # Check if Gini is stable (converged or stuck)
        gini_variance = self._variance(gini_trend)
        if gini_variance < 0.01:
            if recent[-1].strength_gini > 0.7:
                return "converged"
            else:
                return "stuck"

        # Check if oscillating
        if self._is_oscillating(gini_trend):
            return "oscillating"

        return "diverging"

    def calculate_convergence_score(self) -> float:
        """Calculate convergence score based on trajectory."""
        if len(self.health_history) < 2:
            return 0.0

        current = self.health_history[-1]

        # Score based on Gini coefficient and diversity
        gini_score = current.strength_gini
        diversity_score = current.insight_diversity

        # High Gini + moderate diversity = good convergence
        # High Gini + low diversity = echo chamber
        if diversity_score > 0.3:
            return gini_score
        else:
            return gini_score * 0.5  # Penalize low diversity

    def _similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity (placeholder)."""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    def _variance(self, values: List[float]) -> float:
        """Calculate variance of values."""
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)

    def _is_oscillating(self, values: List[float]) -> bool:
        """Check if values are oscillating."""
        if len(values) < 4:
            return False

        # Check for up-down-up-down pattern
        diffs = [values[i+1] - values[i] for i in range(len(values)-1)]
        sign_changes = sum(1 for i in range(len(diffs)-1)
                          if diffs[i] * diffs[i+1] < 0)

        return sign_changes >= len(diffs) / 2

    def stop(self):
        """Stop monitoring."""
        self.active = False
```

**Acceptance Criteria:**
- [ ] Monitor tracks 15+ health metrics
- [ ] Detects echo chambers (consensus clustering)
- [ ] Measures diversity (insights, evidence, sources)
- [ ] Calculates interaction rates (objections, validation, responses)
- [ ] Analyzes convergence trajectory (converging/stuck/oscillating/diverging)
- [ ] Flags warnings and critical issues
- [ ] Runs continuously during swarm execution

---

#### Task 2.2: Add Programmatic Verification Functions
**File:** `swarm/core/verification.py` (NEW)

**Purpose:**
Programmatic quality checks for all signal types.

**Implementation:**
```python
"""Programmatic verification functions for signal quality."""

from typing import List, Optional, Dict
from .signal_store import SignalStore, Signal
import re


class SignalVerifier:
    """Verify signal quality programmatically."""

    def __init__(self, signal_store: SignalStore):
        self.signal_store = signal_store

    def verify_insight_quality(self, insight: Signal) -> Dict[str, any]:
        """Verify insight is based on actual observations, not hallucination."""
        # Get claimed observations
        obs_ids = insight.metadata.get('observation_ids', [])

        if not obs_ids:
            return {
                'valid': False,
                'reason': 'no_observations',
                'score': 0.0
            }

        # Verify observations exist
        observations = []
        for obs_id in obs_ids:
            obs = self.signal_store.get_signal(obs_id)
            if obs:
                observations.append(obs)

        if len(observations) < len(obs_ids) * 0.5:
            return {
                'valid': False,
                'reason': 'missing_observations',
                'score': 0.2
            }

        # Check if insight content relates to observations
        relatedness_scores = []
        for obs in observations:
            score = self._calculate_relatedness(insight.content, obs.content)
            relatedness_scores.append(score)

        avg_relatedness = sum(relatedness_scores) / len(relatedness_scores) if relatedness_scores else 0.0

        if avg_relatedness < 0.3:
            return {
                'valid': False,
                'reason': 'low_relatedness',
                'score': avg_relatedness
            }

        # Check for source diversity
        source_docs = set()
        for obs in observations:
            source = obs.metadata.get('source_document', '')
            source_docs.add(source)

        source_diversity = len(source_docs) / len(observations) if observations else 0.0

        # Overall quality score
        quality_score = (avg_relatedness * 0.6 + source_diversity * 0.4)

        return {
            'valid': quality_score >= 0.5,
            'reason': 'verified' if quality_score >= 0.5 else 'low_quality',
            'score': quality_score,
            'details': {
                'observation_count': len(observations),
                'avg_relatedness': avg_relatedness,
                'source_diversity': source_diversity
            }
        }

    def verify_evidence_relevance(self, evidence: Signal) -> Dict[str, any]:
        """Verify evidence actually supports parent insight."""
        if not evidence.parent:
            return {'valid': False, 'reason': 'no_parent', 'score': 0.0}

        parent = self.signal_store.get_signal(evidence.parent)
        if not parent:
            return {'valid': False, 'reason': 'parent_not_found', 'score': 0.0}

        # Check relevance
        relevance = self._calculate_relatedness(evidence.content, parent.content)

        # Check for supporting vs contradicting
        is_supporting = self._is_supporting(evidence.content, parent.content)

        if not is_supporting:
            # This might be counter-evidence, which is valid
            # but should be marked differently
            return {
                'valid': True,
                'reason': 'counter_evidence',
                'score': relevance,
                'is_supporting': False
            }

        return {
            'valid': relevance >= 0.4,
            'reason': 'verified' if relevance >= 0.4 else 'low_relevance',
            'score': relevance,
            'is_supporting': True
        }

    def verify_objection_substantiveness(self, objection: Signal) -> Dict[str, any]:
        """Verify objection is substantive, not generic."""
        content = objection.content

        # Check length
        if len(content) < 80:
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
                           if re.search(phrase, content, re.IGNORECASE))

        # Check for specific details
        has_numbers = bool(re.search(r'\d+', content))
        has_proper_nouns = bool(re.search(r'\b[A-Z][a-z]+\b', content))
        has_technical_terms = len([w for w in content.split() if len(w) > 10]) > 2

        specificity_score = sum([has_numbers, has_proper_nouns, has_technical_terms]) / 3.0

        # Check for alternative explanation
        has_alternative = any(phrase in content.lower() for phrase in [
            'instead', 'alternative', 'rather', 'actually', 'in fact'
        ])

        # Overall quality
        if generic_count > 2 and not has_alternative:
            return {'valid': False, 'reason': 'too_generic', 'score': 0.3}

        quality_score = (specificity_score * 0.5 +
                        (1.0 if has_alternative else 0.0) * 0.5)

        return {
            'valid': quality_score >= 0.4,
            'reason': 'verified' if quality_score >= 0.4 else 'low_specificity',
            'score': quality_score,
            'details': {
                'has_numbers': has_numbers,
                'has_proper_nouns': has_proper_nouns,
                'has_technical_terms': has_technical_terms,
                'has_alternative': has_alternative
            }
        }

    def verify_consistency_across_insights(self, insights: List[Signal]) -> Dict[str, any]:
        """Check if insights contradict each other."""
        contradictions = []

        for i, insight1 in enumerate(insights):
            for insight2 in insights[i+1:]:
                if self._are_contradictory(insight1.content, insight2.content):
                    contradictions.append({
                        'insight1_id': insight1.id,
                        'insight2_id': insight2.id,
                        'insight1': insight1.content[:100],
                        'insight2': insight2.content[:100]
                    })

        consistency_score = 1.0 - (len(contradictions) / max(len(insights), 1))

        return {
            'consistent': len(contradictions) == 0,
            'contradictions': contradictions,
            'consistency_score': consistency_score
        }

    def _calculate_relatedness(self, text1: str, text2: str) -> float:
        """Calculate semantic relatedness (placeholder for actual embedding similarity)."""
        # For now, use word overlap
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        overlap = len(words1 & words2)
        union = len(words1 | words2)

        return overlap / union if union > 0 else 0.0

    def _is_supporting(self, evidence: str, claim: str) -> bool:
        """Check if evidence supports claim (vs contradicts)."""
        # Look for negation markers in evidence
        negation_markers = [
            'not', 'no', 'never', 'contrary', 'opposite', 'despite',
            'however', 'but', 'although', 'contradicts', 'refutes'
        ]

        evidence_lower = evidence.lower()
        negation_count = sum(1 for marker in negation_markers
                            if marker in evidence_lower)

        # If many negation markers, likely contradicting
        return negation_count < 3

    def _are_contradictory(self, text1: str, text2: str) -> bool:
        """Check if two texts contradict (simplified heuristic)."""
        # Look for opposite sentiment or negation patterns
        # This is a placeholder - real implementation would use NLI model

        # Extract key claims (simplified)
        claims1 = self._extract_claims(text1)
        claims2 = self._extract_claims(text2)

        for c1 in claims1:
            for c2 in claims2:
                if self._are_opposite(c1, c2):
                    return True

        return False

    def _extract_claims(self, text: str) -> List[str]:
        """Extract claims from text (simplified)."""
        # Split on sentence boundaries
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 20]

    def _are_opposite(self, claim1: str, claim2: str) -> bool:
        """Check if two claims are opposite (simplified)."""
        # Check for negation + similar words
        c1_lower = claim1.lower()
        c2_lower = claim2.lower()

        c1_has_not = any(neg in c1_lower for neg in ['not', 'no', 'never'])
        c2_has_not = any(neg in c2_lower for neg in ['not', 'no', 'never'])

        # If one has negation and other doesn't, and they share words
        if c1_has_not != c2_has_not:
            words1 = set(c1_lower.split())
            words2 = set(c2_lower.split())
            overlap = len(words1 & words2)

            if overlap > 3:  # Share significant words
                return True

        return False
```

**Acceptance Criteria:**
- [ ] Can verify insight quality (observation grounding, relatedness)
- [ ] Can verify evidence relevance (supports parent insight)
- [ ] Can verify objection substantiveness (not generic)
- [ ] Can check consistency across insights (detect contradictions)
- [ ] Returns detailed verification reports with scores and reasons

---

#### Task 2.3: Integrate Verification into Agent Workflows
**Files:** `swarm/agents/forager.py`, `swarm/agents/gatherer.py`, `swarm/agents/hater.py`

**Required Changes:**

**In Forager (swarm/agents/forager.py):**
```python
from ..core.verification import SignalVerifier

class Forager:
    def __init__(self, ...):
        # ... existing init ...
        self.verifier = None  # Set during run

    async def run(self, signal_store: SignalStore, llm: SimpleLLM, ...):
        self.verifier = SignalVerifier(signal_store)

        # ... existing run logic ...

        # After generating insight, VERIFY before depositing
        insight_content = await self.generate_insight(cluster, llm)

        # Create temporary signal for verification
        temp_signal = Signal(
            id="temp",
            type="INSIGHT",
            content=insight_content,
            strength=0.5,
            timestamp=time.time(),
            depositor=self.agent_id,
            metadata={'observation_ids': [obs.id for obs in cluster]}
        )

        # Verify quality
        verification = self.verifier.verify_insight_quality(temp_signal)

        if verification['valid']:
            # Use verification score as strength
            strength = verification['score']
            signal_id = signal_store.deposit(...)
            print(f"[FORAGER] {self.agent_id} verified insight (score: {strength:.2f})")
        else:
            print(f"[FORAGER] {self.agent_id} rejected low-quality insight: {verification['reason']}")
            # Don't deposit
```

**In Gatherer (swarm/agents/gatherer.py):**
```python
# After retrieving evidence, verify relevance before depositing
evidence_content = result['content']

# Create temp signal for verification
temp_evidence = Signal(
    id="temp",
    type="EVIDENCE",
    content=evidence_content,
    strength=0.6,
    timestamp=time.time(),
    depositor=self.agent_id,
    parent=insight.id
)

verification = self.verifier.verify_evidence_relevance(temp_evidence)

if verification['valid']:
    strength = verification['score']
    signal_store.deposit(
        signal_type="EVIDENCE",
        content=evidence_content,
        strength=strength,
        depositor=self.agent_id,
        parent=insight.id
    )
    print(f"[GATHERER] Verified evidence (relevance: {strength:.2f})")
else:
    print(f"[GATHERER] Rejected irrelevant evidence: {verification['reason']}")
```

**In Hater (swarm/agents/hater.py):**
```python
# After generating objection, verify substantiveness
objection_content = await self.generate_contradiction(target, llm)

temp_objection = Signal(
    id="temp",
    type="OBJECTION",
    content=objection_content,
    strength=0.5,
    timestamp=time.time(),
    depositor=self.agent_id,
    parent=target.id
)

verification = self.verifier.verify_objection_substantiveness(temp_objection)

if verification['valid']:
    strength = verification['score']
    signal_store.deposit(...)
    print(f"[HATER] Verified objection (quality: {strength:.2f})")
else:
    print(f"[HATER] Rejected generic objection: {verification['reason']}")
    # Regenerate with stronger prompt
    objection_content = await self.generate_stronger_objection(target, llm)
```

**Acceptance Criteria:**
- [ ] Foragers verify insights before depositing
- [ ] Gatherers verify evidence relevance before depositing
- [ ] Haters verify objection substantiveness before depositing
- [ ] Low-quality signals are rejected with clear reasons logged
- [ ] Verification scores are used to set signal strength (not fixed values)

---

### **PHASE 3: ENHANCE AGENT INTERACTION & DIALOGUE**
*Priority: HIGH - Required for "smooth interaction"*

#### Task 3.1: Add Agent Response Mechanism
**File:** `swarm/core/signal_store.py`

**Current Problem:**
Agents deposit signals but never respond to each other directly.

**Required Changes:**
- [ ] Add response tracking to Signal:
  ```python
  @dataclass
  class Signal:
      # ... existing fields ...
      responses: List[str] = field(default_factory=list)  # IDs of direct responses
      is_response_to: Optional[str] = None  # ID of signal this responds to
  ```
- [ ] Add response methods to SignalStore:
  ```python
  def deposit_response(self, signal_type: str, content: str, strength: float,
                      depositor: str, responding_to: str,
                      metadata: Optional[dict] = None) -> Optional[str]:
      """Deposit a signal that directly responds to another signal."""
      # Verify responding_to signal exists
      target = self.get_signal(responding_to)
      if not target:
          return None

      # Create response signal
      signal_id = self.deposit(
          signal_type=signal_type,
          content=content,
          strength=strength,
          depositor=depositor,
          parent=responding_to,  # Parent link
          metadata=metadata
      )

      if signal_id:
          # Update target's response list
          target.responses.append(signal_id)

          # Mark this signal as a response
          response_signal = self.signals[signal_id]
          response_signal.is_response_to = responding_to

      return signal_id

  def get_responses(self, signal_id: str) -> List[Signal]:
      """Get all direct responses to a signal."""
      signal = self.get_signal(signal_id)
      if not signal:
          return []

      return [self.signals[resp_id] for resp_id in signal.responses
              if resp_id in self.signals]

  def get_dialogue_thread(self, signal_id: str, max_depth: int = 10) -> List[Signal]:
      """Get full dialogue thread (signal + all responses recursively)."""
      thread = []

      def collect_thread(sig_id: str, depth: int):
          if depth >= max_depth:
              return

          sig = self.get_signal(sig_id)
          if not sig:
              return

          thread.append(sig)

          # Get responses
          for resp_id in sig.responses:
              collect_thread(resp_id, depth + 1)

      collect_thread(signal_id, 0)
      return thread
  ```

**Acceptance Criteria:**
- [ ] Signals can track direct responses
- [ ] `deposit_response()` method creates response links
- [ ] `get_responses()` retrieves all direct responses
- [ ] `get_dialogue_thread()` retrieves full conversation chains

---

#### Task 3.2: Enable Forager-Hater Dialogue
**Files:** `swarm/agents/forager.py`, `swarm/agents/hater.py`

**Purpose:**
When a hater challenges an insight, the forager who created it should be able to defend or refine it.

**Implementation in Forager:**
```python
async def defend_insights(self, signal_store: SignalStore, llm: SimpleLLM):
    """Check if any of my insights were challenged and defend them."""
    # Find insights I created
    my_insights = [s for s in signal_store.get_all_signals()
                   if s.type == "INSIGHT" and s.depositor == self.agent_id]

    for insight in my_insights:
        # Check for objections
        objections = [s for s in signal_store.get_responses(insight.id)
                     if s.type in ["OBJECTION", "COUNTER_EVIDENCE"]]

        if objections:
            # Check if I already responded
            my_responses = [obj for obj in objections
                           if any(r.depositor == self.agent_id
                                 for r in signal_store.get_responses(obj.id))]

            # Only respond to unanswered objections
            unanswered = [obj for obj in objections
                         if not any(r.depositor == self.agent_id
                                   for r in signal_store.get_responses(obj.id))]

            for objection in unanswered:
                # Generate defense or acknowledgment
                defense = await self.generate_defense(insight, objection, llm)

                if defense:
                    # Deposit as response
                    signal_store.deposit_response(
                        signal_type="INSIGHT",  # Refined insight or rebuttal
                        content=defense,
                        strength=0.6,
                        depositor=self.agent_id,
                        responding_to=objection.id
                    )
                    print(f"[FORAGER] {self.agent_id} defended {insight.id} against {objection.id}")

async def generate_defense(self, insight: Signal, objection: Signal,
                          llm: SimpleLLM) -> Optional[str]:
    """Generate defense or acknowledge valid criticism."""
    # Get original observations
    obs_ids = insight.metadata.get('observation_ids', [])
    observations = [signal_store.get_signal(oid) for oid in obs_ids]
    observation_content = '\n'.join(f"- {obs.content}" for obs in observations if obs)

    prompt = f"""You discovered this insight:
    "{insight.content}"

    Based on these observations:
    {observation_content}

    But an adversarial agent raised this objection:
    "{objection.content}"

    Task:
    1. Is the objection valid? Does it identify a real weakness?
    2. If valid, acknowledge and refine your insight to address it
    3. If invalid, defend your insight with evidence from the observations
    4. Be intellectually honest - admit if you were wrong

    Response:"""

    response = await llm.generate(prompt, max_tokens=200, temperature=0.7)
    return response.strip() if response else None
```

**Implementation in Hater:**
```python
async def engage_in_dialogue(self, signal_store: SignalStore, llm: SimpleLLM):
    """Check if anyone responded to my objections and follow up."""
    # Find objections I created
    my_objections = [s for s in signal_store.get_all_signals()
                    if s.type in ["OBJECTION", "COUNTER_EVIDENCE"]
                    and s.depositor == self.agent_id]

    for objection in my_objections:
        # Check for responses
        responses = signal_store.get_responses(objection.id)

        if responses:
            # Someone engaged! Continue the dialogue
            latest_response = responses[-1]  # Get most recent

            # Check if I already countered this response
            my_counters = [r for r in signal_store.get_responses(latest_response.id)
                          if r.depositor == self.agent_id]

            if not my_counters:
                # Generate counter-response
                counter = await self.generate_counter_response(
                    objection, latest_response, llm
                )

                if counter:
                    signal_store.deposit_response(
                        signal_type="OBJECTION",
                        content=counter,
                        strength=0.7,
                        depositor=self.agent_id,
                        responding_to=latest_response.id
                    )
                    print(f"[HATER] {self.agent_id} continued dialogue on {objection.id}")

async def generate_counter_response(self, objection: Signal, response: Signal,
                                   llm: SimpleLLM) -> Optional[str]:
    """Generate counter-response to a defense."""
    prompt = f"""You raised this objection:
    "{objection.content}"

    The original author responded:
    "{response.content}"

    Task:
    1. Did they address your objection adequately?
    2. If yes, acknowledge their point and suggest how to strengthen their insight
    3. If no, press the objection further with specific examples
    4. Be adversarial but constructive

    Counter-response:"""

    counter = await llm.generate(prompt, max_tokens=200, temperature=0.85)
    return counter.strip() if counter else None
```

**Acceptance Criteria:**
- [ ] Foragers check for objections to their insights
- [ ] Foragers generate defenses or acknowledge valid criticisms
- [ ] Haters check for responses to their objections
- [ ] Haters continue dialogue by responding to defenses
- [ ] Dialogue threads can go 3-5 exchanges deep
- [ ] Both sides are intellectually honest (admit when wrong)

---

### **PHASE 4: ROLE SPECIALIZATION & EFFECTIVENESS METRICS**
*Priority: MEDIUM - Required for "each agent excels at its job"*

#### Task 4.1: Add Agent Effectiveness Tracking
**File:** `swarm/core/agent_metrics.py` (NEW)

**Purpose:**
Track how well each agent type is performing its role.

**Implementation:**
```python
"""Agent effectiveness metrics for swarm optimization."""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from ..core.signal_store import SignalStore


@dataclass
class AgentPerformanceMetrics:
    """Performance metrics for an individual agent."""
    agent_id: str
    agent_type: str

    # Activity metrics
    signals_deposited: int = 0
    signals_rejected: int = 0  # Failed verification
    responses_generated: int = 0

    # Quality metrics
    avg_signal_strength: float = 0.0
    avg_signal_visits: float = 0.0  # How often others interact with this agent's signals
    avg_verification_score: float = 0.0

    # Impact metrics
    signals_amplified_by_others: int = 0  # Others corroborated
    signals_decayed_by_others: int = 0  # Others challenged
    dialogue_threads_started: int = 0
    dialogue_depth: float = 0.0  # Avg responses per signal

    # Role-specific metrics
    role_effectiveness: float = 0.0  # 0.0-1.0

    # Computed score
    overall_effectiveness: float = 0.0


class AgentMetricsTracker:
    """Track agent effectiveness metrics."""

    def __init__(self):
        self.agent_metrics: Dict[str, AgentPerformanceMetrics] = {}

    def record_signal_deposit(self, agent_id: str, agent_type: str,
                             signal_id: str, strength: float,
                             verification_score: Optional[float] = None):
        """Record that an agent deposited a signal."""
        metrics = self._get_or_create_metrics(agent_id, agent_type)
        metrics.signals_deposited += 1

        # Update averages
        total = metrics.signals_deposited + metrics.signals_rejected
        metrics.avg_signal_strength = (
            (metrics.avg_signal_strength * (total - 1) + strength) / total
        )

        if verification_score is not None:
            metrics.avg_verification_score = (
                (metrics.avg_verification_score * (total - 1) + verification_score) / total
            )

    def record_signal_rejection(self, agent_id: str, agent_type: str,
                               reason: str, verification_score: float):
        """Record that an agent's signal was rejected."""
        metrics = self._get_or_create_metrics(agent_id, agent_type)
        metrics.signals_rejected += 1

        # Update averages
        total = metrics.signals_deposited + metrics.signals_rejected
        metrics.avg_verification_score = (
            (metrics.avg_verification_score * (total - 1) + verification_score) / total
        )

    def record_response(self, agent_id: str, agent_type: str,
                       response_to: str):
        """Record that an agent responded to another signal."""
        metrics = self._get_or_create_metrics(agent_id, agent_type)
        metrics.responses_generated += 1

    def calculate_effectiveness(self, agent_id: str, agent_type: str,
                               signal_store: SignalStore) -> float:
        """Calculate role-specific effectiveness."""
        metrics = self._get_or_create_metrics(agent_id, agent_type)

        # Get agent's signals
        agent_signals = [s for s in signal_store.get_all_signals()
                        if s.depositor == agent_id]

        if not agent_signals:
            return 0.0

        # Update visit metrics
        total_visits = sum(s.visits for s in agent_signals)
        metrics.avg_signal_visits = total_visits / len(agent_signals)

        # Update dialogue metrics
        signals_with_responses = [s for s in agent_signals
                                 if signal_store.get_responses(s.id)]
        metrics.dialogue_threads_started = len(signals_with_responses)

        if signals_with_responses:
            total_responses = sum(len(signal_store.get_responses(s.id))
                                for s in signals_with_responses)
            metrics.dialogue_depth = total_responses / len(signals_with_responses)

        # Calculate role-specific effectiveness
        if agent_type == "Scout":
            # Scouts: effectiveness = observation quality + diversity
            effectiveness = self._calculate_scout_effectiveness(agent_signals, signal_store)
        elif agent_type == "Forager":
            # Foragers: effectiveness = insight quality + novelty + validation
            effectiveness = self._calculate_forager_effectiveness(agent_signals, signal_store)
        elif agent_type == "Gatherer":
            # Gatherers: effectiveness = evidence quality + validation rate
            effectiveness = self._calculate_gatherer_effectiveness(agent_signals, signal_store)
        elif agent_type == "Critic":
            # Critics: effectiveness = accuracy + calibration
            effectiveness = self._calculate_critic_effectiveness(agent_signals, signal_store)
        elif agent_type == "Hater":
            # Haters: effectiveness = objection impact + dialogue engagement
            effectiveness = self._calculate_hater_effectiveness(agent_signals, signal_store)
        else:
            effectiveness = 0.5

        metrics.role_effectiveness = effectiveness

        # Overall effectiveness
        metrics.overall_effectiveness = (
            metrics.role_effectiveness * 0.5 +
            metrics.avg_verification_score * 0.3 +
            min(metrics.dialogue_depth / 2.0, 1.0) * 0.2
        )

        return metrics.overall_effectiveness

    def _calculate_scout_effectiveness(self, signals: List,
                                      signal_store: SignalStore) -> float:
        """Scout effectiveness = observation quality + contribution to insights."""
        if not signals:
            return 0.0

        # Check how many insights reference these observations
        observation_ids = [s.id for s in signals]
        insights = signal_store.get_all_signals()
        insights = [i for i in insights if i.type == "INSIGHT"]

        referenced_count = 0
        for obs_id in observation_ids:
            for insight in insights:
                obs_refs = insight.metadata.get('observation_ids', [])
                if obs_id in obs_refs:
                    referenced_count += 1
                    break  # Count once per observation

        reference_rate = referenced_count / len(observation_ids)

        # Scout effectiveness = verification score + reference rate
        avg_score = sum(s.strength for s in signals) / len(signals)

        return avg_score * 0.5 + reference_rate * 0.5

    def _calculate_forager_effectiveness(self, signals: List,
                                        signal_store: SignalStore) -> float:
        """Forager effectiveness = insight quality + validation + novelty."""
        insights = [s for s in signals if s.type == "INSIGHT"]
        if not insights:
            return 0.0

        # Quality = avg strength
        avg_strength = sum(i.strength for i in insights) / len(insights)

        # Validation = % with evidence
        validated_count = 0
        for insight in insights:
            evidence = signal_store.get_descendants(insight.id, "EVIDENCE")
            if len(evidence) >= 2:
                validated_count += 1
        validation_rate = validated_count / len(insights)

        # Novelty = diversity score (how different from other insights)
        # Higher visits = more novel/interesting
        avg_visits = sum(i.visits for i in insights) / len(insights)
        novelty_score = min(avg_visits / 5.0, 1.0)

        return avg_strength * 0.4 + validation_rate * 0.4 + novelty_score * 0.2

    def _calculate_gatherer_effectiveness(self, signals: List,
                                         signal_store: SignalStore) -> float:
        """Gatherer effectiveness = evidence quality + relevance."""
        evidence = [s for s in signals if s.type == "EVIDENCE"]
        if not evidence:
            return 0.0

        # Quality = avg strength
        avg_strength = sum(e.strength for e in evidence) / len(evidence)

        # Relevance = % that actually helped validate insights
        # (Check if parent insights have high strength)
        relevant_count = 0
        for ev in evidence:
            if ev.parent:
                parent = signal_store.get_signal(ev.parent)
                if parent and parent.strength >= 0.7:
                    relevant_count += 1
        relevance_rate = relevant_count / len(evidence)

        return avg_strength * 0.5 + relevance_rate * 0.5

    def _calculate_critic_effectiveness(self, signals: List,
                                       signal_store: SignalStore) -> float:
        """Critic effectiveness = calibration (how accurate were strength adjustments)."""
        # This is tricky - need to track if critic's adjustments were correct
        # For now, use dialogue engagement as proxy
        critiques = [s for s in signals if s.type == "CRITIQUE"]
        if not critiques:
            return 0.0

        # Check if critiques sparked dialogue
        engaged_count = 0
        for critique in critiques:
            responses = signal_store.get_responses(critique.id)
            if responses:
                engaged_count += 1

        engagement_rate = engaged_count / len(critiques)

        # Check avg strength (higher = more confident)
        avg_strength = sum(c.strength for c in critiques) / len(critiques)

        return engagement_rate * 0.6 + avg_strength * 0.4

    def _calculate_hater_effectiveness(self, signals: List,
                                      signal_store: SignalStore) -> float:
        """Hater effectiveness = objection impact + dialogue engagement."""
        objections = [s for s in signals if s.type in ["OBJECTION", "COUNTER_EVIDENCE"]]
        if not objections:
            return 0.0

        # Impact = did target's strength decrease?
        impact_count = 0
        for objection in objections:
            if objection.parent:
                target = signal_store.get_signal(objection.parent)
                if target:
                    # Check if target's strength is below average
                    # (hard to track historical strength without metadata)
                    if target.strength < 0.6:
                        impact_count += 1

        impact_rate = impact_count / len(objections)

        # Engagement = did anyone respond?
        engaged_count = 0
        for objection in objections:
            responses = signal_store.get_responses(objection.id)
            if responses:
                engaged_count += 1

        engagement_rate = engaged_count / len(objections)

        # Hater effectiveness = impact + engagement
        return impact_rate * 0.6 + engagement_rate * 0.4

    def _get_or_create_metrics(self, agent_id: str,
                               agent_type: str) -> AgentPerformanceMetrics:
        """Get or create metrics for an agent."""
        if agent_id not in self.agent_metrics:
            self.agent_metrics[agent_id] = AgentPerformanceMetrics(
                agent_id=agent_id,
                agent_type=agent_type
            )
        return self.agent_metrics[agent_id]

    def get_metrics(self, agent_id: str) -> Optional[AgentPerformanceMetrics]:
        """Get metrics for an agent."""
        return self.agent_metrics.get(agent_id)

    def get_all_metrics(self) -> List[AgentPerformanceMetrics]:
        """Get all agent metrics."""
        return list(self.agent_metrics.values())

    def print_report(self):
        """Print effectiveness report for all agents."""
        print("\n" + "=" * 80)
        print("AGENT EFFECTIVENESS REPORT")
        print("=" * 80)

        # Group by agent type
        by_type: Dict[str, List[AgentPerformanceMetrics]] = {}
        for metrics in self.agent_metrics.values():
            if metrics.agent_type not in by_type:
                by_type[metrics.agent_type] = []
            by_type[metrics.agent_type].append(metrics)

        for agent_type, agents in by_type.items():
            print(f"\n--- {agent_type}s ---")

            # Sort by effectiveness
            agents.sort(key=lambda a: a.overall_effectiveness, reverse=True)

            # Print top 3 and bottom 3
            print("\nTop performers:")
            for metrics in agents[:3]:
                print(f"  {metrics.agent_id}:")
                print(f"    Overall effectiveness: {metrics.overall_effectiveness:.2f}")
                print(f"    Role effectiveness: {metrics.role_effectiveness:.2f}")
                print(f"    Signals deposited: {metrics.signals_deposited}")
                print(f"    Signals rejected: {metrics.signals_rejected}")
                print(f"    Avg strength: {metrics.avg_signal_strength:.2f}")
                print(f"    Dialogue depth: {metrics.dialogue_depth:.2f}")

            if len(agents) > 6:
                print("\nBottom performers:")
                for metrics in agents[-3:]:
                    print(f"  {metrics.agent_id}:")
                    print(f"    Overall effectiveness: {metrics.overall_effectiveness:.2f}")
                    print(f"    Role effectiveness: {metrics.role_effectiveness:.2f}")
                    print(f"    Signals deposited: {metrics.signals_deposited}")

            # Type averages
            avg_effectiveness = sum(a.overall_effectiveness for a in agents) / len(agents)
            print(f"\n{agent_type} average effectiveness: {avg_effectiveness:.2f}")
```

**Acceptance Criteria:**
- [ ] Tracks individual agent performance metrics
- [ ] Calculates role-specific effectiveness (different for each agent type)
- [ ] Identifies top and bottom performers by type
- [ ] Prints effectiveness reports
- [ ] Can be used to adjust agent populations dynamically

---

### **PHASE 5: SELF-HEALING & AUTOMATIC RECOVERY**
*Priority: MEDIUM - Required for "never fail" guarantee*

#### Task 5.1: Add Self-Healing to SwarmMonitor
**File:** `swarm/agents/monitor.py`

**Purpose:**
Automatically fix detected issues.

**Implementation:**
```python
async def trigger_self_healing(self, signal_store: SignalStore,
                               metrics: SwarmHealthMetrics,
                               agent_pools: Dict[str, list]) -> None:
    """Trigger self-healing actions based on health issues."""
    print(f"\n[MONITOR] 🔧 Initiating self-healing...")

    # Issue 1: Low objection rate (haters not doing their job)
    if metrics.objection_rate < 0.1:
        print(f"[MONITOR]   Spawning additional haters (objection rate: {metrics.objection_rate:.2f})")
        # Signal to spawn more haters
        # This would require coordination with the orchestrator
        pass

    # Issue 2: Echo chambers detected
    if metrics.echo_chamber_risk > 0.5:
        print(f"[MONITOR]   Boosting diversity mechanisms (echo risk: {metrics.echo_chamber_risk:.2f})")
        # Increase exploration bonus
        signal_store.exploration_bonus = min(0.8, signal_store.exploration_bonus * 1.5)

        # Decay consensus clusters
        for cluster in metrics.groupthink_clusters:
            for insight_id in cluster:
                insight = signal_store.get_signal(insight_id)
                if insight:
                    insight.strength *= 0.8  # Decay by 20%
                    print(f"[MONITOR]     Decayed echo chamber insight {insight_id}")

    # Issue 3: Low validation rate
    if metrics.validation_rate < 0.5:
        print(f"[MONITOR]   Boosting gatherer priority (validation rate: {metrics.validation_rate:.2f})")
        # This would signal orchestrator to increase gatherer iterations
        pass

    # Issue 4: Convergence stuck
    if metrics.trajectory == "stuck":
        print(f"[MONITOR]   Injecting diversity (convergence stuck)")
        # Boost weak signals
        all_signals = signal_store.get_all_signals()
        weak_signals = [s for s in all_signals if s.strength < 0.4]
        for signal in weak_signals:
            signal.strength = min(1.0, signal.strength * 1.3)
            print(f"[MONITOR]     Boosted weak signal {signal.id}")

    # Issue 5: Low diversity
    if metrics.insight_diversity < 0.3:
        print(f"[MONITOR]   Forcing diversity (insight diversity: {metrics.insight_diversity:.2f})")
        # Increase diversity threshold
        signal_store.diversity_threshold = min(0.98, signal_store.diversity_threshold * 1.05)
```

**Acceptance Criteria:**
- [ ] Detects and responds to low objection rate (spawns more haters)
- [ ] Detects and responds to echo chambers (decays clusters, boosts exploration)
- [ ] Detects and responds to low validation (prioritizes gatherers)
- [ ] Detects and responds to stuck convergence (boosts weak signals)
- [ ] Detects and responds to low diversity (raises diversity threshold)
- [ ] All actions logged clearly

---

#### Task 5.2: Add Dynamic Agent Population Adjustment
**File:** `swarm/monolith_breaking.py`

**Purpose:**
Adjust agent populations based on effectiveness metrics and health issues.

**Implementation:**
```python
async def adaptive_agent_management(
    signal_store: SignalStore,
    agent_metrics: AgentMetricsTracker,
    monitor: SwarmMonitor,
    llm: SimpleLLM,
    current_agents: Dict[str, list],
    max_adjustment_rate: float = 0.2
) -> Dict[str, list]:
    """Adjust agent populations based on effectiveness."""
    # Get latest health metrics
    if not monitor.health_history:
        return current_agents

    latest_health = monitor.health_history[-1]

    adjustments = {}

    # If low objection rate, increase haters
    if latest_health.objection_rate < 0.1:
        current_hater_count = len(current_agents.get('haters', []))
        new_count = int(current_hater_count * (1 + max_adjustment_rate))

        print(f"[ADAPTIVE] Increasing haters from {current_hater_count} to {new_count}")

        # Spawn new haters
        new_haters = [
            Hater(f"Hater_adaptive_{i}", task_prompt="Challenge emerging consensus")
            for i in range(new_count - current_hater_count)
        ]

        # Launch new haters
        for hater in new_haters:
            asyncio.create_task(
                hater.run(signal_store, llm, max_actions=200, temperature=0.85)
            )

        current_agents['haters'].extend(new_haters)
        adjustments['haters'] = new_count

    # If low validation rate, increase gatherers
    if latest_health.validation_rate < 0.5:
        current_gatherer_count = len(current_agents.get('gatherers', []))
        new_count = int(current_gatherer_count * (1 + max_adjustment_rate))

        print(f"[ADAPTIVE] Increasing gatherers from {current_gatherer_count} to {new_count}")

        new_gatherers = [
            Gatherer(f"Gatherer_adaptive_{i}")
            for i in range(new_count - current_gatherer_count)
        ]

        for gatherer in new_gatherers:
            asyncio.create_task(
                gatherer.run(signal_store, llm, max_actions=150)
            )

        current_agents['gatherers'].extend(new_gatherers)
        adjustments['gatherers'] = new_count

    # If effectiveness is very low for a type, spawn more high performers
    for agent_type, agents in current_agents.items():
        type_metrics = [agent_metrics.get_metrics(a.agent_id)
                       for a in agents if agent_metrics.get_metrics(a.agent_id)]

        if type_metrics:
            avg_effectiveness = sum(m.overall_effectiveness for m in type_metrics) / len(type_metrics)

            if avg_effectiveness < 0.3:
                print(f"[ADAPTIVE] Low {agent_type} effectiveness ({avg_effectiveness:.2f})")
                # This is a signal that we need to review agent implementations
                # For now, just log

    return adjustments

# Integrate into main loop
async def run_document_swarm(...):
    # ... existing setup ...

    # Create monitor
    monitor = SwarmMonitor()
    monitor_task = asyncio.create_task(
        monitor.run(signal_store, max_iterations=100)
    )

    # Create metrics tracker
    agent_metrics = AgentMetricsTracker()

    # Track all agents
    current_agents = {
        'scouts': scouts,
        'foragers': foragers,
        'gatherers': gatherers,
        'critics': critics,
        'haters': haters
    }

    # Launch adaptive management (runs periodically)
    async def adaptive_loop():
        for i in range(20):  # Check every ~30 seconds for 10 minutes
            await asyncio.sleep(30)
            adjustments = await adaptive_agent_management(
                signal_store, agent_metrics, monitor, llm,
                current_agents, max_adjustment_rate=0.2
            )
            if adjustments:
                print(f"[ADAPTIVE] Made adjustments: {adjustments}")

    adaptive_task = asyncio.create_task(adaptive_loop())

    # ... existing agent execution ...

    # Wait for all tasks
    await asyncio.gather(
        *scout_tasks,
        *forager_tasks,
        *validation_tasks,
        monitor_task,
        adaptive_task,
        return_exceptions=True
    )
```

**Acceptance Criteria:**
- [ ] Monitors agent effectiveness continuously
- [ ] Spawns additional agents when effectiveness is low
- [ ] Prioritizes high-performing agent types
- [ ] Adjusts agent populations based on health metrics
- [ ] Logs all adjustments clearly
- [ ] Never reduces agents below minimum thresholds

---

## 📊 SUMMARY: TRANSFORMATION

### **Before:**
- Critics count evidence but don't read it (accountants)
- Haters outnumbered 75:1 (100 actions vs 7,500)
- Haters attack strongest signals (wrong targets)
- No agent dialogue (only one-way deposits)
- No programmatic verification
- Echo chambers form easily
- No swarm health monitoring
- No self-healing

### **After:**
- Critics generate reasoned critiques with full context
- Haters run 200+ iterations (match foragers in power)
- Haters target consensus clusters (groupthink detection)
- Agent dialogue enabled (foragers defend, haters follow up)
- All signals programmatically verified before deposit
- Echo chambers detected and actively decayed
- Real-time swarm health monitoring
- Automatic self-healing (spawn agents, boost diversity, etc.)
- Agent effectiveness tracked and optimized
- Never-failing swarm mesh with programmatic guarantees

---

## 🎯 SUCCESS CRITERIA

The swarm mesh is successful when:

1. **✓ Objection rate ≥ 15%** - Every ~6-7 insights gets challenged
2. **✓ Insight diversity ≥ 0.4** - Multiple distinct perspectives represented
3. **✓ Echo chamber risk < 0.3** - Minimal consensus clustering
4. **✓ Validation rate ≥ 60%** - Most insights externally validated
5. **✓ Hater effectiveness ≥ 0.5** - Objections have measurable impact
6. **✓ Critic effectiveness ≥ 0.6** - Critiques spark dialogue
7. **✓ Dialogue depth ≥ 2.0** - Average 2+ responses per challenged signal
8. **✓ Zero critical failures** - Swarm never crashes due to agent failures
9. **✓ Self-healing triggers < 3** - Minimal health issues requiring intervention
10. **✓ Convergence trajectory = "converging"** - Making steady progress toward truth

---

## 🗺️ IMPLEMENTATION PRIORITY

**Week 1 (CRITICAL):**
- Task 1.1: Enhanced context for critics
- Task 1.2: Transform haters into consensus challengers
- Task 2.1: Create SwarmMonitor

**Week 2 (HIGH):**
- Task 1.3: Hater persistence
- Task 2.2: Programmatic verification
- Task 2.3: Integrate verification

**Week 3 (HIGH):**
- Task 3.1: Agent response mechanism
- Task 3.2: Forager-hater dialogue

**Week 4 (MEDIUM):**
- Task 4.1: Agent effectiveness tracking
- Task 5.1: Self-healing
- Task 5.2: Dynamic population adjustment

**Total Estimated Effort:** 3-4 weeks of focused development

---

**END OF ROADMAP**
