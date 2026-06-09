# Agent Architecture Audit - Complete Report

**Date**: 2025-11-14
**Session**: claude/debug-function-argument-mismatches-011CV5EncuCf8JXsKhFWrXHT

## Executive Summary

Conducted comprehensive audit of all 5 existing agents in the stigmergic swarm system. **All existing agents justified their existence** with distinct, non-overlapping roles. Identified 2 **CRITICAL** missing roles and implemented them:

1. **Validator** - Fact-checking and source verification
2. **Pruner** - Active signal quality management

The swarm now has **7 agents** with clear role separation and easy add/remove capability via unified structure.

---

## Part 1: Existing Agents - Justification Analysis

### ✓ Scout (scout.py) - **KEEP**

**Role**: Initial exploration - generates SOLUTION/DRAFT signals

**Strengths**:
- Dynamic web retrieval integration (450 char context snippets)
- Quality checks: repetition detection (30% unique threshold), minimum length (15 chars)
- Token optimized (70 tokens for fast exploration)
- Event-driven with web search capability

**Weaknesses**:
- Heuristic strength assessment (word counting)

**Verdict**: **ESSENTIAL** - No redundancy with other agents
**Action**: None required

---

### ✓ Forager (forager.py) - **KEEP**

**Role**: Elaborates scout signals - creates IMPLEMENTATION/CHALLENGE signals

**Strengths**:
- Event-driven (waits for input signals before acting)
- Pattern discovery across observations (document mode)
- Signal verification system (min_quality_score=0.4)
- Can defend insights when challenged
- Token optimized (100 tokens)

**Weaknesses**:
- Mixed modes (document pattern finding vs. creative elaboration)

**Verdict**: **ESSENTIAL** - Unique role building argument structure
**Action**: None required

---

### ✓ Critic (critic.py) - **KEEP**

**Role**: Quality evaluation - adjusts signal strength (0.6x-1.5x multiplier)

**Strengths**:
- Aggressive differentiation (0.6x-1.5x range for strong boosting/decay)
- Enhanced mode with full provenance context
- Stratified sampling (weak, medium, strong signals)
- Generates reasoned critiques with LLM
- Deposits CRITIQUE signals for transparency

**Weaknesses**:
- Multiplier calculation uses simple word counting
- No fact-checking (only evaluates argument quality, not factual accuracy)

**Verdict**: **ESSENTIAL** - Distinct from Validator (quality vs. accuracy)
**Action**: None required

---

### ✓ Hater (hater.py) - **KEEP**

**Role**: Adversarial testing - generates OBJECTION/COUNTER_EVIDENCE signals

**Strengths**:
- Consensus cluster targeting (detects groupthink)
- Substantiveness verification (80+ chars, specific details required)
- Dialogue capability (responds to defenses)
- Consensus weakness analysis (evidence_diversity, source_diversity)
- Enhanced prompt for stronger contradictions

**Weaknesses**:
- 120 tokens may limit nuanced objections

**Verdict**: **ESSENTIAL** - Unique adversarial role, prevents groupthink
**Action**: None required

---

### ✓ Synthesizer (synthesizer.py) - **KEEP**

**Role**: Final consolidation - produces coherent answer from full discourse

**Strengths**:
- Shows full argument graph (signal + children + critiques)
- Considers FULL discourse structure (not just isolated signals)
- 200 tokens for comprehensive synthesis
- Direct answer to original question

**Weaknesses**:
- Only runs at end (no intermediate synthesis)

**Verdict**: **ESSENTIAL** - No other agent produces final answer
**Action**: None required

---

## Part 2: Missing Roles Identified

Analyzed discourse flow: `Scout → Forager → Critic → Hater → Synthesizer`

| Agent | Gap | Impact | Priority | Status |
|-------|-----|--------|----------|--------|
| **Validator** | No fact-checking or source verification | **HIGH** | **CRITICAL** | ✓ Implemented |
| **Pruner** | No active signal quality management | **HIGH** | **CRITICAL** | ✓ Implemented |
| Connector | No cross-thread relationship finding | MEDIUM | LOW | Future work |
| Amplifier | No explicit consensus tracking | LOW | LOW | Future work |
| Diversifier | No perspective gap detection | LOW | LOW | Future work |

---

## Part 3: New Agents Implemented

### ✓ Validator (validator.py) - **NEW**

**Role**: Fact-checking and source verification

**Distinct from Critic**:
- Critic evaluates **ARGUMENT QUALITY** (logic, coherence, strength)
- Validator verifies **FACTUAL ACCURACY** (claims, sources, evidence)

**Key Features**:
- Targets signals with factual claims (numbers, proper nouns, citations)
- Generates VERIFICATION signals (score 0.4-0.8)
- Boosts verified signals by 1.2x, decays unverified by 0.75x
- Low temperature (0.5) for factual checking
- Verification format:
  ```
  ACCURACY: [HIGH/MEDIUM/LOW]
  REASONING: [1-2 sentences]
  ```

**Signal Flow**:
```
SOLUTION (Scout)
  ↳ IMPLEMENTATION (Forager)
    ↳ VERIFICATION (Validator) ← Checks factual accuracy
    ↳ CRITIQUE (Critic) ← Checks argument quality
    ↳ OBJECTION (Hater) ← Adversarial testing
```

**Configuration**:
- `NUM_VALIDATORS = 1` in config.py
- Max tokens: 120
- Min strength: 0.3
- Runs per round: ITERATIONS_PER_ROUND

---

### ✓ Pruner (pruner.py) - **NEW**

**Role**: Active signal quality management

**Distinct from Passive Decay**:
- Decay is passive (automatic strength reduction over time)
- Pruner is active (intelligent removal based on multiple criteria)

**Removal Criteria**:
1. **Weak signals**: strength < 0.15 (PRUNE_THRESHOLD)
2. **Stale signals**: age > 120s AND visits ≤ 1
3. **Duplicates**: similarity > 0.85 to stronger signal of same type
4. **Orphaned**: parent signal was removed

**Key Features**:
- No LLM dependency (pure signal store analysis)
- Tracks total pruned count across rounds
- Runs every 3 seconds (longer delay than other agents)
- Similarity detection using SequenceMatcher (fast string comparison)

**Configuration**:
- `NUM_PRUNERS = 1` in config.py
- Min strength: 0.15 (PRUNE_THRESHOLD)
- Staleness threshold: 120s
- Similarity threshold: 0.85
- Runs per round: ITERATIONS_PER_ROUND // 3

---

## Part 4: Code Changes Summary

### Files Modified:

1. **swarm/core/config.py**
   - Added `NUM_VALIDATORS = 1`
   - Added `NUM_PRUNERS = 1`

2. **run_task.py**
   - Imported `Validator` and `Pruner`
   - Updated agent count display (line 305-307)
   - Added validator creation (line 446-449)
   - Added pruner creation (line 451-454)
   - Added validator launching (line 490-497)
   - Added pruner launching (line 499-503)

3. **swarm/agents/validator.py** - **NEW FILE**
   - 250 lines
   - Fact-checking logic with accuracy levels (HIGH/MEDIUM/LOW)
   - Signal strength adjustment based on verification
   - Targets claims with numbers, proper nouns, citations

4. **swarm/agents/pruner.py** - **NEW FILE**
   - 180 lines
   - Four removal criteria (weak, stale, duplicate, orphaned)
   - SequenceMatcher-based similarity detection
   - Statistics tracking (total_pruned counter)

---

## Part 5: Unified Agent Structure

All agents follow consistent patterns for easy add/remove:

### Agent Lifecycle:
```python
class NewAgent:
    def __init__(self, agent_id: str, ...):
        self.agent_id = agent_id
        self.active = True
        self.actions_taken = 0

    async def run(self, signal_store: SignalStore, llm: SimpleLLM,
                  max_actions: int = None):
        while self.active and (max_actions is None or
                               self.actions_taken < max_actions):
            # Agent behavior
            self.actions_taken += 1
            await asyncio.sleep(random.uniform(0.3, 0.7))

    def stop(self):
        self.active = False
```

### Integration Pattern:
```python
# 1. Add to config.py
NUM_NEW_AGENTS = 2

# 2. Import in run_task.py
from swarm.agents.new_agent import NewAgent

# 3. Create agents in round loop
new_agents = [
    NewAgent(f"NewAgent_R{round_num}_{i}", ...)
    for i in range(NUM_NEW_AGENTS)
]

# 4. Launch agents
for agent in new_agents:
    tasks.append(asyncio.create_task(
        agent.run(signal_store, llm, max_actions=ITERATIONS_PER_ROUND)
    ))
```

---

## Part 6: Testing & Validation

### Hyper Test Mode:
- Fast end-to-end validation (~60 seconds)
- 2 rounds instead of 3
- 5 iterations per round instead of 20
- Validates full pipeline: Scouts → Foragers → Critics → Haters → **Validators** → **Pruners** → Synthesis

### Run Command:
```bash
python run_task.py hyper_test
```

### Expected Output:
```
✓ HYPER TEST PASSED - All pipeline components working!

Validated:
  ✓ Model loading and generation
  ✓ Web search and information retrieval
  ✓ Scout signal generation
  ✓ Forager elaboration (IMPLEMENTATION/CHALLENGE)
  ✓ Critic evaluation and boosting
  ✓ Hater objections
  ✓ Validator fact-checking (NEW)
  ✓ Pruner signal management (NEW)
  ✓ Round-based iterative refinement
  ✓ Full discourse graph synthesis
```

---

## Part 7: Agent Interaction Map

```
┌─────────────────────────────────────────────────────────────┐
│                       SWARM ARCHITECTURE                     │
└─────────────────────────────────────────────────────────────┘

EXPLORATION PHASE:
  Scout (4x) ──[SOLUTION]──> Signal Store
                                  │
                                  ↓
ELABORATION PHASE:
  Forager (4x) ──[IMPLEMENTATION]──> Signal Store
               ──[CHALLENGE]──────> Signal Store
                                  │
                                  ↓
EVALUATION PHASE:
  Critic (2x) ──[CRITIQUE]──> Signal Store (adjust strength 0.6x-1.5x)
  Validator (1x) ──[VERIFICATION]──> Signal Store (check facts)
                                  │
                                  ↓
ADVERSARIAL PHASE:
  Hater (2x) ──[OBJECTION]──> Signal Store (challenge consensus)
                                  │
                                  ↓
MAINTENANCE PHASE:
  Pruner (1x) ──[REMOVE]──> Signal Store (weak/stale/duplicates)
                                  │
                                  ↓
SYNTHESIS PHASE:
  Synthesizer (1x) ──[FINAL_ANSWER]──> User
```

---

## Part 8: Future Enhancements (Optional)

### Low-Priority Agents:

1. **Connector** (Medium Impact)
   - Role: Find relationships between disparate signals
   - Signal Type: CONNECTION
   - When: If synthesis quality needs improvement

2. **Amplifier** (Low Impact)
   - Role: Boost signals with multi-agent agreement
   - Signal Type: CONSENSUS
   - When: If consensus tracking needed

3. **Diversifier** (Low Impact)
   - Role: Identify perspective gaps
   - Signal Type: ALTERNATIVE_PERSPECTIVE
   - When: If perspective coverage needs improvement

---

## Conclusion

✓ **All 5 existing agents justified** - No redundancy found
✓ **2 critical gaps filled** - Validator and Pruner added
✓ **Unified structure maintained** - Easy to add/remove agents
✓ **Full integration complete** - Ready for testing

The swarm now has **7 specialized agents** with clear role separation:
1. Scout - Initial exploration
2. Forager - Elaboration
3. Critic - Quality evaluation
4. Hater - Adversarial testing
5. **Validator** - Fact-checking (NEW)
6. **Pruner** - Signal management (NEW)
7. Synthesizer - Final answer

Each agent has a distinct, justified role. No overlaps detected.
