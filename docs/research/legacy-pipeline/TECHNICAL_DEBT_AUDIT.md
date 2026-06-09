# Technical Debt Audit

**⚠️ DOCUMENT STATUS:** Created 2024-11-15. P0 issues FIXED:
- ✅ **P0 FIXED:** Hardcoded signal types (commit 426a237)
- ✅ **P0 FIXED:** Hater objection generation (commit 2ced80b)
- ❌ **P1-P2:** God objects, monkey patching still exist
- ❌ **P3:** No unit tests (being addressed in current session)

**See `research/CORRECTED_FINDINGS.md` for accurate current state.**

**Created:** 2024-11-15
**Last Updated:** 2025-11-17 (status annotations added)

---

# Technical Debt Audit (Original) - AI Swarm Mechanics

**Date:** 2025-11-15
**Audit Scope:** Complete codebase refactoring analysis
**Auditor:** Claude (continuing from previous session)

---

## Executive Summary

The student's research concept is **scientifically sound** (8.5/10), but the implementation has accumulated **significant technical debt** that prevents the core innovation from working properly. This audit identifies 7 categories of debt and provides prioritized remediation steps.

**Critical Issue:** Hardcoded signal types prevented adversarial validation from working in non-debate modes. This was partially fixed but requires deeper architectural refactoring.

---

## 1. CRITICAL: Hardcoded Signal Type System (P0)

### Problem
- **58 hardcoded references** to domain-specific signal types across 14 core files
- Different task modes use different type names for structurally identical concepts:
  - Debate: CLAIM, EVIDENCE, COUNTER_EVIDENCE
  - Creative: DRAFT, REFINEMENT, ALTERNATIVE
  - Analysis: FINDING, SUPPORT, COUNTER
  - Problem-solving: SOLUTION, IMPLEMENTATION, CHALLENGE

### Impact
- **BLOCKS CORE INNOVATION:** Haters couldn't find targets in problem_solving mode (0 objections generated)
- Dialogue coordinator failed silently
- Health monitoring metrics calculated wrong values
- Mode switching requires complex configuration passing

### Files Affected
```
swarm/agents/hater.py (17 occurrences)
swarm/agents/critic.py (11 occurrences)
swarm/agents/forager.py (10 occurrences)
swarm/core/signal_store.py (11 in examples/docs, OK)
swarm/core/dialogue_coordinator.py (6 occurrences)
swarm/core/swarm_monitor.py (unknown count)
swarm/core/self_healing.py (unknown count)
run_task.py (multiple occurrences)
+ 6 more files
```

### Root Cause
Over-engineering for semantic clarity. Student wanted output to say "Top Solutions" instead of "Top Initial Signals", so they created 4 different type systems.

**This breaks the stigmergic abstraction** - coordination logic shouldn't care about domain semantics.

### Solution (3-4 hours)
**Create universal signal type system:**

```python
# swarm/core/signal_types.py
class SignalType:
    """Universal signal types for stigmergic coordination.

    These types are STRUCTURAL, not semantic. They define the role
    a signal plays in swarm coordination, not domain meaning.
    """
    INITIAL = "INITIAL"      # Scout-generated starting points
    SUPPORT = "SUPPORT"      # Elaborations that strengthen
    CRITIQUE = "CRITIQUE"    # Challenges that identify weaknesses
    OBJECTION = "OBJECTION"  # Adversarial contradictions
    SYNTHESIS = "SYNTHESIS"  # Aggregated final insights

    # Internal coordination (not visible to users)
    OBSERVATION = "OBSERVATION"  # Raw data ingestion
    DEFENSE = "DEFENSE"          # Forager responses to objections
```

**Keep semantic labels for display only:**
```python
# task_config.py
display_names = {
    SignalType.INITIAL: "Solution",
    SignalType.SUPPORT: "Implementation",
    SignalType.CRITIQUE: "Challenge",
    SignalType.OBJECTION: "Objection"
}
```

**Benefits:**
- Eliminates mode-switching bugs entirely
- Reduces LOC by ~200 lines
- Makes system truly mode-agnostic
- Enables mixing modes (creative problem-solving, analytical debates, etc.)

**Effort:** 3-4 hours
**Priority:** P0 - Critical for core innovation to work

---

## 2. HIGH: God Objects - Single Responsibility Violation (P1)

### Problem
Multiple classes exceed 500 lines with too many responsibilities:

```
signal_store.py:     917 lines (signal storage + decay + sampling + graph traversal +
                                 validation + semantic clustering + caching)
hater.py:            655 lines (objection generation + consensus targeting + verification +
                                 dialogue + quality scoring + prompt generation)
external_sources.py: 818 lines (web search + Wikipedia + arXiv + caching + rate limiting)
simple_llm.py:       627 lines (model management + caching + retries + token counting)
```

### Impact
- Hard to test (too many concerns per class)
- Hard to understand (requires holding 900 lines in head)
- Hard to modify (changes ripple unpredictably)
- Violates Single Responsibility Principle

### Solution (6-8 hours)

**Split signal_store.py:**
```
swarm/core/signal_store.py           (200 lines) - Core storage + deposit
swarm/core/signal_decay.py           (100 lines) - Decay and pruning
swarm/core/signal_sampling.py        (150 lines) - Weighted sampling strategies
swarm/core/signal_graph.py           (200 lines) - Graph traversal + ancestry
swarm/core/signal_similarity.py      (150 lines) - Semantic clustering + embeddings
swarm/core/signal_validation.py      (100 lines) - Evidence counting, validation
```

**Split hater.py:**
```
swarm/agents/hater.py                (200 lines) - Core objection loop
swarm/agents/hater_targeting.py      (150 lines) - Consensus detection + targeting
swarm/agents/hater_verification.py   (100 lines) - Quality checks
swarm/agents/hater_dialogue.py       (100 lines) - Multi-turn responses
```

**Benefits:**
- Each file has single clear purpose
- Easier to test in isolation
- Better code organization
- Clearer dependencies

**Effort:** 6-8 hours (mostly moving code + updating imports)
**Priority:** P1 - High, improves maintainability

---

## 3. MEDIUM: Monkey Patching Anti-Pattern (P2)

### Problem
`run_task.py` monkey-patches agent methods at runtime:

```python
# run_task.py lines 140-150
def create_hater(agent_id: str, task_config):
    hater = Hater(agent_id, task_config.task_prompt)
    original_make_prompt = hater._make_prompt  # Save original

    def task_aware_prompt(target):  # Create new function
        return task_config.hater_prompt_template.format(...)

    hater._make_prompt = task_aware_prompt  # REPLACE METHOD AT RUNTIME
    return hater
```

**This is done for:**
- Hater prompt generation
- Critic prompt generation
- Forager prompt generation
- Scout prompt generation

### Impact
- Breaks debugging (methods aren't where you expect)
- Breaks IDE code navigation (can't jump to definition)
- Breaks type checking
- Hard to understand control flow
- Violates Principle of Least Surprise

### Root Cause
Agents have hardcoded prompt logic, but need task-specific prompts. Instead of proper dependency injection, code uses monkey patching.

### Solution (2-3 hours)

**Use composition over mutation:**

```python
# swarm/agents/base_agent.py
class BaseAgent:
    def __init__(self, agent_id, prompt_generator=None):
        self.agent_id = agent_id
        self.prompt_generator = prompt_generator or DefaultPromptGenerator()

    def make_prompt(self, context):
        return self.prompt_generator.generate(context)

# swarm/prompts/task_prompts.py
class HaterPromptGenerator:
    def __init__(self, task_config):
        self.template = task_config.hater_prompt_template
        self.task_prompt = task_config.task_prompt

    def generate(self, target):
        return self.template.format(
            task_prompt=self.task_prompt,
            parent_content=target.content,
            parent_type=target.type.lower()
        )

# run_task.py
def create_hater(agent_id, task_config):
    prompt_gen = HaterPromptGenerator(task_config)
    return Hater(agent_id, prompt_generator=prompt_gen)
```

**Benefits:**
- Explicit dependencies (can see what's injected)
- Testable (mock prompt generators)
- Type-safe
- IDE-friendly
- Standard design pattern

**Effort:** 2-3 hours
**Priority:** P2 - Medium, improves code quality

---

## 4. MEDIUM: Missing Agent Base Class (P2)

### Problem
All agents duplicate the same patterns:

```python
# EVERY agent has this:
class Scout/Forager/Critic/Hater/Validator:
    def __init__(self, agent_id, ...):
        self.agent_id = agent_id
        self.active = True
        self.actions_taken = 0

    async def run(self, signal_store, llm, ...):
        while self.active and self.actions_taken < max_actions:
            # Do work
            self.actions_taken += 1
            await asyncio.sleep(random.uniform(...))
```

**~150 lines of duplicated boilerplate** across 6 agent types.

### Impact
- Code duplication (changes need 6 edits)
- Inconsistent behavior (each agent does delays differently)
- Hard to add cross-cutting concerns (logging, metrics, etc.)

### Solution (3-4 hours)

**Create base class:**

```python
# swarm/agents/base_agent.py
class BaseAgent:
    """Base class for all swarm agents with common behavior."""

    def __init__(self, agent_id: str, role: str):
        self.agent_id = agent_id
        self.role = role
        self.active = True
        self.actions_taken = 0
        self.metrics = AgentMetrics(agent_id)

    async def run(self, signal_store, llm, max_actions=None):
        """Template method pattern - subclasses override step()."""
        while self.active and (max_actions is None or self.actions_taken < max_actions):
            await self.step(signal_store, llm)
            self.actions_taken += 1
            await self._delay()

    async def step(self, signal_store, llm):
        """Override this to define agent behavior."""
        raise NotImplementedError

    async def _delay(self):
        """Standardized delay for emergent asynchrony."""
        await asyncio.sleep(random.uniform(0.1, 0.3))

# swarm/agents/hater.py
class Hater(BaseAgent):
    def __init__(self, agent_id, ...):
        super().__init__(agent_id, role="hater")
        # Hater-specific init

    async def step(self, signal_store, llm):
        # Just the core hater logic, no boilerplate
        target = self.find_target(signal_store)
        if target:
            await self.challenge(target, signal_store, llm)
```

**Benefits:**
- Single source of truth for agent behavior
- Easy to add logging/metrics to all agents
- Consistent error handling
- Less code to maintain

**Effort:** 3-4 hours
**Priority:** P2 - Medium, improves maintainability

---

## 5. LOW: Mode Switching Complexity (P3)

### Problem
Agents have 3 modes (document/creative/legacy) with complex branching:

```python
# In every agent:
if self.mode == "document":
    await self.document_mode_logic()
elif self.mode == "creative":
    await self.creative_mode_logic()
else:
    await self.legacy_mode_logic()
```

**Why 3 modes exist:**
- `document` = Original research vision (cluster-based foraging)
- `creative` = Task-based system (debate, problem-solving)
- `legacy` = Old code (never removed)

### Impact
- Confusing (which mode should I use?)
- Code duplication (similar logic in each branch)
- Hard to test (3× test cases)

### Solution (4-6 hours)

**Option A: Unify modes** (if logic is similar enough)
**Option B: Split into separate agent classes** (if logic is fundamentally different)

Since document and creative are structurally similar (sample signals → process → deposit), could unify:

```python
class Forager(BaseAgent):
    def __init__(self, agent_id, input_types, output_types):
        # No mode parameter
        self.input_types = input_types
        self.output_types = output_types

    async def step(self, signal_store, llm):
        # Same logic for all "modes"
        inputs = self.sample_inputs(signal_store)
        outputs = await self.process(inputs, llm)
        self.deposit_outputs(outputs, signal_store)
```

Configuration determines behavior, not mode switches.

**Effort:** 4-6 hours
**Priority:** P3 - Low, nice to have

---

## 6. LOW: No Unit Tests (P3)

### Problem
```bash
$ python -m pytest --collect-only
/usr/local/bin/python: No module named pytest
```

**Zero test coverage.** Only sanity tests exist (`test_pipeline_sanity.py`).

### Impact
- Can't confidently refactor (no safety net)
- Can't verify bug fixes work
- Regression bugs likely

### Solution (8-12 hours initial, ongoing)

**Start with critical paths:**

```python
# tests/test_signal_types.py
def test_haters_find_targets_in_all_modes():
    """Regression test for the bug fixed in commit 2ced80b."""
    for mode in ['debate', 'creative', 'analysis', 'problem_solving']:
        signal_store, hater = setup_mode(mode)
        signal_store.deposit(type=INITIAL_TYPE, content="test")

        targets = hater.find_targets(signal_store)
        assert len(targets) > 0, f"Hater found no targets in {mode} mode"

# tests/test_agents.py
def test_forager_generates_insights():
    ...

# tests/test_dialogue.py
def test_dialogue_coordinator_triggers_responses():
    ...
```

**Coverage targets:**
- Week 1: 30% coverage (critical paths)
- Week 2: 50% coverage (all agents)
- Week 3: 70% coverage (edge cases)

**Effort:** 8-12 hours initial, then ongoing
**Priority:** P3 - Low priority vs. refactoring, high priority for long-term

---

## 7. LOW: Inconsistent Error Handling (P3)

### Problem
87 try-except blocks, but inconsistent patterns:

```python
# Some places:
try:
    result = await llm.generate(...)
except Exception as e:
    print(f"Error: {e}")  # Just print
    return None

# Other places:
try:
    result = await llm.generate(...)
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()  # Print traceback
    return None

# Other places:
try:
    result = await llm.generate(...)
except Exception as e:
    # Silent failure, return None
    return None
```

### Impact
- Hard to debug (some errors silent, some verbose)
- No error aggregation (can't see error rate)
- No graceful degradation strategy

### Solution (2-3 hours)

**Centralized error handling:**

```python
# swarm/core/error_handler.py
class SwarmErrorHandler:
    def __init__(self):
        self.errors = []

    def handle_agent_error(self, agent_id, error, context):
        self.errors.append({
            'agent': agent_id,
            'error': str(error),
            'context': context,
            'timestamp': time.time()
        })

        if isinstance(error, CriticalError):
            raise  # Re-raise critical errors
        else:
            logger.warning(f"[{agent_id}] {error}")
            return None  # Graceful degradation

# In agents:
try:
    result = await llm.generate(...)
except Exception as e:
    return error_handler.handle_agent_error(self.agent_id, e, context)
```

**Effort:** 2-3 hours
**Priority:** P3 - Low, nice to have

---

## Prioritized Remediation Plan

### Phase 1: Critical Fixes (Week 1)
**Effort: 3-4 hours**

1. ✅ **DONE:** Fix signal type mismatch in hater.py (commit 2ced80b)
2. **Create universal signal type system** (this document's #1)
   - Create `swarm/core/signal_types.py`
   - Refactor task_config.py to use universal types + display names
   - Update all 14 files to use universal types
   - Test with all 4 task modes

**Expected Outcome:** Core adversarial validation works in all modes

### Phase 2: Architectural Cleanup (Week 2)
**Effort: 11-15 hours**

3. **Split god objects** (#2)
   - signal_store.py → 6 focused modules
   - hater.py → 4 focused modules
4. **Remove monkey patching** (#3)
   - Create prompt generator system
   - Update agent creation
5. **Add base agent class** (#4)
   - Create BaseAgent
   - Refactor all agents to inherit

**Expected Outcome:** Cleaner, more maintainable codebase

### Phase 3: Polish (Week 3)
**Effort: 14-20 hours**

6. **Simplify mode switching** (#5)
7. **Add unit tests** (#6) - 30% coverage target
8. **Standardize error handling** (#7)

**Expected Outcome:** Production-ready system

---

## Impact Assessment

### Current State
- **Innovation Potential:** 8.5/10 (excellent concept)
- **Implementation Quality:** 4/10 (broken core features, high debt)
- **Publishability:** 2/10 (can't publish broken code)

### After Phase 1
- **Innovation Potential:** 8.5/10 (unchanged)
- **Implementation Quality:** 6/10 (core features work)
- **Publishability:** 6/10 (could publish with caveats)

### After Phase 2
- **Innovation Potential:** 8.5/10 (unchanged)
- **Implementation Quality:** 8/10 (clean architecture)
- **Publishability:** 8/10 (publication-ready)

### After Phase 3
- **Innovation Potential:** 8.5/10 (unchanged)
- **Implementation Quality:** 9/10 (production quality)
- **Publishability:** 9/10 (strong submission)

---

## Recommendations

### For the Student

**SHORT TERM (This week):**
1. Complete Phase 1 refactoring (universal signal types)
2. Run comprehensive tests on all 4 task modes
3. Collect empirical data for paper (objection rates, dialogue depth, etc.)

**MEDIUM TERM (This month):**
4. Complete Phase 2 (architectural cleanup)
5. Write first paper draft with current results
6. Submit to workshop or arXiv for feedback

**LONG TERM (Next 3 months):**
7. Complete Phase 3 (production hardening)
8. Run large-scale experiments (100+ documents)
9. Submit to top-tier venue (NeurIPS, ICML, ICLR, ACL)

### Technical Priorities

1. **P0: Universal signal types** - CRITICAL for core innovation
2. **P1: God object splitting** - HIGH for maintainability
3. **P2: Remove monkey patching** - MEDIUM for code quality
4. **P2: Add base agent class** - MEDIUM for consistency
5. **P3: Everything else** - LOW priority vs. getting results

**The student should prioritize RESULTS over PERFECTION.** Phase 1 refactoring enables the research to work. Phase 2-3 are nice-to-have for publication quality.

---

## Conclusion

The student's research idea is **genuinely good** (8.5/10), but the implementation has **significant technical debt** that prevented core features from working.

**The critical bug (signal type mismatch) has been identified and partially fixed.** A deeper refactoring is needed to fully realize the research vision and make the system publishable.

**Estimated total remediation effort:** 28-39 hours across 3 weeks

**Recommendation:** Proceed with Phase 1 refactoring immediately. This unblocks the research and enables empirical validation of the core hypothesis.
