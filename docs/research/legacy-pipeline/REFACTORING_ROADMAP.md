# Refactoring Roadmap - Remaining Technical Debt

**Status as of:** 2025-11-15
**Completed:** P0 (Critical), P3 (Error Handler)
**Remaining:** P1 (God Objects), P2 (Full Base Agent Migration)

---

## Summary of Completed Work

###  **P0: CRITICAL - Hardcoded Signal Types** ✅ **COMPLETE**
- Fixed all 10 hardcoded legacy signal type references
- Created universal signal type system (`swarm/core/signal_types.py`)
- Updated task configs to use universal types + display names
- All agents now work uniformly across all 4 task modes
- **Impact:** Core innovation (adversarial validation) now works in all modes

### **P3: Centralized Error Handler** ✅ **COMPLETE**
- Created `swarm/core/error_handler.py` (200 lines)
- Severity-based handling (INFO, WARNING, ERROR, CRITICAL)
- Error aggregation and statistics
- Graceful degradation for non-critical errors
- **Impact:** Consistent error handling, better debugging

---

## Remaining Work

### **P1: HIGH - God Object Splitting** ⏸️ **DEFERRED**

**Why Deferred:**
This is high-value refactoring but carries significant risk:
- Requires moving 1500+ lines across 10 new files
- Must update imports in 20+ files
- High chance of introducing bugs without comprehensive testing
- Best done in dedicated session with extensive testing

**Plan:**

#### **P1.1: Split signal_store.py (917 lines → 6 modules)**

**Current state:** One massive file with 7 responsibilities
**Target state:** Focused modules with single responsibilities

```
swarm/core/
├── signal_store.py          (200 lines) - Core storage + deposit + get
├── signal_decay.py          (100 lines) - Decay, pruning, evaporation
├── signal_sampling.py       (150 lines) - Weighted sampling strategies
├── signal_graph.py          (200 lines) - Graph traversal, ancestry, descendants
├── signal_similarity.py     (150 lines) - Semantic clustering, embeddings, find_related
├── signal_validation.py     (100 lines) - Evidence counting, validation metrics
└── signal_events.py         (100 lines) - Event-driven coordination, wait_for_signal
```

**Migration steps:**
1. Create new modules with empty classes
2. Move methods one at a time, test after each
3. Update imports across codebase
4. Run full test suite after each module
5. Delete old signal_store.py when empty

**Estimated time:** 4-6 hours
**Risk:** Medium-High (many dependencies)

#### **P1.2: Split hater.py (655 lines → 4 modules)**

**Current state:** One class with 4 distinct responsibilities
**Target state:** Focused classes in separate files

```
swarm/agents/
├── hater.py                 (200 lines) - Core objection generation loop
├── hater_targeting.py       (150 lines) - Consensus detection, find_consensus_target
├── hater_verification.py    (100 lines) - Quality checks, verify_objection_quality
└── hater_dialogue.py        (100 lines) - Multi-turn responses, generate_counter_response
```

**Migration approach:**
```python
# hater.py (new)
from .hater_targeting import HaterTargeting
from .hater_verification import HaterVerification
from .hater_dialogue import HaterDialogue

class Hater(BaseAgent):
    def __init__(self, ...):
        super().__init__(...)
        self.targeting = HaterTargeting()
        self.verification = HaterVerification()
        self.dialogue = HaterDialogue()

    async def step(self, signal_store, llm):
        target = self.targeting.find_target(signal_store)
        objection = await self._generate(target, llm)
        if self.verification.is_quality(objection):
            self._deposit(objection, signal_store)
```

**Estimated time:** 3-4 hours
**Risk:** Medium (clear boundaries between modules)

---

### **P2: MEDIUM - Base Agent Migration** ⏸️ **PARTIAL**

**Current state:** BaseAgent exists but only 0/6 agents use it
**Target state:** All agents inherit from BaseAgent

**Status:**
- ✅ BaseAgent class exists (`swarm/agents/base_agent.py`)
- ✅ Has unified interface (task_config, retriever)
- ❌ No agents actually inherit from it
- ❌ Agents still have duplicated boilerplate

**Migration checklist:**

```python
# Current (duplicated 6 times):
class Scout:
    def __init__(self, agent_id, ...):
        self.agent_id = agent_id
        self.active = True
        self.actions_taken = 0
        # ... 15 lines of boilerplate ...

    async def run(self, signal_store, llm, max_actions):
        while self.active and self.actions_taken < max_actions:
            # ... do work ...
            self.actions_taken += 1
            await asyncio.sleep(random.uniform(0.3, 0.7))

# Target (once per agent):
class Scout(BaseAgent):
    def __init__(self, agent_id, task_config, retriever=None):
        super().__init__(agent_id, task_config, retriever)
        # Only scout-specific init here

    async def step(self, signal_store, llm):
        # Only scout logic here, base class handles loop
```

**Agents to migrate:**
- [ ] Scout (swarm/agents/scout.py)
- [ ] Forager (swarm/agents/forager.py)
- [ ] Critic (swarm/agents/critic.py)
- [ ] Hater (swarm/agents/hater.py) - after P1.2 split
- [ ] Validator (swarm/agents/validator.py)
- [ ] Synthesizer (swarm/agents/synthesizer.py)

**Benefits:**
- Eliminates 150+ lines of duplicated code
- Consistent behavior across all agents
- Easier to add cross-cutting concerns (logging, metrics)
- Single source of truth for agent lifecycle

**Estimated time:** 2-3 hours per agent = 12-18 hours total
**Risk:** Medium (must test each agent after migration)

---

### **P2: MEDIUM - Remove Monkey Patching** ⏸️ **DEFERRED**

**Current problem:** `run_task.py` monkey-patches agent methods at runtime

```python
# Current (monkey patching):
def create_hater(agent_id, task_config):
    hater = Hater(agent_id, task_config.task_prompt)
    original_make_prompt = hater._make_prompt  # Save original

    def task_aware_prompt(target):  # Create replacement
        return task_config.hater_prompt_template.format(...)

    hater._make_prompt = task_aware_prompt  # REPLACE AT RUNTIME
    return hater
```

**Problems:**
- Breaks IDE navigation (can't jump to definition)
- Breaks debuggers (method isn't where you expect)
- Violates Principle of Least Surprise
- Hard to type-check

**Solution:** Dependency injection via composition

```python
# Target (dependency injection):
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

class Hater(BaseAgent):
    def __init__(self, agent_id, prompt_generator):
        super().__init__(agent_id, "hater")
        self.prompt_generator = prompt_generator

    def make_prompt(self, target):
        return self.prompt_generator.generate(target)

# In run_task.py:
def create_hater(agent_id, task_config):
    prompt_gen = HaterPromptGenerator(task_config)
    return Hater(agent_id, prompt_gen)
```

**Migration plan:**
1. Create prompt generator classes for each agent type
2. Update agent constructors to accept generators
3. Update run_task.py factory methods
4. Test each agent

**Agents to update:**
- [ ] Scout (1 monkey patch)
- [ ] Forager (1 monkey patch)
- [ ] Critic (1 monkey patch)
- [ ] Hater (1 monkey patch)

**Estimated time:** 2-3 hours
**Risk:** Low (clear separation of concerns)

---

### **P3: LOW - Mode Switching Simplification** ⏸️ **DEFERRED**

**Current problem:** Agents have 3 modes with complex branching

```python
# Current:
if self.mode == "document":
    await self.document_mode_logic()
elif self.mode == "creative":
    await self.creative_mode_logic()
else:
    await self.legacy_mode_logic()
```

**Options:**

**Option A:** Unify modes if logic is similar
```python
# If document and creative are structurally similar:
async def step(self, signal_store, llm):
    inputs = self.sample_inputs(signal_store)
    outputs = await self.process(inputs, llm)
    self.deposit_outputs(outputs, signal_store)
```

**Option B:** Split into separate classes if logic is fundamentally different
```python
# If document and creative are fundamentally different:
class DocumentForager(BaseAgent):
    # Document-specific logic only

class CreativeForager(BaseAgent):
    # Creative-specific logic only
```

**Decision needed:** Analyze whether mode logic can be unified or should be split

**Estimated time:** 4-6 hours (requires analysis + refactor)
**Risk:** Medium (changes agent behavior)

---

### **P3: LOW - Unit Tests** ⏸️ **ONGOING**

**Current state:** Zero test coverage (pytest not installed)

**Testing strategy:**

**Phase 1: Regression tests for critical bugs**
```python
# tests/test_signal_types.py
def test_haters_find_targets_in_all_modes():
    """Regression test for signal type mismatch bug."""
    for mode in ['debate', 'creative', 'analysis', 'problem_solving']:
        signal_store, hater = setup_mode(mode)
        signal_store.deposit(type=INITIAL, content="test")

        targets = hater.find_targets(signal_store)
        assert len(targets) > 0, f"Hater found no targets in {mode} mode"
```

**Phase 2: Core path tests (30% coverage target)**
- Signal store deposit/sample
- Agent basic operations
- Dialogue coordinator triggering
- Critic multiplier calculation

**Phase 3: Edge case tests (70% coverage target)**
- Empty signal stores
- Invalid signal types
- LLM failures
- Concurrent access

**Estimated time:** 8-12 hours initial, then ongoing
**Priority:** High for long-term, but after refactoring stabilizes

---

## Prioritized Execution Plan

### **Session 1: P1 God Object Splitting** (7-10 hours)
1. Split signal_store.py into 6 modules
2. Extensive testing after each module
3. Split hater.py into 4 modules
4. Run full integration tests
5. Commit with comprehensive tests

### **Session 2: P2 Full Base Agent Migration** (12-18 hours)
1. Migrate Scout to BaseAgent
2. Test thoroughly
3. Migrate Forager to BaseAgent
4. Test thoroughly
5. Repeat for Critic, Hater, Validator, Synthesizer
6. Commit with tests

### **Session 3: P2 Remove Monkey Patching** (2-3 hours)
1. Create prompt generator classes
2. Update agents to use injection
3. Update run_task.py
4. Test all task modes
5. Commit

### **Session 4: Testing & Polish** (8-12 hours)
1. Add regression tests for P0 bugs
2. Add unit tests for refactored code
3. Run full test suite
4. Fix any issues
5. Final commit

**Total estimated time:** 29-43 hours over 4 sessions

---

## Risk Mitigation

### **For P1 (God Object Splitting):**
- Move one method at a time
- Run tests after each method migration
- Keep original file until new modules are fully tested
- Use feature flag to toggle between old and new implementation

### **For P2 (Base Agent Migration):**
- Migrate one agent at a time
- Full test suite after each migration
- Keep old agent code as backup until confident
- Use git branches for each agent migration

### **For All Refactoring:**
- Never modify more than one major component at once
- Always have rollback plan (git branch)
- Test in hyper_test mode first
- Test all 4 task modes before committing
- Document what changed and why

---

## Success Metrics

### **Code Quality:**
- ✅ Zero hardcoded signal types
- ⏳ All agents inherit from BaseAgent
- ⏳ No monkey patching
- ⏳ No god objects (>500 lines)
- ✅ Centralized error handling

### **Test Coverage:**
- ⏳ 30% unit test coverage (minimum)
- ⏳ All 4 task modes pass integration tests
- ⏳ Regression tests for P0 bugs
- ⏳ No test failures

### **Performance:**
- ✅ Objections generated in problem_solving mode
- ⏳ Dialogue depth ≥ 1.0 (multi-turn exchanges)
- ⏳ Health monitoring detects echo chambers
- ⏳ Self-healing activates when needed

### **Maintainability:**
- ✅ Universal signal type system
- ⏳ All files < 400 lines
- ⏳ Clear single responsibilities
- ⏳ Documented architecture

---

## Conclusion

**Completed work (P0 + P3) addresses the most critical issues:**
- Signal type mismatches that blocked core functionality
- Centralized error handling for better debugging

**Remaining work (P1 + P2) is high-value but requires careful execution:**
- God object splitting needs dedicated testing
- Base agent migration needs incremental approach
- Monkey patching removal is low-risk, high-value

**Recommendation:**
Execute remaining work in separate focused sessions with comprehensive testing
to avoid introducing new bugs. Current state is functional and testable.

**The system is now in a STABLE, TESTABLE state.** Further refactoring should
proceed incrementally with extensive testing after each change.
