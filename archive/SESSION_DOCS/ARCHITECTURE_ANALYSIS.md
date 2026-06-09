# ARCHITECTURE ANALYSIS - Evidence-Based Assessment

**Generated:** 2025-11-19
**Purpose:** Understand current architecture before making changes
**Warning:** I lack full context. This is initial discovery, not final truth.

---

## Current Structure

### Directory Organization

```
swarm/
├── agents/          (10 files) - Agent implementations
├── core/            (14 files) - Core coordination & state
├── documents/       (2 files)  - Document processing
├── knowledge/       (1 file)   - Knowledge management
├── llm/             (7 files)  - LLM provider abstractions
├── retrieval/       (5 files)  - Information retrieval
└── validation/      (4 files)  - Validation & knowledge bases

+ Root entry points:
  - run_task.py (947 lines) - Main orchestrator
  - run_task_wrapper.py - Wrapper
  - swarm_evaluation.py - Benchmarks
```

### Module Responsibilities (Initial Understanding)

#### swarm/core/ - Central Coordination
- `signal_store.py` (1,023 lines) ⚠️ LARGEST FILE
  - Stigmergic signal storage
  - Event-driven coordination
  - Signal sampling, decay, amplification
  - Provenance graph traversal
  - Semantic similarity

- `spatial_signal_store.py` (579 lines) - Spatial extension
- `task_config.py` (345 lines) - Task configuration
- `round_coordinator.py` - Round execution
- `dialogue_coordinator.py` - Agent dialogue
- `agent_metrics.py` (419 lines) - Metrics tracking
- `verification.py` (355 lines) - Verification logic
- Other: config, error handling, monitoring, self-healing

#### swarm/agents/ - Agent Implementations
- `scout.py` (366 lines) - Exploration agents
- `forager.py` (359 lines) - Evidence gathering
- `critic.py` (398 lines) - Evaluation agents
- `hater.py` (671 lines) ⚠️ LARGE - Adversarial agents
- `validator.py` - Validation agents
- `pruner.py` - Pruning agents
- `synthesizer.py` - Synthesis agents
- `simple_scout.py` (384 lines) - Phase 2 spatial scouts
- `base_agent.py` - Base class

#### swarm/llm/ - LLM Abstraction
- `simple_llm.py` (627 lines) - Main LLM interface
- `pool.py` (335 lines) - LLM pooling
- `provider.py`, `simple_provider.py`, `vllm_provider.py` - Providers
- `factory.py`, `batcher.py` - Factory & batching

#### swarm/retrieval/ - Information Gathering
- `advanced_retriever.py` (451 lines) - Deep retrieval
- `search_engine.py` (418 lines) - Search interface
- `knowledge_processor.py` (370 lines) - Knowledge processing
- `dynamic_retriever.py`, `simple_web_search.py` - Web search

#### swarm/validation/ - Validation & Knowledge
- `external_sources.py` (830 lines) ⚠️ LARGE - External validation
- `dynamic_knowledge_base.py` (385 lines) - Dynamic KB
- `real_validator.py`, `format_validator.py` - Validators

---

## Dependency Analysis

### Most Depended-On Modules
1. **llm** - Used by 10 files (central dependency)
2. **core** - Used by 5 files (coordination hub)
3. **agents** - Used by 2 files
4. **retrieval** - Used by 2 files
5. **validation** - Used by 1 file

### Key Observation
`signal_store.py` is used by 17 files across the codebase:
- All 9 agent files
- 7 core coordination files
- run_task.py (main entry)

**Question:** Is this actually a problem?
- ✅ Makes sense - it's the shared coordination mechanism
- ❓ Does it create bottlenecks? UNKNOWN - need profiling
- ❓ Is it hard to maintain? UNKNOWN - need developer feedback
- ❓ Does size (1,023 lines) cause issues? UNKNOWN - need evidence

---

## Architecture Pattern: Stigmergic Coordination

### How It Works (Based on Code Reading)

```
1. run_task.py creates:
   - SignalStore (shared environment)
   - Agents (scouts, foragers, critics, haters, etc.)
   - LLM (language model)
   - Coordinators (round, dialogue)

2. Agents interact through SignalStore:
   - deposit(signal_type, content, strength, ...) - Add signals
   - sample_weighted(signal_type, n) - Sample by strength
   - get_ancestors/descendants(signal_id) - Traverse provenance
   - wait_for_signal(signal_type) - Async event coordination
   - decay_all(), prune_weak() - Lifecycle management

3. Coordination flow:
   - Scouts explore → deposit INITIAL/DRAFT signals
   - Foragers build on signals → deposit SUPPORT/EVIDENCE
   - Critics evaluate → adjust strength, deposit CRITIQUE
   - Haters challenge → deposit OBJECTION/COUNTER
   - Validators check → deposit validation signals
   - Pruners clean up → remove weak signals

4. Event-driven:
   - Agents await signal_store.wait_for_signal(type)
   - When signal deposited → asyncio.Event wakes waiters
   - No polling, no busy-waiting
```

### Key Architectural Decisions

**Good Decisions (Evidence-Based):**
1. ✅ Event-driven coordination (no polling)
2. ✅ Async/await throughout (non-blocking)
3. ✅ Composition over monkey-patching (fixed in Session 4)
4. ✅ Provenance tracking (parent-child links)
5. ✅ Signal decay/amplification (stigmergic dynamics)

**Questionable Decisions (Need Evidence):**
1. ❓ Global lock in SignalStore (threading.Lock)
   - **Claim:** Might serialize all operations
   - **Evidence:** NONE - need profiling to confirm
   - **Alternative:** Might be fine if operations are fast

2. ❓ Single SignalStore instance for all agents
   - **Claim:** Could be a bottleneck
   - **Evidence:** NONE - need measurement
   - **Alternative:** Might be necessary for coordination

3. ❓ signal_store.py is 1,023 lines
   - **Claim:** Too large, hard to maintain
   - **Evidence:** NONE - no developer complaints
   - **Alternative:** Might be appropriate for central component

---

## Potential Issues (Hypotheses, Not Facts)

### Hypothesis 1: Threading Lock Bottleneck
**Claim:** Global lock in signal_store.py serializes operations

**Evidence for:**
- Line 57 in signal_store.py: `self._lock = Lock()`
- Many methods use `with self._lock:`
- All agents share one SignalStore

**Evidence against:**
- Operations might be fast enough that lock contention is negligible
- Async event-driven means agents wait, not spin

**Validation needed:**
- [ ] Profile lock wait time
- [ ] Measure operation duration inside lock
- [ ] Compare throughput with/without lock (if possible)
- [ ] Check if lock is actually contended

**Decision:** NO CHANGES until profiling proves this is a bottleneck

---

### Hypothesis 2: signal_store.py Too Large
**Claim:** 1,023 lines is too large, should be split

**Evidence for:**
- 1,023 lines is larger than other files
- Has multiple responsibilities (sampling, decay, graph, events, similarity)

**Evidence against:**
- All responsibilities are related to signal coordination
- Module is cohesive (everything about signals)
- No reported maintenance issues
- Previous attempt to split it was reverted as net negative

**Validation needed:**
- [ ] Survey: Is this file hard to navigate?
- [ ] Metrics: How often is it modified?
- [ ] Test: Can we find things in it easily?
- [ ] Compare: Are smaller files actually easier to maintain?

**Decision:** NO CHANGES - previous split was fake modularization

---

### Hypothesis 3: run_task.py Too Complex
**Claim:** 947 lines, many responsibilities

**Evidence for:**
- 947 lines is large
- Creates agents, coordinates rounds, manages lifecycle
- Has conditional imports for phases

**Evidence against:**
- It's the main orchestrator - should coordinate
- Complexity might be inherent to task
- Code is mostly configuration/setup

**Validation needed:**
- [ ] Read through it - is it actually confusing?
- [ ] Measure cyclomatic complexity
- [ ] Check if different tasks share code

**Decision:** DEFER - need to read it first

---

### Hypothesis 4: LLM Caching Issues
**Claim:** LLM cache might cause problems

**Evidence for:**
- Code has creative vs. non-creative cache settings
- Cache size configurable

**Evidence against:**
- Already has configuration for this
- Creative mode uses smaller cache intentionally

**Validation needed:**
- [ ] Measure cache hit rates
- [ ] Compare quality with/without cache
- [ ] Check memory usage of cache

**Decision:** DEFER - seems already handled

---

## What I DON'T Know (Gaps in Understanding)

### Critical Unknowns
1. **Performance:** No profiling data exists
   - What's slow? Unknown
   - What's fast? Unknown
   - Bottlenecks? Unknown

2. **User Experience:** No user feedback
   - Is anything confusing? Unknown
   - What causes errors? Unknown
   - What's painful to use? Unknown

3. **Maintenance:** No historical data
   - Which files change frequently? Unknown
   - Which bugs recur? Unknown
   - What's hard to debug? Unknown

4. **Memory:** No memory profiling
   - Does it leak? Unknown (fixed leaks found, but more?)
   - How much RAM needed? Unknown
   - Growth over time? Unknown

5. **Correctness:** No integration tests
   - Do full workflows work? Assumed yes
   - Do edge cases break? Unknown
   - Regressions caught? Unknown

### Dependencies I Haven't Examined
- How do agents actually interact?
- What's the Round coordinator doing?
- How does verification work?
- What's in the validation module?
- How does retrieval integrate?

**⚠️ I LACK CONTEXT. I've read ~500 lines of ~22,000.**

---

## Evidence-Based Next Steps

### Step 1: Profile Before Optimizing
**Why:** We don't know what's slow
**How:** Add cProfile to run_task.py, run typical workload
**Time:** 2-3 hours
**Risk:** Low
**Value:** High IF bottlenecks found

**Validation:**
Before: "I think X might be slow"
After: "X takes 47% of runtime, here's proof"

---

### Step 2: Read More Code
**Why:** I've only read 2% of the codebase
**How:** Read key files: agents/*, round_coordinator, verification
**Time:** 4-6 hours
**Risk:** None (just reading)
**Value:** Understanding before changing

**Validation:**
Before: "I don't know how this works"
After: "I understand the data flow"

---

### Step 3: Run It
**Why:** Understanding requires seeing it work
**How:** Run different task types, observe behavior
**Time:** 1-2 hours
**Risk:** None (just observing)
**Value:** Empirical understanding

**Validation:**
Before: "It probably does X"
After: "It does X, here's the output"

---

### Step 4: Measure Memory
**Why:** Unknown memory footprint
**How:** Memory profiler on long runs
**Time:** 1-2 hours
**Risk:** Low
**Value:** Medium IF memory is constrained

---

### Step 5: Add Integration Tests
**Why:** No tests for full workflows
**How:** Test end-to-end scenarios
**Time:** 6-8 hours
**Risk:** Low
**Value:** High (catch regressions, build confidence)

---

## What NOT To Do

### ❌ Don't Refactor signal_store.py (Again)
**Why:**
- Already tried and reverted
- No evidence it's a problem
- Size alone isn't a problem
- Cohesive module is good

**When to reconsider:**
- After profiling shows bottleneck
- After developer feedback shows pain
- After finding actual bugs caused by size

---

### ❌ Don't Add Abstractions
**Why:**
- No proven need
- Adds complexity
- We don't know pain points yet

**When to reconsider:**
- After third copy-paste of same code
- After proven extension point needed
- After simplification demonstrated

---

### ❌ Don't Optimize Locks/Async
**Why:**
- No evidence it's slow
- Could make it worse
- Premature optimization

**When to reconsider:**
- After profiling shows lock contention
- After measuring current throughput
- After benchmarking alternatives

---

## Self-Critique Checklist

Before proposing ANY change:
- [ ] Have I read the relevant code?
- [ ] Do I understand how it works now?
- [ ] Do I have evidence of a problem?
- [ ] Have I measured the impact?
- [ ] Is this the simplest solution?
- [ ] Will this make maintenance easier?
- [ ] Can I prove the benefit?
- [ ] Am I adding or removing complexity?
- [ ] Did I consider doing nothing?

---

## Current Status

**What I Know:**
- Architecture is modular (agents, core, llm, retrieval, validation)
- SignalStore is central coordination mechanism
- Event-driven async design
- Largest files: signal_store (1,023), run_task (947), hater (671)

**What I Don't Know:**
- Performance characteristics
- Actual bottlenecks
- User pain points
- Memory usage
- Maintenance difficulty

**Recommendation:**
**GATHER EVIDENCE before changing anything.**

Start with profiling, reading, and running - not coding.
