# STRUCTURE REFERENCE - Quick Architecture Lookup

**Purpose:** Fast reference to prevent context loss while reading code
**When to use:** Before reading any file, after getting confused
**Update:** As understanding improves

---

## Quick Facts

**Total Lines:** ~22,000 across 44 Python files
**Main Entry:** run_task.py (947 lines)
**Central Hub:** swarm/core/signal_store.py (1,023 lines)
**Architecture:** Stigmergic coordination (agents + shared signal environment)

---

## Directory Map (One-Line Summaries)

```
swarm/
├── agents/        (10 files) - Agent types: Scout, Forager, Critic, Hater, etc.
├── core/          (14 files) - SignalStore, coordinators, metrics, config
├── llm/           (7 files)  - LLM provider abstraction (Phi-2, vLLM, etc.)
├── retrieval/     (5 files)  - Web search, knowledge ingestion
├── validation/    (4 files)  - External validation, knowledge bases
├── documents/     (2 files)  - Document processing
└── knowledge/     (1 file)   - Knowledge management
```

---

## Core Data Flow (Simplified)

```
1. run_task.py creates:
   SignalStore → Shared environment
   Agents → Scout, Forager, Critic, Hater, Pruner, etc.
   LLM → Language model (Phi-2)

2. Agents interact via SignalStore:
   Scout.explore() → deposit(DRAFT)
   Forager.gather() → sample(DRAFT) → deposit(SUPPORT/EVIDENCE)
   Critic.evaluate() → sample(DRAFT) → deposit(CRITIQUE)
   Hater.challenge() → sample(DRAFT) → deposit(OBJECTION)

3. Coordination:
   RoundCoordinator → Manages round execution
   SignalStore events → Async notification (no polling)

4. Lifecycle:
   deposit() → wait_for_signal() → sample_weighted() → decay_all() → prune_weak()
```

---

## Key Classes (What They Do)

### SignalStore (swarm/core/signal_store.py)
**Purpose:** Shared pheromone environment for stigmergic coordination
**Key Methods:**
- `deposit(type, content, strength, ...)` - Add signal
- `sample_weighted(type, n)` - Sample by strength + exploration bonus
- `wait_for_signal(type, timeout)` - Async event waiting
- `get_ancestors/descendants(id)` - Provenance traversal
- `decay_all()` - Evaporation (stigmergy)
- `prune_weak()` - Remove weak signals

**State:**
- `signals: Dict[str, Signal]` - All signals
- `_lock: Lock` - Thread safety
- `_signal_events: Dict[str, asyncio.Event]` - Event coordination

### Signal (dataclass)
**Purpose:** Individual pheromone deposit
**Fields:**
- `id: str` - Unique identifier
- `type: str` - DRAFT, SUPPORT, CRITIQUE, OBJECTION, etc.
- `content: str` - The actual idea/evidence/critique
- `strength: float` - 0.0-1.0 (decays over time)
- `timestamp: float` - When deposited
- `depositor: str` - Agent ID
- `parent: Optional[str]` - Provenance link
- `visits: int` - Corroboration counter

---

## Agent Types (Brief)

| Agent | Input | Output | Purpose |
|-------|-------|--------|---------|
| Scout | - | DRAFT/INITIAL | Explore, generate initial ideas |
| Forager | DRAFT | SUPPORT/EVIDENCE | Build on ideas, gather evidence |
| Critic | DRAFT | CRITIQUE | Evaluate quality, adjust strength |
| Hater | DRAFT | OBJECTION/COUNTER | Challenge, find flaws |
| Validator | Any | VALIDATION | Fact-check against external sources |
| Pruner | - | - | Remove weak/orphaned signals |
| Synthesizer | Multiple | SYNTHESIS | Combine signals into coherent output |

---

## Signal Types (By Task Mode)

### Creative Mode
- Initial: DRAFT
- Support: SUPPORT
- Counter: COUNTER
- Final: CREATIVE_RESULT

### Debate Mode
- Initial: THESIS
- Support: EVIDENCE
- Counter: OBJECTION
- Final: DEBATE_CONCLUSION

### Analysis Mode
- Initial: OBSERVATION
- Support: ANALYSIS
- Counter: CRITIQUE
- Final: ANALYSIS_RESULT

### Problem Solving Mode
- Initial: IDEA
- Support: SUPPORT
- Counter: OBJECTION
- Final: SOLUTION

---

## Coordination Mechanisms

### Event-Driven (No Polling)
```python
# Agent waits
await signal_store.wait_for_signal("DRAFT", timeout=10.0)

# SignalStore notifies on deposit
def deposit(...):
    # ... add signal ...
    self._signal_events["DRAFT"].set()  # Wake waiters
```

### Round-Based Execution
```python
RoundCoordinator:
  for round in range(num_rounds):
    - Run scouts (parallel)
    - Run foragers (parallel)
    - Run critics (parallel)
    - Run haters (parallel)
    - Run pruner
    - Decay all signals
```

### Async/Await Throughout
- All agent methods are `async def`
- LLM calls are `await llm.generate(...)`
- No blocking operations (no time.sleep, only asyncio.sleep)

---

## Configuration (swarm/core/config.py)

**Key Settings:**
- `MODEL_NAME = "microsoft/phi-2"` (2.7B params)
- `NUM_SCOUTS = 4` (exploration)
- `NUM_FORAGERS = 4` (development)
- `NUM_CRITICS = 2` (evaluation)
- `NUM_HATERS = 2` (adversarial)
- `MAX_ITERATIONS = 50` (per round)
- `DECAY_RATE = 0.05` (5% strength loss per iteration)
- `PRUNE_THRESHOLD = 0.15` (remove below this)

---

## Common Patterns

### Agent Skeleton
```python
class SomeAgent:
    async def do_work(self, signal_store, llm):
        # 1. Sample signals
        signals = signal_store.sample_weighted("TYPE", n=3)

        # 2. Process with LLM
        for signal in signals:
            prompt = self._make_prompt(signal)
            response = await llm.generate(prompt, temp=0.7)

            # 3. Deposit result
            signal_store.deposit(
                signal_type="OUTPUT_TYPE",
                content=response,
                strength=self._calculate_strength(response),
                parent=signal.id  # Provenance
            )
```

### Provenance Traversal
```python
# Get all evidence for an idea
evidence = signal_store.get_descendants(idea.id, "SUPPORT")

# Get original idea from critique
original = signal_store.get_ancestors(critique.id, "DRAFT")

# Get synthesis signals connecting two ideas
connecting = signal_store.get_connecting_signals(idea_a.id, idea_b.id, "SYNTHESIS")
```

---

## File Dependencies (Who Uses What)

**signal_store.py** is imported by:
- All agents (scout, forager, critic, hater, validator, pruner, synthesizer)
- All coordinators (round, dialogue)
- run_task.py (main entry)
- swarm_monitor.py, verification.py, agent_metrics.py

**SimpleLLM** is imported by:
- All agents
- run_task.py
- Coordinators

**Task types** flow through:
- task_config.py (definitions)
- run_task.py (selection)
- Agents (via task_config injection)

---

## Known Issues (From Previous Analysis)

### Fixed ✓
- Blocking I/O (time.sleep → asyncio.sleep)
- Memory leaks (unbounded caches, missing embedding cleanup)
- Monkey patching (replaced with composition)
- Race conditions (unsafe dict access)

### Hypotheses (Unproven)
- Lock contention? (No profiling data)
- signal_store.py too large? (No maintenance complaints)
- run_task.py too complex? (Haven't read it fully)

---

## Questions to Ask While Reading

### Any File
- What does this file DO? (One sentence)
- What data does it own?
- Who calls it?
- Who does it call?
- What could go wrong?

### Agent Files
- What signals does it consume?
- What signals does it produce?
- How does it sample? (weighted, stratified, cluster?)
- How does it calculate strength?
- Does it use provenance (parent links)?

### Core Files
- What state does it manage?
- Is it thread-safe? (locks?)
- Is it async-safe? (await?)
- What invariants must hold?
- How does it notify others?

---

## Update Log

**2025-11-19:** Initial creation after architecture mapping
- Mapped 7 modules, 44 files
- Identified signal flow
- Listed agent types and responsibilities
- Next: Deep reading of scout.py, forager.py, critic.py
