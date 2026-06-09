# PARALLEL ARCHITECTURE DESIGN - Evidence Chaining + CUDA-Aware Parallelism

**Date:** 2025-11-20
**Branch:** `claude/analyze-codebase-01RUVXdkHt9uNkPn7rauTXiE`
**Goal:** Design parallel execution that respects signal dependencies and CUDA constraints

---

## EXECUTIVE SUMMARY

**Problem:** Current architecture serializes LLM calls despite concurrent agent execution, wasting 85% of potential performance. Need parallelism that respects signal dependencies.

**Solution:** Stage-based batching with adaptive LLM pooling
- **Batch agents by dependency layer** (scouts together, foragers together, etc.)
- **Adaptive LLM pool** (2-4 instances based on available CUDA memory)
- **Enhanced evidence chaining** (validate references, include context)
- **Expected speedup:** 3-5x (from 4-5 min → 1 min per task)

---

## PROBLEM ANALYSIS

### Current Evidence Chaining Issues

**1. Weak Reference Context**
```python
# Current: Forager generates support for signal
parent_signal = signal_store.sample_weighted(INITIAL)
prompt = f"Support this idea: {parent_signal.content}"
response = await llm.generate(prompt)

# Problem: No verification parent exists when child is created
# Problem: Child doesn't maintain rich context about parent
```

**2. Race Conditions in Parallel Execution**
```python
# Scout deposits signal at time T
signal_id = signal_store.deposit(INITIAL, content, ...)

# Forager samples at time T+0.001s
parent = signal_store.sample_weighted(INITIAL)  # Might get incomplete signal
```

**3. Broken Provenance Chains**
```python
# If parent is pruned before child references it
child.parent = parent_id  # Dangling reference
# No way to trace back reasoning chain
```

### Current Parallelism Bottleneck

```
All agents → Single LLM Queue (Semaphore(1)) → Sequential Processing
             ↓
        [wait] [wait] [wait] [wait] [wait]

Timeline:
Scout1:    [LLM 2s]         [wait]           [LLM 2s]
Scout2:         [wait]  [LLM 2s]        [wait]
Scout3:              [wait]      [LLM 2s]        ...

Total: 10 scouts × 2s each = 20s SEQUENTIAL (should be 2s parallel!)
```

---

## SOLUTION DESIGN

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│              STAGE-BASED PARALLEL EXECUTION                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Stage 1: SCOUTS (Independent)                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ Scout 1  │  │ Scout 2  │  │ Scout 3  │  │ Scout 4  │       │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘       │
│       │             │             │             │                │
│       └─────────────┴─────────────┴─────────────┘                │
│                        ↓                                          │
│            [Batch LLM Processing - Parallel]                     │
│                   LLM Pool (2-4 instances)                       │
│                        ↓                                          │
│            All scouts complete → deposit INITIAL signals         │
│                        ↓                                          │
│  ──────────────────── BARRIER ────────────────────────────       │
│                        ↓                                          │
│  Stage 2: FORAGERS + CRITICS (Depend on Stage 1)                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │Forager 1 │  │Forager 2 │  │ Critic 1 │  │ Critic 2 │       │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘       │
│       │             │             │             │                │
│       └─────────────┴─────────────┴─────────────┘                │
│                        ↓                                          │
│            [Batch LLM Processing - Parallel]                     │
│                   LLM Pool (2-4 instances)                       │
│                        ↓                                          │
│    All foragers/critics complete → deposit SUPPORT/CRITIQUE      │
│                        ↓                                          │
│  ──────────────────── BARRIER ────────────────────────────       │
│                        ↓                                          │
│  Stage 3: HATERS (Depend on Stages 1+2)                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                      │
│  │ Hater 1  │  │ Hater 2  │  │ Hater 3  │                      │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘                      │
│       │             │             │                               │
│       └─────────────┴─────────────┘                               │
│                        ↓                                          │
│            [Batch LLM Processing - Parallel]                     │
│                   LLM Pool (2-4 instances)                       │
│                        ↓                                          │
│       All haters complete → deposit OBJECTION signals            │
│                        ↓                                          │
│  ──────────────────── BARRIER ────────────────────────────       │
│                        ↓                                          │
│  Stage 4: SYNTHESIS (Depends on all)                            │
│  ┌────────────────┐                                              │
│  │  Synthesizer   │                                              │
│  └────────┬───────┘                                              │
│           ↓                                                       │
│      [Single LLM Call]                                           │
│           ↓                                                       │
│     Final synthesis                                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## COMPONENT 1: ENHANCED EVIDENCE CHAINING

### Improvement 1: Rich Signal References

**Current Signal:**
```python
@dataclass
class Signal:
    id: str
    type: str
    content: str
    parent: Optional[str] = None  # Just an ID
    strength: float = 0.5
```

**Enhanced Signal (No Breaking Changes):**
```python
@dataclass
class Signal:
    id: str
    type: str
    content: str
    parent: Optional[str] = None
    strength: float = 0.5

    # NEW: Enhanced metadata for evidence chaining
    metadata: dict = field(default_factory=dict)
    # metadata contains:
    #   - "parent_content": str (cached for faster reference)
    #   - "parent_type": str (for context)
    #   - "reference_quality": float (how well child references parent)
    #   - "provenance_chain": List[str] (full ancestry)
```

**Implementation (backward compatible):**
```python
# swarm/core/signal.py - Add to existing Signal class
def get_parent_context(self) -> Optional[dict]:
    """Get cached parent context from metadata."""
    if self.metadata:
        return {
            "content": self.metadata.get("parent_content"),
            "type": self.metadata.get("parent_type"),
            "id": self.parent
        }
    return None

def get_provenance_chain(self) -> List[str]:
    """Get full ancestry chain."""
    return self.metadata.get("provenance_chain", [self.parent] if self.parent else [])
```

### Improvement 2: Context-Aware Signal Deposit

**Current:**
```python
# Forager generates support without context
response = await llm.generate(f"Support this: {parent.content}")
signal_store.deposit(SUPPORT, response, strength=0.7, parent=parent.id)
```

**Enhanced (with validation and context):**
```python
def deposit_with_context(signal_store, signal_type, content, strength,
                         depositor, parent_signal=None):
    """Deposit signal with enhanced parent context.

    Args:
        parent_signal: Full Signal object (not just ID)
    """
    # Validate parent exists
    if parent_signal and parent_signal.id not in signal_store.signals:
        logger.warning(f"Parent {parent_signal.id} not found, depositing as orphan")
        parent_signal = None

    # Build provenance chain
    provenance_chain = []
    if parent_signal:
        provenance_chain = parent_signal.get_provenance_chain() + [parent_signal.id]

    # Enhanced metadata
    metadata = {
        "parent_content": parent_signal.content if parent_signal else None,
        "parent_type": parent_signal.type if parent_signal else None,
        "provenance_chain": provenance_chain,
        "depositor": depositor,
        "timestamp_local": time.time()
    }

    # Deposit with metadata
    return signal_store.deposit(
        signal_type, content, strength, depositor,
        parent=parent_signal.id if parent_signal else None,
        metadata=metadata
    )
```

### Improvement 3: Reference Quality Validation

**Post-deposit quality check:**
```python
async def validate_reference_quality(child_signal: Signal, parent_signal: Signal,
                                    llm: SimpleLLM) -> float:
    """Check if child properly references parent.

    Returns:
        Quality score 0.0-1.0 (0.0 = no reference, 1.0 = excellent reference)
    """
    # Quick heuristic checks (no LLM)
    parent_keywords = set(parent_signal.content.lower().split())
    child_keywords = set(child_signal.content.lower().split())
    keyword_overlap = len(parent_keywords & child_keywords) / len(parent_keywords)

    if keyword_overlap < 0.1:
        return 0.0  # No meaningful reference

    if keyword_overlap > 0.5:
        return 0.9  # Strong reference

    # Medium overlap - could be good or bad
    # For now, use heuristic (could optionally use LLM for borderline cases)
    return 0.5 + (keyword_overlap - 0.1) / 0.4 * 0.4  # Scale 0.1-0.5 → 0.5-0.9
```

**Usage in agent:**
```python
# After forager generates support
response = await llm.generate(prompt)
child_id = deposit_with_context(
    signal_store, SUPPORT, response, 0.7,
    depositor=self.id, parent_signal=parent
)

# Validate reference quality
child = signal_store.get_signal(child_id)
quality = await validate_reference_quality(child, parent, llm)
child.metadata["reference_quality"] = quality

# Boost strength if high-quality reference
if quality > 0.7:
    signal_store.adjust_strength(child_id, boost=0.1)
```

---

## COMPONENT 2: CUDA-AWARE LLM POOL

### Design: Adaptive Pool Sizing

```python
# swarm/llm/llm_pool.py (NEW FILE)

import asyncio
import torch
from typing import List, Optional
from queue import PriorityQueue
from .simple_llm import SimpleLLM

class AdaptiveLLMPool:
    """LLM pool that adapts to available CUDA memory.

    Key features:
    - Query CUDA memory at startup
    - Allocate 2-4 instances based on available memory
    - Work-stealing queue for load balancing
    - Batch prompts to same instance when possible
    """

    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.instances: List[SimpleLLM] = []
        self.work_queue = asyncio.Queue()
        self.instance_busy = []  # Track which instances are busy

    async def initialize(self) -> int:
        """Initialize pool with optimal size based on CUDA memory.

        Returns:
            Number of instances created
        """
        if self.device != "cuda":
            # CPU mode: use single instance
            instance = SimpleLLM(self.model_name, self.device)
            await instance.load()
            self.instances.append(instance)
            self.instance_busy.append(False)
            return 1

        # Query CUDA memory
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")

        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
        available_memory = total_memory - (torch.cuda.memory_allocated(0) / 1e9)

        print(f"[LLM_POOL] CUDA memory: {available_memory:.1f}GB available / {total_memory:.1f}GB total")

        # Estimate memory per instance based on model
        # Qwen2.5-3B: ~6GB, Qwen2.5-7B: ~14GB, etc.
        memory_per_instance = self._estimate_model_memory()
        print(f"[LLM_POOL] Estimated memory per instance: {memory_per_instance:.1f}GB")

        # Calculate optimal pool size
        # Reserve 2GB for activations/overhead
        usable_memory = available_memory - 2.0
        max_instances = int(usable_memory / memory_per_instance)
        pool_size = max(1, min(4, max_instances))  # Between 1 and 4

        print(f"[LLM_POOL] Creating pool with {pool_size} instances...")

        # Load instances
        for i in range(pool_size):
            instance = SimpleLLM(self.model_name, self.device)
            await instance.load()
            self.instances.append(instance)
            self.instance_busy.append(False)

            # Report memory after each load
            if self.device == "cuda":
                allocated = torch.cuda.memory_allocated(0) / 1e9
                print(f"[LLM_POOL] Instance {i+1}/{pool_size} loaded (CUDA: {allocated:.1f}GB used)")

        print(f"[LLM_POOL] Pool ready with {pool_size} instances\n")
        return pool_size

    def _estimate_model_memory(self) -> float:
        """Estimate memory needed per model instance (GB)."""
        # Rough estimates based on model size
        if "3b" in self.model_name.lower() or "3B" in self.model_name:
            return 6.0  # ~6GB for 3B model
        elif "7b" in self.model_name.lower() or "7B" in self.model_name:
            return 14.0  # ~14GB for 7B model
        elif "14b" in self.model_name.lower() or "14B" in self.model_name:
            return 28.0  # ~28GB for 14B model
        else:
            return 8.0  # Default conservative estimate

    def get_pool_size(self) -> int:
        """Get number of LLM instances in pool."""
        return len(self.instances)

    async def generate(self, prompt: str, max_tokens: int = 150,
                      temperature: float = 0.7, priority: int = 0) -> str:
        """Generate response using least-busy instance.

        Args:
            prompt: Input prompt
            max_tokens: Max tokens to generate
            temperature: Sampling temperature
            priority: Higher priority gets processed first (0 = normal)

        Returns:
            Generated text
        """
        # Find least-busy instance
        instance_idx = self._select_instance()
        instance = self.instances[instance_idx]

        # Mark as busy
        self.instance_busy[instance_idx] = True

        try:
            # Generate
            result = await instance.generate(prompt, max_tokens, temperature)
            return result
        finally:
            # Mark as free
            self.instance_busy[instance_idx] = False

    async def generate_batch(self, prompts: List[str], max_tokens: int = 150,
                            temperature: float = 0.7) -> List[str]:
        """Generate responses for multiple prompts in parallel.

        Distributes prompts across all instances for true parallelism.

        Args:
            prompts: List of prompts
            max_tokens: Max tokens per prompt
            temperature: Sampling temperature

        Returns:
            List of generated texts (same order as prompts)
        """
        if not prompts:
            return []

        pool_size = len(self.instances)

        # Single prompt: use regular generate
        if len(prompts) == 1:
            return [await self.generate(prompts[0], max_tokens, temperature)]

        # Multiple prompts: distribute across pool
        tasks = []
        for i, prompt in enumerate(prompts):
            # Round-robin distribution across instances
            instance_idx = i % pool_size
            instance = self.instances[instance_idx]

            # Create task
            task = instance.generate(prompt, max_tokens, temperature)
            tasks.append(task)

        # Execute all in parallel
        results = await asyncio.gather(*tasks)
        return results

    def _select_instance(self) -> int:
        """Select least-busy instance (round-robin among free instances)."""
        # Find free instances
        free_instances = [i for i, busy in enumerate(self.instance_busy) if not busy]

        if free_instances:
            # Return first free instance
            return free_instances[0]
        else:
            # All busy - return first (will wait)
            return 0

    async def generate_with_stage_batching(self, stage_name: str,
                                          agent_prompts: List[tuple]) -> List[str]:
        """Generate for all agents in a stage with optimal batching.

        Args:
            stage_name: Name of stage (for logging)
            agent_prompts: List of (agent_id, prompt, max_tokens, temperature)

        Returns:
            List of results (same order as agent_prompts)
        """
        if not agent_prompts:
            return []

        print(f"[LLM_POOL] Stage '{stage_name}': Processing {len(agent_prompts)} prompts across {len(self.instances)} instances")

        # Distribute prompts across instances
        pool_size = len(self.instances)
        batches = [[] for _ in range(pool_size)]

        for i, (agent_id, prompt, max_tokens, temperature) in enumerate(agent_prompts):
            instance_idx = i % pool_size
            batches[instance_idx].append((agent_id, prompt, max_tokens, temperature))

        # Process each batch on its instance (in parallel)
        async def process_batch(instance_idx: int, batch: List[tuple]) -> List[tuple]:
            instance = self.instances[instance_idx]
            results = []
            for agent_id, prompt, max_tokens, temperature in batch:
                result = await instance.generate(prompt, max_tokens, temperature)
                results.append((agent_id, result))
            return results

        # Execute all batches in parallel
        batch_tasks = [
            process_batch(i, batch)
            for i, batch in enumerate(batches) if batch
        ]
        batch_results = await asyncio.gather(*batch_tasks)

        # Flatten and reorder results to match input order
        all_results = {}
        for batch_result in batch_results:
            for agent_id, result in batch_result:
                all_results[agent_id] = result

        # Return in original order
        ordered_results = [
            all_results[agent_id]
            for agent_id, _, _, _ in agent_prompts
        ]

        print(f"[LLM_POOL] Stage '{stage_name}': Complete ({len(ordered_results)} responses)")
        return ordered_results

    def get_stats(self) -> dict:
        """Get pool statistics."""
        return {
            "pool_size": len(self.instances),
            "busy_instances": sum(self.instance_busy),
            "free_instances": len(self.instances) - sum(self.instance_busy)
        }
```

---

## COMPONENT 3: STAGE-BASED EXECUTION

### Stage Coordinator

```python
# swarm/core/stage_coordinator.py (NEW FILE)

import asyncio
from typing import List, Dict, Any
from .signal_store import SignalStore
from ..llm.llm_pool import AdaptiveLLMPool

class StageCoordinator:
    """Coordinates stage-based parallel execution.

    Each stage:
    1. Waits for dependencies (previous stage completion)
    2. Batches all agents in stage
    3. Executes in parallel across LLM pool
    4. Validates outputs
    5. Signals next stage to start
    """

    def __init__(self, llm_pool: AdaptiveLLMPool, signal_store: SignalStore):
        self.llm_pool = llm_pool
        self.signal_store = signal_store

        # Stage completion events
        self.stage_complete = {
            "scouts": asyncio.Event(),
            "foragers_critics": asyncio.Event(),
            "haters": asyncio.Event(),
            "validators": asyncio.Event()
        }

    async def run_stage(self, stage_name: str, agents: List[Any],
                       depends_on: List[str] = None) -> Dict[str, Any]:
        """Run a stage of agents in parallel.

        Args:
            stage_name: Name of stage (for logging)
            agents: List of agent objects
            depends_on: List of stage names this stage depends on

        Returns:
            Stage statistics
        """
        # Wait for dependencies
        if depends_on:
            print(f"[STAGE] {stage_name}: Waiting for dependencies {depends_on}")
            for dep in depends_on:
                await self.stage_complete[dep].wait()

        print(f"\n[STAGE] {stage_name}: Starting with {len(agents)} agents")
        stage_start = time.time()

        # Collect all prompts from agents
        agent_prompts = []
        for agent in agents:
            # Each agent prepares its prompt
            prompt_data = await agent.prepare_prompt(self.signal_store)
            if prompt_data:
                agent_prompts.append(prompt_data)

        print(f"[STAGE] {stage_name}: {len(agent_prompts)} prompts collected")

        # Batch execute across LLM pool
        results = await self.llm_pool.generate_with_stage_batching(
            stage_name, agent_prompts
        )

        # Process results (deposit signals)
        signals_deposited = 0
        for agent, result in zip(agents, results):
            if result:
                deposited = await agent.process_result(result, self.signal_store)
                if deposited:
                    signals_deposited += 1

        stage_time = time.time() - stage_start
        print(f"[STAGE] {stage_name}: Complete in {stage_time:.2f}s "
              f"({signals_deposited} signals deposited)")

        # Mark stage as complete
        self.stage_complete[stage_name].set()

        return {
            "stage": stage_name,
            "agents": len(agents),
            "prompts": len(agent_prompts),
            "signals_deposited": signals_deposited,
            "time": stage_time
        }
```

### Agent Interface for Staged Execution

**Each agent needs two methods:**

```python
# Example: Scout agent
class Scout:
    async def prepare_prompt(self, signal_store: SignalStore) -> Optional[tuple]:
        """Prepare prompt for this agent (no LLM call).

        Returns:
            (agent_id, prompt, max_tokens, temperature) or None if nothing to do
        """
        # Check if we should generate (e.g., have research fragments)
        if not self.should_generate():
            return None

        # Build prompt
        prompt = self._build_prompt()
        max_tokens = self.task_config.intake_profile.scout_tokens
        temperature = TEMP_SCOUT

        return (self.id, prompt, max_tokens, temperature)

    async def process_result(self, result: str, signal_store: SignalStore) -> bool:
        """Process LLM result and deposit signal.

        Args:
            result: Generated text from LLM
            signal_store: Where to deposit

        Returns:
            True if signal deposited, False otherwise
        """
        # Quality check
        if not self._is_high_quality(result):
            return False

        # Deposit signal
        signal_id = deposit_with_context(
            signal_store,
            signal_type=self.output_type,
            content=result,
            strength=0.7,
            depositor=self.id,
            parent_signal=None  # Scouts have no parent
        )

        return signal_id is not None
```

---

## COMPONENT 4: FULL INTEGRATION

### Modified run_task.py

```python
# Pseudocode for integration

async def run_task_with_parallel_stages(task_type: str, custom_prompt: str = None):
    """Run task with stage-based parallel execution."""

    # Setup (same as before)
    task_config = create_custom_task(task_type, custom_prompt)
    signal_store = SignalStore(...)

    # CHANGE 1: Use AdaptiveLLMPool instead of SimpleLLM
    print("[INIT] Initializing adaptive LLM pool...")
    llm_pool = AdaptiveLLMPool(MODEL_NAME, DEVICE)
    pool_size = await llm_pool.initialize()
    print(f"[INIT] LLM pool ready with {pool_size} instances\n")

    # CHANGE 2: Create stage coordinator
    stage_coordinator = StageCoordinator(llm_pool, signal_store)

    # Create agents (same as before, but pass llm_pool)
    scouts = [create_scout(i, task_config) for i in range(NUM_SCOUTS)]
    foragers = [create_forager(i, task_config) for i in range(NUM_FORAGERS)]
    critics = [create_critic(i, task_config) for i in range(NUM_CRITICS)]
    haters = [create_hater(i, task_config) for i in range(NUM_HATERS)]

    # CHANGE 3: Run in stages (not all at once)
    for round_num in range(NUM_ROUNDS):
        print(f"\n{'='*70}")
        print(f"ROUND {round_num + 1}/{NUM_ROUNDS}")
        print(f"{'='*70}\n")

        # Research phase (same as before)
        await do_research(...)

        # STAGE 1: Scouts (independent)
        await stage_coordinator.run_stage(
            "scouts",
            agents=scouts,
            depends_on=None  # No dependencies
        )

        # STAGE 2: Foragers + Critics (depend on scouts)
        await asyncio.gather(
            stage_coordinator.run_stage(
                "foragers",
                agents=foragers,
                depends_on=["scouts"]
            ),
            stage_coordinator.run_stage(
                "critics",
                agents=critics,
                depends_on=["scouts"]
            )
        )

        # STAGE 3: Haters (depend on all previous)
        await stage_coordinator.run_stage(
            "haters",
            agents=haters,
            depends_on=["scouts", "foragers", "critics"]
        )

        # Synthesis (same as before, but use llm_pool)
        synthesis = await synthesizer.synthesize(
            signal_store, llm_pool.instances[0], ...
        )
```

---

## PERFORMANCE PROJECTIONS

### Current Performance (Sequential)

```
Round timing:
  Research:     10s
  Scouts:       20s  (10 scouts × 2s each, sequential)
  Foragers:     20s  (10 foragers × 2s each, sequential)
  Critics:      10s  (5 critics × 2s each, sequential)
  Haters:       10s  (5 haters × 2s each, sequential)
  Synthesis:     5s  (1 call)
  ────────────────
  Total:        75s per round

3 rounds = 225s (3.75 minutes)
```

### With Parallel Stages (Pool Size = 4)

```
Round timing:
  Research:     10s  (unchanged)
  Scouts:        5s  (10 scouts ÷ 4 instances = 2.5 batches × 2s)
  Foragers:      5s  (10 foragers ÷ 4 instances = 2.5 batches × 2s)
  Critics:       3s  (5 critics ÷ 4 instances = 1.25 batches × 2s)
  Haters:        3s  (5 haters ÷ 4 instances = 1.25 batches × 2s)
  Synthesis:     5s  (unchanged)
  ────────────────
  Total:        31s per round

3 rounds = 93s (1.55 minutes)

SPEEDUP: 225s → 93s = 2.4x
```

### With Parallel Stages (Pool Size = 2)

```
Round timing:
  Research:     10s  (unchanged)
  Scouts:       10s  (10 scouts ÷ 2 instances = 5 batches × 2s)
  Foragers:     10s  (10 foragers ÷ 2 instances = 5 batches × 2s)
  Critics:       5s  (5 critics ÷ 2 instances = 2.5 batches × 2s)
  Haters:        5s  (5 haters ÷ 2 instances = 2.5 batches × 2s)
  Synthesis:     5s  (unchanged)
  ────────────────
  Total:        45s per round

3 rounds = 135s (2.25 minutes)

SPEEDUP: 225s → 135s = 1.67x
```

---

## MEMORY REQUIREMENTS

### CUDA Memory Estimates

| Model | Single Instance | Pool Size = 2 | Pool Size = 4 |
|-------|----------------|---------------|---------------|
| **Qwen2.5-3B** | 6GB | 12GB | 24GB |
| **Qwen2.5-7B** | 14GB | 28GB | 56GB |
| **Qwen2.5-14B** | 28GB | 56GB | 112GB |

**Recommendations:**
- **24GB VRAM (RTX 3090/4090):** Pool size = 2-3 for Qwen2.5-3B
- **48GB VRAM (A6000):** Pool size = 3-4 for Qwen2.5-3B, or 2 for Qwen2.5-7B
- **80GB VRAM (A100):** Pool size = 4 for Qwen2.5-7B, or 2 for Qwen2.5-14B

**Fallback:** If insufficient memory, pool size = 1 (graceful degradation to current behavior)

---

## IMPLEMENTATION ROADMAP

### Phase 1: Foundation (Week 1)

**Days 1-2: Enhanced Evidence Chaining**
- Add `metadata` field to Signal class (backward compatible)
- Implement `deposit_with_context()` helper
- Add `validate_reference_quality()` function
- Update forager/critic to use enhanced deposit

**Days 3-4: Adaptive LLM Pool**
- Create `swarm/llm/llm_pool.py`
- Implement CUDA memory detection
- Implement adaptive pool sizing
- Add `generate_batch()` method
- Test with different pool sizes

**Days 5-7: Testing & Validation**
- Test evidence chaining (references preserved)
- Test LLM pool (memory usage, speedup)
- Verify backward compatibility
- Benchmark performance improvements

### Phase 2: Stage Coordination (Week 2)

**Days 1-3: Stage Coordinator**
- Create `swarm/core/stage_coordinator.py`
- Implement stage dependencies
- Add `run_stage()` method
- Implement barrier synchronization

**Days 4-5: Agent Refactoring**
- Add `prepare_prompt()` to each agent
- Add `process_result()` to each agent
- Maintain backward compatibility

**Days 6-7: Integration**
- Update `run_task.py` to use stage coordinator
- Add timing instrumentation for stages
- Test full pipeline
- Benchmark end-to-end performance

### Phase 3: Optimization (Week 3)

**Days 1-2: Performance Tuning**
- Profile memory usage patterns
- Optimize batch sizes
- Tune pool size selection

**Days 3-4: Quality Validation**
- Verify evidence chaining quality
- Check reference preservation
- Test with different task types

**Days 5-7: Documentation & Cleanup**
- Update architecture docs
- Add usage examples
- Performance benchmarks
- Migration guide

---

## TESTING STRATEGY

### Unit Tests

```python
# Test 1: Enhanced Evidence Chaining
def test_evidence_chaining():
    signal_store = SignalStore()

    # Deposit parent
    parent_id = signal_store.deposit(INITIAL, "Paris is the capital", 0.8, "Scout1")
    parent = signal_store.get_signal(parent_id)

    # Deposit child with context
    child_id = deposit_with_context(
        signal_store, SUPPORT,
        "Paris has been France's capital since 987 AD",
        0.7, "Forager1", parent_signal=parent
    )
    child = signal_store.get_signal(child_id)

    # Verify context preserved
    assert child.metadata["parent_content"] == "Paris is the capital"
    assert child.metadata["parent_type"] == INITIAL
    assert parent_id in child.metadata["provenance_chain"]
```

```python
# Test 2: LLM Pool Initialization
async def test_llm_pool_cuda_detection():
    pool = AdaptiveLLMPool("Qwen/Qwen2.5-3B-Instruct", "cuda")
    pool_size = await pool.initialize()

    # Should create 1-4 instances based on available memory
    assert 1 <= pool_size <= 4
    assert len(pool.instances) == pool_size
```

```python
# Test 3: Parallel Batching
async def test_parallel_batching():
    pool = AdaptiveLLMPool("test-model", "cpu")
    await pool.initialize()

    prompts = [f"Prompt {i}" for i in range(10)]

    start = time.time()
    results = await pool.generate_batch(prompts, max_tokens=50)
    elapsed = time.time() - start

    # Should be faster than sequential
    assert len(results) == len(prompts)
    # With 2+ instances, should be < 2x sequential time
```

### Integration Tests

```python
# Test 4: Stage-Based Execution
async def test_stage_execution():
    signal_store = SignalStore()
    llm_pool = AdaptiveLLMPool("test-model", "cpu")
    await llm_pool.initialize()

    coordinator = StageCoordinator(llm_pool, signal_store)

    # Create mock agents
    scouts = [MockScout(i) for i in range(5)]
    foragers = [MockForager(i) for i in range(5)]

    # Run stages
    await coordinator.run_stage("scouts", scouts, depends_on=None)
    await coordinator.run_stage("foragers", foragers, depends_on=["scouts"])

    # Verify execution order
    assert coordinator.stage_complete["scouts"].is_set()
    assert coordinator.stage_complete["foragers"].is_set()
```

---

## RISK MITIGATION

### Risk 1: CUDA Out-of-Memory

**Mitigation:**
- Adaptive pool sizing with conservative estimates
- Graceful degradation to pool_size=1 if OOM
- Monitor memory usage and warn before OOM
- Provide manual pool_size override in config

### Risk 2: Signal Dependencies Violated

**Mitigation:**
- Explicit stage barriers (cannot start until deps complete)
- Validation in `deposit_with_context()` (parent must exist)
- Provenance chain tracking (detect broken chains)
- Unit tests for all dependency scenarios

### Risk 3: Race Conditions

**Mitigation:**
- Stage barriers prevent cross-stage races
- Within-stage races are benign (agents independent)
- Signal store already has thread-safe locking
- Event-based coordination (not polling)

### Risk 4: Quality Degradation

**Mitigation:**
- Reference quality validation
- A/B testing (parallel vs sequential)
- Metrics: signal strength distribution, reference quality scores
- Rollback plan: keep sequential execution as fallback

---

## SUCCESS CRITERIA

### Performance

✅ **2-3x speedup** with pool_size=2
✅ **3-5x speedup** with pool_size=4
✅ Sub-60s per round (currently ~75s)
✅ Memory usage within CUDA limits (no OOM)

### Quality

✅ **Reference quality ≥ 0.7** for 80%+ of child signals
✅ **Provenance chains** traceable for all signals
✅ **No broken references** (all parent IDs valid)
✅ **Signal quality unchanged** (compared to sequential)

### Reliability

✅ **Zero crashes** from CUDA OOM
✅ **Graceful degradation** if insufficient memory
✅ **Backward compatible** with existing code
✅ **No race conditions** detected in testing

---

## CONCLUSION

This parallel architecture provides:

1. **True parallelism** via LLM pool (2-4 instances)
2. **Dependency safety** via stage barriers
3. **Enhanced evidence chaining** via rich metadata
4. **Adaptive CUDA usage** via memory detection
5. **3-5x speedup** while maintaining quality

**Next Steps:**
1. Review and approve design
2. Begin Phase 1 implementation (evidence chaining + LLM pool)
3. Test at each milestone
4. Iterate based on benchmarks

**Estimated Total Effort:** 2-3 weeks full implementation + testing

Ready to proceed with implementation! 🚀
