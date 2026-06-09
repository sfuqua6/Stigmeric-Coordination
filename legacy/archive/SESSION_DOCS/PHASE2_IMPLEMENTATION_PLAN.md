# Phase 2 Implementation Plan - Agent Refactoring for Parallel Execution

**Date:** 2025-11-20
**Status:** Stage Coordinator Complete, Agent Refactoring Ready
**Branch:** `claude/analyze-codebase-01RUVXdkHt9uNkPn7rauTXiE`

---

## COMPLETED: StageCoordinator ✅

**File:** `swarm/core/stage_coordinator.py` (235 lines)

### Key Features:

**1. Dependency Management:**
- Stage registration with `asyncio.Event()` barriers
- Automatic waiting for dependencies before stage start
- Parallel execution of stages with same dependencies

**2. Batch Execution:**
- `run_stage()`: Execute all agents in a stage with LLM pool
- `run_parallel_stages()`: Run multiple independent stages together
- Automatic batching via `AdaptiveLLMPool.generate_with_stage_batching()`

**3. Statistics & Monitoring:**
- Per-stage metrics: agents, prompts, signals, time
- Deposit efficiency tracking (signals deposited / prompts generated)
- Comprehensive summary with `get_summary()`

**4. Agent Interface:**
- `StagedAgent` mixin class defines interface:
  - `prepare_prompt()`: Build prompt without LLM call
  - `process_result()`: Deposit signal from LLM result

---

## TODO: Agent Refactoring Pattern

### Template Pattern (Apply to All Agents):

**Step 1: Add StagedAgent Interface**

```python
from ..core.stage_coordinator import StagedAgent

class Scout(StagedAgent):  # Add mixin
    """Scout agent..."""

    # Keep existing methods...

    # ADD: Staged execution support
    async def prepare_prompt(self, signal_store: SignalStore) -> Optional[tuple]:
        """Prepare prompt without LLM call (staged execution).

        Returns:
            (agent_id, prompt, max_tokens, temperature) or None
        """
        # Check if should generate
        if not self._should_generate():
            return None

        # Collect context (fragments, web search, etc.)
        context = await self._collect_context(signal_store)

        # Build prompt
        prompt = self._make_prompt(context)

        # Get token allocation from config
        max_tokens = self.task_config.intake_profile.scout_tokens if self.task_config else 150

        # Return prompt data (no LLM call yet!)
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

        # Quality check
        strength = self.assess_strength_creative(result)
        if strength < self.min_strength:
            return False

        # Deposit with enhanced context
        from ..core.signal_store import deposit_with_context
        signal_id = deposit_with_context(
            signal_store,
            signal_type=self.signal_type,
            content=result.strip(),
            strength=strength,
            depositor=self.agent_id,
            parent_signal=None  # Scouts have no parent
        )

        if signal_id:
            logger.info(f"{self.agent_id} deposited {signal_id} (strength={strength:.2f})")
            return True
        return False
```

### Refactoring Checklist for Each Agent:

#### **Scout** (swarm/agents/scout.py)
- [ ] Add `StagedAgent` mixin
- [ ] Extract context collection from `explore_creative()`
- [ ] Implement `prepare_prompt()`:
  - Check fragment availability
  - Collect web search context if needed
  - Build prompt with `_make_prompt()`
  - Return (agent_id, prompt, max_tokens, temperature)
- [ ] Implement `process_result()`:
  - Quality check with `assess_strength_creative()`
  - Use `deposit_with_context()` instead of `signal_store.deposit()`
  - Track cumulative insights
- [ ] Keep `run()` method for backward compatibility

#### **Forager** (swarm/agents/forager.py)
- [ ] Add `StagedAgent` mixin
- [ ] Implement `prepare_prompt()`:
  - Sample parent signal from signal_store
  - Determine action (SUPPORT vs CRITIQUE)
  - Build prompt with parent context
  - Include provenance chain in prompt
  - Return (agent_id, prompt, max_tokens, temperature)
- [ ] Implement `process_result()`:
  - Quality check
  - Use `deposit_with_context(parent_signal=parent)`
  - Validate reference quality with `validate_reference_quality()`
  - Boost strength if high-quality reference (>0.7)
- [ ] Keep `run()` method for backward compatibility

#### **Critic** (swarm/agents/critic.py)
- [ ] Add `StagedAgent` mixin
- [ ] Implement `prepare_prompt()`:
  - Sample signal to critique
  - Build provenance context (ancestors + descendants)
  - Build prompt with full context
  - Return (agent_id, prompt, max_tokens, temperature)
- [ ] Implement `process_result()`:
  - Parse quality score from result
  - Use `deposit_with_context(parent_signal=target)`
  - Adjust parent strength based on critique
- [ ] Keep `run()` method for backward compatibility

#### **Hater** (swarm/agents/hater.py)
- [ ] Add `StagedAgent` mixin
- [ ] Implement `prepare_prompt()`:
  - Sample target signal (INITIAL, SUPPORT, or CRITIQUE)
  - Check for consensus clusters (semantic similarity)
  - Build objection prompt with target context
  - Return (agent_id, prompt, max_tokens, temperature)
- [ ] Implement `process_result()`:
  - Quality check (substantive objection?)
  - Use `deposit_with_context(parent_signal=target)`
  - Validate references parent's claims
- [ ] Keep `run()` method for backward compatibility

---

## Integration into run_task.py

### Current Architecture (Sequential):

```python
# Current: All agents run concurrently but LLM calls serialize
async def run_task(...):
    llm = SimpleLLM(...)  # Single instance

    # Create agents
    scouts = [Scout(...) for i in range(NUM_SCOUTS)]
    foragers = [Forager(...) for i in range(NUM_FORAGERS)]
    ...

    # All run together (but LLM serializes)
    tasks = []
    for scout in scouts:
        tasks.append(scout.run(signal_store, llm, max_actions=20))
    for forager in foragers:
        tasks.append(forager.run(signal_store, llm, max_actions=20))
    ...

    await asyncio.gather(*tasks)  # Pseudo-parallel (LLM bottleneck)
```

### New Architecture (Staged with LLM Pool):

```python
# New: Staged execution with true parallelism
async def run_task(...):
    # CHANGE 1: Use AdaptiveLLMPool instead of SimpleLLM
    from swarm.llm.llm_pool import AdaptiveLLMPool
    llm_pool = AdaptiveLLMPool(MODEL_NAME, DEVICE, enable_cache=ENABLE_LLM_CACHE)
    pool_size = await llm_pool.initialize()
    print(f"[INIT] LLM pool ready with {pool_size} instances\n")

    # CHANGE 2: Create StageCoordinator
    from swarm.core.stage_coordinator import StageCoordinator
    stage_coordinator = StageCoordinator(llm_pool, signal_store)

    # Create agents (same as before)
    scouts = [Scout(...) for i in range(NUM_SCOUTS)]
    foragers = [Forager(...) for i in range(NUM_FORAGERS)]
    critics = [Critic(...) for i in range(NUM_CRITICS)]
    haters = [Hater(...) for i in range(NUM_HATERS)]

    # CHANGE 3: Run in stages instead of all at once
    for round_num in range(NUM_ROUNDS):
        print(f"\nROUND {round_num + 1}/{NUM_ROUNDS}")

        # Stage 1: Scouts (no dependencies)
        await stage_coordinator.run_stage(
            "scouts",
            agents=scouts,
            depends_on=None,
            max_iterations=ITERATIONS_PER_ROUND
        )

        # Stage 2: Foragers + Critics (depend on scouts)
        await stage_coordinator.run_parallel_stages([
            ("foragers", foragers, ["scouts"], ITERATIONS_PER_ROUND),
            ("critics", critics, ["scouts"], ITERATIONS_PER_ROUND)
        ])

        # Stage 3: Haters (depend on foragers + critics)
        await stage_coordinator.run_stage(
            "haters",
            agents=haters,
            depends_on=["foragers", "critics"],
            max_iterations=HATER_ACTIONS_PER_ROUND
        )

        # Reset for next round
        stage_coordinator.reset()

    # Show stage summary
    print("\n" + stage_coordinator.get_summary())
```

### Benefits of Staged Execution:

✅ **True Parallelism:** LLM pool enables parallel processing
✅ **Dependency Safety:** Stage barriers prevent race conditions
✅ **Better Observability:** Per-stage metrics and timing
✅ **Backward Compatible:** Old `run()` methods still work
✅ **Gradual Migration:** Can migrate agents one at a time

---

## Expected Performance Improvement

### Baseline (Current Sequential):

```
Round timing with 10 scouts, 10 foragers, 5 critics, 5 haters:
  Scouts:    20s (10 × 2s sequential)
  Foragers:  20s (10 × 2s sequential)
  Critics:   10s (5 × 2s sequential)
  Haters:    10s (5 × 2s sequential)
  Total:     60s per round
```

### With Pool Size = 2:

```
Round timing (same agents):
  Scouts:    10s (10 ÷ 2 = 5 batches × 2s)
  Foragers:  10s (10 ÷ 2 = 5 batches × 2s)
  Critics:    5s (5 ÷ 2 = 2.5 batches × 2s)
  Haters:     5s (5 ÷ 2 = 2.5 batches × 2s)
  Total:     30s per round

  Speedup: 60s → 30s = 2x
```

### With Pool Size = 4:

```
Round timing (same agents):
  Scouts:     5s (10 ÷ 4 = 2.5 batches × 2s)
  Foragers:   5s (10 ÷ 4 = 2.5 batches × 2s)
  Critics:    3s (5 ÷ 4 = 1.25 batches × 2s)
  Haters:     3s (5 ÷ 4 = 1.25 batches × 2s)
  Total:     16s per round

  Speedup: 60s → 16s = 3.75x
```

---

## Implementation Timeline

### Week 1 (Current):
- ✅ Day 1: StageCoordinator implementation
- ⏳ Day 2-3: Agent refactoring (Scout, Forager, Critic, Hater)
- ⏳ Day 4: Integration into run_task.py
- ⏳ Day 5: Testing with hyper_test mode

### Week 2:
- Day 1: Performance benchmarking
- Day 2-3: Evidence chaining enhancements (provenance-aware prompts)
- Day 4: Reference quality enforcement
- Day 5: Documentation and examples

---

## Testing Strategy

### Unit Tests:

```python
# Test StageCoordinator dependency management
async def test_stage_dependencies():
    coordinator = StageCoordinator(llm_pool, signal_store)

    # Register stages
    coordinator.register_stage("stage_a")
    coordinator.register_stage("stage_b")

    # Stage B depends on Stage A
    async def run_a():
        await asyncio.sleep(1)
        coordinator.stage_complete["stage_a"].set()

    async def run_b():
        start = time.time()
        await coordinator.stage_complete["stage_a"].wait()
        elapsed = time.time() - start
        assert elapsed >= 1.0  # Waited for stage A

    await asyncio.gather(run_a(), run_b())
```

```python
# Test agent staged execution
async def test_scout_staged_execution():
    scout = Scout("test_scout", INITIAL, "test task")
    signal_store = SignalStore()

    # Prepare prompt
    prompt_data = await scout.prepare_prompt(signal_store)
    assert prompt_data is not None
    agent_id, prompt, max_tokens, temperature = prompt_data

    # Simulate LLM result
    result = "This is a test insight about the task."

    # Process result
    deposited = await scout.process_result(result, signal_store)
    assert deposited == True

    # Verify signal deposited
    signals = signal_store.get_signals_by_type(INITIAL)
    assert len(signals) == 1
```

### Integration Tests:

```python
# Test full staged pipeline
async def test_staged_pipeline():
    # Setup
    llm_pool = AdaptiveLLMPool("test-model", "cpu")
    await llm_pool.initialize()
    signal_store = SignalStore()
    coordinator = StageCoordinator(llm_pool, signal_store)

    # Create agents
    scouts = [Scout(f"scout_{i}", INITIAL, "test") for i in range(3)]
    foragers = [Forager(f"forager_{i}", INITIAL, SUPPORT, "test") for i in range(3)]

    # Run stages
    await coordinator.run_stage("scouts", scouts, None, max_iterations=1)
    await coordinator.run_stage("foragers", foragers, ["scouts"], max_iterations=1)

    # Verify signals
    initial_signals = signal_store.get_signals_by_type(INITIAL)
    support_signals = signal_store.get_signals_by_type(SUPPORT)

    assert len(initial_signals) > 0
    assert len(support_signals) > 0

    # Verify dependencies
    for support in support_signals:
        assert support.parent in [s.id for s in initial_signals]
```

---

## Risk Mitigation

### Risk 1: Backward Compatibility Breaks

**Mitigation:**
- Keep existing `run()` methods unchanged
- Add `prepare_prompt()` and `process_result()` as new methods
- StageCoordinator checks for method existence before calling
- Gradual migration: Can use staged execution for some agents, old execution for others

### Risk 2: Quality Degradation

**Mitigation:**
- A/B test: Run same task with old vs new architecture
- Compare signal quality metrics (avg strength, diversity, reference quality)
- Ensure deposit efficiency remains >70%
- Monitor with SwarmMonitor health metrics

### Risk 3: LLM Pool Memory Issues

**Mitigation:**
- AdaptiveLLMPool already handles CUDA OOM gracefully
- Falls back to single instance if insufficient memory
- Test with different pool sizes (1, 2, 4) to find optimal
- Monitor memory usage during execution

### Risk 4: Race Conditions in Signal Dependencies

**Mitigation:**
- Stage barriers prevent cross-stage races
- Each stage completes fully before next stage starts
- Agents in same stage operate on independent signals (no shared state)
- Signal store already has thread-safe locking

---

## Success Criteria

### Performance:
- ✅ 2x speedup with pool_size=2
- ✅ 3-4x speedup with pool_size=4
- ✅ Graceful degradation to 1x if memory constrained

### Quality:
- ✅ Signal strength distribution unchanged (or improved)
- ✅ Reference quality ≥0.7 for 70%+ of signals
- ✅ Provenance chains complete and traceable
- ✅ No broken parent references

### Reliability:
- ✅ Zero crashes from CUDA OOM
- ✅ Stage dependencies respected (no race conditions)
- ✅ Backward compatible with existing code
- ✅ Passes all integration tests

---

## Next Steps

1. **Review this plan** - Approve approach before refactoring all agents
2. **Refactor agents** - Apply template pattern to Scout, Forager, Critic, Hater
3. **Integrate into run_task.py** - Replace SimpleLLM with AdaptiveLLMPool + StageCoordinator
4. **Test** - Run hyper_test mode, verify correctness
5. **Benchmark** - Measure actual speedup with different pool sizes
6. **Document** - Update README with new architecture

**Estimated Remaining Effort:** 6-8 hours to complete Phase 2

Ready to proceed with agent refactoring? Or would you like to review/adjust the approach first?
