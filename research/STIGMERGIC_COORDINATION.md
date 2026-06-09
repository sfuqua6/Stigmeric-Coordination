# Stigmergic Coordination Principles

**Date:** 2025-11-14
**Commit:** `9cfa91e`
**Philosophy:** Pure event-driven emergence, not time-based synchronization

---

## What is Stigmergy?

**Stigmergy** is a mechanism of indirect coordination between agents through traces left in the environment.

### Key Principles

1. **Indirect Communication** - Agents don't talk directly, they modify the environment
2. **Event-Driven** - Agents react to environmental changes, not scheduled times
3. **Emergent Behavior** - Complex patterns emerge from simple local rules
4. **Decentralized** - No central coordinator or global clock
5. **Asynchronous** - Agents operate independently at their own pace

### Natural Examples

- **Ants:** Leave pheromone trails that guide other ants
- **Termites:** Build complex mounds by reacting to local structures
- **Slime Molds:** Form networks by following chemical gradients

---

## Why NOT Use Timeouts?

### The Problem with Time-Based Coordination

```python
# ❌ WRONG: Time-based (synchronous thinking)
await wait_for_signal(timeout=30.0)  # "Wait 30 seconds for scouts"
if not got_signal:
    await asyncio.sleep(2.0)  # "Wait 2 more seconds before retry"
```

**Why this is wrong:**
- Assumes agents finish in predictable time
- Creates artificial synchronization points
- Violates decentralization (global timing assumption)
- Not emergent (coordinator decides when to give up)
- Fragile (breaks if timing assumptions change)

### The Stigmergic Way

```python
# ✅ CORRECT: Event-driven (emergent)
while active:
    if not has_signals():
        await wait_for_signal(timeout=1.0)  # Quick poll for responsiveness
        continue  # Immediately retry - no artificial delays

    process_signals()
    await asyncio.sleep(random.uniform(0.1, 0.3))  # Natural asynchrony
```

**Why this works:**
- No assumptions about timing
- Agents wake when signals appear (event-driven)
- Immediate retry if no work (no forced waits)
- Natural load balancing from random delays
- Truly emergent coordination

---

## Implementation: Pure Event-Driven

### Signal Store as Environment

The `SignalStore` is the shared environment where agents deposit "pheromones":

```python
class SignalStore:
    def deposit(self, signal_type, content, strength, depositor):
        # Add signal to environment
        self.signals[signal_id] = Signal(...)

        # Wake up waiting agents (event notification)
        self._signal_events[signal_type].set()  # ⚡ Event trigger
```

### Agents as Reactive Entities

Agents don't schedule or plan - they react to environmental state:

#### Scout (Signal Producer)

```python
async def run(self, signal_store, llm):
    while active:
        idea = await explore_creative(llm)  # Generate content
        if idea and strength >= threshold:
            signal_store.deposit("DRAFT", idea, strength, self.id)
            # No checking if anyone is listening - just deposit and move on
        await asyncio.sleep(random.uniform(0.1, 0.5))  # Natural pace
```

**Stigmergic:** Scouts don't know or care if foragers exist. They just deposit signals.

#### Forager (Signal Consumer/Transformer)

```python
async def run(self, signal_store, llm):
    while active:
        if not signal_store.has_signals("DRAFT"):
            # Wait for event (signals appear in environment)
            await signal_store.wait_for_signal("DRAFT", timeout=1.0)
            continue  # Immediately check again - no delay

        # Signals exist - process them
        drafts = signal_store.sample_weighted("DRAFT", n=3)
        refinement = await generate_refinement(drafts, llm)
        signal_store.deposit("REFINEMENT", refinement, strength, self.id)

        await asyncio.sleep(random.uniform(0.1, 0.3))  # Natural asynchrony
```

**Stigmergic:** Foragers don't wait for scouts. They react when DRAFT signals appear.

#### Critic (Signal Evaluator)

```python
async def run(self, signal_store, llm):
    while active:
        if not signal_store.has_signals("DRAFT"):
            await signal_store.wait_for_signal("DRAFT", timeout=1.0)
            continue

        drafts = signal_store.sample_weighted("DRAFT", n=3)
        critique = await evaluate(drafts, llm)
        signal_store.deposit("CRITIQUE", critique, strength, self.id)

        await asyncio.sleep(random.uniform(0.1, 0.3))
```

**Stigmergic:** Critics don't coordinate with foragers. They both react to DRAFTs independently.

---

## Key Design Decisions

### 1. Short Poll Intervals (1 second)

```python
await wait_for_signal(type, timeout=1.0)  # Not 5s, not 30s - just 1s
```

**Why 1 second?**
- Short enough to be responsive
- Long enough to avoid busy-waiting
- Event wakes agent immediately anyway (timeout rarely reached)
- Just a failsafe for missed events

**Not a "timeout" in the traditional sense** - it's a poll interval.

### 2. Immediate Retry on No Signals

```python
if not has_signals():
    await wait_for_signal(timeout=1.0)
    continue  # ← Immediate retry, no sleep/delay
```

**Why immediate?**
- No assumption about when signals will appear
- Natural emergence - agents check as fast as they can
- Event system ensures efficiency (no CPU waste)
- Lets the system find its own rhythm

### 3. Random Delays After Processing

```python
await asyncio.sleep(random.uniform(0.1, 0.3))  # After successful work
```

**Why random?**
- Prevents synchronized oscillations (all agents acting at once)
- Natural load distribution
- Mimics biological systems (ants don't move in lockstep)
- Helps with resource contention (LLM semaphore)

### 4. Fail Silently, Retry Naturally

```python
if not input_signals:
    return  # ← Just return, will retry on next iteration

# NOT:
# print("[ERROR] No signals found!")
# await asyncio.sleep(2.0)  # ← Artificial delay
```

**Why silent failure?**
- Agents try, fail, retry - it's normal
- No centralized error handling needed
- Reduces noise in logs
- Stigmergic systems are naturally fault-tolerant

---

## Emergence in Action

### Timeline of Typical Execution

```
Time    Event                           Why It Works
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
0.0s    All agents start
        Scouts: Start LLM generation
        Foragers: Poll for DRAFT signals (none yet)
        Critics: Poll for DRAFT signals (none yet)

0.1s    Foragers timeout (1s poll), retry immediately
        Critics timeout (1s poll), retry immediately
        (No artificial delays - they keep trying)

5.0s    Scout #1 finishes generation
        → Deposits DRAFT_0000
        → Event wakes ALL waiting foragers/critics

5.1s    Foragers wake up: "DRAFT signals available!"
        → Process DRAFT_0000
        → Generate REFINEMENT_0001
        Critics wake up: "DRAFT signals available!"
        → Process DRAFT_0000
        → Generate CRITIQUE_0002

7.0s    Scout #2 finishes generation
        → Deposits DRAFT_0003
        → Another event triggers

...     System continues emerging naturally
        No central coordination needed
        Each agent just reacts to signals
```

### What Makes This Emergent?

1. **No Global Clock** - Agents don't coordinate on time
2. **Local Rules** - Each agent only looks at signals
3. **Event-Driven** - Reactions triggered by environment changes
4. **Asynchronous** - Random delays prevent lock-step
5. **Self-Organizing** - Load balances naturally

---

## Comparison: Before vs After

### Before (Time-Based) ❌

```python
# Forager waiting
if not has_signals():
    got_signal = await wait_for_signal(timeout=30.0)  # ← Assume 30s is enough

    if not got_signal:  # ← Timeout expired
        print("Still waiting...")  # ← Noise
        await asyncio.sleep(2.0)  # ← Artificial delay
        continue

# Process signals...
await asyncio.sleep(random.uniform(0.3, 0.8))  # ← Larger delays
```

**Problems:**
- Hardcoded 30-second assumption (what if scouts need 40s?)
- 2-second retry delay (why 2? arbitrary)
- Verbose logging (confuses users)
- Not truly emergent (timeout is a form of coordination)

### After (Event-Driven) ✅

```python
# Forager waiting
if not has_signals():
    await wait_for_signal(timeout=1.0)  # ← Quick poll
    continue  # ← Immediate retry

# Process signals...
await asyncio.sleep(random.uniform(0.1, 0.3))  # ← Minimal delay
```

**Advantages:**
- No timing assumptions
- Immediate retry (emergent timing)
- Silent (natural failures)
- Smaller delays (faster iteration)
- Truly stigmergic (pure event-driven)

---

## Performance Characteristics

### Event-Driven Efficiency

```python
# When no signals exist:
await wait_for_signal(timeout=1.0)
  ↓
await asyncio.Event.wait()  # ← Sleep efficiently (no CPU)
  ↓
Wakes immediately when signal deposited  # ← Event.set()
```

**CPU Usage:**
- Agents waiting: ~0% CPU (async sleep)
- Agent wakes: Instant (event notification)
- No polling overhead (event-driven)

### Natural Load Balancing

```python
# Multiple foragers compete for signals
forager_1: await asyncio.sleep(random.uniform(0.1, 0.3))  # ← 0.23s
forager_2: await asyncio.sleep(random.uniform(0.1, 0.3))  # ← 0.17s
forager_3: await asyncio.sleep(random.uniform(0.1, 0.3))  # ← 0.29s
```

**Result:**
- Foragers process signals at different times
- No synchronized "thundering herd"
- Natural distribution of work
- Better GPU/LLM utilization

---

## Failure Modes & Resilience

### What If Scouts Never Produce?

```python
# Foragers keep trying
while active and actions_taken < max_actions:
    if not has_signals():
        await wait_for_signal(timeout=1.0)
        continue
    # ...
```

**Result:**
- Foragers poll every 1 second
- Eventually hit `max_actions` limit
- No crashes, no errors
- System gracefully terminates

### What If Events Are Missed?

```python
await wait_for_signal(timeout=1.0)  # ← Failsafe timeout
```

**Result:**
- Timeout ensures agents don't block forever
- They check `has_signals()` on wake
- Event system is redundant (belt + suspenders)
- System is robust to event delivery failures

### What If LLM Is Slow?

```python
# Scouts generate at their own pace
result = await llm.generate(...)  # ← May take 5-30 seconds
```

**Result:**
- Other agents wait for signals to appear
- No timeout failures
- System naturally adapts to LLM speed
- Emergent timing based on actual performance

---

## Stigmergic Patterns in Code

### 1. Amplification (Positive Feedback)

```python
# Strong signals get sampled more often
signals = signal_store.sample_weighted(type, n=3)
  ↓
# Probability ∝ signal.strength
  ↓
# Popular signals attract more attention
```

### 2. Decay (Negative Feedback)

```python
# Weak signals fade away
signal.strength *= (1.0 - decay_rate)
  ↓
# Old/weak signals naturally removed
  ↓
# System forgets irrelevant information
```

### 3. Exploration Bonus (Diversity)

```python
# Under-visited signals get bonus weight
exploration_weight = bonus * (1.0 - visit_ratio)
  ↓
# Minority opinions get heard
  ↓
# Prevents echo chambers
```

### 4. Semantic Clustering (Similarity)

```python
# Similar signals cluster together
cluster = signal_store.sample_cluster(type, similarity_threshold=0.7)
  ↓
# Related ideas naturally group
  ↓
# Emergent topic organization
```

---

## Best Practices

### ✅ DO

1. **Use events for coordination**
   ```python
   await wait_for_signal(type, timeout=1.0)
   ```

2. **Fail silently and retry**
   ```python
   if not signals:
       return  # Try again next iteration
   ```

3. **Use random delays**
   ```python
   await asyncio.sleep(random.uniform(0.1, 0.3))
   ```

4. **Check environment state**
   ```python
   if not has_signals():
       # React to absence
   ```

5. **Deposit and forget**
   ```python
   signal_store.deposit(type, content, strength, self.id)
   # Don't wait for acknowledgment
   ```

### ❌ DON'T

1. **Don't use long timeouts**
   ```python
   await wait_for_signal(type, timeout=30.0)  # ❌ Too long
   ```

2. **Don't add artificial delays**
   ```python
   await asyncio.sleep(2.0)  # ❌ Why 2 seconds?
   ```

3. **Don't log every retry**
   ```python
   print("Still waiting...")  # ❌ Noise
   ```

4. **Don't synchronize agents**
   ```python
   await barrier.wait()  # ❌ Centralized coordination
   ```

5. **Don't assume timing**
   ```python
   # ❌ Assumes agents finish in 10s
   await asyncio.sleep(10.0)
   check_results()
   ```

---

## Summary

### Stigmergic Principles Honored

| Principle | Implementation |
|-----------|----------------|
| **Event-Driven** | Agents wake when signals appear (asyncio.Event) |
| **Emergent** | No centralized timing or coordination |
| **Asynchronous** | Random delays (0.1-0.3s) for natural distribution |
| **Fail-Safe** | Agents retry without errors, silent failures |
| **Decentralized** | Each agent only looks at local signals |
| **Self-Organizing** | Natural load balancing from asynchrony |

### Code Simplicity

- **-45 lines** of coordination code removed
- **No timeouts** longer than 1 second
- **No artificial delays** between retries
- **No verbose logging** of normal operations
- **Pure event-driven** coordination

### Result

A truly stigmergic system where:
- Agents deposit signals and move on
- Agents react when signals appear
- Complex behavior emerges from simple rules
- No central coordinator needed
- Natural fault tolerance
- Honors biological inspiration

---

**Philosophy:** Let the swarm emerge. Don't force it to synchronize.

**Commit:** `9cfa91e` - Pure event-driven stigmergic coordination
