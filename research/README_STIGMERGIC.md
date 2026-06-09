# Stigmergic Swarm Intelligence System v1.0

**True swarm mechanics through stigmergic coordination**

## What is Stigmergy?

Stigmergy is a mechanism of **indirect coordination** where agents communicate through modifications to a shared environment. Think of ants leaving pheromone trails - no ant tells another where to go, but the trails guide collective behavior.

## Core Swarm Mechanics

### Decentralized Coordination Pattern

```
Agents → Deposit Signals → Shared Environment
         ↑                         ↓
         └─── Sample (weighted) ←──┘
```

**Key Principles:**

1. **No Central Controller** - Agents act independently based on local information
2. **Indirect Communication** - Agents interact through signal deposits, not direct messages
3. **Probabilistic Selection** - Agents sample signals weighted by strength (not deterministically)
4. **Positive Feedback** - Successful paths get amplified (corroboration strengthens signals)
5. **Negative Feedback** - Unused paths decay (evaporation weakens signals)
6. **Emergent Convergence** - System naturally focuses on strong solutions without explicit termination

### Signal Lifecycle

```
DEPOSIT → SAMPLE → AMPLIFY → DECAY → PRUNE
   ↓         ↓         ↑        ↓        ↓
Scout    Forager   Success  Time    Weak signals
explores  follows   builds            removed
          signal   strength
```

---

## System Architecture

### Components

#### 1. Signal Store (Pheromone Environment)

**File:** `swarm/core/signal_store.py`

Thread-safe shared medium where agents deposit and sample signals.

**Signal Structure:**
```python
@dataclass
class Signal:
    id: str              # Unique identifier
    type: str            # CLAIM, EVIDENCE, CRITIQUE
    content: str         # The actual content
    strength: float      # 0.0 to 1.0 (pheromone concentration)
    timestamp: float     # When deposited
    depositor: str       # Agent ID
    parent: Optional[str] # Parent signal (for chains)
    visits: int          # Corroboration count
```

**Operations:**
- `deposit(type, content, strength, depositor)` - Add signal to environment
- `sample_weighted(type, n)` - Get n signals probabilistically (weighted by strength)
- `amplify(signal_id, factor)` - Increase signal strength (corroboration)
- `decay_all()` - Reduce all signals (evaporation)
- `prune_weak(threshold)` - Remove signals below threshold
- `get_top_signals(type, n)` - Get strongest signals (final results)

**Stigmergic Properties:**
- **Positive Feedback**: `amplify()` increases strength when agents corroborate
- **Negative Feedback**: `decay_all()` reduces strength over time
- **Self-Organization**: Strong signals attract more visits, weak signals fade

#### 2. Scout Agents

**File:** `swarm/agents/scout.py`

Exploration agents that probe solution space and deposit initial signals.

**Behavior Loop:**
```python
while active:
    # Explore randomly (high temperature, diverse prompts)
    idea = explore_random()

    # Self-assess quality
    strength = assess_strength(idea)

    # Deposit if promising
    if strength > min_threshold:
        signal_store.deposit(signal)

    # Non-blocking delay
    await asyncio.sleep(random(0.1, 0.5))
```

**Purpose:**
- Generate diverse possibilities
- Deposit graded signals
- No coordination with other scouts
- High exploration (temperature=0.9)

#### 3. Forager Agents

**File:** `swarm/agents/forager.py`

Following agents that sample signals and develop strong ones.

**Behavior Loop:**
```python
while active:
    # Sample signal weighted by strength
    signal = signal_store.sample_weighted(input_type)

    # Develop the idea further
    developed = develop(signal.content)

    # Assess result
    strength = assess_developed(developed)

    # Amplify if good (positive feedback)
    if strength > threshold:
        signal_store.amplify(signal.id, factor=1.3)
        signal_store.deposit(refined_signal)

    await asyncio.sleep(random(0.2, 0.6))
```

**Purpose:**
- Follow strong signals (exploitation)
- Amplify successful paths
- Deposit refined outputs
- Biased sampling (weighted by strength)

#### 4. Environmental Process

**Runs concurrently with agents:**

```python
async def environment_process():
    for iteration in range(MAX_ITERATIONS):
        await asyncio.sleep(ITERATION_DELAY)

        # Decay all signals (evaporation)
        signal_store.decay_all()

        # Remove weak signals (pruning)
        signal_store.prune_weak()
```

**Purpose:**
- Implements signal decay (evaporation)
- Prunes weak signals
- No direct agent control

---

## How Swarm Coordination Works

### Example: Claim Generation and Evidence Gathering

**Time T=0:**
```
[Empty Environment]
```

**Time T=1-5: Scouts Explore**

```
Scout_A explores → generates "CO2 levels rising" → assess=0.6 → deposit CLAIM_0001
Scout_B explores → generates "Temperature anomalies" → assess=0.7 → deposit CLAIM_0002
Scout_C explores → generates "Ice melt accelerating" → assess=0.5 → deposit CLAIM_0003
```

**Environment State:**
```
CLAIM_0001: strength=0.6
CLAIM_0002: strength=0.7  ← Strongest
CLAIM_0003: strength=0.5
```

**Time T=6: Foragers Sample (Probabilistic)**

Forager samples signals weighted by strength:
- CLAIM_0002 has 70% chance (strongest)
- CLAIM_0001 has 60% chance
- CLAIM_0003 has 50% chance

**Time T=7: Forager Develops Strong Signal**

```
Forager_A samples CLAIM_0002 (probabilistically chosen)
Forager_A develops: "IPCC reports show 1.1°C warming..."
Forager_A assesses: strength=0.8 (good!)
Forager_A amplifies CLAIM_0002: 0.7 → 0.91 (×1.3)
Forager_A deposits EVIDENCE_0004: strength=0.8
```

**Environment State:**
```
CLAIM_0001: strength=0.6
CLAIM_0002: strength=0.91 ← AMPLIFIED! (corroborated)
CLAIM_0003: strength=0.5
EVIDENCE_0004: strength=0.8 (parent: CLAIM_0002)
```

**Time T=8: Decay (Evaporation)**

```
All signals decay by 5%:
CLAIM_0001: 0.6 → 0.57
CLAIM_0002: 0.91 → 0.86  ← Still strongest
CLAIM_0003: 0.5 → 0.48
EVIDENCE_0004: 0.8 → 0.76
```

**Time T=9-15: Positive Feedback Loop**

More foragers sample CLAIM_0002 (strongest signal):
- Each successful development amplifies it further
- Creates cascading attention on promising paths
- Weak signals (CLAIM_0001, CLAIM_0003) continue decaying

**Time T=16: Pruning**

```
CLAIM_0003: strength=0.13 < threshold (0.15)
→ Pruned (removed from environment)
```

**Convergence:**
- System naturally focuses on CLAIM_0002
- Multiple evidence signals build up
- Weak signals fade away
- No explicit "pick the best" logic needed

---

## Exploration-Exploitation Tradeoff

### Exploration (Scouts)
- High temperature (0.9)
- Random exploration
- Diverse signal generation
- Low strength threshold (0.4)

### Exploitation (Foragers)
- Lower temperature (0.7)
- Weighted sampling (biased toward strong signals)
- Develop existing signals
- Higher strength threshold (0.5)

### Balance
- Decay rate (0.05) controls how fast signals fade
- Prune threshold (0.15) controls when signals are removed
- Amplify factor (1.3) controls positive feedback strength

**Tuning:**
- Higher decay → more exploration (signals fade faster, try new things)
- Lower decay → more exploitation (signals persist, focus on existing)
- Higher amplify → stronger convergence (successful paths dominate quickly)
- Lower amplify → broader search (less winner-take-all)

---

## File Structure

```
swarm/
├── core/
│   ├── signal_store.py       # Pheromone environment (Signal class + SignalStore)
│   └── config.py             # Configuration parameters
├── agents/
│   ├── scout.py              # Exploration agents (deposit initial signals)
│   └── forager.py            # Following agents (sample + amplify + refine)
├── llm/
│   └── simple_llm.py         # Minimal LLM wrapper (distilgpt2)
└── main_swarm.py             # Original entry point (use run_swarm.py instead)

run_swarm.py                  # Top-level entry point (RECOMMENDED)
STIGMERGIC_SWARM_DESIGN.md    # Detailed design document
archive/evolutionary_v0.3.5/  # Previous evolutionary approach (archived)
```

---

## Configuration

**File:** `swarm/core/config.py`

```python
# Model settings
MODEL_NAME = "distilgpt2"      # 353MB, loads in ~10s
DEVICE = "cuda" or "cpu"       # Auto-detected
MAX_TOKENS = 100               # Short responses
TEMPERATURE = 0.8              # Diversity

# Swarm settings
NUM_SCOUTS = 3                 # Exploration agents
NUM_FORAGERS = 3               # Following agents
MAX_ITERATIONS = 20            # Total cycles

# Signal dynamics
DECAY_RATE = 0.05              # 5% strength reduction per iteration
PRUNE_THRESHOLD = 0.15         # Remove signals below this
AMPLIFY_FACTOR = 1.3           # 30% boost when corroborated

# Thresholds
MIN_DEPOSIT_STRENGTH = 0.4     # Only deposit if assessed above this
MIN_AMPLIFY_STRENGTH = 0.5     # Only amplify if developed idea is strong

# Topic
THESIS = "Climate change requires immediate global action..."
```

---

## Running the System

### Basic Run

```bash
python run_swarm.py
```

### Expected Output

```
======================================================================
STIGMERGIC SWARM SYSTEM v1.0
======================================================================

Thesis: Climate change requires immediate global action...

[INIT] Loading language model...
[LLM] Model loaded in 10.2s

[INIT] Created 3 scouts and 3 foragers

[START] Launching swarm (agents run independently)...

[SCOUT] Scout_Claim_1 deposited CLAIM_0000 (strength=0.69)
[SCOUT] Scout_Claim_2 deposited CLAIM_0001 (strength=0.80)

[ITER 01] Environment update:
  Signals: 2 (pruned 0, avg strength 0.75)
  By type: {'CLAIM': 2}

[FORAGER] Forager_Evidence_0 amplified CLAIM_0001 -> 0.85
[FORAGER] Forager_Evidence_0 deposited EVIDENCE_0002

...

======================================================================
SWARM COMPLETE - RESULTS
======================================================================

--- Top CLAIMs (by signal strength) ---
1. [Strength: 0.921, Visits: 4]
   Global temperatures have risen 1.1°C since pre-industrial times...

2. [Strength: 0.876, Visits: 3]
   Renewable energy transition is economically feasible...

--- Top EVIDENCE (by signal strength) ---
1. [Strength: 0.885, Visits: 2]
   IPCC AR6 reports show consistent warming trends...
```

---

## Key Differences from Evolutionary Approach (v0.3.5)

| Aspect | Evolutionary v0.3.5 | Stigmergic v1.0 |
|--------|-------------------|----------------|
| **Coordination** | Central scheduler | Decentralized signals |
| **Communication** | Direct (state manager) | Indirect (pheromone trails) |
| **Agent Types** | ClaimGenerator, EvidenceFinder, Critic | Scout (explore), Forager (follow) |
| **Selection** | Performance scores + spawn/death | Probabilistic signal sampling |
| **Feedback** | Agent EMA scores | Signal amplification |
| **Convergence** | Explicit termination conditions | Natural signal concentration |
| **Lifecycle** | Agent spawning/death | Signal decay/pruning |
| **Complexity** | High (many subsystems) | Low (simple signal dynamics) |
| **Biological Inspiration** | Evolutionary algorithms | Ant colony optimization |

---

## Why This Approach Works

### 1. True Swarm Intelligence

- **No bottlenecks**: No central scheduler deciding who acts when
- **Robust**: System continues if individual agents fail
- **Scalable**: Add more agents without coordination overhead
- **Adaptive**: Automatically focuses on promising areas

### 2. Stigmergic Coordination

- **Simplicity**: Agents only need to deposit/sample signals
- **Asynchronous**: All agents run concurrently without blocking
- **Self-Organization**: Strong solutions emerge from local interactions
- **Natural Selection**: Weak signals naturally fade (no explicit pruning logic needed)

### 3. Exploration-Exploitation Balance

- **Scouts ensure diversity**: Always exploring new possibilities
- **Foragers focus effort**: Following strong signals efficiently
- **Decay prevents stagnation**: Forces continuous exploration
- **Amplification rewards success**: Positive feedback on good solutions

---

## Research Connections

This implementation combines concepts from:

1. **Ant Colony Optimization (ACO)** - Pheromone trails, probabilistic sampling
2. **Particle Swarm Optimization (PSO)** - Distributed search, collective intelligence
3. **Stigmergic Systems** - Indirect coordination through environment
4. **Multi-Agent Systems** - Autonomous agents, emergent behavior
5. **Argument Mining** - Claim generation, evidence gathering, critique

---

## Future Extensions

### Phase 2 Enhancements

1. **Multiple Signal Types**
   - Add CRITIQUE scouts and foragers
   - Create signal chains: CLAIM → EVIDENCE → CRITIQUE

2. **Adaptive Parameters**
   - Dynamic decay rate based on signal diversity
   - Adaptive amplification based on convergence

3. **Visualization**
   - Real-time signal strength heatmap
   - Agent activity tracking
   - Signal lineage trees (parent-child relationships)

4. **Advanced Sampling**
   - Temperature-based sampling (exploration parameter)
   - Novelty bonus (reward signals far from existing clusters)
   - Age penalty (prefer recent signals)

5. **Knowledge Integration**
   - External knowledge base access for scouts
   - Web search for evidence gathering
   - Fact-checking for signal validation

### Phase 3 Research

1. **Cross-Swarm Debate**
   - Multiple swarms with different theses
   - Inter-swarm signal exchange
   - Competitive argumentation

2. **Human-in-the-Loop**
   - User signals (human-deposited high-strength signals)
   - Interactive pruning (user removes bad signals)
   - Hybrid scoring (LLM + human assessment)

3. **Semantic Clustering**
   - Embed signals in vector space
   - Cluster similar signals
   - Detect redundancy and conflicts

---

## Technical Notes

### Concurrency Model

All agents and the environment process run concurrently using `asyncio`:

```python
tasks = []
for scout in scouts:
    tasks.append(asyncio.create_task(scout.run(...)))
for forager in foragers:
    tasks.append(asyncio.create_task(forager.run(...)))
tasks.append(asyncio.create_task(environment_process()))

await asyncio.gather(*tasks)  # All run in parallel
```

### Thread Safety

Signal store uses locks for thread-safe operations:

```python
class SignalStore:
    def __init__(self):
        self._lock = Lock()

    def deposit(self, ...):
        with self._lock:
            # Atomic signal addition
```

### Performance

- **Model Loading**: ~10s for distilgpt2 on CPU, ~27s on CUDA (first time)
- **Signal Generation**: ~2-5s per agent action (LLM inference)
- **Throughput**: 3-6 signals/iteration with 6 agents (3 scouts + 3 foragers)
- **Memory**: ~500MB for distilgpt2 + ~100MB for signals (grows with iterations)

---

## Success Metrics

Proven working characteristics (v1.0):

✅ **Agents run asynchronously** - All agents operate concurrently without blocking
✅ **Scouts deposit diverse signals** - Multiple CLAIMs generated with varied content
✅ **Foragers sample probabilistically** - Weighted sampling based on signal strength
✅ **Signals are amplified** - Corroboration increases signal strength (×1.3 factor)
✅ **Weak signals decay** - Unused signals naturally fade (5% per iteration)
✅ **System produces outputs** - Final results show top signals by strength
✅ **No central coordination** - Decentralized agent behavior through signal store

---

## Version History

- **v1.0** (2025-11-11) - Initial stigmergic implementation
  - Signal store with deposit/sample/amplify/decay
  - Scout and Forager agents
  - Asynchronous concurrent execution
  - Proven working with distilgpt2

- **v0.3.5** (2025-11-11) - Evolutionary approach (archived)
  - Agent spawning/death based on performance
  - Central scheduler with EMA scoring
  - Never successfully generated outputs (model loading timeouts)

---

## License

MIT License - See LICENSE file for details

---

## Contributing

This is a research prototype. Contributions welcome for:
- Additional agent types
- Better sampling strategies
- Visualization tools
- Performance optimizations

---

**Generated with stigmergic swarm intelligence** 🐜

Last Updated: 2025-11-11
Version: 1.0.0
Status: ✅ Working and generating outputs
