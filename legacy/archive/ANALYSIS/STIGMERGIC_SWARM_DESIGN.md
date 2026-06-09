# Stigmergic Swarm Architecture v1.0

## Core Swarm Mechanics

**Stigmergy**: Indirect coordination through environmental modifications. Agents deposit signals ("pheromones") that guide other agents' behavior.

### Key Principles

1. **Decentralized Coordination** - No central scheduler, agents act independently
2. **Indirect Interaction** - Agents communicate through shared signal store
3. **Signal Deposition** - Scouts explore and deposit graded signals
4. **Signal Sampling** - Foragers sample signals probabilistically
5. **Amplification** - Corroboration strengthens signals (positive feedback)
6. **Decay** - Unused signals fade over time (evaporation)
7. **Exploration-Exploitation** - Randomness vs following strong signals

---

## System Architecture

```
                    SHARED SIGNAL STORE
                    (Pheromone Environment)
                           |
        +------------------+------------------+
        |                  |                  |
    SCOUT               SCOUT             FORAGER
   (explore)           (explore)          (follow)
        |                  |                  |
        +--- deposit ------+                  |
        |    signals       |                  |
        |                  |                  |
        +------ sample <---+------------------+
                 (probabilistic)
```

### Components

#### 1. Signal Store
- Thread-safe shared medium
- Stores signals with: {id, type, content, strength, timestamp}
- Operations:
  - `deposit_signal(signal)` - Add new signal
  - `sample_signals(type, n)` - Get n signals weighted by strength
  - `amplify_signal(id)` - Increase signal strength (corroboration)
  - `decay_all(rate)` - Reduce all signal strengths over time
  - `prune_weak(threshold)` - Remove signals below threshold

#### 2. Scout Agents
- Explore solution space randomly or with weak bias
- Generate diverse possibilities
- Deposit signals with initial strength based on self-assessment
- Types:
  - `ArgumentScout` - Explores claim directions
  - `EvidenceScout` - Explores support angles
  - `CriticScout` - Explores challenge angles

#### 3. Forager Agents
- Sample signals probabilistically (weighted by strength)
- Follow strong signals to develop ideas
- Amplify signals they corroborate (positive feedback)
- Deposit refined signals back
- Types:
  - `ArgumentForager` - Develops claims
  - `EvidenceForager` - Builds supporting evidence
  - `CriticForager` - Develops critiques

#### 4. Signal Decay
- Automatic evaporation: `strength *= (1 - decay_rate)` each cycle
- Removes signals below threshold
- Ensures only actively reinforced signals persist

---

## Agent Behavior

### Scout Behavior Loop
```python
while active:
    # Explore randomly or with weak bias
    idea = explore_random()

    # Assess quality
    strength = self_assess(idea)

    # Deposit signal if promising
    if strength > threshold:
        signal_store.deposit({
            "type": self.signal_type,
            "content": idea,
            "strength": strength,
            "depositor": self.id
        })

    # Small random wait
    await asyncio.sleep(random.uniform(0.1, 0.5))
```

### Forager Behavior Loop
```python
while active:
    # Sample signals probabilistically (weighted by strength)
    signal = signal_store.sample_weighted(self.signal_type)

    if signal:
        # Develop the signal's idea
        developed = develop(signal.content)

        # Assess result
        strength = assess(developed)

        # Amplify original if good
        if strength > threshold:
            signal_store.amplify(signal.id, factor=1.2)

            # Deposit refined signal
            signal_store.deposit({
                "type": self.output_type,
                "content": developed,
                "strength": strength,
                "parent": signal.id
            })

    await asyncio.sleep(random.uniform(0.1, 0.5))
```

### System Loop
```python
# Decay signals every cycle
signal_store.decay_all(decay_rate=0.05)

# Prune weak signals
signal_store.prune_weak(threshold=0.1)

# All agents run concurrently, asynchronously
# No central scheduling, no turn-taking
```

---

## Debate Application

### Signal Types

1. **CLAIM_DIRECTION** - Potential argument angles
   - Scouts explore diverse claims
   - Foragers develop strong directions into full claims

2. **EVIDENCE_LEAD** - Potential evidence sources
   - Scouts suggest evidence angles
   - Foragers build detailed evidence

3. **CRITIQUE_ANGLE** - Potential challenges
   - Scouts identify weaknesses
   - Foragers develop critiques

### Convergence

**Natural emergent convergence:**
- Strong signals attract more foragers
- More corroboration → stronger signals → more attention
- Weak signals decay and disappear
- System naturally focuses on promising areas

**No explicit termination needed:**
- Run for fixed time or iterations
- Or until signal diversity drops below threshold
- Or until top signals stabilize

---

## Implementation Simplifications (v1.0)

### Minimal Working System

1. **Single LLM** - Share one distilgpt2 instance
2. **Simple Signals** - Just {id, type, text, strength, timestamp}
3. **Two Agent Types** - Scouts (explore) and Foragers (follow)
4. **Basic Decay** - Linear decay each iteration
5. **Output Collection** - Best signals at end are results

### File Structure

```
swarm/
├── core/
│   ├── signal_store.py      # Pheromone environment
│   └── config.py             # Simple config
├── agents/
│   ├── scout.py              # Exploration agents
│   └── forager.py            # Following agents
├── llm/
│   └── simple_llm.py         # Minimal LLM wrapper
└── main_swarm.py             # Entry point
```

---

## Key Differences from Evolutionary Approach

| Aspect | Evolutionary (v0.3.5) | Stigmergic (v1.0) |
|--------|----------------------|-------------------|
| Coordination | Central scheduler | Decentralized signals |
| Communication | Direct (state manager) | Indirect (signal store) |
| Selection | Performance scores + spawn/death | Probabilistic signal sampling |
| Feedback | Agent scores (EMA) | Signal strength amplification |
| Convergence | Termination conditions | Natural signal concentration |
| Complexity | High (lifecycle, scoring, scheduling) | Low (signals + simple agents) |

---

## Success Criteria

1. Agents run asynchronously without central coordination
2. Scouts deposit diverse signals
3. Foragers sample and amplify strong signals
4. Weak signals decay and disappear
5. System produces focused outputs (top N signals)
6. **Actually generates results** (unlike v0.3.5)

---

Version: 1.0.0
Date: 2025-11-11
Status: Design ready for implementation
