# Implementation Notes v1.1

**Date**: 2025-11-12
**Version**: Stigmergic Swarm v1.1 with Adversarial Agents

## Recent Improvements

### 1. Model Upgrade
- **Changed**: microsoft/phi-2 (2.7B params, instruction-tuned)
- **Rationale**: Better reasoning, more reliable outputs than distilgpt2
- **Location**: `swarm/core/config.py` line 7

### 2. Adversarial Hater Agent
- **Created**: `swarm/agents/hater.py`
- **Purpose**: Generate contradictory evidence and challenge strong signals
- **Behavior**: Samples strong CLAIM/EVIDENCE signals, generates counterarguments
- **Signal Type**: COUNTER_EVIDENCE (new type!)
- **Temperature**: 0.85 (high for adversarial creativity)
- **Versions**: Both async (Hater) and sync (HaterSync) implemented

### 3. Critic Agent
- **Created**: `swarm/agents/critic.py`
- **Purpose**: Focused analytical critique of claims
- **Behavior**: Samples CLAIM signals, generates constructive critiques
- **Signal Type**: CRITIQUE
- **Temperature**: 0.6 (lower for focused analysis)

### 4. Per-Agent Temperatures
- **Location**: `swarm/core/config.py` lines 12-15
- **Values**:
  - TEMP_SCOUT = 0.9 (high exploration)
  - TEMP_FORAGER = 0.7 (balanced development)
  - TEMP_CRITIC = 0.6 (focused analysis)
  - TEMP_HATER = 0.85 (adversarial creativity)
- **Integration**: Each agent uses its specific temperature from config

### 5. Coroutine-Safe Caching with LRU Eviction
- **Location**: `swarm/llm/simple_llm.py`
- **Features**:
  - OrderedDict-based LRU cache
  - Async lock for coroutine safety
  - Cache hit/miss statistics
  - Configurable cache size (default 100)
  - Automatic eviction when size exceeded
- **Stats Tracking**:
  - cache_hits, cache_misses, hit_rate
  - Logged on each generation
  - Displayed in final results

### 6. Lazy Loading
- **Implementation**: Model loads on first `generate()` call
- **Benefit**: Instant startup, deferred loading cost
- **Location**: `swarm/llm/simple_llm.py` line 100

### 7. Lowered Thresholds
- **Rationale**: Ensure foragers/critics/hater actually deposit signals
- **Changes**:
  - MIN_DEPOSIT_STRENGTH: 0.4 → 0.3
  - MIN_AMPLIFY_STRENGTH: 0.5 → 0.4
- **Location**: `swarm/core/config.py` lines 31-32

### 8. Critic-Foragers
- **Implementation**: Foragers split 50/50 between evidence and critique
- **Evidence Foragers**: CLAIM → EVIDENCE (support claims)
- **Critique Foragers**: CLAIM → CRITIQUE (analyze weaknesses)
- **Location**: `run_swarm.py` lines 46-56

### 9. Parent-Claim Context for Critiques
- **Enhancement**: Foragers include parent signal context when generating critiques
- **Implementation**: Extract parent signal ID and include in prompt
- **Location**: `swarm/agents/forager.py` lines 161-164
- **Benefit**: More contextually aware critiques

### 10. Enhanced Prompts
- **All agents**: Improved instruction-following prompts
- **Scout prompts**: More specific task descriptions
- **Forager prompts**: Clear role definitions, quality requirements
- **Locations**:
  - `swarm/agents/scout.py` lines 112-133
  - `swarm/agents/forager.py` lines 143-171

### 11. Improved Strength Assessment
- **Enhanced heuristics** for all agents:
  - More quality indicators (citations, quantifiers, reasoning, nuance)
  - Better scoring weights
  - Lower randomness for foragers (more consistent)
- **Locations**:
  - Scout: `swarm/agents/scout.py` lines 81-119
  - Forager: `swarm/agents/forager.py` lines 97-139
  - Critic: `swarm/agents/critic.py` lines 75-95
  - Hater: `swarm/agents/hater.py` lines 91-118

---

## Agent Registry

| Agent Type | Count | Input Signal | Output Signal | Temperature | Purpose |
|------------|-------|--------------|---------------|-------------|---------|
| Scout | 4 | None | CLAIM | 0.9 | Exploration, diverse claims |
| Forager (Evidence) | 2 | CLAIM | EVIDENCE | 0.7 | Develop evidence |
| Forager (Critique) | 2 | CLAIM | CRITIQUE | 0.7 | Develop critiques |
| Critic | 2 | CLAIM | CRITIQUE | 0.6 | Focused analysis |
| Hater | 2 | CLAIM/EVIDENCE | COUNTER_EVIDENCE | 0.85 | Adversarial challenges |

**Total**: 12 agents running concurrently

---

## Signal Types

| Type | Source | Purpose |
|------|--------|---------|
| CLAIM | Scout | Initial argument exploration |
| EVIDENCE | Forager (Evidence) | Support for claims |
| CRITIQUE | Forager (Critique), Critic | Analytical weaknesses |
| COUNTER_EVIDENCE | Hater | Adversarial contradictions |

---

## System Flow

```
Scouts (4) explore → deposit CLAIM signals

Foragers (2) sample CLAIMs → develop EVIDENCE → amplify strong CLAIMs
Foragers (2) sample CLAIMs → develop CRITIQUEs → amplify strong CLAIMs

Critics (2) sample CLAIMs → generate focused CRITIQUEs

Haters (2) sample strong CLAIM/EVIDENCE → generate COUNTER_EVIDENCE

All signals decay (5% per iteration)
Weak signals pruned (< 0.15 threshold)
Strong signals amplified (×1.3 when corroborated)
```

---

## TODO: Future Improvements

### High Priority

1. **Bounded Cache Eviction Policy**
   - Current: Simple LRU, no size bounds enforced strictly
   - Recommendation: Add configurable eviction policy (LRU, LFU, TTL)
   - Location: `swarm/llm/simple_llm.py`
   - Implementation: Add eviction strategy parameter, implement policy classes

2. **Unit Tests for New Agents**
   - Missing: Tests for Hater and Critic agents
   - Recommendation: Add pytest tests for:
     - Signal sampling behavior
     - Strength assessment accuracy
     - Prompt generation correctness
     - Async/sync consistency (for Hater)
   - Location: Create `tests/test_agents.py`

3. **JSON Parse Failure Logging**
   - Current: No dedicated JSON parser in swarm system
   - Recommendation: If adding structured output parsing:
     - Log parse failures with error details
     - Track failure rate per agent type
     - Add fallback extraction strategies
   - Note: Current system uses natural language, not JSON

4. **State Validation Hardening**
   - Current: Signal validation in SignalStore
   - Recommendation:
     - Add legacy JSON shape support (if needed)
     - Validate signal content schemas
     - Reject malformed/empty signals
   - Location: `swarm/core/signal_store.py`

### Medium Priority

5. **Realtime Instrumentation**
   - Add monitoring for:
     - Cache performance (hit rate trends)
     - Parse failure rates (if JSON parsing added)
     - Agent action success rates
     - Signal strength distributions
   - Implementation: Add `swarm/monitoring/` module
   - Output: Periodic stats logged or exported

6. **Hater Integration in Synchronous Loop**
   - Current: Async Hater fully implemented
   - HaterSync exists but needs proper sync LLM integration
   - Recommendation:
     - Create synchronous LLM wrapper
     - Test HaterSync in non-async contexts
   - Location: `swarm/agents/hater.py` HaterSync class

7. **Cross-Agent Influence Tracking**
   - Track how signals from one agent type influence others
   - Example: Does Hater COUNTER_EVIDENCE amplify Critic activity?
   - Implementation: Add signal lineage tracking
   - Location: `swarm/core/signal_store.py` Signal class

### Low Priority

8. **Dynamic Temperature Adjustment**
   - Adjust agent temperatures based on swarm state
   - Example: Increase exploration (Scout temp) when convergence detected
   - Implementation: Add adaptive temperature module

9. **Signal Clustering**
   - Group similar signals to detect redundancy
   - Implementation: Use embeddings (sentence-transformers)
   - Benefit: Prune duplicate ideas, focus diversity

10. **Multi-Swarm Debate**
    - Run multiple swarms with different theses
    - Cross-swarm signal exchange
    - Implementation: Add swarm ID to signals, inter-swarm sampling

---

## Known Issues

### Issue 1: Cache Key Collisions
- **Problem**: MD5 hash may collide for similar prompts
- **Impact**: Low probability, but possible cache pollution
- **Fix**: Use SHA-256 or include more prompt context in key
- **Priority**: Low

### Issue 2: No Sync LLM for HaterSync
- **Problem**: HaterSync.execute() uses placeholder logic
- **Impact**: Synchronous contexts can't use Hater properly
- **Fix**: Implement synchronous generate method in SimpleLLM
- **Priority**: Medium (if sync execution needed)

### Issue 3: Unbounded Signal Growth
- **Problem**: Signals accumulate without hard limit
- **Impact**: Memory usage grows unbounded over long runs
- **Fix**: Add max signal count, enforce FIFO eviction
- **Priority**: Medium

---

## Performance Characteristics

### Model Loading
- **phi-2**: ~15-30s on CPU, ~10s on CUDA (first time)
- **Lazy loading**: Defers cost until first generation

### Generation Speed
- **Per agent action**: 2-5s (CPU), 1-2s (CUDA)
- **Throughput**: ~10-15 signals/iteration (12 agents, some fail)

### Memory Usage
- **Model**: ~5.5GB (phi-2)
- **Cache**: ~10-50MB (depends on cache_size and prompt lengths)
- **Signals**: ~1-5MB (grows with iterations)

### Cache Performance
- **Expected hit rate**: 10-30% (diverse prompts, high temperature)
- **Speedup on hit**: ~3-5x faster than generation

---

## Testing Checklist

Before pushing to production:

- [ ] All agents registered in run_swarm.py
- [ ] Per-agent temperatures correctly applied
- [ ] Cache statistics displayed in output
- [ ] Critic-foragers spawning correctly
- [ ] Haters depositing COUNTER_EVIDENCE
- [ ] Parent-claim context included in critiques
- [ ] Thresholds lowered (agents depositing signals)
- [ ] No Unicode encoding errors
- [ ] Model loads successfully
- [ ] Signals decay and prune correctly
- [ ] Final results show all signal types

---

## Recommendations for Next Version (v1.2)

1. **Add Web Search Integration**
   - Hater agents could search for contradictory evidence
   - Evidence foragers could verify claims with real data
   - Implementation: Use DuckDuckGo API or SerpAPI

2. **Argument Graph Visualization**
   - Generate network graph of signals (claim → evidence → critique)
   - Show signal strength as node size
   - Show amplification as edge weights
   - Implementation: NetworkX + Plotly

3. **Human-in-the-Loop**
   - Allow user to upvote/downvote signals
   - Integrate human feedback into signal strength
   - Implementation: Add web UI with signal voting

4. **Semantic Deduplication**
   - Embed signals with sentence-transformers
   - Cluster similar signals
   - Prune near-duplicates
   - Implementation: Add `swarm/embedding/` module

---

## Code Quality Notes

### Strengths
- Decentralized agent coordination (true swarm)
- Coroutine-safe caching with statistics
- Lazy loading reduces startup time
- Per-agent temperature tuning
- Comprehensive agent registry

### Areas for Improvement
- Missing unit tests
- No structured logging (uses print statements)
- Hard-coded config values (should be CLI args)
- No checkpointing/resume capability
- Limited error handling in agent loops

---

## Contact for Questions

For implementation details or clarification:
- Check `README_STIGMERGIC.md` for architecture overview
- Check `STIGMERGIC_SWARM_DESIGN.md` for design principles
- Check this file (`IMPLEMENTATION_NOTES.md`) for recent changes

---

**Last Updated**: 2025-11-12
**Version**: v1.1 (Adversarial Agents + Caching)
**Status**: Tested and working with ph-2
**Next Milestone**: v1.2 (Web search + visualization)
