# CLAUDE GUIDANCE - Comprehensive Improvement Roadmap

**Generated:** 2025-11-18
**Purpose:** Complete TODO list for improving AI Swarm Mechanics to production quality
**Based on:** Deep codebase analysis of 22,701 lines across 44 Python files

---

## Table of Contents

1. [Critical Fixes (Do First)](#critical-fixes)
2. [High Priority Improvements](#high-priority-improvements)
3. [Code Quality & Architecture](#code-quality--architecture)
4. [Performance Optimizations](#performance-optimizations)
5. [Testing & Validation](#testing--validation)
6. [Documentation & Examples](#documentation--examples)
7. [Feature Completeness](#feature-completeness)
8. [Research & Publication Path](#research--publication-path)

---

## Critical Fixes (Do First)

### 1. Fix Blocking I/O in Async Context ⚡ QUICK WIN
**Time:** 30 minutes
**Impact:** 5-10% throughput improvement, prevents event loop blocking
**Priority:** P0 - CRITICAL

**Problem:**
- `swarm/retrieval/search_engine.py:51` uses `time.sleep()` in async function
- `swarm/retrieval/web_scraper.py:63` uses `time.sleep()` in async function
- Blocks entire event loop, prevents concurrent agent execution

**Fix:**
```python
# BEFORE
import time
time.sleep(delay)

# AFTER
import asyncio
await asyncio.sleep(delay)
```

**Files to change:**
- `swarm/retrieval/search_engine.py` line 51
- `swarm/retrieval/web_scraper.py` line 63

---

### 2. Fix Unbounded Cache Memory Leak ⚡ QUICK WIN
**Time:** 15 minutes
**Impact:** Prevent unbounded memory growth
**Priority:** P0 - CRITICAL

**Problem:**
- `swarm/validation/external_sources.py:65-66` has unbounded caches
- `self._search_cache: Dict[str, Any] = {}` grows forever
- `self._scrape_cache: Dict[str, str] = {}` grows forever
- No eviction policy = memory leak

**Fix:**
```python
# BEFORE
self._search_cache: Dict[str, Any] = {}
self._scrape_cache: Dict[str, str] = {}

# AFTER
from collections import OrderedDict
self._search_cache = OrderedDict()  # Use maxsize in get/set
self._scrape_cache = OrderedDict()
self._max_cache_size = 1000  # Add eviction on deposit

# In cache set:
if len(self._search_cache) > self._max_cache_size:
    self._search_cache.popitem(last=False)  # LRU eviction
```

**Files to change:**
- `swarm/validation/external_sources.py` lines 65-66, 105-108, 147-150

---

### 3. Critic Agent Doesn't Generate Signals 🔥 MAJOR GAP
**Time:** 6-8 hours
**Impact:** Core functionality missing, prevents true evaluation workflow
**Priority:** P0 - CRITICAL

**Problem:**
Critics currently only adjust signal strength (multiply by 0.8-1.2), they don't generate CRITIQUE signals that explain WHY a signal is weak/strong.

**Current behavior** (swarm/agents/critic.py:85-150):
```python
async def evaluate_insights(self, signal_store, llm):
    targets = signal_store.sample_weighted(self.evaluate_type, n=3)
    for signal in targets:
        multiplier = self._calculate_quality_multiplier(signal.content)
        signal.strength *= multiplier  # ONLY adjusts strength!
```

**Needed behavior:**
```python
async def evaluate_insights(self, signal_store, llm):
    targets = signal_store.sample_stratified(self.evaluate_type,
                                             weak=2, medium=2, strong=2)
    for signal in targets:
        # Get full context
        evidence = signal_store.get_descendants(signal.id, "SUPPORT")
        objections = signal_store.get_descendants(signal.id, "OBJECTION")

        # Generate LLM-based critique
        critique_text = await self._generate_critique(signal, evidence, objections, llm)

        # Deposit CRITIQUE signal
        signal_store.deposit(
            signal_type="CRITIQUE",
            content=critique_text,
            parent=signal.id,
            strength=self._calculate_critique_strength(critique_text)
        )

        # Adjust parent strength based on critique
        signal.strength *= self._extract_quality_multiplier(critique_text)
```

**Implementation plan:**
1. Add `sample_stratified()` to signal_store.py (stratified by strength)
2. Add `_generate_critique()` method with provenance context
3. Modify `evaluate_insights()` to deposit CRITIQUE signals
4. Add critique quality scoring
5. Test with debate mode

**Files to change:**
- `swarm/agents/critic.py` - major refactor of lines 85-250
- `swarm/core/signal_store.py` - add sample_stratified method
- `tests/` - add critic signal generation tests

**Reference:**
See `research/COMPREHENSIVE_IMPLEMENTATION_ANALYSIS.md` lines 125-219 for detailed analysis

---

## High Priority Improvements

### 4. Remove Monkey Patching Anti-Pattern
**Time:** 2-3 hours
**Impact:** IDE navigation works, type checking works, debugging easier
**Priority:** P1 - HIGH

**Problem:**
Task-specific prompts are injected by replacing agent methods at runtime:

```python
# run_task.py:58-71
def create_scout(agent_id: str, task_config):
    scout = Scout(agent_id, signal_type=task_config.signal_types["initial"])

    # MONKEY PATCH: Replace method at runtime
    original_make_prompt = scout._make_prompt
    def custom_make_prompt(search_context=None):
        return task_config.scout_prompt_template.format(...)
    scout._make_prompt = custom_make_prompt  # BAD!
```

**Solution:** Use composition instead of mutation
```python
# AFTER (use TaskConfig injection)
class Scout:
    def __init__(self, agent_id: str, signal_type: str, prompt_config: PromptConfig):
        self.prompt_config = prompt_config  # Injected, not mutated

    def _make_prompt(self, search_context=None):
        return self.prompt_config.scout_template.format(...)  # Use config
```

**Files to change:**
- `run_task.py` lines 58-71, 85-98, 136-154, 168-176
- `swarm/agents/scout.py` - add prompt_config parameter
- `swarm/agents/forager.py` - add prompt_config parameter
- `swarm/agents/hater.py` - add prompt_config parameter
- `swarm/core/task_config.py` - create PromptConfig dataclass

**Benefits:**
- IDE can navigate to method definitions
- Type checkers work (mypy, pyright)
- Debuggers can set breakpoints properly
- Code is self-documenting

---

### 5. Refactor God Objects
**Time:** 8-10 hours
**Impact:** Testability, maintainability, code clarity
**Priority:** P1 - HIGH

**God Objects Identified:**
1. **signal_store.py** (951 lines) - 7 responsibilities
2. **hater.py** (656 lines) - 5 responsibilities
3. **external_sources.py** (818 lines) - 4 responsibilities
4. **simple_llm.py** (627 lines) - 4 responsibilities
5. **run_task.py** (1,012 lines) - orchestration + creation + synthesis

---

#### 5.1. Refactor signal_store.py (951 lines → 5 modules)

**Current responsibilities:**
1. Signal storage (dict + list operations)
2. Signal decay (strength decay over time)
3. Signal sampling (weighted, type-based, clustering)
4. Graph traversal (ancestors, descendants)
5. Event notification (asyncio.Event management)
6. Caching (ancestor/descendant cache)
7. Similarity checking (embeddings, clustering)

**Proposed split:**
```
swarm/core/
  signal_store.py (200 lines) - Core storage + events
  signal_decay.py (100 lines) - Decay logic
  signal_sampling.py (200 lines) - Weighted, stratified sampling
  signal_graph.py (200 lines) - Parent-child traversal
  signal_clustering.py (200 lines) - Embeddings, similarity
```

**Benefits:**
- Each module has single responsibility
- Easier to test in isolation
- Easier to understand and modify
- Can optimize each concern independently

---

#### 5.2. Refactor hater.py (656 lines → 4 modules)

**Current responsibilities:**
1. Target selection (consensus detection, challenge prioritization)
2. Objection generation (LLM prompting)
3. Quality verification (substantiveness scoring)
4. Dialogue coordination (incomplete)
5. Scoring and metrics

**Proposed split:**
```
swarm/agents/
  hater.py (200 lines) - Main agent loop
  hater_targeting.py (150 lines) - Consensus + priority logic
  hater_generation.py (150 lines) - LLM prompting strategies
  hater_verification.py (100 lines) - Quality scoring
```

---

#### 5.3. Refactor external_sources.py (818 lines → 4 modules)

**Current responsibilities:**
1. Wikipedia search (API calls)
2. Web search (search engine integration)
3. Web scraping (HTML parsing)
4. Rate limiting (delays, quotas)
5. Caching (multiple cache dictionaries)

**Proposed split:**
```
swarm/validation/
  external_sources.py (150 lines) - Main coordinator
  wikipedia_client.py (200 lines) - Wikipedia API
  web_search_client.py (200 lines) - Web search
  web_scraper.py (200 lines) - HTML parsing
  rate_limiter.py (100 lines) - Shared rate limiting
```

---

#### 5.4. Refactor simple_llm.py (627 lines → 4 modules)

**Current responsibilities:**
1. Model loading (transformers, quantization)
2. LRU caching (OrderedDict + async locks)
3. Generation (inference, timeouts, retries)
4. Token counting and validation

**Proposed split:**
```
swarm/llm/
  simple_llm.py (200 lines) - Main interface
  model_loader.py (200 lines) - Loading, quantization, device
  generation_cache.py (150 lines) - LRU cache + locks
  token_utils.py (100 lines) - Counting, validation
```

---

### 6. Add Semantic Caching for LLM Calls
**Time:** 6 hours
**Impact:** 20-40% fewer LLM calls, major cost savings
**Priority:** P1 - HIGH

**Problem:**
Current cache requires exact string match:
```python
# swarm/llm/simple_llm.py:207
cache_key = f"{prompt}|{temperature}|{max_tokens}"
if cache_key in self._cache:  # Exact match only!
    return self._cache[cache_key]
```

**Hit rate:** ~20-30% (only repeated prompts)

**Solution:** Semantic similarity caching
```python
# Embed prompts, find similar cached results
prompt_embedding = self.embed_prompt(prompt)
for cached_key, cached_value in self._cache.items():
    cached_embedding = cached_value["embedding"]
    similarity = cosine_similarity(prompt_embedding, cached_embedding)
    if similarity > 0.95:  # Very similar prompt
        return cached_value["result"]
```

**Hit rate:** ~50-70% (semantically similar prompts)

**Implementation:**
1. Add prompt embedding (use same model as signal clustering)
2. Store embeddings alongside cached results
3. Add similarity threshold parameter (0.90-0.98)
4. Benchmark cache hit rate before/after

**Files to change:**
- `swarm/llm/simple_llm.py` lines 200-230 (cache implementation)
- Add tests for semantic cache hits

---

## Code Quality & Architecture

### 7. Add Type Hints Throughout Codebase
**Time:** 4-6 hours
**Impact:** IDE autocomplete, type checking, self-documenting code
**Priority:** P2 - MEDIUM

**Current state:** ~60% of functions have type hints
**Target:** 95%+ coverage

**Focus areas:**
1. All agent classes (scout, forager, critic, hater, validator, pruner)
2. Signal store methods
3. LLM methods
4. Task config structures

**Tools:**
```bash
# Check current coverage
mypy swarm/ --strict

# Add missing hints incrementally
# Use typing: Optional, List, Dict, Union, Callable, etc.
```

---

### 8. Extract Configuration to Environment Variables
**Time:** 2 hours
**Impact:** Deployment flexibility, containerization-ready
**Priority:** P2 - MEDIUM

**Problem:**
All config is hardcoded in `swarm/core/config.py`:
```python
MODEL_NAME = "microsoft/phi-2"  # Can't change without editing code
LLM_CACHE_SIZE = 500
NUM_SCOUTS = 4
```

**Solution:**
```python
import os

MODEL_NAME = os.getenv("SWARM_MODEL_NAME", "microsoft/phi-2")
LLM_CACHE_SIZE = int(os.getenv("SWARM_CACHE_SIZE", "500"))
NUM_SCOUTS = int(os.getenv("SWARM_NUM_SCOUTS", "4"))
```

**Add:**
- `.env.example` file with all variables
- `python-dotenv` for loading
- Documentation of all env vars

---

### 9. Add Logging Framework
**Time:** 3-4 hours
**Impact:** Production debugging, performance monitoring
**Priority:** P2 - MEDIUM

**Problem:**
Currently using `print()` statements everywhere:
```python
print(f"[SCOUT] Generated observation: {content[:100]}")
print(f"[HATER] Deposited objection")
```

**Issues:**
- Can't control verbosity
- Can't route to files
- No structured logging
- No log levels

**Solution:**
```python
import logging

logger = logging.getLogger(__name__)
logger.info("Generated observation", extra={"agent_id": self.agent_id, "content_length": len(content)})
logger.debug("Deposited objection", extra={"parent": parent_id})
```

**Add:**
- `swarm/utils/logging_config.py` with structured logging
- Log levels: DEBUG, INFO, WARNING, ERROR
- JSON logging for production
- File + console handlers

---

### 10. Add Error Handling for External API Failures
**Time:** 2 hours
**Impact:** Robustness, graceful degradation
**Priority:** P2 - MEDIUM

**Problem:**
Wikipedia/web search failures can crash agents:
```python
# swarm/validation/external_sources.py:105
result = wikipedia.search(query)  # Can raise exceptions
```

**Solution:**
```python
try:
    result = wikipedia.search(query, results=5)
except (ConnectionError, TimeoutError, wikipedia.exceptions.WikipediaException) as e:
    logger.warning(f"Wikipedia search failed: {e}")
    return []  # Graceful degradation
```

**Add error handling for:**
- Wikipedia API failures
- Web search timeouts
- Scraping errors
- Rate limit exceeded

---

## Performance Optimizations

### 11. Batch Embedding Computation
**Time:** 4 hours
**Impact:** 2-3x faster embedding via GPU batching
**Priority:** P2 - MEDIUM

**Problem:**
Embeddings computed one at a time:
```python
# swarm/core/signal_store.py:151
new_embedding = self.embedding_model.encode(content)  # Single item
```

**Solution:**
```python
# Collect multiple signals
batch_contents = [s.content for s in signals]

# Compute in parallel on GPU
batch_embeddings = self.embedding_model.encode(batch_contents, batch_size=32)

# Assign back
for signal, embedding in zip(signals, batch_embeddings):
    self.signal_embeddings[signal.id] = embedding
```

**Benefits:**
- GPU parallelization (2-3x faster)
- Amortize transfer overhead
- Better memory coalescing

---

### 12. Optimize Pruner O(n²) Similarity Check
**Time:** 2 hours
**Impact:** 5-10% faster pruning for large signal sets
**Priority:** P3 - LOW

**Problem:**
```python
# swarm/agents/pruner.py - check all pairs for duplicates
for i, sig1 in enumerate(signals):
    for j, sig2 in enumerate(signals[i+1:]):
        if similarity(sig1, sig2) > 0.85:
            # Mark duplicate
```

**Complexity:** O(n²) where n = number of signals

**Solution:** Use approximate nearest neighbors
```python
from annoy import AnnoyIndex

# Build index of embeddings
index = AnnoyIndex(embedding_dim, 'angular')
for i, signal in enumerate(signals):
    index.add_item(i, signal_embedding)
index.build(10)

# Find duplicates via nearest neighbors
for i, signal in enumerate(signals):
    neighbors = index.get_nns_by_item(i, 5)
    for j in neighbors:
        if similarity > 0.85:
            # Mark duplicate
```

**Complexity:** O(n log n) build + O(log n) per query

---

### 13. Add Temporal Filtering to Similarity Checks
**Time:** 1 hour
**Impact:** 5-10% faster deposit operations
**Priority:** P3 - LOW

**Problem:**
Similarity check compares against ALL signals of same type:
```python
same_type = [s for s in self.signals.values() if s.type == signal_type]
# Could be 100+ signals, most are old and irrelevant
```

**Solution:**
```python
# Only check recent signals (last 5 minutes)
recent_cutoff = time.time() - 300
same_type = [s for s in self.signals.values()
             if s.type == signal_type and s.timestamp > recent_cutoff]
```

**Rationale:**
- Duplicates are typically deposited close in time
- Old signals likely already decayed/pruned
- 80% reduction in comparisons

---

## Testing & Validation

### 14. Add Integration Tests for Full Round Execution
**Time:** 6-8 hours
**Impact:** Empirical validation of swarm mechanics
**Priority:** P1 - HIGH

**Current testing:**
- 10 unit tests for signal mechanics (no LLM)
- 556 sanity tests (basic checks)
- No integration tests

**Needed tests:**

#### Test 1: Full Single Round
```python
def test_full_round_execution():
    """Test that a complete round executes successfully."""
    # Setup
    store = SignalStore()
    llm = MockLLM()  # Controlled responses
    scouts, foragers, critics, haters = create_agents()

    # Execute round
    await run_single_round(scouts, foragers, critics, haters, store, llm)

    # Assertions
    assert len(store.get_signals("INITIAL")) >= 4  # Scouts deposited
    assert len(store.get_signals("SUPPORT")) >= 2  # Foragers elaborated
    assert len(store.get_signals("OBJECTION")) >= 1  # Haters challenged

    # Check provenance
    for support in store.get_signals("SUPPORT"):
        assert support.parent is not None  # Linked to INITIAL
```

#### Test 2: Multi-Round Quality Improvement
```python
def test_multi_round_quality_improvement():
    """Test that signal quality improves across rounds."""
    rounds_data = []

    for round_num in range(3):
        # Run round
        await run_round(...)

        # Measure quality metrics
        avg_strength = calculate_avg_strength(store)
        diversity = calculate_diversity(store)
        objection_rate = len(objections) / len(claims)

        rounds_data.append({
            "round": round_num,
            "avg_strength": avg_strength,
            "diversity": diversity,
            "objection_rate": objection_rate
        })

    # Quality should improve or stabilize
    assert rounds_data[2]["avg_strength"] >= rounds_data[0]["avg_strength"]
    assert rounds_data[2]["diversity"] >= 0.3  # Maintained diversity
```

#### Test 3: Synthesis Quality
```python
def test_synthesis_incorporates_evidence():
    """Test that synthesis includes supporting evidence."""
    # Setup signals with known content
    claim = store.deposit("INITIAL", "Climate action reduces emissions")
    evidence1 = store.deposit("SUPPORT", "Study shows 30% reduction", parent=claim)
    evidence2 = store.deposit("SUPPORT", "Policy X decreased CO2", parent=claim)

    # Generate synthesis
    synthesis = await synthesizer.synthesize(store, llm)

    # Assertions
    assert "30% reduction" in synthesis or "Study" in synthesis
    assert "Policy X" in synthesis or "CO2" in synthesis
    assert len(synthesis) >= 200  # Substantive
```

---

### 15. Add Benchmarking Against RAG Baselines
**Time:** 8-12 hours
**Impact:** Quantify swarm advantages with real data
**Priority:** P1 - HIGH (for publication)

**Benchmark tasks:**
1. TruthfulQA subset (hallucination detection)
2. MMLU subset (multi-turn reasoning)
3. Document QA (RAG comparison)

**Metrics:**
- Accuracy vs single-LLM baseline
- Diversity of responses
- Provenance completeness
- Adversarial validation rate

**Implementation:**
```python
# tests/benchmarks/truthfulqa_test.py
def test_swarm_vs_baseline_truthfulness():
    """Compare swarm to single-LLM on TruthfulQA questions."""
    questions = load_truthfulqa_sample(n=50)

    # Baseline: Single LLM
    baseline_answers = [llm.generate(q) for q in questions]
    baseline_score = evaluate_truthfulness(baseline_answers)

    # Swarm: Full multi-agent run
    swarm_answers = [run_swarm_task("analysis", q) for q in questions]
    swarm_score = evaluate_truthfulness(swarm_answers)

    # Swarm should be more truthful (haters challenge hallucinations)
    assert swarm_score > baseline_score

    # Log results for paper
    log_benchmark_results("truthfulqa", baseline_score, swarm_score)
```

---

### 16. Add Property-Based Testing
**Time:** 4 hours
**Impact:** Find edge cases automatically
**Priority:** P2 - MEDIUM

**Use hypothesis library:**
```python
from hypothesis import given, strategies as st

@given(st.floats(min_value=0.0, max_value=1.0))
def test_signal_strength_bounds(initial_strength):
    """Signal strength should stay in [0, 1] after decay."""
    signal = Signal("test", "INITIAL", "content", strength=initial_strength)

    # Apply decay multiple times
    for _ in range(100):
        signal.strength *= 0.99

    assert 0.0 <= signal.strength <= 1.0

@given(st.lists(st.text(min_size=1, max_size=100), min_size=1, max_size=20))
def test_clustering_handles_all_inputs(contents):
    """Clustering should handle any valid content list."""
    store = SignalStore(use_semantic_clustering=True)

    # Deposit all signals
    for content in contents:
        store.deposit("INITIAL", content)

    # Should not crash, should return valid clusters
    clusters = store.get_clusters("INITIAL")
    assert isinstance(clusters, list)
```

---

## Documentation & Examples

### 17. Create Architecture Diagrams
**Time:** 3-4 hours
**Impact:** Clarity for contributors
**Priority:** P2 - MEDIUM

**Diagrams needed:**

1. **Signal Flow Diagram**
   - Scout → INITIAL signal
   - Forager → SUPPORT/CRITIQUE signals (parent: INITIAL)
   - Hater → OBJECTION signals (parent: INITIAL)
   - Critic → Adjusts strength
   - Synthesizer → SYNTHESIS (aggregates all)

2. **Agent Interaction Diagram**
   - Event-driven coordination
   - Signal deposit triggers events
   - Agents wait for specific signal types

3. **System Architecture Diagram**
   - Core (signal_store, config, task_config)
   - Agents (scout, forager, critic, hater, validator, pruner, synthesizer)
   - LLM (simple_llm, factory, providers)
   - Retrieval (advanced_retriever, search, scrape)
   - Validation (external_sources, verification)

**Tools:** Mermaid (in markdown), diagrams.net, or PlantUML

---

### 18. Add Custom Task Configuration Guide
**Time:** 2 hours
**Impact:** Enable users to create new task types
**Priority:** P2 - MEDIUM

**Current gap:**
Users don't know how to create custom tasks beyond the 4 built-in types (debate, creative, analysis, problem_solving).

**Guide should include:**
1. TaskConfig dataclass structure
2. Signal type mapping (structural → display names)
3. Prompt template variables
4. Example: Creating a "code_review" task type
5. Testing custom task configs

**Location:** `docs/CUSTOM_TASK_GUIDE.md`

---

### 19. Add Real-World Examples
**Time:** 6 hours
**Impact:** Demonstrates capabilities, helps users get started
**Priority:** P2 - MEDIUM

**Examples to create:**

1. **Research Paper Analysis**
   ```python
   # examples/research_paper_analysis.py
   task = "Analyze the key contributions of 'Attention Is All You Need'"
   result = run_swarm_task("analysis", task, knowledge_docs=["transformer_paper.pdf"])
   ```

2. **Code Review Debate**
   ```python
   # examples/code_review.py
   code = load_code("pull_request_123.py")
   task = f"Review this code for bugs, performance issues, and style:\n{code}"
   result = run_swarm_task("analysis", task)
   ```

3. **Product Requirements Analysis**
   ```python
   # examples/requirements_analysis.py
   prd = load_document("product_requirements.md")
   task = f"Identify gaps and conflicts in these requirements:\n{prd}"
   result = run_swarm_task("analysis", task)
   ```

---

### 20. Document Signal Flow with Examples
**Time:** 2 hours
**Impact:** Helps users understand stigmergic coordination
**Priority:** P2 - MEDIUM

**Create:** `docs/SIGNAL_FLOW_GUIDE.md`

**Content:**
- What is stigmergic communication?
- Signal types and their purposes
- Parent-child relationships
- Example signal chains with actual content
- How strength affects selection
- Provenance tracking examples

---

## Feature Completeness

### 21. Complete or Remove Dialogue Coordinator
**Time:** 6 hours (complete) or 30 minutes (remove)
**Impact:** Clarity on feature scope
**Priority:** P2 - MEDIUM

**Current state:**
- `swarm/core/dialogue_coordinator.py` exists
- Minimal implementation
- Referenced in hater.py but not fully integrated

**Options:**

**Option A: Complete Implementation**
- Add structured dialogue turns between haters and foragers
- Track dialogue state (who said what, when)
- Add dialogue quality metrics
- Test dialogue improves signal quality

**Option B: Remove**
- Delete `dialogue_coordinator.py`
- Remove references from hater.py
- Update docs to remove dialogue feature

**Recommendation:** Remove for now (not core to stigmergic model), add later if needed

---

### 22. Complete or Archive SimpleScout + SpatialSignalStore
**Time:** 8 hours (complete) or 1 hour (archive)
**Impact:** Reduce complexity or validate experimental features
**Priority:** P3 - LOW

**Current state:**
- Both fully implemented (963 combined lines)
- Behind feature flags (USE_SIMPLE_SCOUTS, USE_SPATIAL_STORE)
- Not tested against standard agents
- Unclear if spatial locality improves quality

**Options:**

**Option A: Benchmark and Document**
- Run controlled experiments: SimpleScout vs Scout
- Measure quality differences
- Document when to use spatial features
- Add to publication if beneficial

**Option B: Archive**
- Move to `archive/experimental/`
- Remove from main codebase
- Reduce maintenance burden
- Simplify system for users

**Recommendation:** Archive unless there's research value

---

### 23. Add Self-Healing Coordinator or Remove
**Time:** 8 hours (complete) or 30 minutes (remove)
**Impact:** Adaptive agent population or simpler system
**Priority:** P3 - LOW

**Current state:**
- `swarm/core/self_healing.py` partially implemented
- Can spawn agents dynamically
- Can detect problems (low objection rate, echo chambers)
- Not fully tested with async agent lifecycle

**Options:**

**Option A: Complete**
- Test dynamic agent spawning
- Add agent retirement (remove idle agents)
- Add metrics for when to spawn/retire
- Validate self-healing improves outcomes

**Option B: Remove**
- Delete self_healing.py
- Use fixed agent populations
- Simpler, more predictable

**Recommendation:** Remove for now, add later as advanced feature

---

### 24. Add Stratified Sampling to Signal Store
**Time:** 2 hours
**Impact:** Enables critic to sample weak/medium/strong signals
**Priority:** P1 - HIGH (needed for critic fix)

**Current:**
Only `sample_weighted()` which biases toward strong signals

**Needed:**
```python
def sample_stratified(
    self,
    signal_type: str,
    weak: int = 0,
    medium: int = 0,
    strong: int = 0
) -> List[Signal]:
    """Sample signals stratified by strength.

    Args:
        signal_type: Type of signals to sample
        weak: Number of weak signals (strength < 0.4)
        medium: Number of medium signals (0.4 <= strength < 0.7)
        strong: Number of strong signals (strength >= 0.7)

    Returns:
        List of sampled signals from each stratum
    """
    signals = self.get_signals(signal_type)

    weak_signals = [s for s in signals if s.strength < 0.4]
    medium_signals = [s for s in signals if 0.4 <= s.strength < 0.7]
    strong_signals = [s for s in signals if s.strength >= 0.7]

    result = []
    result.extend(random.sample(weak_signals, min(weak, len(weak_signals))))
    result.extend(random.sample(medium_signals, min(medium, len(medium_signals))))
    result.extend(random.sample(strong_signals, min(strong, len(strong_signals))))

    return result
```

**Location:** `swarm/core/signal_store.py`

---

## Research & Publication Path

### 25. Empirical Evaluation for Publication
**Time:** 12-16 hours
**Impact:** Publication-ready results
**Priority:** P1 - HIGH (if pursuing publication)

**Evaluation plan:**

#### Experiment 1: Swarm vs Single-LLM Baseline
- **Task:** 50 questions from TruthfulQA, MMLU, document QA
- **Baselines:** GPT-3.5, GPT-4 (via API), Phi-2 (same model as swarm)
- **Metrics:** Accuracy, truthfulness, hallucination rate
- **Hypothesis:** Swarm with adversarial validation > single-LLM

#### Experiment 2: Ablation Studies
Test contribution of each agent type:
- **No haters:** How much does accuracy drop?
- **No critics:** How does quality suffer?
- **No validators:** Hallucination rate increase?
- **Hypothesis:** Each agent type contributes measurably

#### Experiment 3: Scalability
- **Agents:** 2, 4, 8, 16 scouts (vary agent count)
- **Metrics:** Quality vs runtime tradeoff
- **Hypothesis:** Quality plateaus at ~8-12 agents

#### Experiment 4: Real Document Corpus
- **Dataset:** Scientific papers, news articles, technical docs
- **Task:** Extract insights, verify claims, generate summaries
- **Metrics:** Human evaluation of quality
- **Hypothesis:** Swarm outperforms RAG baselines

---

### 26. Write Research Paper
**Time:** 20-30 hours
**Impact:** Publication in ACL/EMNLP/NeurIPS or JAIR
**Priority:** P1 - HIGH (if pursuing publication)

**Paper outline:**

1. **Abstract** (200 words)
   - Problem: LLM hallucinations, lack of provenance
   - Solution: Stigmergic multi-agent swarm
   - Results: X% improvement over baselines

2. **Introduction** (1-1.5 pages)
   - Motivation: Need for verifiable, multi-perspective AI
   - Contributions: Event-driven stigmergy, adversarial validation, provenance

3. **Related Work** (1-1.5 pages)
   - Multi-agent LLM systems (AutoGPT, BabyAGI, etc.)
   - Retrieval-augmented generation (RAG)
   - Adversarial training and red teaming
   - Stigmergic coordination in nature

4. **Methodology** (2-3 pages)
   - Architecture overview
   - Agent types and roles
   - Signal flow and stigmergic coordination
   - Event-driven execution model

5. **Experiments** (2-3 pages)
   - Datasets and baselines
   - Evaluation metrics
   - Ablation studies
   - Scalability analysis

6. **Results** (1-2 pages)
   - Tables and figures
   - Statistical significance tests
   - Error analysis

7. **Discussion** (1 page)
   - Why swarm works
   - Limitations
   - Future work

8. **Conclusion** (0.5 pages)

**Target venues:**
- ACL/EMNLP (NLP focus)
- NeurIPS (multi-agent systems track)
- JAIR (journal, longer format)

---

## Priority Matrix

### Must Do Before v1.0 Release
- [x] Fix blocking I/O (30 min)
- [x] Fix unbounded caches (15 min)
- [ ] Critic signal generation (6-8 hours)
- [ ] Remove monkey patching (2-3 hours)
- [ ] Add integration tests (6-8 hours)

### Should Do for Production Quality
- [ ] Refactor god objects (8-10 hours)
- [ ] Semantic caching (6 hours)
- [ ] Type hints (4-6 hours)
- [ ] Environment variables (2 hours)
- [ ] Logging framework (3-4 hours)
- [ ] Error handling (2 hours)

### Nice to Have for Polish
- [ ] Architecture diagrams (3-4 hours)
- [ ] Custom task guide (2 hours)
- [ ] Real-world examples (6 hours)
- [ ] Signal flow guide (2 hours)
- [ ] Performance optimizations (7 hours total)

### For Research/Publication
- [ ] Empirical benchmarking (12-16 hours)
- [ ] Ablation studies (8 hours)
- [ ] Paper writing (20-30 hours)

---

## Total Effort Estimates

**Critical Fixes:** 7-9 hours
**High Priority:** 24-31 hours
**Code Quality:** 23-30 hours
**Performance:** 7 hours
**Testing:** 18-24 hours
**Documentation:** 13-14 hours
**Features:** 16-24 hours
**Research:** 32-46 hours

**Total to Production Quality:** ~80-100 hours
**Total to Publication:** ~130-170 hours

---

## Implementation Order Recommendation

### Phase 1: Critical Fixes (1 day)
1. Fix blocking I/O (30 min)
2. Fix unbounded caches (15 min)
3. Add stratified sampling (2 hours)
4. Critic signal generation (6-8 hours)

### Phase 2: Code Quality (1 week)
5. Remove monkey patching (2-3 hours)
6. Add type hints (4-6 hours)
7. Logging framework (3-4 hours)
8. Environment variables (2 hours)
9. Error handling (2 hours)

### Phase 3: Testing & Validation (1 week)
10. Integration tests (6-8 hours)
11. Property-based tests (4 hours)
12. Benchmarking setup (8-12 hours)

### Phase 4: Architecture (2 weeks)
13. Refactor signal_store (8-10 hours)
14. Refactor hater (4-6 hours)
15. Refactor external_sources (4-6 hours)
16. Refactor simple_llm (4-6 hours)

### Phase 5: Performance (3 days)
17. Semantic caching (6 hours)
18. Batch embeddings (4 hours)
19. Optimize pruner (2 hours)
20. Temporal filtering (1 hour)

### Phase 6: Documentation (1 week)
21. Architecture diagrams (3-4 hours)
22. Custom task guide (2 hours)
23. Real-world examples (6 hours)
24. Signal flow guide (2 hours)

### Phase 7: Research (3-4 weeks)
25. Empirical evaluation (12-16 hours)
26. Paper writing (20-30 hours)

---

## Success Metrics

### Code Quality Metrics
- [ ] Type hint coverage > 95%
- [ ] No pylint warnings
- [ ] All tests passing (unit + integration)
- [ ] Code coverage > 80%

### Performance Metrics
- [ ] Event loop never blocked (no time.sleep in async)
- [ ] LLM cache hit rate > 50%
- [ ] Average round time < 3 minutes
- [ ] Memory usage < 8GB for standard config

### Research Metrics
- [ ] Swarm accuracy > single-LLM baseline on TruthfulQA
- [ ] Adversarial validation reduces hallucinations by >20%
- [ ] Provenance tracking enables 100% claim verification
- [ ] System scales linearly with agent count (up to 16 agents)

---

## Getting Started

To start working through this roadmap:

1. **Review this document** - Understand all items
2. **Choose a phase** - Start with Phase 1 (critical fixes)
3. **Track progress** - Use TodoWrite tool or GitHub issues
4. **Update docs** - Keep SESSION_REFACTORING_LOG.md updated
5. **Commit frequently** - Small, focused commits with clear messages

**Questions or need clarification on any item?** Review the detailed analysis in:
- `COMPREHENSIVE_CODEBASE_ANALYSIS.md` (deep dive into code)
- `ANALYSIS_SUMMARY.txt` (quick reference)
- `research/CORRECTED_FINDINGS.md` (known issues)
- `SESSION_REFACTORING_LOG.md` (recent improvements)

---

**Last Updated:** 2025-11-18
**Maintainer:** AI Swarm Mechanics Development Team
**Total Items:** 26 major improvements across 7 categories
