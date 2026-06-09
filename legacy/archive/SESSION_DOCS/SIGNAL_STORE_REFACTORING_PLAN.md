# Signal Store Refactoring Plan

**Current Status:** signal_store.py has 1,023 lines with 7 distinct responsibilities
**Goal:** Split into 5 focused modules with single responsibilities
**Estimated Effort:** 8-10 hours
**Priority:** High - Core module affects entire codebase

---

## Current Responsibilities Analysis

### 1. Core Storage & Basic Operations (~200 lines)
**Methods:**
- `__init__()` - Initialize store with configuration
- `deposit()` - Add new signal to store
- `get_signal()` - Retrieve signal by ID
- `get_all_signals()` - Get all signals
- `get_top_signals()` - Get N strongest signals
- `clear()` - Clear all signals
- `get_stats()` - Statistics

**Keeps:** Dictionary storage, basic CRUD, locking

### 2. Event Management (~100 lines)
**Methods:**
- `wait_for_signal()` - Async wait for signal type
- `clear_signal_event()` - Clear event flags
- `has_signals()` - Check if signals exist
- Event creation in `deposit()`

**Responsibility:** asyncio.Event coordination for agent synchronization

### 3. Sampling (~250 lines)
**Methods:**
- `sample_weighted()` - Weighted random sampling
- `sample_stratified()` - Stratified by strength
- `sample_cluster()` - Semantic clustering sampling
- `find_related_signals()` - Similarity-based search

**Responsibility:** Different signal selection strategies

### 4. Decay & Strength Management (~150 lines)
**Methods:**
- `decay_all()` - Apply time-based decay
- `prune_weak()` - Remove weak signals
- `amplify()` - Increase signal strength
- `boost_contrarian_signals()` - Anti-echo boost

**Responsibility:** Signal lifecycle and strength dynamics

### 5. Graph Traversal (~250 lines)
**Methods:**
- `get_ancestors()` - Traverse parent chain
- `get_descendants()` - Traverse child tree
- `get_children()` - Direct children only
- `get_connecting_signals()` - Path between signals
- `_invalidate_cache_for_signal()` - Cache management

**Responsibility:** Parent-child provenance traversal

### 6. Similarity & Embeddings (~200 lines)
**Methods:**
- `_initialize_embedding_model()` - Load sentence transformers
- `_check_similarity()` - Cosine/string similarity
- Embedding storage in `signal_embeddings` dict

**Responsibility:** Semantic similarity for duplicate detection

### 7. Dialogue & Validation (~100 lines)
**Methods:**
- `deposit_response()` - Link signals in dialogue
- `get_responses()` - Get direct responses
- `get_dialogue_thread()` - Full conversation
- `get_validation_status()` - Check support/objections
- `get_unvalidated_signals()` - Find signals needing validation

**Responsibility:** Dialogue threading and validation tracking

---

## Proposed Module Split

### Module 1: `signal_store.py` (Core - ~250 lines)

**Purpose:** Core storage, basic operations, coordination hub

**Keeps:**
```python
class Signal:  # Dataclass
class SignalStore:
    def __init__(...)
    def deposit(...)  # Delegates to other modules
    def get_signal(...)
    def get_all_signals(...)
    def get_top_signals(...)
    def clear(...)
    def get_stats(...)

    # Composition - delegate to modules
    self.events = SignalEvents(...)
    self.sampler = SignalSampler(...)
    self.decay_manager = SignalDecay(...)
    self.graph = SignalGraph(...)
    self.similarity = SignalSimilarity(...)
```

**Role:** Orchestrator that coordinates other modules

---

### Module 2: `signal_events.py` (NEW - ~100 lines)

**Purpose:** Event-driven coordination between agents

**Exports:**
```python
class SignalEvents:
    def __init__(self):
        self._signal_events: Dict[str, asyncio.Event] = {}
        self._new_signal_event = asyncio.Event()

    async def wait_for_signal(self, signal_type: str, timeout: Optional[float]) -> bool
    def clear_signal_event(self, signal_type: str) -> None
    def has_signals(self, signal_type: str, signals: Dict) -> bool
    def notify_deposit(self, signal_type: str) -> None
```

**Usage in SignalStore:**
```python
# In deposit():
signal_id = self._store_signal(signal)
self.events.notify_deposit(signal_type)  # Trigger events
return signal_id
```

---

### Module 3: `signal_sampling.py` (NEW - ~250 lines)

**Purpose:** Signal selection strategies

**Exports:**
```python
class SignalSampler:
    def __init__(self, diversity_threshold: float, exploration_bonus: float):
        ...

    def sample_weighted(self, signals: List[Signal], n: int) -> List[Signal]
    def sample_stratified(self, signals: List[Signal], weak: int, medium: int,
                         strong: int, ...) -> List[Signal]
    def sample_cluster(self, signals: List[Signal], size: int,
                      diversity_weight: float, ...) -> List[Signal]
    def find_related(self, signal: Signal, candidates: List[Signal],
                    threshold: float, n: int) -> List[Signal]
```

**Usage in SignalStore:**
```python
def sample_weighted(self, signal_type: str, n: int) -> List[Signal]:
    signals = [s for s in self.signals.values() if s.type == signal_type]
    return self.sampler.sample_weighted(signals, n)
```

---

### Module 4: `signal_decay.py` (NEW - ~150 lines)

**Purpose:** Signal lifecycle and strength dynamics

**Exports:**
```python
class SignalDecay:
    def __init__(self, decay_rate: float, prune_threshold: float):
        self.decay_rate = decay_rate
        self.prune_threshold = prune_threshold

    def decay_all(self, signals: Dict[str, Signal],
                  contrarian_types: Optional[List[str]],
                  contrarian_boost: float) -> int

    def prune_weak(self, signals: Dict[str, Signal],
                   embeddings: Dict[str, Any]) -> int

    def amplify(self, signal: Signal, factor: float) -> bool

    def boost_contrarian(self, signals: Dict[str, Signal],
                        contrarian_types: List[str],
                        boost_factor: float) -> int
```

**Usage in SignalStore:**
```python
def decay_all(self, contrarian_types=None, contrarian_boost=1.10):
    return self.decay_manager.decay_all(
        self.signals, contrarian_types, contrarian_boost
    )
```

---

### Module 5: `signal_graph.py` (NEW - ~250 lines)

**Purpose:** Provenance graph traversal with caching

**Exports:**
```python
class SignalGraph:
    def __init__(self):
        self._ancestor_cache: Dict[tuple, List[Signal]] = {}
        self._descendant_cache: Dict[tuple, List[Signal]] = {}

    def get_ancestors(self, signal_id: str, signals: Dict[str, Signal],
                     target_type: Optional[str]) -> List[Signal]

    def get_descendants(self, signal_id: str, signals: Dict[str, Signal],
                       target_type: Optional[str]) -> List[Signal]

    def get_children(self, signal_id: str, signals: Dict[str, Signal],
                    child_type: Optional[str]) -> List[Signal]

    def get_connecting_signals(self, signal_id_a: str, signal_id_b: str,
                              signals: Dict[str, Signal]) -> List[Signal]

    def invalidate_cache_for_signal(self, signal_id: str,
                                    parent_id: Optional[str],
                                    signals: Dict[str, Signal]) -> None
```

**Usage in SignalStore:**
```python
def get_ancestors(self, signal_id: str, target_type=None):
    return self.graph.get_ancestors(signal_id, self.signals, target_type)
```

---

### Module 6: `signal_similarity.py` (NEW - ~200 lines)

**Purpose:** Semantic similarity for duplicate detection

**Exports:**
```python
class SignalSimilarity:
    def __init__(self, use_semantic: bool, diversity_threshold: float):
        self.use_semantic = use_semantic
        self.diversity_threshold = diversity_threshold
        self.embedding_model = None
        self.embeddings: Dict[str, Any] = {}
        if use_semantic:
            self._initialize_embedding_model()

    def _initialize_embedding_model(self) -> None

    def check_similarity(self, content1: str, content2: str,
                        embedding1: Any, embedding2: Any) -> float

    def compute_embedding(self, content: str) -> Optional[Any]

    def store_embedding(self, signal_id: str, content: str) -> None

    def remove_embedding(self, signal_id: str) -> None

    def clear_all_embeddings(self) -> None
```

**Usage in SignalStore:**
```python
def deposit(self, ...):
    # Check for duplicates
    new_embedding = self.similarity.compute_embedding(content)
    for existing in same_type:
        existing_embedding = self.similarity.embeddings.get(existing.id)
        similarity = self.similarity.check_similarity(
            content, existing.content, new_embedding, existing_embedding
        )
        if similarity >= self.similarity.diversity_threshold:
            # Reject duplicate
            return None
```

---

## Dialogue & Validation

**Decision:** Keep in `signal_store.py` for now

**Rationale:**
- Only 100 lines
- Tightly coupled to core operations
- Not used heavily enough to warrant separate module
- Can extract later if needed

---

## Migration Strategy

### Phase 1: Create New Modules (No Breaking Changes)

**Step 1:** Create empty module files
```bash
touch swarm/core/signal_events.py
touch swarm/core/signal_sampling.py
touch swarm/core/signal_decay.py
touch swarm/core/signal_graph.py
touch swarm/core/signal_similarity.py
```

**Step 2:** Copy code to new modules (don't delete from signal_store.py yet)
- Extract methods with minimal modifications
- Add proper imports
- Create classes with clear interfaces

**Step 3:** Add composition to SignalStore
```python
class SignalStore:
    def __init__(self, ...):
        # NEW: Delegate to specialized modules
        self.events = SignalEvents()
        self.sampler = SignalSampler(diversity_threshold, exploration_bonus)
        self.decay_manager = SignalDecay(decay_rate, prune_threshold)
        self.graph = SignalGraph()
        self.similarity = SignalSimilarity(use_semantic_clustering, diversity_threshold)
```

**Step 4:** Update methods to delegate
```python
# BEFORE
def sample_weighted(self, signal_type: str, n: int):
    # 50 lines of logic
    ...

# AFTER
def sample_weighted(self, signal_type: str, n: int):
    signals = [s for s in self.signals.values() if s.type == signal_type]
    return self.sampler.sample_weighted(signals, n)
```

---

### Phase 2: Test & Validate

**Run all tests:**
```bash
pytest tests/ -v
```

**Manual testing:**
- Run run_task.py with debate config
- Run document processing
- Check signal counts match
- Verify no crashes

---

### Phase 3: Remove Duplicated Code

**Once confident:**
- Delete old method bodies from SignalStore
- Keep delegation stubs
- Remove unused imports
- Update docstrings

---

### Phase 4: Update Documentation

**Files to update:**
- README.md - Note module split
- ARCHITECTURE.md - Document new structure
- Docstrings - Point to new modules

---

## Benefits After Refactoring

### Code Organization
- **Before:** 1,023 lines in one file
- **After:** 6 files averaging ~200 lines each

### Testability
- Can test sampling strategies independently
- Can mock graph traversal for other tests
- Can test decay logic in isolation

### Maintainability
- Clear responsibility boundaries
- Easier to understand each module
- Easier to optimize specific concerns

### Performance
- Can optimize graph traversal independently
- Can swap embedding models without touching storage
- Can cache more aggressively per module

---

## Risks & Mitigation

### Risk 1: Breaking Existing Code

**Mitigation:**
- Keep delegation layer in SignalStore
- No API changes - only internal refactoring
- Comprehensive testing before removing old code

### Risk 2: Performance Regression

**Mitigation:**
- Profile before and after
- Keep hot paths in SignalStore if needed
- Can inline critical methods if necessary

### Risk 3: Circular Dependencies

**Mitigation:**
- Clear dependency hierarchy:
  - signal_similarity.py (no deps)
  - signal_graph.py (no deps)
  - signal_decay.py (no deps)
  - signal_sampling.py (depends on similarity)
  - signal_events.py (no deps)
  - signal_store.py (depends on all)

---

## Implementation Checklist

### Preparation
- [ ] Create backup branch
- [ ] Run full test suite (baseline)
- [ ] Profile performance (baseline)

### Module Creation
- [ ] Create signal_events.py
- [ ] Create signal_sampling.py
- [ ] Create signal_decay.py
- [ ] Create signal_graph.py
- [ ] Create signal_similarity.py

### Code Migration
- [ ] Extract event methods to SignalEvents
- [ ] Extract sampling methods to SignalSampler
- [ ] Extract decay methods to SignalDecay
- [ ] Extract graph methods to SignalGraph
- [ ] Extract similarity methods to SignalSimilarity

### Integration
- [ ] Add composition to SignalStore.__init__
- [ ] Update all methods to delegate
- [ ] Update imports across codebase
- [ ] Fix any broken references

### Testing
- [ ] Run unit tests
- [ ] Run integration tests
- [ ] Manual smoke testing
- [ ] Performance profiling

### Cleanup
- [ ] Remove duplicated code from signal_store.py
- [ ] Update docstrings
- [ ] Update documentation
- [ ] Code review

### Commit Strategy
- Commit 1: Create new modules (empty)
- Commit 2: Add code to new modules (keep duplicates)
- Commit 3: Add composition to SignalStore
- Commit 4: Update methods to delegate
- Commit 5: Remove duplicated code
- Commit 6: Update documentation

---

## Timeline Estimate

| Phase | Tasks | Time |
|-------|-------|------|
| **Preparation** | Branch, tests, profiling | 30 min |
| **Module Creation** | Create 5 files | 15 min |
| **Code Migration** | Extract ~800 lines | 3 hours |
| **Integration** | Composition + delegation | 2 hours |
| **Testing** | Comprehensive validation | 1.5 hours |
| **Cleanup** | Remove duplication, docs | 1 hour |
| **Buffer** | Unexpected issues | 2 hours |
| **Total** | | **10 hours** |

---

## Success Criteria

✅ **Functional:**
- All existing tests pass
- No behavior changes
- Same outputs for same inputs

✅ **Structural:**
- signal_store.py < 300 lines
- Each new module < 250 lines
- Clear single responsibility per module

✅ **Performance:**
- No regression in benchmarks
- Memory usage unchanged
- Latency within 5%

✅ **Quality:**
- Type hints on all methods
- Docstrings on all classes
- Clear module boundaries

---

## Future Enhancements (Post-Refactoring)

### Easy Optimizations
1. **Parallel sampling** - Can now optimize sampler independently
2. **Faster graph traversal** - Can use specialized data structures
3. **Better embedding caching** - Can add LRU per embedding type

### New Features
1. **Pluggable samplers** - Easy to add new sampling strategies
2. **Custom decay policies** - Easy to experiment with decay
3. **Graph algorithms** - Easy to add PageRank, clustering, etc.

---

**Status:** Plan created, ready for implementation
**Estimated ROI:** High - Core module affects entire system
**Recommended Approach:** Incremental commits, thorough testing
