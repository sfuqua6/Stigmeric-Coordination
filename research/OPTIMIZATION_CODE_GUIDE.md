# AI Swarm Mechanics - Optimization Code Guide

## Quick Reference: Before & After Code Examples

---

## 1. SEMAPHORE OPTIMIZATION (HIGH PRIORITY)

### Before (simple_llm.py:47)
```python
self._generation_semaphore = asyncio.Semaphore(3)
```

### After - Option A: Fixed Increase
```python
# For RTX 3060 (6GB): Use 6-8 concurrent
# For RTX 4090 (24GB): Use 12-16 concurrent
# For CPU: Use 2-3 concurrent
self._generation_semaphore = asyncio.Semaphore(6)
```

### After - Option B: Dynamic Scaling
```python
@staticmethod
def _calculate_optimal_semaphore(device: str, available_agents: int) -> int:
    """Calculate optimal semaphore limit based on device and agents."""
    if device == "cuda":
        # VRAM heuristic: allow ~1 concurrent per 2 agents, cap at 8-12
        return min(max(available_agents // 2, 3), 12)
    else:  # CPU
        return 2  # CPU is much slower

# In __init__:
self._generation_semaphore = asyncio.Semaphore(
    self._calculate_optimal_semaphore(device, num_agents=10)  # or get from config
)
```

### After - Option C: Configurable
```python
# In config.py
LLM_CONCURRENT_GENERATIONS = 6  # Increase from default 3

# In simple_llm.py
from ..core.config import LLM_CONCURRENT_GENERATIONS
self._generation_semaphore = asyncio.Semaphore(LLM_CONCURRENT_GENERATIONS)
```

**Expected Impact:** +40% throughput

---

## 2. EMBEDDING LAZY COMPUTATION (HIGH PRIORITY)

### Before (signal_store.py:146-165)
```python
def deposit(self, signal_type: str, content: str, strength: float,
            depositor: str, parent: Optional[str] = None, metadata: Optional[dict] = None) -> Optional[str]:
    with self._lock:
        # EXPENSIVE: Compute embedding for every new signal
        new_embedding = None
        if self.use_semantic_clustering and self.embedding_model is not None:
            new_embedding = self.embedding_model.encode(content)  # ~100ms

        # EXPENSIVE: Check against ALL existing signals
        same_type = [s for s in self.signals.values() if s.type == signal_type]
        for existing in same_type:
            existing_embedding = self.signal_embeddings.get(existing.id)
            similarity = self._check_similarity(
                content, existing.content,
                embedding1=new_embedding,
                embedding2=existing_embedding
            )
            if similarity >= self.diversity_threshold:
                existing.strength = min(1.0, existing.strength * 1.1)
                existing.visits += 1
                return None

        # ... deposit signal ...
        if new_embedding is not None:
            self.signal_embeddings[signal_id] = new_embedding
```

### After - Option A: Lazy on First Use
```python
def deposit(self, signal_type: str, content: str, strength: float,
            depositor: str, parent: Optional[str] = None, metadata: Optional[dict] = None) -> Optional[str]:
    with self._lock:
        # DON'T compute embedding yet - do it lazily on first comparison
        
        # FAST: Use string similarity only for diversity check
        same_type = [s for s in self.signals.values() if s.type == signal_type]
        
        # Filter to recent signals only (temporal optimization)
        cutoff_time = time.time() - 300  # Last 5 minutes
        recent_same_type = [s for s in same_type if s.timestamp > cutoff_time]
        
        for existing in recent_same_type:
            # Use fast string similarity first
            fast_similarity = SequenceMatcher(None, 
                content.lower(), 
                existing.content.lower()
            ).ratio()
            
            # Only compute embeddings if fast check is inconclusive
            if 0.7 < fast_similarity < 0.95:
                # Need semantic check - now compute embeddings
                if not hasattr(self, '_embedding_cache'):
                    self._embedding_cache = {}
                
                new_emb = self._get_cached_embedding(content)
                existing_emb = self.signal_embeddings.get(existing.id)
                
                if existing_emb is not None:
                    semantic_sim = self._cosine_similarity(new_emb, existing_emb)
                    if semantic_sim >= self.diversity_threshold:
                        existing.strength = min(1.0, existing.strength * 1.1)
                        existing.visits += 1
                        return None
            elif fast_similarity >= self.diversity_threshold:
                existing.strength = min(1.0, existing.strength * 1.1)
                existing.visits += 1
                return None
        
        # Deposit signal WITHOUT computing embedding yet
        signal_id = f"{signal_type}_{self._next_id:04d}"
        self._next_id += 1
        
        signal = Signal(
            id=signal_id, type=signal_type, content=content,
            strength=strength, timestamp=time.time(),
            depositor=depositor, parent=parent, metadata=metadata or {}
        )
        
        self.signals[signal_id] = signal
        # Store embedding later (on first query/comparison)
        
        return signal_id

def _get_cached_embedding(self, content: str):
    """Get embedding with caching to avoid recomputation."""
    cache_key = hashlib.md5(content.encode()).hexdigest()
    
    if cache_key not in self._embedding_cache:
        if self.embedding_model is not None:
            self._embedding_cache[cache_key] = self.embedding_model.encode(content)
    
    return self._embedding_cache[cache_key]
```

### After - Option B: Batch Embeddings
```python
async def deposit_batch(self, signals_data: List[dict]) -> List[Optional[str]]:
    """Deposit multiple signals with batched embedding computation."""
    # Collect all new signals
    new_signals = []
    embedding_tasks = []
    
    for signal_data in signals_data:
        # Create signal without embedding
        signal = Signal(
            id=f"{signal_data['type']}_batch_{len(new_signals):04d}",
            type=signal_data['type'],
            content=signal_data['content'],
            strength=signal_data['strength'],
            timestamp=time.time(),
            depositor=signal_data['depositor'],
            parent=signal_data.get('parent'),
            metadata=signal_data.get('metadata', {})
        )
        new_signals.append(signal)
        embedding_tasks.append(signal_data['content'])
    
    # Compute all embeddings in parallel if available
    with self._lock:
        if self.use_semantic_clustering and self.embedding_model is not None:
            # Use batch encoding if model supports it
            if hasattr(self.embedding_model, 'encode_multi_process'):
                embeddings = self.embedding_model.encode_multi_process(embedding_tasks)
            else:
                embeddings = [self.embedding_model.encode(content) for content in embedding_tasks]
            
            # Store embeddings and deposit
            signal_ids = []
            for signal, embedding in zip(new_signals, embeddings):
                self.signals[signal.id] = signal
                self.signal_embeddings[signal.id] = embedding
                signal_ids.append(signal.id)
            
            return signal_ids
```

**Expected Impact:** +20-30% throughput

---

## 3. CACHE INVALIDATION OPTIMIZATION (HIGH PRIORITY)

### Before (signal_store.py:187-189)
```python
# Clear ALL caches on every signal deposit
self._ancestor_cache.clear()
self._descendant_cache.clear()
```

### After - Selective Invalidation
```python
def deposit(self, signal_type: str, content: str, strength: float,
            depositor: str, parent: Optional[str] = None, metadata: Optional[dict] = None) -> Optional[str]:
    with self._lock:
        # ... deposit logic ...
        signal_id = f"{signal_type}_{self._next_id:04d}"
        self.signals[signal_id] = signal
        
        # SELECTIVE INVALIDATION: Only invalidate affected caches
        self._invalidate_caches_for_signal(signal_id, parent)
        
        return signal_id

def _invalidate_caches_for_signal(self, signal_id: str, parent_id: Optional[str] = None):
    """Invalidate only affected cache entries, not entire cache."""
    # Clear cache entries that involve this signal
    keys_to_remove = []
    
    # Check ancestor cache
    for cache_key in self._ancestor_cache.keys():
        signal_id_in_key, target_type = cache_key
        if signal_id_in_key == signal_id or signal_id_in_key == parent_id:
            keys_to_remove.append(cache_key)
    
    for key in keys_to_remove:
        del self._ancestor_cache[key]
    
    # Check descendant cache
    keys_to_remove = []
    for cache_key in self._descendant_cache.keys():
        signal_id_in_key, target_type = cache_key
        if signal_id_in_key == signal_id or signal_id_in_key == parent_id:
            keys_to_remove.append(cache_key)
    
    for key in keys_to_remove:
        del self._descendant_cache[key]

# Alternative: LRU Cache with Size Limit
from functools import lru_cache

class SignalStore:
    def __init__(self, ...):
        # ... existing init ...
        self._cache_max_size = 100
        
        # Use functools.lru_cache for automatic management
        self._get_ancestors_cached = lru_cache(maxsize=self._cache_max_size)(
            self._get_ancestors_impl
        )
    
    def get_ancestors(self, signal_id: str, target_type: Optional[str] = None) -> List[Signal]:
        """Get ancestors with LRU cache."""
        return self._get_ancestors_cached(signal_id, target_type)
    
    def _get_ancestors_impl(self, signal_id: str, target_type: Optional[str]) -> List[Signal]:
        """Actual implementation of get_ancestors."""
        # ... existing logic ...
        pass
    
    def deposit(self, ...):
        # Clear LRU cache on deposit
        self._get_ancestors_cached.cache_clear()
        # ... rest of deposit logic ...
```

**Expected Impact:** +10-15% throughput

---

## 4. ASYNC RATE LIMITING (HIGH PRIORITY)

### Before (search_engine.py:51, web_scraper.py:63)
```python
# BLOCKING - stops entire event loop
def _rate_limit_wait(self):
    elapsed = time.time() - self.last_request_time
    if elapsed < self.rate_limit:
        time.sleep(self.rate_limit - elapsed)  # BLOCKS!
```

### After
```python
async def _rate_limit_wait(self):  # Make async!
    elapsed = time.time() - self.last_request_time
    if elapsed < self.rate_limit:
        await asyncio.sleep(self.rate_limit - elapsed)  # Non-blocking!
    self.last_request_time = time.time()

# Update all call sites:
# Before:
response = self.session.get(url)  # Synchronous

# After:
async def scrape_async(self, url: str):
    """Async version of scrape."""
    loop = asyncio.get_event_loop()
    await self._rate_limit_wait()  # Async rate limit
    response = await loop.run_in_executor(
        None,
        self.session.get,
        url
    )
    return response
```

**Expected Impact:** +5-10% throughput

---

## 5. SLEEP DELAYS TO EVENT-DRIVEN (HIGH PRIORITY)

### Before (scout.py:101, forager.py:62, 88)
```python
while self.active and self.actions_taken < max_actions:
    idea = await self.explore_creative(llm)
    if idea:
        signal_store.deposit(...)
    self.actions_taken += 1
    await asyncio.sleep(random.uniform(0.1, 0.5))  # Fixed delay every iteration!
```

### After - Option A: Event-Driven
```python
async def run(self, signal_store: SignalStore, llm: SimpleLLM,
              min_strength: float = 0.5, max_actions: int = 1):
    """Event-driven instead of time-driven."""
    while self.active and self.actions_taken < max_actions:
        # Only delay if generation failed (backoff)
        idea = await self.explore_creative(llm)
        
        if idea:
            strength = self.assess_strength_creative(idea)
            if strength >= min_strength:
                signal_store.deposit(
                    signal_type=self.signal_type,
                    content=idea,
                    strength=strength,
                    depositor=self.agent_id
                )
                # Success - minimal delay
                await asyncio.sleep(0.01)  # Just to yield control
            else:
                # Weak idea - back off slightly
                await asyncio.sleep(random.uniform(0.05, 0.1))
        else:
            # Generation failed - back off more
            await asyncio.sleep(random.uniform(0.1, 0.3))
        
        self.actions_taken += 1
```

### After - Option B: Configuration-Driven
```python
# In config.py
AGENT_LOOP_DELAY = 0.0  # Set to 0 for production (no artificial delays)
                        # Set to 0.2 for debugging (natural timing)

# In agents
class Scout:
    def __init__(self, ...):
        from ..core.config import AGENT_LOOP_DELAY
        self.loop_delay = AGENT_LOOP_DELAY
    
    async def run(self, ...):
        while self.active and self.actions_taken < max_actions:
            idea = await self.explore_creative(llm)
            if idea:
                signal_store.deposit(...)
            self.actions_taken += 1
            
            if self.loop_delay > 0:
                await asyncio.sleep(self.loop_delay)
```

### After - Option C: Signal-Driven
```python
async def run(self, signal_store: SignalStore, llm: SimpleLLM):
    """Wait for required input signals instead of fixed delays."""
    while self.active and self.actions_taken < max_actions:
        # Wait for required input signal type (if needed)
        if hasattr(self, 'input_type'):
            await signal_store.wait_for_signal(self.input_type, timeout=1.0)
        
        # Generate
        idea = await self.explore_creative(llm)
        
        if idea:
            signal_store.deposit(...)
        
        self.actions_taken += 1
```

**Expected Impact:** +20-40% throughput

---

## 6. EXTERNAL SOURCE CACHE BOUNDS (MEDIUM PRIORITY)

### Before (external_sources.py:66, 284)
```python
class WikipediaSource(ExternalSource):
    def __init__(self):
        self.cache = {}  # UNBOUNDED
    
    def verify(self, claim: str) -> Dict:
        cache_key = ' '.join(sorted(key_terms))
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        result = await self._real_wikipedia_lookup(...)
        self.cache[cache_key] = result  # Never evicted!
        return result
```

### After - LRU Cache
```python
from collections import OrderedDict

class WikipediaSource(ExternalSource):
    def __init__(self, cache_size: int = 100):
        """Initialize with bounded cache."""
        self.cache = OrderedDict()
        self.cache_size = cache_size
    
    async def verify(self, claim: str) -> Dict:
        cache_key = ' '.join(sorted(key_terms))
        
        if cache_key in self.cache:
            # Move to end (LRU)
            self.cache.move_to_end(cache_key)
            return self.cache[cache_key]
        
        result = await self._real_wikipedia_lookup(...)
        
        # Store with eviction
        self.cache[cache_key] = result
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)  # Remove oldest
        
        return result
```

**Expected Impact:** +5-10% memory savings

---

## 7. SIGNAL PRUNING CLEANUP (MEDIUM PRIORITY)

### Before (signal_store.py:459-474)
```python
def prune_weak(self) -> int:
    to_remove = [
        sid for sid, signal in self.signals.items()
        if signal.strength < self.prune_threshold
    ]
    
    for sid in to_remove:
        del self.signals[sid]
        # Embeddings NOT cleaned!
        # Metadata NOT cleaned!
    
    return len(to_remove)
```

### After - Complete Cleanup
```python
def prune_weak(self) -> int:
    """Remove signals below pruning threshold with complete cleanup."""
    to_remove = [
        sid for sid, signal in self.signals.items()
        if signal.strength < self.prune_threshold
    ]
    
    with self._lock:
        for sid in to_remove:
            signal = self.signals[sid]
            
            # Remove signal
            del self.signals[sid]
            
            # ALSO remove embedding
            self.signal_embeddings.pop(sid, None)
            
            # Clean up parent relationship
            if signal.parent and signal.parent in self.signals:
                parent = self.signals[signal.parent]
                if sid in parent.responses:
                    parent.responses.remove(sid)
            
            # Clean up child relationships
            for child_id in signal.responses:
                if child_id in self.signals:
                    self.signals[child_id].is_response_to = None
    
    return len(to_remove)
```

**Expected Impact:** +5-10% memory savings

---

## Performance Testing Script

```python
import time
import asyncio
from swarm.core.signal_store import SignalStore

async def benchmark_deposit_speed():
    """Benchmark signal deposit performance."""
    store = SignalStore(use_semantic_clustering=True)
    
    # Create 100 observations first
    for i in range(100):
        await asyncio.to_thread(
            store.deposit,
            "OBSERVATION",
            f"Sample observation {i}",
            0.7,
            "scout_0"
        )
    
    # Time new deposit
    start = time.time()
    for i in range(100, 110):
        await asyncio.to_thread(
            store.deposit,
            "OBSERVATION",
            f"Sample observation {i}",
            0.7,
            "scout_0"
        )
    elapsed = time.time() - start
    
    print(f"10 deposits with 100 existing signals: {elapsed:.2f}s")
    print(f"Average per deposit: {elapsed/10*1000:.0f}ms")
    print(f"Expected: <50ms (with optimizations), >300ms (without)")

if __name__ == "__main__":
    asyncio.run(benchmark_deposit_speed())
```

---

## Implementation Priority

**Quickest Wins (30 minutes):**
1. Semaphore from 3 to 6
2. Change time.sleep to asyncio.sleep in search_engine.py
3. Add embedding cleanup in prune_weak()

**High Impact (2 hours):**
1. Lazy embedding computation
2. Selective cache invalidation
3. Remove fixed sleep delays

**Medium Impact (4 hours):**
1. Add temporal filtering to similarity checks
2. Bounded external source caches
3. Separate queue timeout from generation timeout

