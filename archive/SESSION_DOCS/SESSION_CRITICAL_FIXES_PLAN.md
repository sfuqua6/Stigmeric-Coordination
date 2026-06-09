# Session Planning: Critical Fixes Implementation

**Date:** 2025-11-20
**Branch:** claude/analyze-codebase-01RUVXdkHt9uNkPn7rauTXiE
**Objective:** Fix critical issues in ai_swarm_mechanics codebase

## Executive Summary

I've been asked to fix critical issues in the codebase:
1. Embedding memory leaks
2. Replace 272 print statements with structured logging
3. Fix bare exception handlers
4. Complete Critic agent to generate CRITIQUE signals
5. Optimize O(n²) signal similarity to O(log n) using FAISS

Before implementing, I must carefully review each change to ensure no regressions.

---

## Issue 1: Embedding Memory Leak

### Problem Analysis

**Location:** `swarm/core/signal_store.py:68` and `swarm/agents/pruner.py:141`

**Current Behavior:**
- Signal embeddings are stored in `self.signal_embeddings: Dict[str, Any] = {}` (line 68)
- When signals are pruned via `signal_store.prune_weak()`, embeddings ARE cleaned up (line 576-577)
- When signals are pruned via `pruner.py`, embeddings ARE cleaned up (line 141-142)
- **BUT** there's no cleanup in other deletion paths

**Evidence from Code Review:**
```python
# signal_store.py:559-579 (prune_weak method)
def prune_weak(self) -> int:
    with self._lock:
        to_remove = [sid for sid, signal in self.signals.items()
                    if signal.strength < self.prune_threshold]
        for sid in to_remove:
            del self.signals[sid]
            # BUGFIX: Clean up embedding to prevent memory leak
            if sid in self.signal_embeddings:
                del self.signal_embeddings[sid]  # ✅ CLEANUP EXISTS
        return len(to_remove)

# signal_store.py:626-631 (clear method)
def clear(self) -> None:
    with self._lock:
        self.signals.clear()
        self.signal_embeddings.clear()  # ✅ CLEANUP EXISTS
        self._next_id = 0

# pruner.py:140-142
if signal_id in signal_store.signal_embeddings:
    del signal_store.signal_embeddings[signal_id]  # ✅ CLEANUP EXISTS
```

**Wait - Let me reconsider!** Looking at the code more carefully:
- `prune_weak()` - ✅ Already cleans up embeddings
- `clear()` - ✅ Already cleans up embeddings
- `pruner.py` - ✅ Already cleans up embeddings

**Where's the leak?** The leak happens if:
1. Signals are deleted through direct dictionary manipulation (bypassing methods)
2. Cache invalidation doesn't clean up embeddings
3. No centralized delete method

### My Doubts

🤔 **Question 1:** Is there actually a leak, or is it already fixed?
- Looking at the code comments "BUGFIX", it seems someone already attempted to fix this
- All deletion paths I can see DO clean up embeddings
- **BUT** - there's no centralized `delete_signal()` method, which is dangerous

🤔 **Question 2:** What if signals are removed via other paths?
- Response cleanup? (deposit_response - no deletion path visible)
- Graph operations? (get_ancestors, get_descendants - read-only)
- Direct manipulation? (Could external code do `signal_store.signals.pop(id)`?)

🤔 **Question 3:** Should I trust the analysis report?
- The report says there's a leak
- But the code shows cleanup in all paths I can find
- Perhaps the leak is in **potential future code paths**?

### Solution Strategy

**Approach:** Create a centralized `delete_signal()` method to ensure consistency

**Why this approach:**
1. **Defensive programming:** Even if no current leak, prevents future leaks
2. **Single responsibility:** One method handles all cleanup logic
3. **Maintainability:** Easy to find and update deletion logic
4. **Safety:** Internal methods use this instead of direct deletion

**Implementation:**
```python
# Add to SignalStore class
def delete_signal(self, signal_id: str) -> bool:
    """Delete a signal and all associated data.

    Ensures cleanup of:
    - Signal from signals dict
    - Embedding from signal_embeddings dict
    - Cache entries involving this signal

    Args:
        signal_id: Signal ID to delete

    Returns:
        True if signal was deleted, False if not found
    """
    with self._lock:
        if signal_id not in self.signals:
            return False

        # Get signal info before deletion (for cache cleanup)
        signal = self.signals[signal_id]

        # Delete signal
        del self.signals[signal_id]

        # Delete embedding (prevent memory leak)
        if signal_id in self.signal_embeddings:
            del self.signal_embeddings[signal_id]

        # Clean up cache entries
        # Remove ancestor cache entries
        keys_to_remove = [k for k in self._ancestor_cache.keys()
                         if k[0] == signal_id]
        for key in keys_to_remove:
            del self._ancestor_cache[key]

        # Remove descendant cache entries
        keys_to_remove = [k for k in self._descendant_cache.keys()
                         if k[0] == signal_id]
        for key in keys_to_remove:
            del self._descendant_cache[key]

        return True
```

**Then refactor existing deletion paths to use this:**
- `prune_weak()` - use `delete_signal()`
- `pruner.py` - use `delete_signal()`
- Future code - forced to use safe API

### Risks

⚠️ **Risk 1:** Performance - Centralized method might be slower
- **Mitigation:** Cache cleanup is already O(k) where k = cache size, not worse than current

⚠️ **Risk 2:** Breaking existing behavior
- **Mitigation:** Keep existing methods, just refactor internals to use delete_signal()

⚠️ **Risk 3:** Lock contention
- **Mitigation:** delete_signal() already uses lock, no change

---

## Issue 2: Print Statements → Structured Logging

### Problem Analysis

**Scale:** 219 print() calls across 20 files (from grep count)

**Current State:**
```python
# Everywhere in the codebase
print(f"[PRUNER] {self.agent_id} pruned {pruned_count} signals")
print("[SIGNAL_STORE] Loading semantic embedding model...")
print(f"[CRITIC] {self.agent_id} evaluated {signal.id}")
```

**Problems:**
1. No log levels (can't filter INFO vs DEBUG vs ERROR)
2. No timestamps
3. Can't redirect to files
4. Can't disable specific modules
5. Output pollution in production
6. No structured data for parsing

**Good News:** Logging infrastructure already exists!
- `swarm/core/logging_config.py` provides `get_logger(__name__)`
- Already used in some files (critic.py, etc.)
- Just need to migrate print → logger

### My Doubts

🤔 **Question 1:** Should I migrate ALL print statements?
- Some might be intentional user-facing output
- Some might be in __main__ blocks for CLI output
- Need to distinguish debug output vs user output

🤔 **Question 2:** What log level for each statement?
- `[PRUNER] pruned X signals` → INFO or DEBUG?
- `Loading model...` → INFO (important startup info)
- `Found X signals` → DEBUG (internal details)
- Errors → ERROR obviously

🤔 **Question 3:** Will this break any tests?
- Tests might expect print() output
- Need to check test files

🤔 **Question 4:** Should I preserve the [MODULE] prefix?
- Logger already includes module name
- Redundant to have both `[PRUNER]` and logger name `swarm.agents.pruner`
- **Decision:** Remove [MODULE] prefix, use logger name

### Solution Strategy

**Approach:** Systematic migration with log level mapping

**Log Level Mapping:**
```python
# INFO level (important events user should know)
print(f"[AGENT] Starting...")  → logger.info("Starting...")
print(f"[MODEL] Loading...")   → logger.info("Loading model...")
print(f"[PRUNER] Pruned X")     → logger.info("Pruned %d signals", X)

# DEBUG level (internal details, high volume)
print(f"[SIGNAL] Checking...")  → logger.debug("Checking signal similarity...")
print(f"[CACHE] Hit/miss")      → logger.debug("Cache hit for %s", key)

# WARNING level (issues but not critical)
print(f"[WARNING] ...")         → logger.warning("...")

# ERROR level (problems)
print(f"[ERROR] ...")           → logger.error("...", exc_info=True)
```

**Implementation Plan:**
1. Add `from swarm.core.logging_config import get_logger` to each file
2. Add `logger = get_logger(__name__)` at module level
3. Replace each print() with appropriate logger call
4. Use f-string → %-style formatting for performance: `logger.info("Message %s", var)`

**File Priority (most prints first):**
1. `swarm/llm/simple_llm.py` - 75 prints
2. `swarm/documents/processor.py` - 19 prints
3. `swarm/retrieval/simple_web_search.py` - 13 prints
4. `swarm/core/signal_store.py` - 11 prints
5. `swarm/retrieval/advanced_retriever.py` - 11 prints
... (continue for all 20 files)

### Risks

⚠️ **Risk 1:** Breaking output expectations
- **Mitigation:** Keep user-facing output as print(), only migrate debug output

⚠️ **Risk 2:** Too verbose logging
- **Mitigation:** Use DEBUG for high-volume output, INFO for important events

⚠️ **Risk 3:** Performance impact
- **Mitigation:** Logger is fast, minimal impact. Can disable DEBUG in production.

---

## Issue 3: Bare Exception Handlers

### Problem Analysis

**Location:** `swarm/validation/external_sources.py` - multiple instances

**Pattern Found:**
```python
# Line 598-599
try:
    # symbolic computation
except:
    pass  # Silent failure

# Line 624-625
try:
    result = sympify(expr)
except:
    pass

# Line 654-655, 686-687, etc.
```

**Problems:**
1. Catches ALL exceptions including KeyboardInterrupt, SystemExit
2. Silent failures make debugging impossible
3. No logging of what went wrong
4. Could hide real bugs

### My Doubts

🤔 **Question 1:** Why were these bare excepts used?
- Probably because sympy can raise many different exception types
- Developer wanted to fall through to next pattern
- But this is still wrong approach

🤔 **Question 2:** What exceptions should we catch?
- SymPy can raise: `SyntaxError`, `TypeError`, `ValueError`, `AttributeError`
- Should catch `Exception` (base for all "normal" errors)
- Should NOT catch `BaseException` (includes KeyboardInterrupt)

🤔 **Question 3:** Should we log these?
- YES for debugging
- But at what level? DEBUG or WARNING?
- These are expected failures (pattern not matching), so DEBUG level

### Solution Strategy

**Approach:** Replace bare except with specific Exception catching + logging

**Pattern:**
```python
# OLD (bad)
try:
    result = sympify(expr)
except:
    pass

# NEW (good)
try:
    result = sympify(expr)
except (SyntaxError, TypeError, ValueError) as e:
    logger.debug("Could not parse expression: %s", e)
except Exception as e:
    logger.warning("Unexpected error in symbolic verification: %s", e)
```

**Why this is better:**
1. Still catches all "normal" errors
2. Does NOT catch KeyboardInterrupt/SystemExit
3. Logs errors for debugging
4. Distinguishes expected errors (DEBUG) from unexpected (WARNING)

**Implementation:**
- Add logger to external_sources.py
- Replace each bare except with specific catches
- Add appropriate logging

### Risks

⚠️ **Risk 1:** Missing an exception type
- **Mitigation:** Use `Exception` as final catch-all (still better than bare except)

⚠️ **Risk 2:** Too much logging
- **Mitigation:** Use DEBUG level for expected failures

---

## Issue 4: Complete Critic Agent

### Problem Analysis

**Location:** `swarm/agents/critic.py:65-166`

**Current Behavior:**
- Critic samples signals (stratified sampling ✅)
- Generates critique text via LLM ✅
- Calculates quality multiplier ✅
- **ALREADY deposits CRITIQUE signals!** (lines 113-119, 144-150) ✅
- Adjusts parent signal strength ✅

**Wait - the code ALREADY does this!**

```python
# Line 113-119
critique_id = signal_store.deposit(
    signal_type="CRITIQUE",
    content=critique_text,
    depositor=self.agent_id,
    parent=signal.id,
    strength=quality_score
)
```

**So what's incomplete?**

Looking more carefully:
- The code is wrapped in `if hasattr(self, 'generate_critique_with_context'):`
- This uses **monkey-patched methods** from run_task.py
- The fallback path (lines 130-159) also deposits CRITIQUE signals
- The very last fallback (lines 160-166) does NOT deposit - just adjusts strength

### My Doubts

🤔 **Question 1:** Is this actually incomplete?
- The analysis report says "Critic doesn't generate signals"
- But the code clearly does (lines 113, 144)
- Perhaps the issue is the monkey-patching dependency?

🤔 **Question 2:** What should I "complete"?
- Remove monkey-patching dependency (use task_config)
- Ensure critique generation works without monkey-patching
- The generate_critique method exists (lines 298-357) and uses task_config

🤔 **Question 3:** Is the fallback path (lines 160-166) intentional?
- This is when neither `generate_critique_with_context` nor `generate_critique` exists
- It's a safety fallback - simple length check
- Should this also deposit a CRITIQUE signal? Probably yes for consistency

### Solution Strategy

**Approach:** Ensure CRITIQUE signals are ALWAYS deposited, not just in monkey-patched path

**Changes:**
1. Make fallback path (lines 160-166) also deposit CRITIQUE signals
2. Ensure task_config path works properly
3. Add logging for which path was taken

**Implementation:**
```python
# Line 160-166 - current fallback
else:
    # Fallback: simple validation check
    multiplier = 1.1 if len(signal.content) > 50 else 0.9
    old_strength = signal.strength
    signal.strength *= multiplier
    logger.debug(f"{self.agent_id} evaluated {signal.id} (fallback mode: {multiplier:.2f})")

# NEW - deposit critique signal even in fallback
else:
    # Fallback: simple validation check
    quality_score = 0.6 if len(signal.content) > 50 else 0.4
    critique_text = f"Basic length evaluation: {'adequate' if len(signal.content) > 50 else 'too brief'} ({len(signal.content)} chars)"

    critique_id = signal_store.deposit(
        signal_type="CRITIQUE",
        content=critique_text,
        depositor=self.agent_id,
        parent=signal.id,
        strength=quality_score
    )

    multiplier = 1.1 if quality_score > 0.5 else 0.9
    old_strength = signal.strength
    signal.strength *= multiplier

    logger.debug(f"{self.agent_id} evaluated {signal.id} (fallback mode, deposited {critique_id})")
```

### Risks

⚠️ **Risk 1:** Too many CRITIQUE signals
- **Mitigation:** Fallback is rare, only when LLM methods unavailable

⚠️ **Risk 2:** Low-quality critiques in fallback
- **Mitigation:** Mark with lower strength (0.4-0.6) to indicate uncertainty

---

## Issue 5: O(n²) Signal Similarity → FAISS Optimization

### Problem Analysis

**Location:** `swarm/core/signal_store.py:149-170`

**Current Behavior:**
```python
# Line 149-170
recent_cutoff = time.time() - 300  # 5 minutes
same_type = [s for s in self.signals.values()
            if s.type == signal_type and s.timestamp > recent_cutoff]

# Compute embedding for new signal
if same_type and self.use_semantic_clustering:
    new_embedding = self.embedding_model.encode(content)

# Check each existing signal - O(n) loop
for existing in same_type:
    existing_embedding = self.signal_embeddings.get(existing.id)
    similarity = self._check_similarity(content, existing.content,
                                       new_embedding, existing_embedding)
    if similarity >= self.diversity_threshold:
        # Reject duplicate
        return None
```

**Complexity:**
- O(n) filter for same_type signals
- O(n) loop to check each signal
- Each check is O(d) for embedding comparison (d = embedding dimension = 384)
- **Total: O(n * d) = O(n) per deposit**
- With m deposits: **O(m * n) = O(n²)** for swarm lifetime

**Why this is a problem:**
- For 1000 signals: 1000 comparisons per new signal
- For 10000 signals: 10000 comparisons per new signal
- With 384-dim embeddings: 3.84M floating point operations per signal

### My Doubts

🤔 **Question 1:** Is this actually O(n²) or am I misunderstanding?
- Current analysis: O(n) per deposit, O(m*n) total
- But they have temporal filtering (5 min window)
- So it's O(k) where k = signals in last 5 minutes
- Still could be large though

🤔 **Question 2:** Will FAISS actually help?
- FAISS provides fast nearest neighbor search (IndexFlatIP is O(n) with SIMD, IndexHNSW is O(log n))
- **CORRECTION:** IndexFlatIP is still O(n), just highly optimized with vectorization
- But we need to check ALL similar signals (not just nearest)
- We need ALL signals above threshold, not just top k
- So FAISS might not help much? **WRONG - FAISS can do radius search**

🤔 **Question 3:** Is the optimization worth the complexity?
- FAISS adds dependency (already in requirements.txt ✅)
- FAISS index needs rebuilding when signals change
- Trade-off: faster search vs. slower indexing
- Worth it if swarms have >1000 signals

🤔 **Question 4:** How to keep FAISS index updated?
- Option A: Rebuild entire index on each deposit (expensive)
- Option B: Add to index on deposit, remove on prune
- Option C: Lazy rebuild when staleness threshold reached
- **Best: Option B - incremental updates**

🤔 **Question 5:** Should I optimize only deposit() or also find_related_signals()?
- `deposit()` - called frequently, needs optimization
- `find_related_signals()` (line 637) - also O(n), should optimize
- `sample_cluster()` (line 681) - also O(n), should optimize
- **Optimize all three!**

### Solution Strategy

**Approach:** Add optional FAISS index for semantic similarity with graceful fallback

**Design Decisions:**

1. **Gradual Migration:** Keep string-based similarity as fallback
2. **Optional Feature:** Only use FAISS if available + embeddings enabled
3. **Incremental Index:** Add/remove vectors as signals change
4. **Per-Type Indexing:** Separate FAISS index per signal type (more efficient)

**Data Structure:**
```python
class SignalStore:
    def __init__(self, ...):
        # Existing
        self.signal_embeddings: Dict[str, Any] = {}

        # NEW: FAISS indexes
        self.use_faiss = use_semantic_clustering  # Enable FAISS with embeddings
        self.faiss_indexes: Dict[str, Any] = {}  # signal_type -> FAISS index
        self.faiss_id_maps: Dict[str, List[str]] = {}  # signal_type -> [signal_ids]

        # Initialize FAISS if available
        if self.use_faiss:
            self._initialize_faiss()
```

**Implementation Strategy:**

```python
def _initialize_faiss(self):
    """Initialize FAISS for fast similarity search."""
    try:
        import faiss
        self.faiss_available = True
        logger.info("FAISS available for fast similarity search")
    except ImportError:
        self.faiss_available = False
        logger.warning("FAISS not available, using O(n) similarity search")
        self.use_faiss = False

def _add_to_faiss_index(self, signal_id: str, signal_type: str, embedding: Any):
    """Add signal embedding to FAISS index for its type.

    Args:
        signal_id: Signal ID
        signal_type: Signal type (for separate indexes)
        embedding: Embedding vector (numpy array)
    """
    if not self.use_faiss or not self.faiss_available:
        return

    import faiss
    import numpy as np

    # Create index for this type if doesn't exist
    if signal_type not in self.faiss_indexes:
        embedding_dim = len(embedding)
        # Use inner product (cosine similarity after normalization)
        index = faiss.IndexFlatIP(embedding_dim)
        self.faiss_indexes[signal_type] = index
        self.faiss_id_maps[signal_type] = []

    # Normalize embedding for cosine similarity
    embedding = np.array(embedding, dtype=np.float32)
    embedding = embedding / np.linalg.norm(embedding)

    # Add to index
    self.faiss_indexes[signal_type].add(embedding.reshape(1, -1))
    self.faiss_id_maps[signal_type].append(signal_id)

def _remove_from_faiss_index(self, signal_id: str, signal_type: str):
    """Remove signal from FAISS index.

    FAISS doesn't support removal, so we:
    1. Mark as deleted in id_map (use None)
    2. Rebuild index periodically when too many deleted

    Args:
        signal_id: Signal ID to remove
        signal_type: Signal type
    """
    if not self.use_faiss or signal_type not in self.faiss_id_maps:
        return

    # Find signal in id map
    id_map = self.faiss_id_maps[signal_type]
    if signal_id in id_map:
        idx = id_map.index(signal_id)
        id_map[idx] = None  # Mark as deleted

        # Rebuild if >30% deleted
        deleted_count = id_map.count(None)
        if deleted_count / len(id_map) > 0.3:
            self._rebuild_faiss_index(signal_type)

def _rebuild_faiss_index(self, signal_type: str):
    """Rebuild FAISS index for a signal type (remove deleted entries).

    Args:
        signal_type: Signal type to rebuild
    """
    if signal_type not in self.faiss_id_maps:
        return

    import faiss
    import numpy as np

    # Get non-deleted signals
    id_map = self.faiss_id_maps[signal_type]
    valid_ids = [sid for sid in id_map if sid is not None and sid in self.signals]

    if not valid_ids:
        # No valid signals, clear index
        del self.faiss_indexes[signal_type]
        del self.faiss_id_maps[signal_type]
        return

    # Get embeddings for valid signals
    embeddings = []
    new_id_map = []
    for sid in valid_ids:
        if sid in self.signal_embeddings:
            emb = np.array(self.signal_embeddings[sid], dtype=np.float32)
            emb = emb / np.linalg.norm(emb)  # Normalize
            embeddings.append(emb)
            new_id_map.append(sid)

    if not embeddings:
        del self.faiss_indexes[signal_type]
        del self.faiss_id_maps[signal_type]
        return

    # Create new index
    embedding_dim = len(embeddings[0])
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(np.array(embeddings, dtype=np.float32))

    # Update
    self.faiss_indexes[signal_type] = index
    self.faiss_id_maps[signal_type] = new_id_map

    logger.debug(f"Rebuilt FAISS index for {signal_type}: {len(new_id_map)} signals")

def _find_similar_faiss(self, signal_type: str, embedding: Any,
                        similarity_threshold: float, max_results: int = 100) -> List[str]:
    """Find similar signals using FAISS index.

    Args:
        signal_type: Signal type to search
        embedding: Query embedding (normalized)
        similarity_threshold: Minimum cosine similarity
        max_results: Maximum results to return

    Returns:
        List of signal IDs above similarity threshold
    """
    if (not self.use_faiss or
        signal_type not in self.faiss_indexes or
        not self.faiss_id_maps[signal_type]):
        return []

    import numpy as np

    # Normalize query embedding
    query = np.array(embedding, dtype=np.float32)
    query = query / np.linalg.norm(query)

    # Search FAISS index
    # Since we use IndexFlatIP with normalized vectors, scores are cosine similarities
    k = min(max_results, len(self.faiss_id_maps[signal_type]))
    scores, indices = self.faiss_indexes[signal_type].search(
        query.reshape(1, -1), k
    )

    # Filter by threshold and map back to signal IDs
    similar_ids = []
    for score, idx in zip(scores[0], indices[0]):
        if score >= similarity_threshold:
            signal_id = self.faiss_id_maps[signal_type][idx]
            if signal_id is not None and signal_id in self.signals:
                similar_ids.append(signal_id)

    return similar_ids
```

**Then update deposit() to use FAISS:**

```python
def deposit(self, signal_type: str, content: str, ...) -> Optional[str]:
    with self._lock:
        recent_cutoff = time.time() - 300
        same_type = [s for s in self.signals.values()
                    if s.type == signal_type and s.timestamp > recent_cutoff]

        # Compute embedding if needed
        new_embedding = None
        if same_type and self.use_semantic_clustering and self.embedding_model:
            new_embedding = self.embedding_model.encode(content)

        # Use FAISS for similarity search (O(log n))
        if self.use_faiss and new_embedding is not None:
            similar_ids = self._find_similar_faiss(
                signal_type, new_embedding, self.diversity_threshold
            )
            # Check each similar signal
            for similar_id in similar_ids:
                existing = self.signals.get(similar_id)
                if existing and existing.timestamp > recent_cutoff:
                    # Already know similarity >= threshold from FAISS
                    existing.strength = min(1.0, existing.strength * 1.1)
                    existing.visits += 1
                    logger.info(f"Rejected duplicate {signal_type} via FAISS, amplified {existing.id}")
                    return None
        else:
            # Fallback: O(n) search (original implementation)
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
                    logger.info(f"Rejected duplicate {signal_type} (similarity: {similarity:.2f}), amplified {existing.id}")
                    return None

        # Create signal (existing code)
        signal_id = f"{signal_type}_{self._next_id:04d}"
        self._next_id += 1
        # ... rest of deposit logic ...

        # Store embedding
        if new_embedding is not None:
            self.signal_embeddings[signal_id] = new_embedding
            # Add to FAISS index
            self._add_to_faiss_index(signal_id, signal_type, new_embedding)

        return signal_id
```

**Update delete_signal() to remove from FAISS:**
```python
def delete_signal(self, signal_id: str) -> bool:
    with self._lock:
        if signal_id not in self.signals:
            return False

        signal = self.signals[signal_id]
        signal_type = signal.type

        # Delete from FAISS index
        self._remove_from_faiss_index(signal_id, signal_type)

        # ... rest of deletion ...
```

### Complexity Analysis

**Before (O(n) per deposit):**
- Filter same_type: O(n)
- Check each signal: O(n * d) where d = embedding_dim = 384
- Total per deposit: O(n)
- Total for swarm: O(m * n) where m = deposits

**After with FAISS (O(n) with SIMD per deposit):**
- FAISS search: IndexFlatIP is O(n) with SIMD/optimized linear algebra (10-100x faster constants)
- This is NOT truly O(log n), just much better constants
- IndexIVFFlat would be O(log n) but requires training

**Wait - am I wrong about FAISS complexity?**

Let me reconsider:
- **IndexFlatIP:** Brute force, O(n), but SIMD-optimized (faster constants)
- **IndexIVFFlat:** Inverted file index, O(log n) average, requires training
- **IndexHNSW:** Hierarchical NSW, O(log n), no training needed

**Better approach:** Use IndexHNSWFlat for true O(log n)

```python
def _initialize_faiss_index(self, signal_type: str, embedding_dim: int):
    """Initialize FAISS index for a signal type."""
    import faiss

    # Use HNSW for O(log n) search without training
    # M = number of connections (16 is good default)
    # efConstruction = quality of index (200 is good default)
    index = faiss.IndexHNSWFlat(embedding_dim, 16)
    index.hnsw.efSearch = 32  # Search quality (higher = more accurate)

    return index
```

### Risks

⚠️ **Risk 1:** FAISS not available
- **Mitigation:** Graceful fallback to O(n) search

⚠️ **Risk 2:** Index rebuild overhead
- **Mitigation:** Only rebuild when >30% deleted, amortized O(1)

⚠️ **Risk 3:** Memory overhead
- **Mitigation:** HNSW uses ~2x memory of vectors, acceptable trade-off

⚠️ **Risk 4:** Breaking quality (different results than before)
- **Mitigation:** Use exact search (IndexFlatIP) first, then switch to HNSW
- **Testing:** Verify duplicate detection still works

⚠️ **Risk 5:** Complexity explosion
- **Mitigation:** Start with IndexFlatIP (simple, faster than current)
- **Future:** Upgrade to HNSW if needed

---

## Implementation Order

### Phase 1: Low-Risk Fixes (Do First)
1. **Bare exception handlers** - Clear benefit, low risk
2. **Centralized delete_signal()** - Defensive, prevents future bugs
3. **Print → logging migration** - Clear benefit, systematic

### Phase 2: Medium-Risk Fixes (Do Second)
4. **Critic fallback path** - Small change, low impact

### Phase 3: High-Risk Optimization (Do Last, Test Heavily)
5. **FAISS optimization** - Complex, needs testing

### Rollback Plan
- Each fix in separate commit
- Test after each commit
- If issues found, revert specific commit

---

## Testing Strategy

### Unit Tests
- Test delete_signal() cleans up all data structures
- Test logging output (check log levels)
- Test exception handling (verify specific exceptions caught)
- Test Critic deposits CRITIQUE in all paths

### Integration Tests
- Run full swarm with FAISS enabled
- Compare results with/without FAISS (should be similar)
- Measure performance improvement
- Check memory usage (no leaks)

### Regression Tests
- Existing tests should still pass
- No change in final synthesis output quality
- Signal counts similar to before

---

## Performance Targets

### Memory
- **Before:** Unbounded growth (potential leak)
- **After:** Constant per signal (embedding cleaned up)
- **FAISS overhead:** 2x embedding size for HNSW

### Speed
- **Before:** O(n) similarity check = ~1000 comparisons for 1000 signals
- **After (IndexFlatIP):** O(n) but 10-100x faster (SIMD)
- **After (HNSW):** O(log n) = ~10 comparisons for 1000 signals

### Quality
- **Goal:** No regression in duplicate detection
- **Metric:** Same duplicate rejection rate ±2%

---

## Doubts to Resolve Before Implementation

### Critical Doubts

1. ❓ **Is there actually an embedding leak in current code?**
   - Need to verify by checking all deletion paths
   - May already be fixed
   - **Resolution:** Add centralized method anyway (defensive)

2. ❓ **Should I use IndexFlatIP (simple, O(n) but faster) or HNSW (complex, O(log n))?**
   - IndexFlatIP: Easy to implement, exact results, 10-100x faster than current
   - HNSW: Harder to implement, approximate results, true O(log n)
   - **Decision:** Start with IndexFlatIP, upgrade to HNSW if needed

3. ❓ **Will FAISS change duplicate detection behavior?**
   - FAISS uses cosine similarity (same as current)
   - But floating point differences might cause edge cases
   - **Resolution:** Test thoroughly, compare results

4. ❓ **How to handle backward compatibility?**
   - Old signal stores won't have FAISS indexes
   - **Resolution:** Lazy initialization on first use

### Minor Doubts

5. ❓ **Should logging migration happen all at once or file-by-file?**
   - All at once: Big PR, hard to review
   - File-by-file: Many small PRs, easier to review
   - **Decision:** All at once (automated script, systematic)

6. ❓ **What log level for each print statement?**
   - See "Log Level Mapping" above
   - When in doubt, use DEBUG

---

## Success Criteria

### Must Have ✅
1. No embedding memory leaks (verified by memory profiling)
2. All print() replaced with logger.*
3. No bare except: handlers
4. Critic deposits CRITIQUE signals in all code paths
5. FAISS integration works with fallback

### Should Have 📋
1. 10x performance improvement in similarity search
2. Zero regression in duplicate detection quality
3. All existing tests pass
4. Memory usage acceptable (<2x embeddings)

### Nice to Have 🎯
1. HNSW for O(log n) search
2. Comprehensive logging for debugging
3. Performance metrics tracked

---

## Conclusion

I've carefully analyzed each issue and have a detailed implementation plan. The fixes range from low-risk (logging, exceptions) to high-risk (FAISS optimization). I'll implement in phases, test thoroughly, and be ready to rollback if issues arise.

**Key insight:** Some issues (embedding leak, critic completeness) may already be partially fixed. I'll verify current state before making changes to avoid unnecessary churn.

**Biggest risk:** FAISS optimization could change behavior. Will implement conservatively with IndexFlatIP first.

**Ready to proceed:** Yes, with careful testing after each phase.
