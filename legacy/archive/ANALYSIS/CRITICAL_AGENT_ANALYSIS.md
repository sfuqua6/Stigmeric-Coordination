# Critical Analysis: Agent Interaction Logic Issues

**Date:** 2025-11-13
**Focus:** Deep architectural problems in agent coordination and interaction
**Reference:** CLAUDE_CODE_WEB_PROMPT.md sections on "AREAS NEEDING REFINEMENT"

---

## 🔴 CRITICAL ARCHITECTURAL ISSUES

### Issue 1: Hater Agents Are Completely Broken in Document Mode ⚠️ CRITICAL

**Location:** `swarm/agents/hater.py:37-45`, `swarm/monolith_breaking.py:245-266`

**The Problem:**
```python
# hater.py line 39-40
claim_signals = signal_store.sample_weighted("CLAIM", n=2)
evidence_signals = signal_store.sample_weighted("EVIDENCE", n=2)
```

**Why This Is Broken:**
- Haters search for "CLAIM" and "EVIDENCE" signal types
- **Document mode uses "OBSERVATION" and "INSIGHT" types**
- Haters will NEVER find any targets → infinite loop doing nothing
- Wastes CPU cycles for entire validation phase

**Worse:**
```python
# monolith_breaking.py line 253-266
haters = [
    Hater(f"Hater_{i}", task_prompt="Challenge insights")
    for i in range(num_haters)
]
# ...
# Launch all validation agents
validation_tasks = []
for gatherer in gatherers:
    validation_tasks.append(gatherer.run(...))
for critic in critics:
    validation_tasks.append(critic.run(...))

# Haters might need the original run signature
# For now, skip haters or implement a simple wrapper
# ^^^ HATERS ARE CREATED BUT NEVER RUN!
```

**Impact:**
- Haters are instantiated but tasks never launched
- No contrarian challenges happen
- Echo chamber effect (identified in CLAUDE_CODE_WEB_PROMPT.md:156-157)
- System loses adversarial validation entirely

**Fix Required:**
```python
# Option 1: Update Hater to work with document mode
async def run(self, signal_store: SignalStore, llm: SimpleLLM, ...):
    while self.active and self.actions_taken < max_actions:
        # Sample INSIGHT signals (not CLAIM)
        insights = signal_store.sample_weighted("INSIGHT", n=3)

        if insights:
            # Pick strongest to challenge
            target = max(insights, key=lambda s: s.strength)
            # Generate counter-evidence
            # Deposit as COUNTER or OBJECTION type

# Option 2: Actually run the haters in monolith_breaking.py
for hater in haters:
    validation_tasks.append(
        hater.run(signal_store, llm, max_actions=validation_iterations)
    )
```

---

### Issue 2: Forager Provenance Tracking is Broken ⚠️ CRITICAL

**Location:** `swarm/agents/forager.py:94-103`, `swarm/core/signal_store.py:444-477`

**The Problem:**
```python
# forager.py line 94-103
metadata = {
    'observation_count': len(cluster),
    'observation_ids': observation_ids,  # List of ALL observations
    'source_documents': source_docs
}

signal_id = signal_store.deposit(
    signal_type="INSIGHT",
    parent=cluster[0].id,  # ❌ ONLY FIRST OBSERVATION IS PARENT
    metadata=metadata
)
```

**Why This Breaks:**
```python
# signal_store.py get_validation_status() line 365
observation_ancestors = self.get_ancestors(signal_id, target_type="OBSERVATION")
observation_count = len(observation_ancestors)
```

**The Issue:**
- Forager creates insight from 5 observations
- Sets only `cluster[0]` as parent
- Puts other 4 in metadata (not in parent chain)
- `get_ancestors()` only traverses parent links
- Reports insight is based on **1 observation** when it's actually based on **5**
- Critics undervalue the insight
- Validation score is artificially low

**Impact on Quality:**
- Insights from multiple observations appear weak
- Critics decay them instead of amplifying
- Best insights (cross-document patterns) get suppressed
- System converges to shallow single-source insights

**Fix Required:**
```python
# Option 1: Multi-parent support in Signal dataclass
@dataclass
class Signal:
    # ...
    parent: Optional[str] = None
    parents: List[str] = field(default_factory=list)  # NEW: All parents

# Option 2: Deposit multiple provenance links as separate signals
for obs in cluster:
    signal_store.deposit(
        signal_type="PROVENANCE",
        content=f"Insight {insight_id} derived from {obs.id}",
        strength=0.1,  # Low strength, just metadata
        parent=obs.id,
        metadata={'insight_id': insight_id}
    )

# Option 3: Fix get_validation_status to check metadata
def get_validation_status(self, signal_id):
    # Check metadata for observation_ids as fallback
    signal = self.signals[signal_id]
    observation_ids = signal.metadata.get('observation_ids', [])
    observation_count = len(observation_ids)
    # ...
```

---

### Issue 3: No Priority System for Validation ⚠️ HIGH

**Location:** `swarm/agents/gatherer.py:54-68`, Reference: CLAUDE_CODE_WEB_PROMPT.md:1364-1370

**The Problem:**
```python
# gatherer.py line 54-58
unvalidated = signal_store.get_unvalidated_signals(
    signal_type="INSIGHT",
    min_evidence=2
)
# Process a small batch to avoid rate limiting
batch = unvalidated[:batch_size]  # ❌ JUST TAKES FIRST N
```

**Why This Is Wasteful:**
- All insights treated equally
- Weak insights (1 observation, low strength) get validated same as strong ones
- High-potential insights (5+ observations, cross-document) wait in queue
- Gatherers waste API calls on low-value insights

**From CLAUDE_CODE_WEB_PROMPT.md:**
> "No priority queue for which insights need validation most" (line 1367)
> "Priority validation queue (validate high-potential insights first)" (line 1369)

**Fix Required:**
```python
def get_unvalidated_signals_prioritized(
    self,
    signal_type: str,
    min_evidence: int = 2
) -> List[Signal]:
    """Get unvalidated signals prioritized by potential value."""
    unvalidated = []

    for signal in self.signals.values():
        if signal.type != signal_type:
            continue

        evidence_count = sum(
            1 for s in self.signals.values()
            if s.parent == signal.id and s.type == "EVIDENCE"
        )

        if evidence_count < min_evidence:
            # Calculate priority score
            obs_count = len(signal.metadata.get('observation_ids', []))
            source_count = len(signal.metadata.get('source_documents', []))

            priority = (
                signal.strength * 0.4 +           # Current strength
                obs_count / 10.0 * 0.3 +          # Observation count
                source_count / 5.0 * 0.2 +        # Document diversity
                (1.0 / (signal.visits + 1)) * 0.1 # Novelty
            )

            unvalidated.append((priority, signal))

    # Sort by priority descending
    unvalidated.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in unvalidated]
```

---

### Issue 4: Forager Clustering is Too Primitive ⚠️ HIGH

**Location:** `swarm/core/signal_store.py:289-328`, Reference: CLAUDE_CODE_WEB_PROMPT.md:1344-1357

**The Problem:**
```python
# signal_store.py line 311-327
seed = random.choice(candidates)  # ❌ RANDOM SEED
cluster = [seed]

# Find similar signals
for candidate in candidates:
    if candidate.id != seed.id:
        sim = self._check_similarity(seed.content, candidate.content)
        if sim >= similarity_threshold:
            similarities.append((candidate, sim))

# _check_similarity uses SequenceMatcher - string overlap only!
def _check_similarity(self, content1: str, content2: str) -> float:
    return SequenceMatcher(None, content1.lower(), content2.lower()).ratio()
```

**From CLAUDE_CODE_WEB_PROMPT.md:**
> "Cluster sampling is based on embeddings similarity only" (line 1346)
> "No semantic understanding of what makes a 'good' cluster" (line 1347)
> "May cluster unrelated observations that happen to use similar words" (line 1348)
> "Misses cross-domain patterns that use different terminology" (line 1349)

**Why This Fails:**
```
Observation A: "Global temperatures increased 1.5°C since 1880"
Observation B: "Arctic ice coverage decreased 40% since 1980"
Observation C: "Ocean acidification pH dropped 0.1 units"

String similarity: A-B = 0.15, A-C = 0.12, B-C = 0.18
All below 0.4 threshold → Never clustered!
Semantic relationship: ALL about climate change → Should be clustered!
```

**Impact:**
- Misses cross-domain patterns (the CORE VALUE of the system!)
- Clusters surface-level word matches instead of semantic relationships
- Pattern discovery is shallow
- System reduces to "find documents with similar words" not "find insights"

**Fix Required:**
```python
class SignalStore:
    def __init__(self, ...):
        # Add embedding model
        from sentence_transformers import SentenceTransformer
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.signal_embeddings = {}  # Cache embeddings

    def deposit(self, ...):
        signal_id = ...
        signal = Signal(...)
        self.signals[signal_id] = signal

        # Generate embedding for semantic similarity
        embedding = self.embedding_model.encode(content)
        self.signal_embeddings[signal_id] = embedding

        return signal_id

    def sample_cluster_semantic(
        self,
        signal_type: str,
        size: int = 5,
        similarity_threshold: float = 0.6  # Cosine similarity
    ) -> List[Signal]:
        """Sample cluster using semantic embeddings."""
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        candidates = [s for s in self.signals.values() if s.type == signal_type]
        if len(candidates) < 2:
            return candidates

        # Pick seed (could be random or high-strength)
        seed = random.choice(candidates)
        cluster = [seed]
        seed_embedding = self.signal_embeddings[seed.id].reshape(1, -1)

        # Find semantically similar
        for candidate in candidates:
            if candidate.id == seed.id:
                continue

            cand_embedding = self.signal_embeddings[candidate.id].reshape(1, -1)
            similarity = cosine_similarity(seed_embedding, cand_embedding)[0][0]

            if similarity >= similarity_threshold:
                cluster.append(candidate)

        return cluster[:size]
```

---

### Issue 5: Critics Don't Actually Validate Quality ⚠️ MEDIUM

**Location:** `swarm/agents/critic.py:68-82`, Reference: CLAUDE_CODE_WEB_PROMPT.md:1372-1385

**The Problem:**
```python
# critic.py line 68-82
# Calculate strength multiplier based on validation
multiplier = self.calculate_multiplier(validation)

if multiplier > 1.0:
    # Well-validated: Amplify
    signal_store.amplify(insight.id, factor=multiplier)
```

```python
# calculate_multiplier just counts things
def calculate_multiplier(self, validation: dict) -> float:
    score = validation['validation_score']
    evidence_count = validation['evidence_count']

    # Strong validation = amplification
    if score >= 0.7:
        multiplier = 1.3
    elif score >= 0.5:
        multiplier = 1.15
    # ...
```

**What's Missing:**
- No LLM-based quality check of the evidence
- No verification that evidence actually supports the insight
- No check for contradictions in evidence
- Just counting: 2 evidence = good, 0 evidence = bad

**From CLAUDE_CODE_WEB_PROMPT.md:**
> "Shallow validation (just checks if *something* supports it)" (line 1374)
> "No contradiction detection from external sources" (line 1375)
> "No quality assessment of evidence sources" (line 1377)

**Example Failure:**
```
Insight: "Climate change is accelerating ocean acidification"
Evidence 1: "Ocean pH has decreased" ✓ Relevant
Evidence 2: "Ice cream sales are up" ✗ Irrelevant
Critic sees: 2 evidence → multiply strength by 1.3
Reality: Only 1 relevant evidence → should multiply by 1.05
```

**Fix Required:**
```python
async def evaluate_insights(self, signal_store: SignalStore, llm: SimpleLLM):
    insights = signal_store.sample_weighted("INSIGHT", n=3)

    for insight in insights:
        validation = signal_store.get_validation_status(insight.id)

        # Get actual evidence signals
        evidence_signals = signal_store.get_descendants(
            insight.id,
            target_type="EVIDENCE"
        )

        if evidence_signals:
            # Use LLM to check evidence quality
            relevance_score = await self._assess_evidence_quality(
                insight,
                evidence_signals,
                llm
            )

            # Adjust multiplier based on actual quality
            multiplier = self._calculate_quality_multiplier(
                validation,
                relevance_score
            )
        else:
            multiplier = self.calculate_multiplier(validation)

        # Apply multiplier...

async def _assess_evidence_quality(
    self,
    insight: Signal,
    evidence_signals: List[Signal],
    llm: SimpleLLM
) -> float:
    """Use LLM to assess how well evidence supports insight."""
    prompt = f"""Assess how well this evidence supports the insight.

Insight: {insight.content}

Evidence:
{chr(10).join(f"{i+1}. {e.content}" for i, e in enumerate(evidence_signals))}

Rate each evidence item:
- 2 = Strong support, directly relevant
- 1 = Weak support, tangentially related
- 0 = No support, irrelevant or contradictory

Output format: [score1, score2, ...]"""

    result = await llm.generate(prompt, max_tokens=50, temperature=0.3)

    # Parse scores and calculate average
    # ...
    return average_score
```

---

### Issue 6: Synthesizer Has Insufficient Context ⚠️ MEDIUM

**Location:** `swarm/agents/synthesizer.py:66-84`, `swarm/monolith_breaking.py:333-354`

**The Problem:**
```python
# monolith_breaking.py line 333-354
top_insights = signal_store.get_top_signals("INSIGHT", n=10)

synthesizer = Synthesizer(...)
signal_context = {'INSIGHT': top_insights}

synthesis = await synthesizer.synthesize(
    signal_store,
    llm,
    {'insights': 'INSIGHT'},  # Only insights, no observations/evidence
    temperature=0.6
)
```

```python
# synthesizer.py line 44-49
for signal_type, signals in signal_context.items():
    if signals:
        prompt += f"\n{signal_type}s:\n"
        for i, signal in enumerate(signals[:3], 1):
            preview = signal.content[:150]  # ❌ TRUNCATED!
            prompt += f"  {i}. {preview}...\n"
```

**What's Wrong:**
- Synthesizer only sees top 10 insights
- Content truncated to 150 chars
- No access to supporting observations
- No access to validation evidence
- Can't trace back to source documents
- Loses nuance and detail

**From CLAUDE_CODE_WEB_PROMPT.md:**
> "Synthesizer receives top 10 insights, generates narrative" (line 1402)
> "May lose important details from lower-ranked insights" (line 1404)
> "No fact-checking against source observations" (line 1406)
> "Limited context (only sees insights, not underlying observations)" (line 1407)

**Fix Required:**
```python
# monolith_breaking.py
top_insights = signal_store.get_top_signals("INSIGHT", n=10)

# Enrich with supporting context
enriched_insights = []
for insight in top_insights:
    # Get supporting observations
    observations = signal_store.get_ancestors(
        insight.id,
        target_type="OBSERVATION"
    )

    # Get validation evidence
    evidence = signal_store.get_descendants(
        insight.id,
        target_type="EVIDENCE"
    )

    enriched_insights.append({
        'insight': insight,
        'observations': observations[:5],  # Top 5 observations
        'evidence': evidence[:3]  # Top 3 evidence
    })

synthesis = await synthesizer.synthesize_enriched(
    signal_store,
    llm,
    enriched_insights,
    temperature=0.6
)
```

```python
# synthesizer.py
async def synthesize_enriched(
    self,
    signal_store: SignalStore,
    llm: SimpleLLM,
    enriched_insights: List[dict],
    temperature: float = 0.6
) -> str:
    """Generate synthesis with full context."""

    prompt = f"""Synthesize insights with supporting evidence.

Question: {self.task_prompt}

"""

    for i, item in enumerate(enriched_insights, 1):
        insight = item['insight']
        obs = item['observations']
        evidence = item['evidence']

        prompt += f"\nInsight {i}: {insight.content}\n"
        prompt += f"  Based on {len(obs)} observations:\n"
        for j, o in enumerate(obs[:3], 1):
            prompt += f"    {j}. {o.content[:100]}...\n"

        if evidence:
            prompt += f"  Validated by {len(evidence)} external sources:\n"
            for j, e in enumerate(evidence[:2], 1):
                prompt += f"    {j}. {e.content[:80]}...\n"

    prompt += "\nSynthesize into coherent answer..."

    # Generate with more context
    return await llm.generate(prompt, max_tokens=400, temperature=temperature)
```

---

### Issue 7: No Deduplication Strategy for Foragers ⚠️ MEDIUM

**Location:** `swarm/agents/forager.py:44-47`, Reference: CLAUDE_CODE_WEB_PROMPT.md:1420-1423

**The Problem:**
```python
# forager.py line 44-47
while self.active and self.actions_taken < max_actions:
    await self.find_patterns(signal_store, llm, cluster_size, similarity_threshold)
    self.actions_taken += 1
    await asyncio.sleep(random.uniform(0.3, 0.8))
```

**Why This Wastes Resources:**
- 50 foragers run in parallel
- Each samples clusters randomly
- High probability of overlap:
  - Forager_1 samples [obs1, obs2, obs3, obs4, obs5]
  - Forager_2 samples [obs1, obs2, obs3, obs6, obs7]
  - 60% overlap → likely to generate similar insights
- SignalStore rejects at deposit, but LLM call already wasted

**From CLAUDE_CODE_WEB_PROMPT.md:**
> "No explicit coordination between agents" (line 1420)
> "May duplicate work (multiple agents process same signals)" (line 1421)

**Math:**
```
50 foragers × 150 iterations = 7,500 pattern-finding attempts
Each attempt: sample cluster (5 obs) + LLM call (200 tokens)
If 30% are duplicates rejected: 2,250 wasted LLM calls
At ~0.5s per call: 18.75 minutes wasted
```

**Fix Required:**
```python
class SignalStore:
    def __init__(self, ...):
        self.cluster_lock = asyncio.Lock()
        self.recently_sampled_clusters = {}  # Hash → timestamp

    async def sample_cluster_coordinated(
        self,
        signal_type: str,
        size: int,
        similarity_threshold: float,
        agent_id: str
    ) -> Optional[List[Signal]]:
        """Sample cluster with deduplication."""

        async with self.cluster_lock:
            # Try up to 5 times to find unique cluster
            for attempt in range(5):
                cluster = self.sample_cluster(
                    signal_type, size, similarity_threshold
                )

                # Hash the cluster
                cluster_hash = hash(tuple(sorted(s.id for s in cluster)))

                # Check if recently sampled (within last 60 seconds)
                if cluster_hash in self.recently_sampled_clusters:
                    last_time = self.recently_sampled_clusters[cluster_hash]
                    if time.time() - last_time < 60:
                        continue  # Try again

                # Mark as sampled
                self.recently_sampled_clusters[cluster_hash] = time.time()

                # Clean up old entries
                cutoff = time.time() - 60
                self.recently_sampled_clusters = {
                    h: t for h, t in self.recently_sampled_clusters.items()
                    if t > cutoff
                }

                return cluster

            # After 5 attempts, allow duplicate
            return cluster

# forager.py
async def find_patterns(self, signal_store, llm, cluster_size, similarity_threshold):
    cluster = await signal_store.sample_cluster_coordinated(
        signal_type="OBSERVATION",
        size=cluster_size,
        similarity_threshold=similarity_threshold,
        agent_id=self.agent_id
    )

    if cluster is None or len(cluster) < 2:
        return

    # Continue with pattern finding...
```

---

### Issue 8: No Fact-Checking in Synthesis ⚠️ MEDIUM

**Location:** `swarm/agents/synthesizer.py:90-98`, Reference: CLAUDE_CODE_WEB_PROMPT.md:1405-1414

**The Problem:**
```python
# synthesizer.py line 90-98
result = await llm.generate(prompt, max_tokens=200, temperature=temperature)

if result and len(result.strip()) > 20:
    synthesis = result.strip()
    return synthesis  # ❌ NO VERIFICATION
else:
    return None
```

**What Could Go Wrong:**
- LLM hallucinates connections not in insights
- Synthesis contradicts observations
- Made-up statistics or facts
- Reversed causality

**From CLAUDE_CODE_WEB_PROMPT.md:**
> "No fact-checking against source observations" (line 1406)
> "Could hallucinate connections" (line 1408)
> "Fact verification pass (check synthesis against observations)" (line 1411)

**Example:**
```
Insight 1: "Temperature increased 1.5°C"
Insight 2: "Arctic ice decreased 40%"
Synthesis: "Temperature increased 2.5°C causing 60% ice loss"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           Hallucinated numbers!
```

**Fix Required:**
```python
async def synthesize(self, signal_store, llm, signal_types, temperature=0.6):
    # Generate initial synthesis
    synthesis_draft = await self._generate_synthesis(...)

    # Verify against source observations
    verification_passed = await self._verify_synthesis(
        synthesis_draft,
        signal_store,
        llm
    )

    if not verification_passed:
        # Regenerate with stricter prompt
        synthesis_final = await self._generate_synthesis(
            ...,
            stricter=True
        )
        return synthesis_final

    return synthesis_draft

async def _verify_synthesis(
    self,
    synthesis: str,
    signal_store: SignalStore,
    llm: SimpleLLM
) -> bool:
    """Check synthesis against observations for hallucinations."""

    # Get all observations
    observations = signal_store.get_signals_by_type("OBSERVATION")
    obs_text = "\n".join(f"- {o.content}" for o in observations[:20])

    prompt = f"""Check if this synthesis is faithful to the observations.

Synthesis:
{synthesis}

Source Observations:
{obs_text}

Are there any claims in the synthesis that contradict or are not supported by the observations?
Answer YES if hallucinations found, NO if faithful.
"""

    result = await llm.generate(prompt, max_tokens=10, temperature=0.1)

    return "NO" in result.upper()
```

---

## 🟡 MODERATE ISSUES

### Issue 9: Signal Graph Traversal is Inefficient

**Location:** `swarm/core/signal_store.py:444-513`

**Problem:** Every call to `get_ancestors()` or `get_descendants()` does BFS from scratch.

**Called By:**
- `get_validation_status()` - called by every critic on every iteration
- Critic runs 20 agents × 150 iterations = 3,000 BFS traversals
- Each traversal visits potentially hundreds of signals

**Fix:** Add memoization/caching:
```python
from functools import lru_cache

class SignalStore:
    def __init__(self, ...):
        self._ancestor_cache = {}
        self._descendant_cache = {}

    def deposit(self, ...):
        # ... existing code ...
        # Invalidate cache when new signal added
        self._ancestor_cache.clear()
        self._descendant_cache.clear()

    def get_ancestors(self, signal_id, target_type=None):
        cache_key = (signal_id, target_type)
        if cache_key in self._ancestor_cache:
            return self._ancestor_cache[cache_key]

        # Do traversal
        ancestors = self._compute_ancestors(signal_id, target_type)

        # Cache result
        self._ancestor_cache[cache_key] = ancestors
        return ancestors
```

---

### Issue 10: No Early Stopping for Convergence

**Location:** `swarm/monolith_breaking.py:188-208`, Reference: CLAUDE_CODE_WEB_PROMPT.md:1358-1371

**Problem:**
```python
# monolith_breaking.py
pattern_iterations = 150  # Fixed!
for iteration in range(150):
    # ... run foragers ...
```

**From CLAUDE_CODE_WEB_PROMPT.md:**
> "Fixed iteration counts regardless of convergence state" (line 1360)
> "No early stopping based on insight stability" (line 1361)
> "Implement early stopping when Gini plateaus" (line 1368)

**Fix:**
```python
# Track gini history
gini_history = []
plateau_threshold = 0.02  # Gini change < 2% = plateau
plateau_window = 20  # Check last 20 iterations

for iteration in range(max_iterations):
    # Run foragers...

    # Check convergence every 10 iterations
    if iteration % 10 == 0 and iteration > 0:
        insights = signal_store.get_signals_by_type("INSIGHT")
        if insights:
            strengths = [s.strength for s in insights]
            gini = calculate_gini_coefficient(strengths)
            gini_history.append(gini)

            # Check for plateau
            if len(gini_history) >= plateau_window:
                recent = gini_history[-plateau_window:]
                gini_change = max(recent) - min(recent)

                if gini_change < plateau_threshold:
                    print(f"[EARLY STOP] Gini plateaued at {gini:.3f}")
                    break
```

---

## 📊 SUMMARY TABLE

| Issue | Severity | Impact | Fix Complexity | Lines Affected |
|-------|----------|--------|----------------|----------------|
| #1: Haters broken | 🔴 Critical | No adversarial validation | Low | 50 |
| #2: Provenance broken | 🔴 Critical | Insights undervalued | Medium | 100 |
| #3: No validation priority | 🔴 High | Wasted API calls | Medium | 80 |
| #4: Primitive clustering | 🔴 High | Misses patterns | High | 150 |
| #5: No quality validation | 🟡 Medium | False positives | High | 120 |
| #6: Limited synthesis context | 🟡 Medium | Loses details | Low | 60 |
| #7: Forager duplication | 🟡 Medium | Wasted compute | Medium | 80 |
| #8: No fact-checking | 🟡 Medium | Hallucinations | Medium | 70 |
| #9: Inefficient traversal | 🟡 Low | Slow | Low | 40 |
| #10: No early stopping | 🟡 Low | Slow | Low | 30 |

**Total Lines to Fix:** ~780 lines across 6 files

---

## 🎯 RECOMMENDED FIX PRIORITY

### Phase 1 (Critical - Do Immediately):
1. ✅ Fix Hater signal types (CLAIM/EVIDENCE → INSIGHT)
2. ✅ Actually run Haters in validation loop
3. ✅ Fix provenance tracking (multi-parent or metadata check)
4. ✅ Add validation prioritization

### Phase 2 (High Value):
5. ✅ Implement semantic clustering with embeddings
6. ✅ Add forager deduplication
7. ✅ Enrich synthesizer context

### Phase 3 (Quality Improvements):
8. ✅ Add LLM-based evidence quality checking
9. ✅ Add synthesis fact-checking
10. ✅ Optimize graph traversal with caching

### Phase 4 (Performance):
11. ✅ Early stopping for convergence
12. ✅ Temporal awareness for signals

---

## 📝 NOTES

This analysis reveals the system has significant **interaction bugs** where agents:
- Search for wrong signal types
- Don't run at all
- Lose provenance information
- Duplicate work
- Don't validate quality

The fixes are straightforward but require careful testing to ensure:
- Backward compatibility with creative mode
- No performance regression
- Proper error handling

**Next Steps:** Implement Phase 1 fixes immediately, as they prevent core functionality.
