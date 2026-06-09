# Task Adaptivity Evaluation & Knowledge Intake Scaling

**Date:** 2025-11-20
**Focus:** Evaluate current task adaptivity + Design massive knowledge intake improvements

---

## Part 1: Task Adaptivity Evaluation

### Current State - What Adapts to Task Type?

#### ✅ DOES Adapt (Prompt Templates Only)

**File:** `swarm/core/task_config.py`

| Task Type | Scout Prompt | Forager Prompt | Critic Prompt | Hater Prompt |
|-----------|--------------|----------------|---------------|--------------|
| **Debate** | "Generate a claim" | "Provide evidence" | "Critique claim" | "Counterargument" |
| **Creative** | "Generate creative draft" | "Develop draft" | "Refine draft" | "Alternative approach" |
| **Analysis** | "Generate observation" | "Develop analysis" | "Critique analysis" | "Challenge analysis" |
| **Problem Solving** | "Propose solution" | "Elaborate solution" | "Evaluate solution" | "Alternative solution" |

**Adaptation Level:** **Shallow (Language Only)**
- Changes **what agents say** (prompts, labels)
- Does NOT change **how agents work** (behavior, parameters, strategies)

---

#### ❌ DOES NOT Adapt (Everything Else)

**Knowledge Intake Parameters (FIXED):**
```python
# swarm/retrieval/advanced_retriever.py
target_words_per_round = 100000      # FIXED - same for all tasks
min_sources_per_keyword = 3          # FIXED - same for all tasks
chunk_size = 300                     # FIXED - same for all tasks
overlap = 50                         # FIXED - same for all tasks
```

**Agent Token Allocations (FIXED):**
```python
# Current allocations (after our improvement):
Scout: 150 tokens      # FIXED - same for all tasks
Forager: 100 tokens    # FIXED - same for all tasks
Critic: 150 tokens     # FIXED - same for all tasks
Hater: 120-150 tokens  # FIXED - same for all tasks
Synthesizer: 300-400   # FIXED - same for all tasks
```

**Agent Behavior (FIXED):**
- Number of iterations
- Sampling strategies
- Temperature settings
- Fragment assignment strategy
- Round structure

**Research Strategy (FIXED):**
- Source diversity (always 3 sources)
- Research depth (always 100K words)
- Fragment prioritization (always importance × rarity)
- Knowledge graph depth

---

### Problem: One-Size-Fits-All Approach

#### Example 1: Creative Writing Task

**Current behavior:**
- Ingests 100K words of research (overkill for poetry)
- Scouts use 150 tokens (may need less for haiku, more for novel)
- Same sampling strategy as technical analysis (wrong)

**What it SHOULD do:**
- Less research, more diversity (30K words, many varied sources)
- Variable tokens (50 for haiku, 200 for short story)
- Sample for novelty, not evidence
- Lightweight fact-checking (don't need Wikipedia for creative work)

#### Example 2: Technical Analysis Task

**Current behavior:**
- Ingests 100K words (might not be enough for deep technical topic)
- 3 sources per keyword (insufficient for technical accuracy)
- Same token allocation as creative task (wrong)

**What it SHOULD do:**
- Deep research (250K-500K words for complex topics)
- Many sources (5-10 per keyword for verification)
- Higher tokens for scouts (200+) to preserve technical details
- More rigorous fact-checking
- Source credibility weighting

#### Example 3: Debate Task

**Current behavior:**
- Balanced research (100K words)
- No pro/con separation in sources
- Same fragment assignment as other tasks

**What it SHOULD do:**
- Balanced pro/con research (fetch opposing viewpoints explicitly)
- Separate fragment streams for supporting/opposing evidence
- Higher hater temperature (more adversarial)
- Balanced source representation

---

### Task Adaptivity Requirements Matrix

| Task Type | Research Depth | Source Count | Scout Tokens | Synthesizer Tokens | Fact-Check Rigor | Fragment Strategy |
|-----------|----------------|--------------|--------------|--------------------|--------------------|-------------------|
| **Creative Writing** | Low (30-50K) | Many (breadth) | Variable (50-200) | High (300-400) | Low | Diversity sampling |
| **Technical Analysis** | High (250-500K) | Deep (5-10/kw) | High (200+) | Medium (200-250) | Very High | Depth + quality |
| **Problem Solving** | Medium (150K) | Balanced | Medium (150) | Medium (250) | Medium | Solution-oriented |
| **Debate** | High (200-300K) | Balanced pro/con | High (180) | Medium (250) | High | Pro/con separation |
| **Research Summary** | Very High (500K+) | Comprehensive | Very High (250) | Low (150) | Critical | Coverage maximization |

**Conclusion:** Current system has **0% task adaptivity** for intake parameters.

---

## Part 2: Knowledge Intake Scaling Analysis

### Current Limitations

#### Volume Constraints

**Current capacity:**
```python
target_words_per_round = 100000  # 100K words
min_sources_per_keyword = 3      # 3 sources
```

**For a typical task:**
- 5 keywords extracted
- 5 keywords × 3 sources = 15 sources max
- 100K words ≈ 200-300 pages of text
- **This is SMALL for serious research tasks**

**Comparison to human research:**
- PhD literature review: 500K-2M words
- Technical report: 200K-500K words
- News article: 10K-50K words
- Our system: **100K words (insufficient for deep tasks)**

#### Source Diversity Constraints

**Current sources:**
- Wikipedia (broad, shallow)
- DuckDuckGo (instant answers only)

**Missing sources:**
- arXiv (academic papers)
- PubMed (medical research)
- GitHub (code repositories)
- News aggregators
- Domain-specific databases
- Books (Google Books API)

#### Processing Constraints

**Current chunk size: 300 words**
- May be too small for dense technical content
- Loses context in long-form arguments
- Fragments complex ideas unnecessarily

**Fixed overlap: 50 words**
- Not adaptive to content type
- Mathematical proofs need more overlap
- Creative writing needs less

---

### Proposed Improvements

#### Improvement 1: Task-Adaptive Intake Profiles

**Create intake configuration per task type:**

```python
@dataclass
class IntakeProfile:
    """Knowledge intake profile for task adaptation."""

    # Research volume
    target_words_per_round: int      # 30K - 500K
    max_sources_per_keyword: int     # 3 - 15
    research_rounds: int              # 1 - 3

    # Content processing
    chunk_size: int                   # 200 - 800 words
    chunk_overlap: int                # 30 - 100 words

    # Token allocation
    scout_tokens: int                 # 50 - 250
    forager_tokens: int               # 80 - 150
    critic_tokens: int                # 100 - 200
    synthesizer_tokens: int           # 150 - 400

    # Source preferences
    source_types: List[str]           # ["wikipedia", "arxiv", "news"]
    source_credibility_weight: float  # 0.0 - 1.0

    # Fragment strategy
    fragment_assignment: str          # "round_robin", "clustered", "balanced"
    prioritization: str               # "importance", "rarity", "diversity"

    # Quality thresholds
    min_fragment_quality: float       # 0.3 - 0.8
    fact_check_threshold: float       # 0.5 - 0.9
```

**Example profiles:**

```python
CREATIVE_INTAKE = IntakeProfile(
    target_words_per_round=40000,     # Light research
    max_sources_per_keyword=8,        # Breadth over depth
    research_rounds=1,
    chunk_size=400,
    chunk_overlap=50,
    scout_tokens=120,                 # Medium variability
    synthesizer_tokens=350,           # High for creative synthesis
    source_types=["wikipedia", "news", "general"],
    fragment_assignment="diversity",
    prioritization="rarity",          # Favor unique findings
    fact_check_threshold=0.5          # Relaxed
)

TECHNICAL_INTAKE = IntakeProfile(
    target_words_per_round=300000,    # Deep research
    max_sources_per_keyword=10,       # Depth + verification
    research_rounds=2,                # Multi-round for refinement
    chunk_size=600,                   # Preserve technical context
    chunk_overlap=80,
    scout_tokens=220,                 # High for technical detail
    synthesizer_tokens=250,           # Medium (good signals = less synthesis)
    source_types=["wikipedia", "arxiv", "academic"],
    fragment_assignment="clustered",  # Thematic coherence
    prioritization="quality",         # Favor authoritative sources
    fact_check_threshold=0.8          # Rigorous
)

DEBATE_INTAKE = IntakeProfile(
    target_words_per_round=250000,    # Balanced depth
    max_sources_per_keyword=12,       # Pro + con sources
    research_rounds=2,
    chunk_size=500,
    chunk_overlap=60,
    scout_tokens=180,
    synthesizer_tokens=250,
    source_types=["wikipedia", "news", "opinion", "academic"],
    fragment_assignment="balanced_procon",  # NEW: Separate pro/con
    prioritization="balanced",        # Equal weight to opposing views
    fact_check_threshold=0.75         # High (avoid misinformation)
)

RESEARCH_INTAKE = IntakeProfile(
    target_words_per_round=500000,    # MAXIMUM depth
    max_sources_per_keyword=15,
    research_rounds=3,                # Multi-round refinement
    chunk_size=800,                   # Large chunks for academic papers
    chunk_overlap=100,
    scout_tokens=250,                 # Maximum detail preservation
    synthesizer_tokens=200,           # Low (signals are comprehensive)
    source_types=["wikipedia", "arxiv", "pubmed", "academic"],
    fragment_assignment="knowledge_graph",  # Graph-based prioritization
    prioritization="coverage",        # Maximize topic coverage
    fact_check_threshold=0.85         # Very rigorous
)
```

#### Improvement 2: Massive Intake Scaling (5x Increase)

**Target: 100K → 500K words per round**

**Changes needed:**

1. **Parallel source fetching**
   ```python
   # OLD: Sequential fetching
   for query in queries:
       result = await wikipedia.search(query)

   # NEW: Parallel fetching (5-10x faster)
   tasks = [wikipedia.search(q) for q in queries]
   results = await asyncio.gather(*tasks)
   ```

2. **More sources per keyword**
   ```python
   # OLD: 3 sources
   min_sources_per_keyword = 3

   # NEW: 5-15 sources (task-adaptive)
   max_sources_per_keyword = intake_profile.max_sources_per_keyword
   ```

3. **Multi-round deep research**
   ```python
   # OLD: Single round
   knowledge = await retriever.deep_research_round(keywords, round_num=0)

   # NEW: Multi-round refinement (2-3 rounds)
   for round_num in range(intake_profile.research_rounds):
       knowledge = await retriever.deep_research_round(
           keywords=keywords if round_num == 0 else refined_keywords,
           round_num=round_num,
           previous_synthesis=synthesis if round_num > 0 else ""
       )
       # Extract emerging topics from synthesis
       refined_keywords = extract_emerging_keywords(synthesis)
   ```

4. **Expanded source backends**
   ```python
   # NEW sources to add:
   - ArXivAPI (academic papers)
   - PubMedAPI (medical research)
   - GitHubAPI (code repositories)
   - NewsAPI (current events)
   - BooksAPI (Google Books)
   - ScholarAPI (Google Scholar)
   ```

#### Improvement 3: Intelligent Fragment Management

**Problem:** With 500K words → ~1000+ fragments, scouts can't process all

**Solution: Priority-based fragment selection**

```python
def select_top_fragments(fragments: List[ResearchFragment],
                        top_k: int,
                        strategy: str) -> List[ResearchFragment]:
    """Select top k fragments based on strategy."""

    if strategy == "importance":
        # Prioritize high-importance fragments
        scored = sorted(fragments,
                       key=lambda f: f.importance * (1 + f.rarity),
                       reverse=True)

    elif strategy == "diversity":
        # Maximize keyword diversity (creative tasks)
        scored = maximal_diversity_selection(fragments, top_k)

    elif strategy == "coverage":
        # Maximize topic coverage (research tasks)
        scored = coverage_maximization_selection(fragments, top_k)

    elif strategy == "knowledge_graph":
        # Prioritize high-connectivity fragments
        scored = graph_centrality_selection(fragments, top_k)

    return scored[:top_k]
```

**Fragment budget per scout:**
```python
# With 500K words and 10 scouts:
# 500K words → ~1000 fragments total
# Per scout: 100 fragments (was 10)
# With 150 tokens each: 15,000 tokens per scout (affordable)
```

#### Improvement 4: Knowledge Graph Enhancement

**Current:** Simple keyword → fragments mapping

**Proposed:** Rich knowledge graph with:

1. **Fragment-to-fragment edges**
   - Shared keywords (cosine similarity)
   - Citation relationships (if from academic papers)
   - Temporal ordering (time-series data)
   - Causal relationships (detected from text)

2. **Centrality-based prioritization**
   ```python
   # PageRank for fragments (find central concepts)
   centrality_scores = pagerank(knowledge_graph)

   # Prioritize high-centrality fragments
   critical_fragments = [f for f in fragments
                        if centrality_scores[f.id] > threshold]
   ```

3. **Multi-hop reasoning**
   ```python
   # Scout can traverse graph to build composite insights
   related_fragments = get_neighbors(current_fragment, max_hops=2)
   composite_context = combine_fragments(related_fragments)
   ```

---

## Part 3: Implementation Plan

### Phase 1: Task-Adaptive Configuration (High Priority)

**Estimated effort:** 4-6 hours

**Steps:**
1. Create `IntakeProfile` dataclass in `swarm/core/task_config.py`
2. Add `intake_profile` field to `TaskConfig`
3. Define profiles for each task type (creative, technical, debate, research)
4. Modify `AdvancedRetriever.__init__` to accept `intake_profile`
5. Update all intake parameters to read from profile

**Files to modify:**
- `swarm/core/task_config.py` (add IntakeProfile, define profiles)
- `swarm/retrieval/advanced_retriever.py` (accept profile, use parameters)
- `swarm/retrieval/knowledge_processor.py` (accept chunk_size, overlap from profile)
- `swarm/agents/scout.py` (read scout_tokens from profile)
- `run_task.py` (pass profile to retriever)

**Immediate benefit:** Different tasks get appropriate research depth/breadth

---

### Phase 2: Intake Volume Scaling (High Priority)

**Estimated effort:** 6-8 hours

**Steps:**
1. Implement parallel source fetching (asyncio.gather)
2. Increase default target_words: 100K → 250K (2.5x)
3. Add configurable max_sources_per_keyword
4. Implement fragment selection strategies (top-k by priority)
5. Add fragment budget management (limit per scout)

**Files to modify:**
- `swarm/retrieval/advanced_retriever.py` (parallel fetching, increased limits)
- `swarm/agents/scout.py` (fragment budget handling)
- `understanding_notes/` (document new limits)

**Immediate benefit:** 2.5-5x more knowledge ingested per task

---

### Phase 3: Source Expansion (Medium Priority)

**Estimated effort:** 8-12 hours

**Steps:**
1. Implement ArXivAPI backend (academic papers)
2. Implement NewsAPI backend (current events)
3. Implement ScholarAPI backend (Google Scholar)
4. Add source type filtering per profile
5. Implement source credibility scoring

**Files to create:**
- `swarm/retrieval/arxiv_api.py`
- `swarm/retrieval/news_api.py`
- `swarm/retrieval/scholar_api.py`

**Files to modify:**
- `swarm/retrieval/advanced_retriever.py` (integrate new sources)

**Immediate benefit:** Academic/technical tasks get proper sources

---

### Phase 4: Knowledge Graph Enhancement (Lower Priority)

**Estimated effort:** 10-15 hours

**Steps:**
1. Implement fragment similarity calculation
2. Build adjacency matrix from fragments
3. Implement PageRank/centrality calculation
4. Add graph-based fragment prioritization
5. Enable multi-hop fragment traversal for scouts

**Files to create:**
- `swarm/retrieval/knowledge_graph.py`

**Files to modify:**
- `swarm/retrieval/advanced_retriever.py` (build graph, use for prioritization)
- `swarm/agents/scout.py` (traverse graph for composite context)

**Immediate benefit:** Better fragment selection, composite insights

---

### Phase 5: Multi-Round Research (Lower Priority)

**Estimated effort:** 4-6 hours

**Steps:**
1. Add research_rounds parameter to IntakeProfile
2. Implement round-based keyword refinement
3. Extract emerging topics from synthesis
4. Run subsequent rounds with refined keywords
5. Merge knowledge from all rounds

**Files to modify:**
- `swarm/retrieval/advanced_retriever.py` (multi-round loop)
- `run_task.py` (orchestrate multiple rounds)

**Immediate benefit:** Deeper, more focused research

---

## Part 4: Risk Assessment

### Risk 1: Token Cost Explosion

**Risk:** 500K words → 1000 fragments × 150 tokens = 150K tokens per scout

**Mitigation:**
- Implement fragment selection (only process top 100 per scout)
- Adaptive token allocation (not all fragments need 150 tokens)
- Budget cap per scout (e.g., max 20K tokens)

**Status:** ⚠️ Manageable with selection

### Risk 2: Processing Time

**Risk:** 500K words takes much longer to fetch/process

**Mitigation:**
- Parallel fetching (5-10x speedup)
- Caching (don't re-fetch same sources)
- Progressive processing (start agents while still fetching)
- Timeout limits per source

**Status:** ✅ Parallel fetching solves this

### Risk 3: Quality Dilution

**Risk:** More research != better results if quality is low

**Mitigation:**
- Source credibility scoring
- Fragment quality filtering (min_quality threshold)
- Prioritize authoritative sources
- Fact-checking integration

**Status:** ✅ Quality filters prevent this

### Risk 4: Configuration Complexity

**Risk:** Too many parameters to tune

**Mitigation:**
- Provide sensible defaults per task type
- User doesn't need to configure (profiles handle it)
- Document each parameter clearly
- Provide examples

**Status:** ✅ Profiles abstract complexity

---

## Part 5: Expected Outcomes

### Quantitative Improvements

| Metric | Current | Phase 1 | Phase 2 | Phase 3 | Phase 4 |
|--------|---------|---------|---------|---------|---------|
| **Words per round** | 100K | 100K (adaptive) | 250-500K | 250-500K | 250-500K |
| **Sources per keyword** | 3 | 3-15 (adaptive) | 3-15 | 5-20 | 5-20 |
| **Source types** | 2 | 2 | 2 | 5+ | 5+ |
| **Fragment quality** | Mixed | Filtered | Filtered | Credibility-scored | Graph-ranked |
| **Task adaptivity** | 0% | 80% | 80% | 90% | 95% |

### Qualitative Improvements

**After Phase 1 (Task Profiles):**
- ✅ Creative tasks get light, diverse research
- ✅ Technical tasks get deep, authoritative research
- ✅ Debate tasks get balanced pro/con sources
- ✅ Research tasks get comprehensive coverage

**After Phase 2 (5x Scaling):**
- ✅ Serious tasks get adequate knowledge depth
- ✅ Complex topics fully explored
- ✅ Rare/niche findings discovered
- ✅ Better composite insights

**After Phase 3 (Source Expansion):**
- ✅ Academic tasks access scholarly papers
- ✅ Current events use real-time news
- ✅ Technical analysis verified across sources
- ✅ Source diversity increases reliability

**After Phase 4 (Knowledge Graph):**
- ✅ Central concepts automatically identified
- ✅ Fragment selection optimized
- ✅ Multi-hop reasoning enabled
- ✅ Knowledge connections discovered

---

## Part 6: Immediate Action Items

### Must Implement Now (Critical)

1. **Create IntakeProfile class**
   - Define structure
   - Create 4-5 profiles (creative, technical, debate, research, default)
   - Add to task_config.py

2. **Scale intake to 250K words**
   - Change target_words_per_round default
   - Implement parallel fetching
   - Add fragment selection (top 100 per scout)

3. **Make token allocation adaptive**
   - Read scout_tokens from profile
   - Update scout.py to use profile.scout_tokens
   - Update other agents similarly

### Should Implement Soon (Important)

4. **Add source type configuration**
   - Source type preferences per profile
   - Graceful fallback if source unavailable

5. **Implement fragment prioritization**
   - Top-k selection strategies
   - Quality filtering

### Can Defer (Nice to Have)

6. **Add ArXiv/Scholar sources**
7. **Build knowledge graph**
8. **Multi-round research**

---

## Conclusion

**Current state:** 0% task adaptivity, limited intake (100K words)

**After Phase 1+2:** 80% task adaptivity, 2.5-5x intake (250-500K words)

**Effort:** 10-14 hours for Phase 1+2

**ROI:** Massive - different tasks get appropriate research, complex tasks fully explored

**Recommendation:** Implement Phase 1+2 immediately. Phase 3-5 can be added incrementally based on user needs.
