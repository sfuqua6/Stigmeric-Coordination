# Understanding: scout.py

**Date:** 2025-11-19
**Purpose:** Exploration agent that generates initial signals from research/creativity
**Confidence:** High (read full file, clear intent)
**Lines:** 366

---

## What It Does (One Sentence)

Scout agents explore territory (research fragments or web searches) and deposit initial observations/ideas as signals for other agents to build upon.

---

## Key Components

### Class: Scout
**Purpose:** Exploration agent with two modes:
1. **RAG mode** - Process pre-assigned research fragments (deep research)
2. **Fallback mode** - Extract keywords, web search, generate ideas

**State:**
- `agent_id` - Unique identifier
- `signal_type` - What to deposit ("DRAFT", "INITIAL", "OBSERVATION", etc.)
- `task_prompt` - Task context (legacy, should use task_config)
- `task_config` - Composition pattern for prompts ✓ (no monkey patching)
- `dynamic_retriever` - For keyword extraction and web search
- `assigned_fragments` - List of ResearchFragments (RAG mode)
- `fragment_index` - Current position in fragments
- `active` - Running flag
- `actions_taken` - Counter

### Main Methods

#### `run(signal_store, llm, min_strength, max_actions, web_search_fn)`
**Purpose:** Main loop - explore and deposit signals
- Calls `explore_creative()` to generate ideas
- Calls `assess_strength_creative()` to score ideas
- Deposits if strength >= min_strength
- Repeats up to max_actions times
- **No sleep** - pure event-driven

#### `explore_creative(llm, web_search_fn)`
**Purpose:** Generate one idea using two-tier priority:
1. **Priority 1**: Use assigned research fragments (if available)
   - Process one fragment per call
   - Increments fragment_index
   - Formats fragment as context
2. **Priority 2**: Keyword extraction + web search (legacy)
   - Extract keywords from task_prompt
   - Search and append to temp file
   - Read context snippet

**Returns:** Generated idea string or None

#### `assess_strength_creative(idea)`
**Purpose:** Heuristic strength scoring 0.0-1.0
**Factors:**
- Base: 0.35 + length/250 (up to 0.25)
- +0.12 if has numbers
- +0.15 if has specifics (study, data, research, etc.)
- +0.10 if has citations (IPCC, NASA, journal, etc.)
- +0.08 if has quantifiers (significant, critical, etc.)
- ±0.08 random exploration

**Range:** 0.0-1.0 (clamped)

#### `_make_prompt(search_context)`
**Purpose:** Generate LLM prompt with context
**Strategy:**
1. If task_config exists → use template (composition ✓)
2. Else fallback to inline prompts
3. Different prompts for:
   - Research fragments (rich, evidence-based)
   - Web search context (generic)
   - No context (pure creativity)

### Helper Function

#### `assign_research_to_scouts(scouts, research_fragments)`
**Purpose:** Divide research fragments among scouts (division of labor)
**Strategy:**
- Sort by importance × (1 + rarity)
- Round-robin assignment
- Logs assignments per scout

---

## Dependencies

### Imports
- `asyncio` - Async execution
- `random` - Strength randomization, sampling
- `typing.Optional, List` - Type hints
- `..core.signal_store.SignalStore` - Shared environment
- `..llm.simple_llm.SimpleLLM` - Language model
- `..core.config.TEMP_SCOUT` - Temperature (0.9 - high exploration)

### Imported By
- `run_task.py` - Main orchestrator
- Probably coordinators

---

## Data Flow

```
INPUT:
- assigned_fragments (ResearchFragments) OR
- task_prompt + web_search_fn

PROCESSING:
1. explore_creative()
   → Get context (fragment or web search)
   → Generate prompt via _make_prompt()
   → Call llm.generate(temp=0.9, max_tokens=70, use_cache=False)
   → Validate output (length, uniqueness)

2. assess_strength_creative()
   → Heuristic scoring based on content features

3. run()
   → If strength >= min_strength:
     signal_store.deposit(signal_type, content, strength, depositor)

OUTPUT:
- Signals deposited to signal_store
- Type: DRAFT/INITIAL/OBSERVATION (task-dependent)
```

---

## Signal Interaction

### Reads
- **None** - Scouts don't sample, they only deposit

### Writes
- **signal_type** (configurable: DRAFT, INITIAL, OBSERVATION, IDEA, THESIS)
- Parent: None (scouts create root signals)
- Strength: 0.0-1.0 (heuristic-based)

### Sampling Strategy
- **N/A** - Scouts don't sample

### Provenance
- No parent (root signals)
- Other agents will use scout signals as parents

---

## LLM Usage

### Calls
- `llm.generate(prompt, max_tokens=70, temperature=TEMP_SCOUT, use_cache=False)`

### Temperature
- `TEMP_SCOUT = 0.9` (high exploration)
- Imported from config

### Caching
- **Disabled** (`use_cache=False`)
- Reason: Want diversity, not repetition

### Prompt Construction
- Rich prompts for research fragments
- Generic prompts for web search
- Simple prompts for pure creativity

### Error Handling
- Try/except around llm.generate
- Prints error + traceback
- Returns None on failure

### Output Validation
- Min 15 chars, 2 words
- Max 30% repetition (unique word ratio)
- Checks length, rejects too short

---

## Task Awareness

### Uses task_config ✓
- Composition pattern (no monkey patching)
- Uses `task_config.scout_prompt_template` if available
- Fallback to inline prompts

### Task-Specific Behavior
- Different prompts per task mode (creative, debate, analysis, problem_solving)
- Signal types vary: DRAFT, THESIS, OBSERVATION, IDEA

### Mode/Thesis
- Doesn't use mode directly
- Uses task_prompt and task_config

---

## Potential Issues

### 1. Strength Heuristic is Simplistic [CONFIRMED]
**Evidence:** Lines 213-251 use keyword matching
**Problem:**
- "study" = +0.15 bonus even if nonsense
- Can be gamed with keyword stuffing
- No semantic understanding
**Severity:** Low-Medium (just a heuristic, not critical)
**Alternative:** Could use LLM-based scoring (but slower)

### 2. Fragment Assignment is Stateful [CONFIRMED]
**Evidence:** `fragment_index` increments
**Problem:**
- If scout runs multiple times, continues from where it left off
- Could exhaust fragments
- Not idempotent
**Severity:** Low (probably intentional - progressive intake)
**Risk:** Edge case if scout runs more times than fragments

### 3. No Rate Limiting for Web Search [HYPOTHESIS]
**Evidence:** Calls `web_search_fn` without delay
**Problem:** Could hit rate limits on search API
**Severity:** Unknown (depends on web_search_fn implementation)
**Validation needed:** Check web_search_fn for rate limiting

### 4. Print Statements Instead of Logging [CONFIRMED]
**Evidence:** Lines 62, 67, 73, 82, etc. - many print() calls
**Problem:**
- Can't disable in production
- Can't filter by level
- Mixed with output
**Severity:** Low (usability/debugging issue)
**Fix:** Replace with logging framework (on REALISTIC_NEXT_STEPS.md list)

### 5. Temperature Hardcoded in Config [OBSERVED]
**Evidence:** Imports TEMP_SCOUT from config (0.9)
**Problem:** Can't adjust per-scout
**Severity:** Very Low (probably fine)
**Note:** Could allow per-agent temperature override

---

## Questions Remaining

1. **How does web_search_fn work?**
   - What's the API?
   - Is there rate limiting?
   - Read: `swarm/retrieval/simple_web_search.py`

2. **How are ResearchFragments created?**
   - What's the structure?
   - Where do they come from?
   - Read: `swarm/retrieval/advanced_retriever.py`

3. **How is signal_type chosen?**
   - Who decides DRAFT vs. INITIAL vs. OBSERVATION?
   - Read: `swarm/core/task_config.py`

4. **What happens to deposited signals?**
   - Who samples them?
   - Read: `forager.py`, `critic.py`, `hater.py`

5. **Is 70 max_tokens enough?**
   - Claims are 1-2 sentences
   - Seems reasonable but should verify actual outputs
   - **Could profile:** Average tokens used

---

## Improvement Ideas

### 1. Add Structured Logging [PRIORITY: Medium]
**Description:** Replace print() with logging module
**Evidence:** 20+ print statements scattered throughout
**Risk:** Low (non-breaking change)
**Effort:** 30 minutes
**Validation:** Against REALISTIC_NEXT_STEPS.md criteria:
- ✅ Solves real problem? Yes - debugging, production deployment
- ✅ Measurable benefit? Yes - can disable/filter logs
- ✅ Simplest solution? Yes - stdlib logging
- ❓ Worth effort? Debatable - scout.py alone not enough, should do all files

**Decision:** DEFER - Do all files at once, not piecemeal

---

### 2. Consider LLM-Based Strength Scoring [PRIORITY: Low]
**Description:** Use LLM to score quality instead of keywords
**Evidence:** Current heuristic is simplistic
**Risk:** Medium (slower, more LLM calls)
**Effort:** 2-3 hours
**Validation:**
- ❌ Solves real problem? UNPROVEN - is keyword scoring actually bad?
- ❌ Measurable benefit? Would need A/B test
- ❌ Simplest solution? No - adds complexity
- 🚩 Optimizing without profiling? YES - RED FLAG

**Decision:** DO NOT DO - No evidence keyword scoring is problem

---

### 3. Make Temperature Configurable Per-Agent [PRIORITY: Very Low]
**Description:** Allow temperature override in __init__
**Evidence:** Currently uses global TEMP_SCOUT
**Risk:** Low
**Effort:** 15 minutes
**Validation:**
- ❌ Solves real problem? NO - no one asked for this
- ❌ Measurable benefit? Unknown
- 🚩 "Might be useful later"? YES - YAGNI violation

**Decision:** DO NOT DO - No proven need

---

## Code Quality Observations

### Good ✓
1. **Clear structure** - Methods have single purposes
2. **Good docstrings** - Explain purpose and args
3. **Error handling** - Try/except around LLM calls
4. **Composition pattern** - Uses task_config, no monkey patching
5. **Type hints** - Parameters and returns typed
6. **No blocking** - Uses await, not time.sleep

### Could Improve
1. **Logging** - print() instead of logging module
2. **Magic numbers** - 70 tokens, 0.9 temp, etc. could be constants
3. **Long method** - explore_creative() is 100+ lines (still readable though)
4. **Comments** - Some complex sections could use "why" comments

### Surprising
1. **No caching** - Intentionally disabled for diversity (good!)
2. **High temperature** - 0.9 is very creative (intentional)
3. **Short outputs** - 70 tokens = ~50 words (1-2 sentences as intended)
4. **Stateful fragments** - Progressive intake, not random sampling

---

## Architectural Fit

### Stigmergic Pattern ✓
- Deposits signals (pheromones) ✓
- Doesn't directly communicate with other agents ✓
- Environment (signal_store) mediates ✓

### Division of Labor ✓
- `assign_research_to_scouts()` divides territory ✓
- Each scout processes different fragments ✓
- Like biological scouts exploring different areas ✓

### Event-Driven ✓
- Doesn't poll or sleep ✓
- Pure async/await ✓
- Signal deposition triggers events in signal_store ✓

---

## Notes

### RAG Integration (Phase 5)
The file has extensive comments about "PROPER RAG INTEGRATION":
- Priority 1: Process assigned research fragments
- Priority 2: Fall back to keyword + web search
- This is recent addition (comments suggest iteration)

### Backward Compatibility
Maintains legacy behavior:
- Inline prompts if no task_config
- Keyword extraction if no fragments
- Fallback chains ensure nothing breaks

### Performance Characteristics (Guesses, Need Profiling)
- LLM calls: 1 per explore_creative call
- Max tokens: 70 (fast)
- Temperature: 0.9 (no cache hit benefit)
- Estimated: ~0.5-1s per idea generation (GUESS - needs measurement)

### Testing Gaps (Observed)
- No unit tests found in reading
- Could test:
  - Strength assessment with known inputs
  - Prompt construction with different configs
  - Fragment assignment logic
  - Error handling paths

---

## Action Items from This Reading

### Must Read Next
1. **forager.py** - What happens to scout signals?
2. **task_config.py** - How are signal types decided?
3. **signal_store.py deposit()** - How does deposition work?

### Validate Hypotheses
1. **Web search rate limiting** - Check simple_web_search.py
2. **Fragment structure** - Check advanced_retriever.py
3. **Actual token usage** - Could add instrumentation

### Document Findings
1. ✓ Scout is purely exploratory (no sampling)
2. ✓ Uses composition pattern correctly
3. ✓ RAG integration is two-tiered
4. ✓ Strength is heuristic-based (not LLM)

---

## Self-Critique

### What I'm Confident About
- ✅ Scout's purpose and behavior
- ✅ Data flow (fragment/search → LLM → deposit)
- ✅ Integration with signal_store
- ✅ Task-awareness via composition

### What I'm Uncertain About
- ❓ Is strength heuristic actually problematic? (No evidence)
- ❓ Is 70 tokens optimal? (Seems reasonable, not measured)
- ❓ Fragment assignment performance (Need profiling)
- ❓ How often scouts run (Depends on coordinator)

### What I'm Guessing At
- 🤔 Performance characteristics (no profiling data)
- 🤔 Whether keyword scoring is good enough (no A/B test)
- 🤔 Optimal temperature (0.9 seems high but intentional)

### What I Need to Validate
- [ ] Read forager.py to see what happens to scout signals
- [ ] Read signal_store.py to understand deposit() internals
- [ ] Check if print() is a real problem (developer feedback?)
- [ ] Verify fragment assignment doesn't have edge cases

---

## Conclusion

Scout.py is well-structured, follows stigmergic patterns correctly, and uses composition over monkey patching. The RAG integration shows iterative refinement. Main "issue" is print() instead of logging, but that's a codebase-wide concern, not scout-specific.

**No changes recommended at this time.**

**Confidence in understanding: High (85%)**
- Know what it does ✓
- Know how it fits ✓
- Identified questions ✓
- Marked speculation ✓
