# Implementation Decisions Needed

**Date:** 2025-11-17
**Status:** Awaiting user decisions on verified issues

---

## Critical Re-examination Complete

After thorough code verification, I found my initial analysis had significant errors:

### ❌ What I Got WRONG
1. **Phase 2-3 "dead code"** → Actually runtime-switchable experimental features ✅
2. **Semaphore still 3** → Already fixed to 6 in commit f571603 ✅
3. **Legacy aliases unused** → Actually used in ~10 places across code ✅
4. **Performance analysis current** → Document outdated, semaphore already fixed ✅

### ✅ What I Got RIGHT
1. **Enhanced critic mode unreachable** → 105 lines never executed (verified)
2. **Document mode has no entry point** → ~200 lines with no run_document.py (verified)
3. **Mode/phase proliferation confusing** → 4 overlapping systems (verified)

See `research/CORRECTED_FINDINGS.md` for full details.

---

## Decisions Needed

### DECISION 1: Enhanced Critic Mode Fate

**Current State:**
- Function `evaluate_insights_enhanced()` exists (critic.py:295-400, 105 lines)
- Provides sophisticated evaluation: full provenance, LLM reasoning, quality assessment
- **Never executed** because requires mode="document" but all critics use mode="creative"

**Evidence:**
```python
# critic.py:50-51
if self.enhanced_context:
    await self.evaluate_insights_enhanced(signal_store, llm)  # ← NEVER CALLED

# run_task.py:104
critic = Critic(agent_id, mode="creative", thesis=task_config.task_prompt)  # ← Always creative
```

**Options:**

**A) Activate Enhanced Mode**
```python
# Change run_task.py:104 to use document mode
critic = Critic(agent_id, mode="document", enhanced_context=True, thesis=task_config.task_prompt)
```
- **Pros:** Unlocks sophisticated 105-line evaluation logic
- **Cons:** May require more integration work
- **Effort:** ~1 hour to activate + test

**B) Delete Enhanced Mode**
```python
# Delete critic.py:295-400 (105 lines)
# Delete enhanced_context parameter
# Simplify to just creative mode
```
- **Pros:** Removes dead code, simplifies
- **Cons:** Loses sophisticated evaluation capability
- **Effort:** ~30 minutes to clean up

**C) Keep As-Is (Document for Future)**
- Leave code in place for future document processing mode
- Add comment explaining it's for future use
- **Effort:** ~5 minutes to add comments

**YOUR DECISION:** _____

---

### DECISION 2: Document Mode Fate

**Current State:**
- Document mode exists in 3 agent files (~200 lines total)
- Workflow: Scouts read documents → Extract OBSERVATION signals → Foragers find patterns → Generate INSIGHT signals
- **No entry point** - no run_document.py exists
- Never used in production (only creative mode runs)

**Evidence:**
```bash
$ grep -r "mode=\"document\"" --include="*.py" (outside archive)
# NO RESULTS - never used
```

**Affected Code:**
- scout.py: lines for document mode
- forager.py: lines for document mode
- critic.py: lines 47-55, 185-294

**Options:**

**A) Complete Document Mode**
```python
# Create run_document.py entry point
# Integrate document processing pipeline
# Test with large corpora
```
- **Pros:** Enables document analysis use case (original vision)
- **Cons:** Significant work to complete
- **Effort:** ~8-12 hours to fully implement + test

**B) Delete Document Mode**
```python
# Delete all document mode code (~200 lines)
# Remove mode parameter from agents
# Simplify to task-based only
```
- **Pros:** Removes ~200 lines dead code, simplifies architecture
- **Cons:** Loses document processing capability permanently
- **Effort:** ~2-3 hours to clean up properly

**C) Archive Document Mode**
```bash
# Move document mode code to archive/document_processing/
# Keep for future reference
# Simplify active code
```
- **Pros:** Preserves work, simplifies active code
- **Cons:** Still removes from production
- **Effort:** ~1 hour to archive properly

**D) Keep As-Is (Document for Future)**
- Leave in place for potential future development
- Add TODO comments
- **Effort:** ~10 minutes to document

**YOUR DECISION:** _____

---

### DECISION 3: Phase 4-5 Integration

**Current State:**
- Phase 4 (RealValidator) labeled "PRODUCTION READY" but still behind USE_REAL_VALIDATOR flag
- Phase 5 (AdvancedRetriever) labeled "PRODUCTION READY" but still behind USE_ADVANCED_RETRIEVER flag
- Both set to True by default
- Both work fine

**Evidence:**
```python
# config.py:91-106
USE_REAL_VALIDATOR = True      # Phase 4 - PRODUCTION READY
USE_ADVANCED_RETRIEVER = True  # Phase 5 - PRODUCTION READY
# Why still "phases" if production ready?
```

**Options:**

**A) Integrate Into Core**
```python
# Remove flags, always use these features
# Update config validation
# Remove "Phase" terminology
```
- **Pros:** Clearer production status
- **Cons:** Removes ability to disable
- **Effort:** ~1 hour

**B) Keep Flags, Change Labels**
```python
USE_REAL_VALIDATOR = True      # External validation (recommended)
USE_ADVANCED_RETRIEVER = True  # Advanced knowledge retrieval (recommended)
# Remove "Phase" terminology, keep flags
```
- **Pros:** Keeps configurability, clearer naming
- **Cons:** Still some complexity
- **Effort:** ~10 minutes

**C) Keep As-Is**
- Flags allow easy toggling for testing
- "Phase" is just historical naming
- **Effort:** 0 minutes

**YOUR DECISION:** _____

---

### DECISION 4: Mode Proliferation Cleanup

**Current State:**
- Agent modes: document/creative/legacy (only creative used)
- Task types: debate/creative/analysis/problem_solving (all used)
- Both concepts exist simultaneously

**Evidence:**
```python
# Agent defaults say "document" but always overridden
def __init__(self, mode="document", ...):  # Default never used

# Always passed "creative"
critic = Critic(agent_id, mode="creative", ...)
```

**Options:**

**A) Remove Agent Modes Entirely**
```python
# Depends on Decision 2 (if document mode deleted)
# Simplify to just task_type
class Critic:
    def __init__(self, agent_id: str, task_type: TaskType):
        # No mode parameter
```
- **Pros:** Simplifies significantly
- **Cons:** Removes future extensibility
- **Effort:** ~2-3 hours to refactor all agents

**B) Fix Misleading Defaults**
```python
# If keeping modes, change defaults to match reality
def __init__(self, mode="creative", ...):  # Match actual usage
```
- **Pros:** Less confusing
- **Cons:** Still have redundancy
- **Effort:** ~30 minutes

**C) Keep As-Is**
- Modes provide extensibility
- Task types handle use cases
- **Effort:** 0 minutes

**YOUR DECISION:** _____

---

## What NOT to Change

Based on verification, these are fine as-is:

1. ✅ **Phase 2-3 experimental features** - Keep (intentional runtime switches)
2. ✅ **Legacy signal aliases** - Keep (used in ~10 places)
3. ✅ **simple_scout.py** - Keep (experimental feature, 12KB)
4. ✅ **spatial_signal_store.py** - Keep (experimental feature, 19KB)

---

## Remaining Performance Issues (Note Only)

These are real but may be intentional tradeoffs:

1. **Embedding computation** - Still computed on every deposit (signal_store.py:148)
2. **Cache clearing** - Still clears entire cache on every write (signal_store.py:188-189)
3. **Sleep delays** - Still present in agent loops (scout.py:101, forager.py:62, critic.py:54)

**Estimated impact:** ~20-30% throughput loss (down from 40-60% after semaphore fix)

**Options:**
- Fix now (~2-4 hours per issue)
- Document as known limitations
- Address in future optimization pass

---

## Summary

**Verified Dead Code:** ~300-400 lines (if document mode is abandoned)

**Effort to Clean:**
- Enhanced mode decision: 30 min - 1 hour
- Document mode decision: 1-12 hours (depending on choice)
- Phase labeling: 10 min - 1 hour
- Mode cleanup: 30 min - 3 hours

**Total:** 2-17 hours depending on decisions

**Recommendation:** Start with low-effort decisions (Phase labeling, misleading defaults), then decide on document mode fate.

---

**Please provide your decisions for items 1-4 above.**
