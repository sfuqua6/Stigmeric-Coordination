# Debugging Report: Function and Argument Mismatches

**Date:** 2025-11-13
**Analysis Type:** Cross-file function signature and interface compatibility
**Status:** Analysis Complete - Fixes Pending

---

## Executive Summary

Thorough analysis identified **7 critical issues** involving function/argument mismatches, missing methods, and inconsistent interfaces across the codebase. These issues will cause runtime TypeErrors and AttributeErrors that prevent proper system operation.

---

## Critical Issues Found

### Issue 1: Hater Constructor Signature Mismatch ⚠️ CRITICAL

**Severity:** CRITICAL - Will cause immediate TypeError
**Files Affected:**
- `swarm/monolith_breaking.py:245-247` (caller)
- `swarm/agents/hater.py:13-22` (definition)

**Problem:**
```python
# monolith_breaking.py line 245-247 - INCORRECT CALL
haters = [
    Hater(f"Hater_{i}", task_prompt="Challenge insights")
    for i in range(num_haters)
]

# hater.py line 13-22 - ACTUAL SIGNATURE
def __init__(self, agent_id: str, thesis: str):
    """Initialize hater.

    Args:
        agent_id: Unique agent ID
        thesis: The debate thesis
    """
```

**Issue:** The call passes `task_prompt="Challenge insights"` as a keyword argument, but the constructor expects `thesis` as a positional argument. There is no `task_prompt` parameter.

**Impact:** Raises `TypeError: __init__() got an unexpected keyword argument 'task_prompt'`

**Fix Required:**
```python
# Option 1: Update hater.py constructor to accept task_prompt
def __init__(self, agent_id: str, task_prompt: str = "Challenge insights"):

# Option 2: Update monolith_breaking.py call to pass thesis
haters = [
    Hater(f"Hater_{i}", "Synthesize document insights")
    for i in range(num_haters)
]

# RECOMMENDED: Option 1 - More consistent with other agents
```

---

### Issue 2: DocumentProcessor Missing Method `split_into_sections` ⚠️ CRITICAL

**Severity:** CRITICAL - Will cause AttributeError
**Files Affected:**
- `run_web_ingestion.py:359-363` (caller)
- `swarm/documents/processor.py` (missing method)

**Problem:**
```python
# run_web_ingestion.py line 359 - CALLS NON-EXISTENT METHOD
sections = processor.split_into_sections(
    text=doc['text'],
    source_name=doc['url'],
    metadata={'title': doc['title'], 'url': doc['url']}
)
```

**Issue:** The `DocumentProcessor` class has no `split_into_sections` method. It has:
- `split_on_paragraphs(text)` - returns List[str]
- `group_paragraphs_into_sections(paragraphs)` - returns List[str]
- But no method that takes `text`, `source_name`, and `metadata` and returns DocumentSection objects

**Impact:** Raises `AttributeError: 'DocumentProcessor' object has no attribute 'split_into_sections'`

**Fix Required:** Add the missing method to `DocumentProcessor`:
```python
def split_into_sections(self, text: str, source_name: str,
                       metadata: dict) -> List[DocumentSection]:
    """Split text into DocumentSection objects.

    Args:
        text: Text to split
        source_name: Source document name
        metadata: Metadata dict

    Returns:
        List of DocumentSection objects
    """
    # Implementation needed
```

---

### Issue 3: Missing Function `run_document_swarm_from_sections` ⚠️ HIGH

**Severity:** HIGH - Will fail but has fallback
**Files Affected:**
- `run_web_ingestion.py:390-404` (caller)
- `swarm/monolith_breaking.py` (missing function)

**Problem:**
```python
# run_web_ingestion.py line 390 - CALLS NON-EXISTENT FUNCTION
try:
    result = await run_document_swarm_from_sections(
        sections=all_sections,
        model_name=model_name,
        # ... other args
    )
except NameError:
    # Fallback to regular pipeline
```

**Issue:** The function `run_document_swarm_from_sections` does not exist in `monolith_breaking.py`. Only `run_document_swarm(document_paths: List[str], ...)` exists.

**Impact:** Will catch NameError and fall back to less efficient path (saves temp files, reloads them). Inefficient but functional.

**Fix Required:** Add the missing function to accept sections directly:
```python
async def run_document_swarm_from_sections(
    sections: List[DocumentSection],
    model_name: str = "microsoft/phi-2",
    # ... other parameters
):
    """Run document swarm from pre-loaded sections."""
    # Skip Phase 1 (document loading) since sections provided
    # Jump directly to Phase 2 (infrastructure init)
```

---

### Issue 4: Forager Legacy Method References Missing Attributes ⚠️ MEDIUM

**Severity:** MEDIUM - Only affects legacy creative mode (not used in document processing)
**Files Affected:**
- `swarm/agents/forager.py:240-274` (`_make_prompt` method)

**Problem:**
```python
# forager.py line 242 - REFERENCES UNDEFINED ATTRIBUTES
def _make_prompt(self, signal: Signal) -> str:
    """Generate development prompt based on types."""
    if self.output_type == "CLAIM":  # ❌ self.output_type not defined
        return (f"You are a forager agent developing arguments about:\n"
                f"\"{self.thesis}\"\n\n"  # ❌ self.thesis not defined
```

**Issue:** The `_make_prompt` method references `self.output_type` and `self.thesis` which are never set in `__init__`. The method is only called by `develop()` which is part of legacy creative mode.

**Impact:** Would raise `AttributeError` if called, but this is legacy code not used in document mode.

**Fix Required:**
```python
# Option 1: Remove legacy code (RECOMMENDED)
# Remove _make_prompt and develop methods entirely

# Option 2: Add attributes to __init__ (if legacy mode needs to be preserved)
def __init__(self, agent_id: str, mode: str = "document",
             output_type: str = None, thesis: str = None):
    # ...
    self.output_type = output_type
    self.thesis = thesis
```

---

### Issue 5: Critic Legacy Method References Missing Attribute ⚠️ MEDIUM

**Severity:** MEDIUM - Only affects legacy creative mode
**Files Affected:**
- `swarm/agents/critic.py:159-188` (`generate_critique` method)

**Problem:**
```python
# critic.py line 172 - REFERENCES UNDEFINED ATTRIBUTE
prompt = (f"You are a critic agent analyzing arguments about this thesis:\n"
         f"\"{self.thesis}\"\n\n"  # ❌ self.thesis not defined
```

**Issue:** The `generate_critique` method references `self.thesis` which is never set in `__init__`. This is legacy code from the creative debate mode.

**Impact:** Would raise `AttributeError` if called, but not used in document mode.

**Fix Required:** Same as Issue 4 - either remove legacy code or add missing attribute.

---

### Issue 6: Inconsistent SimpleLLM Constructor Parameter ⚠️ LOW

**Severity:** LOW - Works but inconsistent
**Files Affected:**
- `swarm/monolith_breaking.py:116` (caller)
- `swarm/llm/simple_llm.py:14-15` (definition)

**Problem:**
```python
# monolith_breaking.py line 116
llm = SimpleLLM(model_name, DEVICE, enable_cache=True, cache_size=50)

# simple_llm.py line 14
def __init__(self, model_name: str, device: str, enable_cache: bool = True,
             cache_size: int = 100, use_quantization: bool = None):
```

**Issue:** Call passes `cache_size=50` but definition defaults to `cache_size=100`. Not a bug, but worth noting the intentional override.

**Impact:** None - works as intended. This is just a parameter override.

**Fix Required:** None (this is intentional).

---

### Issue 7: Scout Constructor Backward Compatibility Complexity ⚠️ LOW

**Severity:** LOW - Works but complex
**Files Affected:**
- `swarm/agents/scout.py:18-45` (constructor logic)

**Problem:**
The Scout constructor has complex backward compatibility logic to handle both:
1. New document mode: `Scout(id, section=DocumentSection)`
2. Legacy creative mode: `Scout(id, signal_type, task_prompt)`

```python
# Line 30-45 - Complex branching logic
if isinstance(section_or_signal_type, str):
    # Legacy mode
    self.section = None
    self.mode = "creative"
elif section_or_signal_type is not None:
    # New mode
    self.section = section_or_signal_type
    self.mode = "document"
else:
    # Fallback
    self.section = None
    self.mode = "creative"
```

**Issue:** This works correctly but is complex and could be simplified if legacy mode is no longer needed.

**Impact:** None - works as designed. Code is just more complex than needed.

**Fix Required:** Consider simplifying if legacy creative mode is no longer needed.

---

## Summary of Fixes Required

### Must Fix (Critical - Will cause crashes):
1. ✅ Fix Hater constructor signature mismatch
2. ✅ Add DocumentProcessor.split_into_sections() method
3. ✅ Add monolith_breaking.run_document_swarm_from_sections() function

### Should Fix (Medium - Affects unused legacy code):
4. ⚠️ Fix or remove Forager._make_prompt() and legacy methods
5. ⚠️ Fix or remove Critic.generate_critique() and legacy methods

### Optional (Low - Works but could be cleaner):
6. ℹ️ Document intentional cache_size override
7. ℹ️ Consider simplifying Scout backward compatibility

---

## Testing Strategy

After fixes are applied, test the following:

### Critical Path Tests:
1. Run monolith_breaking.py with sample documents
2. Run web_ingestion.py with a test query
3. Verify Hater agents instantiate correctly
4. Verify DocumentProcessor.split_into_sections works
5. Verify run_document_swarm_from_sections works (if implemented)

### Legacy Code Tests (if preserved):
6. Run run_task.py in creative mode
7. Verify Scout creative mode still works
8. Verify Forager legacy methods work
9. Verify Critic legacy methods work

---

## Recommended Fix Priority

1. **IMMEDIATE:** Issue #1 (Hater constructor) - Blocks basic functionality
2. **IMMEDIATE:** Issue #2 (DocumentProcessor) - Blocks web ingestion
3. **HIGH:** Issue #3 (run_document_swarm_from_sections) - Inefficient fallback
4. **MEDIUM:** Issues #4-5 (Legacy code) - Clean up or fix for consistency
5. **LOW:** Issues #6-7 (Documentation/cleanup) - Nice to have

---

## Files Requiring Changes

1. `swarm/agents/hater.py` - Update constructor signature
2. `swarm/documents/processor.py` - Add split_into_sections method
3. `swarm/monolith_breaking.py` - Add run_document_swarm_from_sections function
4. `swarm/agents/forager.py` - Fix or remove legacy code
5. `swarm/agents/critic.py` - Fix or remove legacy code

---

## End of Report
