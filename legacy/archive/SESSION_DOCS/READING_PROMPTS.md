# READING PROMPTS - Self-Questioning Framework

**Purpose:** Ask these questions while reading to prevent assumptions and maintain rigor
**Usage:** Copy relevant section, answer questions, save to understanding_notes/
**Update:** Add new prompts as patterns emerge

---

## Meta-Prompts (Before Reading Anything)

### Context Check
- [ ] Have I read STRUCTURE_REFERENCE.md recently?
- [ ] Do I know where this file fits in the architecture?
- [ ] What am I trying to learn from this file?
- [ ] What assumptions am I bringing?

### Honesty Check
- [ ] Am I reading to understand, or to confirm what I think I know?
- [ ] Am I willing to be wrong about my hypotheses?
- [ ] Will I mark guesses as guesses?
- [ ] Will I admit when I'm confused?

---

## For ANY File

### Purpose & Responsibility
1. **What does this file DO?** (One clear sentence)
2. **What problem does it solve?**
3. **Why does it exist?** (What would break if it didn't?)
4. **What data/state does it own?**
5. **Is the responsibility clear and single?** (Or doing too much?)

### Dependencies & Coupling
6. **What does it import?** (List modules)
7. **Who imports it?** (Who depends on it?)
8. **Could this create circular dependencies?**
9. **Is the coupling tight or loose?**
10. **What would need to change if this changed?**

### Code Quality
11. **Is it readable?** (Can I understand it in one pass?)
12. **Are names clear?** (Do functions/variables say what they do?)
13. **Are functions short?** (< 30 lines ideally)
14. **Is there duplication?** (DRY violations?)
15. **Are there magic numbers?** (Unexplained constants?)

### Error Handling
16. **What can go wrong?** (List failure modes)
17. **How are errors handled?** (Try/except, validation, assertions?)
18. **Are edge cases handled?** (Empty inputs, None, etc.)
19. **Could this crash silently?** (Swallowed exceptions?)
20. **Are error messages helpful?** (Could I debug from them?)

### Testing & Debugging
21. **Is this testable?** (Can I test it in isolation?)
22. **Are there tests for it?** (Check tests/ directory)
23. **How would I debug this if it broke?** (Logging, print statements?)
24. **Are there comments explaining WHY?** (Not what - why)
25. **Are complex sections documented?** (Non-obvious logic)

### Performance & Resources
26. **Does it do I/O?** (File, network, database?)
27. **Is the I/O async?** (await, not blocking?)
28. **Does it allocate much memory?** (Large data structures?)
29. **Does it have unbounded growth?** (Caches, lists without limits?)
30. **Could this be a bottleneck?** (Guess + mark as "NEED PROFILING")

---

## For AGENT Files (scout.py, forager.py, critic.py, etc.)

### Signal Interaction
31. **What signal types does it READ?** (sample, get_ancestors, etc.)
32. **What signal types does it WRITE?** (deposit)
33. **How does it sample signals?** (weighted, stratified, cluster, top-n?)
34. **Does it use provenance?** (parent parameter in deposit?)
35. **Does it traverse the graph?** (get_ancestors, get_descendants?)

### Agent Behavior
36. **What's its role in the swarm?** (Exploration, development, evaluation, etc.)
37. **How does it decide what to do?** (Sampling strategy, conditions)
38. **How does it calculate signal strength?** (Fixed, LLM-based, heuristic?)
39. **Does it wait for signals?** (wait_for_signal usage?)
40. **How often does it act?** (Every iteration, conditional, event-driven?)

### LLM Usage
41. **Does it call the LLM?** (llm.generate)
42. **What temperature does it use?** (Exploration vs. exploitation)
43. **How are prompts constructed?** (_make_prompt method?)
44. **Does it parse LLM output?** (Extract scores, structured data?)
45. **What if LLM fails?** (Error handling, retries, fallbacks?)

### Task Awareness
46. **Does it use task_config?** (Composition pattern)
47. **Are prompts task-specific?** (Different for creative vs. debate?)
48. **Does mode affect behavior?** (mode="creative" vs. "analytical")
49. **Does it use thesis?** (For focused modes)
50. **Is it flexible across task types?** (Or specialized?)

---

## For CORE Files (signal_store.py, coordinators, etc.)

### State Management
51. **What state does it manage?** (List data structures)
52. **Is state mutable or immutable?** (In-place updates or copies?)
53. **Who can access the state?** (Public, private, protected?)
54. **Are there invariants?** (Conditions that must always hold)
55. **How are invariants maintained?** (Validation, locks, checks?)

### Concurrency & Safety
56. **Is it thread-safe?** (Uses locks?)
57. **Is it async-safe?** (Proper await usage?)
58. **Could there be race conditions?** (Shared state without locks?)
59. **Could there be deadlocks?** (Multiple locks, wrong order?)
60. **Does it use events?** (asyncio.Event coordination?)

### Performance Critical
61. **Is this on the hot path?** (Called frequently?)
62. **Are operations O(1), O(n), O(n²)?** (Algorithm complexity)
63. **Is there caching?** (Memoization, result caching?)
64. **Could caching cause staleness?** (Invalidation strategy?)
65. **Is there unnecessary copying?** (Large data structures)

### API Design
66. **Is the API intuitive?** (Method names clear?)
67. **Are parameters well-named?** (Obvious what they do?)
68. **Are there sensible defaults?** (Optional parameters)
69. **Is it hard to misuse?** (Fail-safe design?)
70. **Is it backward compatible?** (If changing, would it break users?)

---

## For LLM Files (simple_llm.py, providers, etc.)

### Abstraction
71. **What's abstracted?** (Provider-specific details?)
72. **Is the abstraction leaky?** (Provider details bleeding through?)
73. **Can providers be swapped?** (OpenAI, Anthropic, local, etc.)
74. **Is configuration clean?** (How to switch providers?)
75. **Does it handle provider failures?** (Fallbacks, retries?)

### Resource Management
76. **How is the model loaded?** (Memory management)
77. **Is the model unloaded?** (Cleanup?)
78. **How many models in memory?** (Pooling?)
79. **Is GPU memory managed?** (CUDA allocation)
80. **Are there memory leaks?** (Model not freed?)

---

## For RETRIEVAL Files (search, web scraping, etc.)

### External Dependencies
81. **What external services?** (APIs, websites?)
82. **How are failures handled?** (Network errors, rate limits?)
83. **Is there rate limiting?** (Respectful scraping)
84. **Is there caching?** (Avoid redundant requests)
85. **Is data validated?** (Malformed responses?)

### Data Quality
86. **How is relevance determined?** (Ranking, filtering?)
87. **Is content cleaned?** (HTML stripped, normalized?)
88. **Are sources tracked?** (Provenance for verification)
89. **Is there deduplication?** (Same content from multiple sources?)
90. **How much data is retrieved?** (Could it be too much/little?)

---

## For VALIDATION Files (validators, knowledge bases, etc.)

### Correctness
91. **What's being validated?** (Facts, format, logic?)
92. **How is validation performed?** (External APIs, rules, LLM?)
93. **What's the confidence level?** (Scoring mechanism)
94. **Can validation fail?** (Unavailable resources?)
95. **What happens on validation failure?** (Signal strength reduced?)

### Knowledge Management
96. **How is knowledge stored?** (In-memory, database, file?)
97. **How is knowledge updated?** (Learning, manual, external?)
98. **Can knowledge be wrong?** (How is it corrected?)
99. **Is there knowledge decay?** (Temporal validity?)
100. **How is conflict resolved?** (Contradictory facts?)

---

## Critical Thinking Prompts

### Challenge Assumptions
- **What am I assuming is true?** (List assumptions)
- **What would break if my assumption is wrong?**
- **How could I verify this assumption?** (Test, measurement, code trace)
- **Am I confusing correlation with causation?**
- **Am I generalizing from a single example?**

### Identify Gaps
- **What don't I understand yet?** (Confusing parts)
- **What would I need to know to understand this?** (Prerequisites)
- **Where is the documentation?** (Docstrings, comments, external docs)
- **Who would I ask if I could?** (Domain expert, original author)
- **What would I test first to verify my understanding?**

### Spot Problems
- **If I were an attacker, how would I break this?** (Security mindset)
- **If I were a user, what would frustrate me?** (UX mindset)
- **If I were maintaining this in 2 years, what would confuse me?** (Maintenance mindset)
- **What would happen under heavy load?** (Performance mindset)
- **What happens when something fails?** (Failure mindset)

### Propose Improvements
- **Is there a simpler way to do this?** (Simplification)
- **Is there duplicated code that could be abstracted?** (DRY)
- **Could this be more readable?** (Clarity)
- **Could this be more efficient?** (Performance - but MEASURE first)
- **Could this be more robust?** (Error handling)

---

## Documentation Template (After Reading)

Save to `understanding_notes/<filename>.md`:

```markdown
# Understanding: <filename>

**Date:** YYYY-MM-DD
**Purpose:** <One sentence>
**Confidence:** High/Medium/Low

## What It Does
<Clear explanation>

## Key Components
- Component 1: <description>
- Component 2: <description>

## Dependencies
**Imports:** <list>
**Imported by:** <list>

## Data Flow
<Describe inputs → processing → outputs>

## Potential Issues
1. Issue 1: <description> [CONFIRMED/HYPOTHESIS]
2. Issue 2: <description> [CONFIRMED/HYPOTHESIS]

## Questions Remaining
- Question 1
- Question 2

## Improvement Ideas
1. Idea 1: <description> [PRIORITY: High/Med/Low]
   - **Evidence:** <what supports this>
   - **Risk:** <what could go wrong>
   - **Effort:** <hours estimate>

## Notes
<Any other observations>
```

---

## Commit to Using This

Before reading any file:
1. ✅ Read relevant section of this document
2. ✅ Answer questions while reading
3. ✅ Save notes to understanding_notes/
4. ✅ Update STRUCTURE_REFERENCE if needed
5. ✅ Mark speculation clearly

After reading:
1. ✅ Review notes for assumptions
2. ✅ Challenge conclusions
3. ✅ Identify what to verify
4. ✅ Prioritize further reading
5. ✅ Only propose changes with evidence

---

## Anti-Patterns to Avoid While Reading

❌ **Speed reading** - Slow down, understand deeply
❌ **Confirmation bias** - Challenge your hypotheses
❌ **Assuming intent** - Code does what it does, not what author "meant"
❌ **Perfectionism** - Good enough understanding to proceed is okay
❌ **Analysis paralysis** - Can always read more later
❌ **Ignoring tests** - Tests reveal intent and edge cases
❌ **Skipping docs** - Docstrings and comments have context
❌ **Forgetting the user** - What's the actual use case?

---

## Success Criteria

**Good reading session:**
- ✅ Can explain what the file does to someone else
- ✅ Can trace a data flow through the file
- ✅ Can identify what could go wrong
- ✅ Know what I don't know
- ✅ Have specific questions to answer next

**Bad reading session:**
- ❌ "I think I get it" but can't explain
- ❌ Skimmed without understanding
- ❌ Made assumptions without marking them
- ❌ Jumped to conclusions without evidence
- ❌ Didn't identify any questions (means I didn't think deeply enough)

---

## Remember

**The goal is understanding, not speed.**
**Mark guesses as guesses.**
**Admit confusion honestly.**
**Question everything, including these questions.**
