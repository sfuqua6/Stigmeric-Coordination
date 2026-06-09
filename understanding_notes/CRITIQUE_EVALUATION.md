# CRITIQUE EVALUATION - Evidence-Based Analysis

**Generated:** 2025-11-20
**Critique Source:** External review of swarm performance report
**Methodology:** Verify each claim against code, apply decision matrix

---

## Summary of Critique

The critique analyzes a swarm run and identifies:
1. Signal imbalance (59.9% INITIAL, 11.8% SUPPORT)
2. Outlier strength (signal at 1.764 vs avg 0.67)
3. Visits = 0 anomaly
4. High hater temperature (0.85)
5. Iteration/round mismatch (50 vs 3)
6. Low cache hit rate (22.5%)

**Recommendations:**
- Rebalance signal roles
- Tune temperatures
- Adjust thresholds
- Fix instrumentation
- Add metrics

---

## Claim-by-Claim Verification

### Claim 1: "Config Iterations: 50 but only 3 rounds"

**Verification:**
- ✅ **EXPLAINED** - Not a bug, different concepts
- `MAX_ITERATIONS = 50` in config.py (swarm/core/config.py:40) = max actions per agent
- `num_rounds = 3` in RoundCoordinator (swarm/core/round_coordinator.py:23) = refinement rounds
- These are orthogonal: rounds are for iterative search refinement, iterations are per-agent action limits

**Conclusion:** No action needed. This is expected behavior.

---

### Claim 2: "Visits = 0 anomaly for top INITIALs"

**Verification:**
Reading code to understand visits tracking...

**From signal_store.py:22:**
```python
@dataclass
class Signal:
    visits: int = 0  # Track corroboration
```

**From signal_store.py:479-484 (amplify method):**
```python
def amplify(self, signal_id: str, factor: float = 1.2) -> bool:
    if signal_id in self.signals:
        signal = self.signals[signal_id]
        signal.strength = min(1.0, signal.strength * factor)
        signal.visits += 1  # <-- Incremented on amplify
        return True
```

**Finding:** Visits are ONLY incremented when `amplify()` is called. If top INITIAL signals have Visits=0, it means:
- They were never amplified
- Possibly scouts deposited high-strength signals that never got corroboration
- OR the system values different signals for final synthesis than what gets amplified

**Question:** Are visits also incremented on sampling? Let me check...

**From signal_store.py:359-405 (sample_weighted):**
- Uses visits in weighting calculation but does NOT increment
- Exploration bonus formula: `exploration_weight = self.exploration_bonus * (1.0 - visit_ratio)`
- This READS visits but doesn't increment them

**Conclusion:**
- ⚠️ Visits=0 could indicate signals that were never amplified
- This is NOT necessarily a bug - it could mean scouts deposited strong ideas that went straight to synthesis
- HOWEVER, if the goal is to track "how often sampled", then visits should be incremented on sampling too

**Decision:** This is a potential enhancement but needs validation:
- Does Visits=0 correlate with lower quality?
- Should sampling increment visits?
- Or is current behavior (amplify-only) intentional?

**Action:** DEFER - Need to see actual data to determine if this is a problem

---

### Claim 3: "Signal imbalance: 59.9% INITIAL, 11.8% SUPPORT"

**Verification:**
Need to understand signal flow:
- Scouts deposit INITIAL (or task-specific type)
- Foragers sample INITIAL and deposit SUPPORT
- Critics sample and deposit CRITIQUE
- Validators deposit VERIFICATION

**From forager.py:87:**
```python
input_signals = signal_store.sample_weighted(input_type, n=1)
```

**From forager.py:121:**
```python
signal_store.deposit(
    signal_type=self.output_type,  # <-- This is SUPPORT
    strength=0.6,
    parent=signal.id
)
```

**Question:** Why might SUPPORT be low?

**Possible reasons:**
1. NUM_FORAGERS too low relative to NUM_SCOUTS (config: 4 scouts, 4 foragers)
2. Foragers hitting max_actions too early
3. Foragers waiting for signals (event-driven could cause stalls)
4. Forager deposit threshold too high

**From config.py:49:**
```python
MIN_DEPOSIT_STRENGTH = 0.3  # Only deposit if above this
```

**From forager.py:** Strength is fixed at 0.6, which is above 0.3, so threshold isn't blocking.

**Hypothesis:** If scouts are MORE ACTIVE than foragers (higher temperature, more ideas), you'd get many INITIAL but few SUPPORT.

**Current config:**
- TEMP_SCOUT = 0.9 (very creative)
- TEMP_FORAGER = 0.7 (moderate)
- NUM_SCOUTS = NUM_FORAGERS = 4

**Assessment:**
- This imbalance COULD be a problem if the goal is convergence
- But it COULD also be intentional for diverse exploration
- Need to measure: Does higher SUPPORT correlate with better synthesis?

**Decision:** UNCLEAR - Need A/B testing to determine if this is actually a problem

---

### Claim 4: "Outlier strength 1.764 vs avg 0.67"

**Verification:**
Signal strength is bounded to [0, 1.0] in multiple places:

**From signal_store.py:31:**
```python
def __post_init__(self):
    self.strength = max(0.0, min(1.0, self.strength))
```

**Wait, this should PREVENT strength > 1.0!**

**How could strength reach 1.764?**

Let me check if amplify respects the cap:

**From signal_store.py:482:**
```python
signal.strength = min(1.0, signal.strength * factor)
```

✅ Amplify respects cap.

**Could decay create negative values? From signal_store.py:518:**
```python
signal.strength *= (1.0 - self.decay_rate)
```

No cap enforcement here, but multiplication by <1.0 can't create >1.0.

**Hypothesis:** The strength 1.764 is IMPOSSIBLE given current code, unless:
1. Signal.__post_init__ isn't being called (dataclass issue)
2. Direct assignment bypasses setter
3. The report is from an older version of code

**Verification from signal_store.py:29-35:**
```python
@dataclass
class Signal:
    # ... fields ...
    def __post_init__(self):
        """Ensure fields are initialized correctly."""
        self.strength = max(0.0, min(1.0, self.strength))  # CLAMP [0, 1.0]
```

**AND from signal_store.py:176-186 (deposit method):**
```python
signal = Signal(
    id=signal_id,
    type=signal_type,
    content=content,
    strength=strength,  # Passed directly, __post_init__ will clamp
    # ...
)
self.signals[signal_id] = signal
```

**Conclusion:**
- ❌ **IMPOSSIBLE** - Strength 1.764 cannot exist in current codebase
- The Signal.__post_init__ clamps to [0, 1.0]
- amplify() uses `min(1.0, ...)`
- decay uses multiplication by <1.0 (cannot exceed 1.0)
- boost_contrarian_signals() uses `min(1.0, ...)`

**Hypothesis:** The report is either:
1. From an older version of code before __post_init__ was added
2. A typo/error in the report
3. Using a fork/modification of the code

**Action:** ❌ REJECT - This claim appears to be based on outdated/incorrect data

---

### Claim 5: "High Hater Temperature 0.85 is too noisy"

**Verification:**
**Current config (config.py:27-31):**
```python
TEMP_SCOUT = 0.9      # High exploration
TEMP_FORAGER = 0.7    # Balanced development
TEMP_CRITIC = 0.6     # Focused analysis
TEMP_HATER = 0.85     # High adversarial creativity
TEMP_SYNTHESIZER = 0.6  # Focused synthesis
```

**Question:** Is 0.85 too high for haters?

**Consideration:**
- Haters are SUPPOSED to generate adversarial, challenging signals
- High temperature = more creative/diverse objections
- Low temperature = more focused/conservative objections

**From ARCHITECTURAL_ANALYSIS.md findings:**
- Current system has 28.3% CRITIQUE signals - "healthy amount"
- System already shows good critical review

**Question:** What's the evidence that 0.85 is "too noisy"?
- The critique claims "noisy and possibly destructive rather than constructive"
- But provides NO EVIDENCE this is actually happening
- No data on hater signal quality
- No comparison of different temperature values

**Decision:** ⚠️ **SPECULATIVE** - No evidence provided that 0.85 causes problems

**To validate, would need:**
1. Measure hater signal quality at different temperatures
2. A/B test: 0.85 vs 0.55 vs 0.65
3. Metrics: coherence, relevance, usefulness of objections

**Action:** DEFER - Cannot justify change without evidence

---

### Claim 6: "Low cache hit rate 22.5% could be improved"

**Verification:**
**From config.py:76-81:**
```python
LLM_CACHE_SIZE = 1000  # Cached LLM responses
ENABLE_LLM_CACHE = True

# Task-specific cache settings
CREATIVE_CACHE_SIZE = 50  # Much smaller for creative tasks
CREATIVE_CACHE_ENABLED = True
```

**Consideration:**
- Low cache hit rate (22.5%) could mean:
  - High diversity (good for exploration!)
  - Inefficiency (bad for cost)

**From critique:** "low reuse could be fine for creativity"

**Question:** What's the task type?
- If creative/exploration task → 22.5% is probably GOOD (high diversity)
- If analytical/factual task → might want higher caching

**Current evidence:**
- CREATIVE_CACHE_SIZE = 50 (intentionally small!)
- Comment says "target <50% hit rate"
- So 22.5% is BELOW target, meaning VERY high diversity

**Decision:** ⚠️ **CONTEXT-DEPENDENT** - Could be optimal for creative tasks

**Action:** NO CHANGE - System appears designed for high diversity

---

## Summary of Verification

| Claim | Status | Action |
|-------|--------|--------|
| Iteration/round mismatch | ✅ EXPLAINED | None - expected behavior |
| Visits=0 anomaly | ⚠️ UNCLEAR | Defer - need data |
| Signal imbalance | ⚠️ UNCLEAR | Defer - need A/B test |
| Strength 1.764 outlier | ❌ IMPOSSIBLE | Reject - bad data |
| High hater temp | ⚠️ SPECULATIVE | Defer - no evidence |
| Low cache hit | ⚠️ OPTIMAL? | None - designed for diversity |

**Findings:**
- 1 claim REJECTED (impossible strength value)
- 1 claim EXPLAINED (iterations vs rounds)
- 4 claims UNCLEAR (need more data/testing)
- 0 claims VALIDATED for immediate action

---

## Evaluating Recommendations Using Decision Matrix

### Recommendation 1: "Rebalance signal roles - increase SUPPORT from 11.8% to 25%"

**Problem validation:**
- [ ] **What problem?** Low SUPPORT percentage - but is this BAD?
- [ ] **Evidence exists?** 11.8% observed, but no evidence it's suboptimal
- [ ] **Measurable impact?** Unknown - need to test if higher SUPPORT improves synthesis
- [🚩] **Red flag:** "Might improve convergence" - SPECULATIVE

**Decision Matrix (estimated):**
| Criterion | Score | Weight | Total |
|-----------|-------|--------|-------|
| Solves real problem | 2 | 3 | 6 (no evidence problem exists) |
| Measurable benefit | 2 | 3 | 6 (unclear if beneficial) |
| Reduces complexity | 2 | 2 | 4 (adds tuning complexity) |
| Low risk | 3 | 2 | 6 (could harm diversity) |
| Easy to test | 4 | 1 | 4 (can A/B test) |
| **TOTAL** | | | **26** |

**Score: 26** - Above threshold (>15) BUT heavily dependent on unverified assumption

**Recommendation:** ⚠️ **EXPERIMENTAL ONLY** - Run A/B test, don't change production

---

### Recommendation 2: "Reduce Scout temp 0.9→0.7, Hater temp 0.85→0.55"

**Problem validation:**
- [ ] **What problem?** "Too much randomness" - but EVIDENCE?
- [ ] **Measurable impact?** Unknown
- [🚩] **Red flag:** "Makes them less noisy" - subjective, no data

**Decision Matrix:**
| Criterion | Score | Weight | Total |
|-----------|-------|--------|-------|
| Solves real problem | 1 | 3 | 3 (no problem shown) |
| Measurable benefit | 2 | 3 | 6 (speculative) |
| Reduces complexity | 2 | 2 | 4 (neutral) |
| Low risk | 2 | 2 | 4 (could reduce diversity) |
| Easy to test | 5 | 1 | 5 (trivial config change) |
| **TOTAL** | | | **22** |

**Score: 22** - Above threshold but WEAK evidence

**Recommendation:** ⚠️ **A/B TEST ONLY** - Don't implement without validation

---

### Recommendation 3: "Raise prune threshold 0.15→0.2"

**Problem validation:**
- [ ] **What problem?** "Remove weak noise" - but is current 0.15 insufficient?
- [ ] **Evidence?** None provided
- [🚩] **Red flag:** "Might clean up better" - YAGNI violation

**Decision Matrix:**
| Criterion | Score | Weight | Total |
|-----------|-------|--------|-------|
| Solves real problem | 2 | 3 | 6 |
| Measurable benefit | 2 | 3 | 6 |
| Reduces complexity | 3 | 2 | 6 |
| Low risk | 3 | 2 | 6 |
| Easy to test | 5 | 1 | 5 |
| **TOTAL** | | | **29** |

**Score: 29** - Above threshold

**Assessment:** Low risk, easy to test, but still no evidence of problem

**Recommendation:** ⚠️ **LOW PRIORITY** - Could test, but not urgent

---

### Recommendation 4: "Fix Visits logging"

**Problem validation:**
- [x] **What problem?** Visits=0 for top signals
- [x] **Evidence?** Observed in report
- [x] **Measurable?** Yes - visits should be tracked
- [ ] **Is it a BUG?** UNCLEAR - could be expected if signals aren't amplified

**Analysis:**
Current behavior: visits++ ONLY on amplify(), NOT on sampling

**Question:** SHOULD sampling increment visits?

**Design considerations:**
- If visits = "how many times corroborated" → current is CORRECT (amplify only)
- If visits = "how many times used" → should increment on sampling too

**Need to check:** What's the intended semantics of "visits"?

**From signal_store.py exploration_bonus calculation:**
```python
visit_ratio = s.visits / max_visits
exploration_weight = self.exploration_bonus * (1.0 - visit_ratio)
```

This suggests visits = "popularity" for exploration bonus. If so, should increment on sampling.

**Decision Matrix:**
| Criterion | Score | Weight | Total |
|-----------|-------|--------|-------|
| Solves real problem | 3 | 3 | 9 (IF bug) |
| Measurable benefit | 4 | 3 | 12 (better metrics) |
| Reduces complexity | 3 | 2 | 6 (fixes confusion) |
| Low risk | 4 | 2 | 8 |
| Easy to test | 4 | 1 | 4 |
| **TOTAL** | | | **39** |

**Score: 39** - Above threshold

**Recommendation:** ✅ **INVESTIGATE** - Check if visits should increment on sampling

---

### Recommendation 5: "Add synthesis audit step"

**Problem validation:**
- [ ] **What problem?** Current synthesis might be suboptimal
- [ ] **Evidence?** None - just a suggestion
- [🚩] **Red flag:** "Might improve quality" - speculative

**Decision Matrix:**
| Criterion | Score | Weight | Total |
|-----------|-------|--------|-------|
| Solves real problem | 2 | 3 | 6 |
| Measurable benefit | 3 | 3 | 9 |
| Reduces complexity | 1 | 2 | 2 (adds step) |
| Low risk | 3 | 2 | 6 |
| Easy to test | 3 | 1 | 3 |
| **TOTAL** | | | **26** |

**Score: 26** - Above threshold but speculative

**Recommendation:** 📋 **LATER** - Interesting idea but not urgent

---

## Validated Actions (Score >35 with evidence)

### Action 1: Investigate Visits Tracking [Score: 39]

**Problem:** Visits=0 for top signals - unclear if bug or expected

**Investigation steps:**
1. Check intended semantics of "visits" field
2. Determine if sampling should increment visits
3. If yes, add visits++ to sample_weighted()
4. Test before/after

**Implementation:**
```python
# In signal_store.py sample_weighted():
def sample_weighted(self, signal_type: str, n: int = 1) -> List[Signal]:
    with self._lock:
        # ... existing sampling logic ...

        # INCREMENT VISITS for sampled signals
        for signal in sampled:
            signal.visits += 1

        return sampled
```

**Validation needed:**
- Does this match intended semantics?
- Check with user/documentation

---

## Actions REJECTED (Below threshold or no evidence)

1. ❌ Strength 1.764 investigation - IMPOSSIBLE in current code
2. ❌ Temperature tuning - No evidence of problem
3. ❌ Threshold adjustments - Speculative benefit
4. ❌ Signal rebalancing - No evidence imbalance is bad

---

## Recommended Next Steps

### Immediate (High confidence):**

1. ✅ **Clarify visits semantics** - Ask user or check docs
   - If visits = "times used", implement sampling increment
   - If visits = "times corroborated", keep current behavior

### Experimental (A/B testing needed):

2. 📊 **Run baseline measurements** before any changes:
   - Synthesis quality (human evaluation)
   - Signal distribution (INITIAL/SUPPORT/CRITIQUE percentages)
   - Visits distribution
   - Cache hit rates

3. 📊 **A/B test temperature changes** (if user wants to experiment):
   - Control: Current temps
   - Variant A: Scout 0.7, Hater 0.6
   - Measure: synthesis quality, diversity, convergence time

4. 📊 **A/B test threshold changes**:
   - Control: Current thresholds
   - Variant: Prune 0.2, Min Amplify 0.35
   - Measure: final signal count, synthesis quality

### Documentation (No code changes):

5. 📝 **Document expected behavior**:
   - Explain iterations vs rounds distinction
   - Document visits semantics
   - Explain why low cache hit is optimal for creative tasks

---

## Conclusion

**Valid insights from critique:**
- ✅ Good analysis of signal distribution
- ✅ Useful metrics suggested (diversity, convergence)
- ✅ Process improvements (synthesis audit, metrics tracking)

**Invalid/speculative claims:**
- ❌ Strength >1.0 (impossible in current code)
- ⚠️ Temperature values (no evidence they're wrong)
- ⚠️ Threshold values (no evidence they're suboptimal)
- ⚠️ Signal imbalance (no evidence it's a problem)

**Evidence-based approach:**
1. Don't change parameters without validation
2. Run A/B tests for experiments
3. Measure before and after
4. Document intended behavior

**ONE actionable item:** Investigate visits tracking semantics and possibly increment on sampling (Score: 39, evidence-based)
