# Quick Test Mode - Rapid A/B Comparison

Fast testing mode for comparative evaluation during development.

---

## Usage

### Quick Mode (60-90 seconds)
```bash
python comparative_evaluation.py --quick
```

**What it does:**
- ⚡ 3 iterations (instead of 10-20)
- ⚡ 3 scouts, 1 forager, 1 critic (instead of 5 scouts, 2 foragers, 1 critic)
- ⚡ 1 short prompt (instead of 3 comprehensive prompts)
- ✅ Shows full baseline AND swarm outputs
- ✅ Complete quality comparison metrics
- ✅ Saves results to JSON

**Expected runtime:** ~60-90 seconds

---

### Full Mode (5-10 minutes)
```bash
python comparative_evaluation.py
```

**What it does:**
- 📊 10-20 iterations for thorough refinement
- 📊 5 scouts, 2 foragers, 1 critic
- 📊 3 comprehensive test prompts
- 📊 Detailed analysis across multiple domains
- 📊 Full comparative metrics

**Expected runtime:** ~5-10 minutes

---

## What You'll See

### Quick Mode Output

```
⚡ QUICK MODE ENABLED - Fast testing with minimal agents
   - 3 iterations instead of 20
   - 3 scouts instead of 5
   - 1 test prompt instead of 3
   - Expected runtime: ~60-90 seconds

[COMPARATIVE] Initializing baseline and swarm...
[BASELINE] Initializing Phi-2...
[BASELINE] Ready
[COMPARATIVE] Ready for A/B testing

======================================================================
A/B COMPARISON
Prompt: Should AI be regulated? Give pros and cons.
======================================================================

[1/2] Running BASELINE (single-agent)...
[BASELINE] Generating initial response...
[BASELINE] Refinement iteration 1/3...
[BASELINE] Refinement iteration 2/3...
[BASELINE] Complete in 12.5s
[BASELINE] LLM generations: 3
[BASELINE] Response length: 847 chars

[2/2] Running SWARM (multi-agent)...
[SWARM] Running swarm on: Should AI be regulated? Give pros and cons.
[SWARM] Max iterations: 3
[SWARM] Quick mode: True
[SWARM] Quick mode: 3 scouts, 1 foragers, 1 critics
[SWARM] Launching 3 scouts, 1 foragers, 1 critics...
[SWARM] Running for 3 iterations...
[SWARM] [ITER 01] Signals: 3, Avg strength: 0.78
[SWARM] Synthesizing results...
[SWARM] Complete in 38.2s
[SWARM] Signals: 8
[SWARM] LLM generations: 12

[ANALYSIS] Comparing outputs...

======================================================================
COMPARISON SUMMARY
======================================================================

Prompt: Should AI be regulated? Give pros and cons....

📊 QUALITY COMPARISON:
  Overall winner: SWARM
  Swarm wins: 5/8 metrics
  Key improvements:
    • evidence_count: +45.2%
    • reasoning_depth: +31.8%
    • nuance_score: +52.4%

⚡ EFFICIENCY:
  Runtime: 12.5s (baseline) vs 38.2s (swarm)
  LLM generations: 3 vs 12
  Quality per generation: SWARM wins

======================================================================

======================================================================
BASELINE OUTPUT:
======================================================================
[Full baseline response shown here]

======================================================================
SWARM OUTPUT:
======================================================================
[Full swarm response shown here]

======================================================================

======================================================================
OVERALL SUMMARY
======================================================================

Total comparisons: 1
Swarm wins: 5 quality metrics
Baseline wins: 3 quality metrics

Conclusion: SWARM OUTPERFORMS BASELINE

💾 Results saved to: evaluation_results/comparative_results.json

✅ Comparative evaluation complete!
⚡ Quick mode completed - for full evaluation run without --quick flag
```

---

## When to Use Each Mode

### Use Quick Mode (`--quick`) When:
- 🔧 **Development/debugging**: Testing changes to evaluation code
- 🔧 **Sanity checking**: Verifying swarm still works after code changes
- 🔧 **Quick demos**: Showing how comparison works
- 🔧 **Iterating**: Testing different prompts or configurations
- ⏱️ **Limited time**: Need results in under 2 minutes

### Use Full Mode (no flag) When:
- 📊 **Real benchmarking**: Publishing results or making decisions
- 📊 **Multiple prompts**: Testing across different domains
- 📊 **Thorough analysis**: Need comprehensive quality metrics
- 📊 **Final validation**: Before publishing or presenting
- 🎯 **Research**: Gathering data for papers or reports

---

## Output Files

Both modes save results to `evaluation_results/comparative_results.json`:

```json
{
  "timestamp": "2025-11-14T...",
  "quick_mode": true,
  "comparisons": [
    {
      "prompt": "Should AI be regulated?...",
      "baseline_result": {
        "output": "...",
        "runtime": 12.5,
        "llm_generations": 3
      },
      "swarm_result": {
        "output": "...",
        "runtime": 38.2,
        "llm_generations": 12
      },
      "quality_comparison": {
        "evidence_count": {
          "baseline": 8,
          "swarm": 12,
          "improvement_pct": 50.0,
          "winner": "swarm"
        },
        ...
      }
    }
  ]
}
```

---

## Customization

### Custom Prompt (Quick Mode)
```python
# Edit comparative_evaluation.py line 436-438
if args.quick:
    test_prompts = [
        "Your custom quick prompt here"
    ]
```

### Adjust Agent Count
```python
# Edit run_task_wrapper.py line 58-61
if quick_mode:
    num_scouts = 5     # Increase for more thorough (slower)
    num_foragers = 2   # More foragers = more connections
    num_critics = 1    # More critics = more validation
```

### Adjust Iterations
```python
# Edit comparative_evaluation.py line 78 and 92
max_iterations = 5 if self.quick_mode else 10  # Change 5 to your preference
```

---

## Performance Expectations

### Quick Mode
- **Baseline**: ~10-15 seconds (3 iterations × 3-5s per generation)
- **Swarm**: ~30-60 seconds (3 scouts + foragers + critics for 3 iterations)
- **Total**: ~60-90 seconds including comparison

### Full Mode
- **Baseline**: ~30-60 seconds (10 iterations)
- **Swarm**: ~2-4 minutes (5 scouts, 2 foragers, 1 critic for 20 iterations)
- **Total per prompt**: ~3-5 minutes
- **3 prompts**: ~9-15 minutes

---

## Tips

1. **Start with quick mode** to verify everything works
2. **Use quick mode during development** to iterate fast
3. **Run full mode before publishing** results
4. **Watch the outputs** in quick mode to see quality differences
5. **Check JSON results** for detailed metrics

---

## Example Workflow

```bash
# 1. Quick sanity check
python comparative_evaluation.py --quick

# 2. Review outputs (shown in terminal)
# Baseline: Simple response
# Swarm: Evidence-based, multi-perspective analysis

# 3. If swarm wins on quick test, run full evaluation
python comparative_evaluation.py

# 4. Analyze comprehensive results
cat evaluation_results/comparative_results.json | jq '.comparisons[].quality_comparison'

# 5. Share results
# Screenshots of terminal output
# JSON file for detailed analysis
```

---

## Troubleshooting

### Quick mode too fast, not seeing differences?
```bash
# Increase iterations (edit comparative_evaluation.py line 78)
max_iterations = 5 if self.quick_mode else 10
# Change to:
max_iterations = 7 if self.quick_mode else 10
```

### Quick mode still too slow?
```bash
# Reduce agents (edit run_task_wrapper.py line 59)
num_scouts = 3
# Change to:
num_scouts = 2
```

### Want to see signal evolution?
Check the JSON output:
```bash
cat evaluation_results/comparative_results.json | jq '.comparisons[0].swarm_result.debate_quality'
```

---

## What Quick Mode Tests

Even in quick mode, you're testing:
- ✅ Baseline LLM loading and generation
- ✅ Swarm agent creation and coordination
- ✅ Scout exploration
- ✅ Forager elaboration
- ✅ Critic evaluation
- ✅ Signal store mechanics
- ✅ Synthesis generation
- ✅ Quality comparison metrics
- ✅ Full A/B comparison framework

**It's a real test**, just faster with fewer iterations/agents.

---

## See Also

- `COMPARATIVE_BENCHMARKING.md` - Complete A/B testing guide
- `comparative_evaluation.py` - Source code
- `run_task_wrapper.py` - Swarm wrapper
- `single_agent_baseline.py` - Baseline implementation
