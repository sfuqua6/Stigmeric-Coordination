# Get Started in 5 Minutes

## Step 1: Install Dependencies (2 minutes)

```bash
# Basic installation
pip install -r requirements.txt

# Optional: For 75% memory savings (requires CUDA)
pip install bitsandbytes
```

---

## Step 2: Choose Your Mode (30 seconds)

### Option A: Just Run It (Fastest)
```bash
python main_async.py
```
**What you get**: 5-10x faster than original, instant startup

### Option B: Maximum Performance (if you have CUDA)
1. Edit `main_async.py`, find this section:
   ```python
   "use_quantization": False,
   ```
2. Change to:
   ```python
   "use_quantization": True,
   ```
3. Run:
   ```bash
   python main_async.py
   ```
**What you get**: 5-10x faster + 75% less memory (3.5GB vs 14GB)

### Option C: See the Difference
```bash
python demo_comparison.py
```
**What you get**: Side-by-side comparison of all modes

---

## Step 3: Customize (optional)

### Change the Thesis
Edit the thesis in `main_async.py`:
```python
thesis = "Your debate topic here"
```

### Use a Preset Config
```python
from configs.async_optimized import FAST_TEST_CONFIG  # Quick 20-action test
from main_async import run_swarm

run_swarm("Your thesis", FAST_TEST_CONFIG)
```

Available presets:
- `ASYNC_OPTIMIZED_CONFIG` - Best overall (quantization + async)
- `FAST_TEST_CONFIG` - Quick testing (20 actions)
- `CPU_CONFIG` - No GPU needed
- `MAX_QUALITY_CONFIG` - Highest quality output

---

## What to Expect

### First Run
```
SWARM DEBATE SYSTEM - ASYNC v0.3
================================================
Thesis: Social media has done more harm than good to democratic discourse
Async execution: ENABLED
Max concurrent agents: 5
Lazy loading: True
Quantization: False  (or True if you enabled it)
================================================

Initialized swarm with 6 agents

Loading model Qwen/Qwen2.5-7B-Instruct on cuda...
Model loaded successfully!

[ITER 1] Executing 3 agents concurrently...
  [DONE] 2.1s | Success: 3/3 | Failed: 0
    [OK] agent_Cl_0 (score: 0.75)
    [OK] agent_Ev_2 (score: 0.68)
    [OK] agent_Cr_4 (score: 0.71)
...
```

### Final Output
```
[QUALITY METRICS]
Avg claim confidence: 0.78
Evidence coverage: 85%
Critique coverage: 75%

[TOP CLAIMS]
1. Social media creates echo chambers that amplify extreme views
   Confidence: 0.85, Evidence: 2, Critiques: 1
2. Misinformation spreads 6x faster than factual content
   Confidence: 0.82, Evidence: 2, Critiques: 1
...

[SAVE] Output saved to: debate_output_async.json
```

---

## Common Issues

### Issue: "Out of memory"
**Solution**: Enable quantization or reduce concurrent agents
```python
"use_quantization": True,        # 14GB -> 3.5GB
"max_concurrent_agents": 3,      # Reduce from 5 to 3
```

### Issue: "bitsandbytes not found"
**Solution**: Either install it or disable quantization
```bash
# Install
pip install bitsandbytes

# Or disable
"use_quantization": False,
```

### Issue: "No module named 'swarm_debate'"
**Solution**: Make sure you're in the right directory
```bash
cd /path/to/swarmai
python main_async.py
```

### Issue: "CUDA not available"
**Solution**: Use CPU config
```python
from configs.async_optimized import CPU_CONFIG
from main_async import run_swarm
run_swarm("Your thesis", CPU_CONFIG)
```

---

## Next Steps

### Learn More
1. **Quick reference**: Read `QUICKREF.md` (5 min)
2. **Full guide**: Read `README.md` (15 min)
3. **Performance tuning**: Read `ASYNC_GUIDE.md` (20 min)

### Customize
1. Change thesis in `main_async.py`
2. Adjust `max_total_actions` for longer/shorter runs
3. Try different config presets

### Extend
1. Read `ARCHITECTURE.md` to understand design
2. Copy an agent from `swarm_debate/agents/`
3. Add your own agent type

---

## Quick Commands

```bash
# Run async (recommended)
python main_async.py

# Run sync (baseline)
python main.py

# Compare performance
python demo_comparison.py

# Run tests
python test_refactor.py

# Quick 20-action test
python -c "from configs.async_optimized import FAST_TEST_CONFIG; from main_async import run_swarm; run_swarm('Test', FAST_TEST_CONFIG)"
```

---

## Performance Expectations

| Mode | Time for 50 actions | Memory | GPU |
|------|-------------------|--------|-----|
| Sync | ~5-8 minutes | 14GB | Optional |
| Async | ~1-2 minutes | 14GB | Optional |
| Async + Quant | ~1-2 minutes | 3.5GB | Required |

**Speedup: 5-10x** with async
**Memory: 75% less** with quantization

---

## That's It!

You're ready to go. Just run:
```bash
python main_async.py
```

For more details, see:
- `QUICKREF.md` - Quick reference
- `README.md` - Full documentation
- `ASYNC_GUIDE.md` - Performance guide

**Questions?** Check the documentation files above or review the code - it's well commented!

---

**Welcome to the Swarm Debate System v0.3!**
*5-10x faster | 75% less memory | Production ready*
