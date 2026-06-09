# Troubleshooting Guide

## Issues Found During Testing (2025-11-11)

### ❌ CRITICAL: Model Loading Timeouts on CPU

**Problem**: The model times out during download and loading on CPU.

**What I Found**:
- The model download and CPU loading takes 60-90+ seconds
- `phi-2` is ~5.5GB and slow to load without GPU
- First run must download model files
- CPU inference without quantization is extremely slow

**Evidence from Testing**:
```
Loading model microsoft/phi-2 on cpu...
[INFO] Set pad_token to eos_token: <|endoftext|>
Fetching 2 files:   0%|          | 0/2 [00:00<?, ?it/s]
[TIMEOUT after 90 seconds]
```

### ✅ FIXED: Unicode Encoding Errors

**Problem**: Windows console (cp1252) can't display Unicode arrows and emojis.

**Fixed in commit fea1c45**:
- Replaced `→` with `[INFO]`, `[NL]`
- Replaced `✓` with `[OK]`, `[KB]`
- Replaced `⚠️` with `[!]`
- Removed emojis from print statements

**Files Fixed**:
- `swarm_debate/llm/model_manager_async.py`
- `swarm_debate/core/knowledge_base.py`
- `swarm_debate/llm/json_parser.py`
- `main_with_knowledge.py`

## Current Status

### What Works ✅
- Model loading code (no crashes)
- Tokenizer configuration
- Unicode-safe console output
- Knowledge base system
- Agent integration

### What Doesn't Work Yet ❌
- **Model actually generating responses** - Too slow on CPU
- First-time model download times out
- CPU inference is impractically slow

## Solutions

### Option 1: Pre-download the Model (Recommended)

Run this once to download the model:
```bash
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoTokenizer.from_pretrained('microsoft/phi-2'); print('Tokenizer downloaded'); AutoModelForCausalLM.from_pretrained('microsoft/phi-2'); print('Model downloaded')"
```

This will take 5-10 minutes but only needs to be done once.

### Option 2: Use a Smaller Model

Try `distilgpt2` (much smaller, faster on CPU):
```python
config = {
    "model_name": "distilgpt2",  # Only 353MB!
    "device": "cpu",
    ...
}
```

Update `swarm_debate/config/settings.py` line 15.

### Option 3: Use GPU (If Available)

The system will auto-detect CUDA:
```python
import torch
if torch.cuda.is_available():
    # GPU mode (much faster)
    config["use_quantization"] = True  # 4-bit quantization
else:
    # CPU mode (slow)
    config["use_quantization"] = False
```

### Option 4: Increase Timeouts

Edit `swarm_debate/orchestration/executor_async.py` line 40:
```python
output = await asyncio.wait_for(
    agent.execute(state, llm_manager),
    timeout=120.0  # Increase from 30 to 120 seconds
)
```

## Performance Expectations

### First Run (Model Download + Load)
- **CPU**: 5-10 minutes (download) + 60-90s (load) = 6-11 minutes
- **GPU**: 5-10 minutes (download) + 15-30s (load) = 5-11 minutes

### Subsequent Runs (Load Only)
- **CPU**: 60-90 seconds per load
- **GPU**: 15-30 seconds per load

### Generation Speed
- **CPU (phi-2)**: 1-3 tokens/second → ~30-60s per response
- **GPU (phi-2)**: 10-30 tokens/second → ~5-10s per response
- **CPU (distilgpt2)**: 5-10 tokens/second → ~10-20s per response

## Diagnostic Commands

### Check if Model is Downloaded
```bash
python -c "import os; path=os.path.expanduser('~/.cache/huggingface/hub/models--microsoft--phi-2'); print('Downloaded' if os.path.exists(path) else 'Not downloaded')"
```

### Test Model Loading Only
```bash
python -c "from transformers import AutoTokenizer; t=AutoTokenizer.from_pretrained('microsoft/phi-2'); print('Tokenizer OK')"
```

### Test Basic Generation
```bash
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; import torch; t=AutoTokenizer.from_pretrained('microsoft/phi-2'); m=AutoModelForCausalLM.from_pretrained('microsoft/phi-2'); inp=t('Hello', return_tensors='pt'); out=m.generate(**inp, max_new_tokens=5); print(t.decode(out[0]))"
```

## Next Steps

1. **Pre-download the model** (see Option 1 above)
2. **Test with a simpler model first** (distilgpt2)
3. **Consider GPU access** if available
4. **Increase timeouts** if still timing out

## Questions?

Check logs for specific errors:
- Unicode errors → Fixed in v0.3.3
- Timeout errors → Model loading/generation too slow
- Import errors → Missing dependencies
- CUDA errors → Quantization requires GPU

Last updated: 2025-11-11
Version: v0.3.3
