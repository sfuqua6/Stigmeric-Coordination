# Benchmarking Infrastructure

Rigorous evaluation framework for swarm intelligence system.

## Quick Start

```bash
# Run benchmarks
python run_benchmark.py

# This will:
# 1. Load 3 datasets (TruthfulQA, MMLU, GSM8K)
# 2. Run swarm system on each question
# 3. Run baseline (single LLM) on each question
# 4. Compare accuracy, tokens, statistical significance
# 5. Generate verdict
```

## Datasets

### TruthfulQA (5 questions)
- Tests factual accuracy and truthfulness
- Questions about common misconceptions
- Example: "What happens if you crack your knuckles?"

### MMLU (3 questions)
- Multi-task language understanding
- Covers biology, history, physics
- Example: "What is the powerhouse of the cell?"

### GSM8K (3 questions)
- Grade school math reasoning
- Arithmetic, geometry, word problems
- Example: "If a train travels at 60 mph for 3 hours, how far?"

## Evaluation Metrics

### Accuracy
- **Swarm Accuracy**: % correct using swarm system
- **Baseline Accuracy**: % correct using single LLM
- **Improvement**: swarm - baseline

### Statistical Significance
- **p-value**: From t-test comparing results
- **Significant**: p < 0.05

### Efficiency
- **Token Overhead**: swarm_tokens / baseline_tokens
- **Efficiency Ratio**: accuracy per 1000 tokens

## Interpreting Results

### ✓ Success Criteria
```
Improvement > 0.05 (5% better than baseline)
AND
Improvement > (overhead - 1) * 0.02 (justifies token cost)
```

### Example Verdicts

**Case 1: Swarm Wins**
```
Swarm:    0.750 (6/8 correct)
Baseline: 0.625 (5/8 correct)
Delta:    +0.125 ✓
Tokens:   2.3x overhead

Verdict: ✓ Improvement (12.5%) justifies overhead (2.3x)
```

**Case 2: Swarm Loses**
```
Swarm:    0.600 (3/5 correct)
Baseline: 0.800 (4/5 correct)
Delta:    -0.200 ✗
Tokens:   5.2x overhead

Verdict: ✗ Swarm worse than baseline - investigate why
```

**Case 3: Marginal Win**
```
Swarm:    0.680 (17/25 correct)
Baseline: 0.640 (16/25 correct)
Delta:    +0.040 ⚠
Tokens:   10.5x overhead

Verdict: ⚠ Slight improvement but 10x overhead not justified
```

## Extending

### Add More Questions
Edit `benchmarking/datasets.py`:
```python
# TruthfulQALoader.load()
questions.append(BenchmarkQuestion(
    id="tqa_006",
    question="Your question here",
    answer="Ground truth answer",
    category="category",
    metadata={}
))
```

### Add New Dataset
```python
class YourDatasetLoader(DatasetLoader):
    def load(self, limit=None):
        questions = [...]
        return questions

# In datasets.py get_dataset():
loaders['yourdataset'] = YourDatasetLoader
```

### Customize Similarity Threshold
```python
evaluator = SwarmEvaluator(similarity_threshold=0.8)  # Default: 0.7
```

## Current Limitations

1. **Small datasets**: Only 5-11 questions total (for speed)
   - Expand to 50-100 questions for production
2. **Placeholder swarm**: `run_benchmark.py` simulates swarm
   - TODO: Integrate with actual `run_task.py` pipeline
3. **Fuzzy matching**: Uses string similarity for answer checking
   - May miss semantically equivalent answers
   - TODO: Use semantic similarity (embeddings)

## Next Steps

After running benchmarks:

1. **If swarm loses**: Debug before proceeding
   - Check scout signal quality
   - Verify synthesis uses scout outputs
   - Examine token distribution

2. **If swarm wins**: Proceed to Phase 2
   - Redesign scouts as simple agents
   - Add spatial structure
   - Re-benchmark after each change

3. **If marginal**: Analyze cost/benefit
   - Is 2% improvement worth 10x tokens?
   - Where are tokens being wasted?
   - Can we reduce overhead?
