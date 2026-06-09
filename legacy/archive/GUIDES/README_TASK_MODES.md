# Task-Based Stigmergic Swarm

The swarm system now supports multiple task types beyond just debates!

## Quick Start

```bash
# Use predefined tasks
python run_task.py creative
python run_task.py debate
python run_task.py analysis
python run_task.py problem_solving

# Or provide your own prompt
python run_task.py creative "Write a haiku about artificial intelligence"
python run_task.py analysis "What are the societal impacts of social media?"
python run_task.py problem_solving "How can we make cities more sustainable?"
```

## Supported Task Types

### 1. Creative Mode (`creative`)
**Use for**: Poems, stories, creative writing, artistic content

**Default prompt**: "Write a poem on the human condition of disconnection"

**Signal types**:
- `DRAFT` - Initial creative attempts
- `REFINEMENT` - Improvements and enhancements
- `CRITIQUE` - Artistic/craft analysis
- `ALTERNATIVE` - Different creative approaches

**Example**:
```bash
python run_task.py creative "Write a short story about time travel"
```

### 2. Debate Mode (`debate`)
**Use for**: Arguing for/against a thesis, exploring controversies

**Default prompt**: "Climate change requires immediate global action..."

**Signal types**:
- `CLAIM` - Arguments and positions
- `EVIDENCE` - Supporting data and research
- `CRITIQUE` - Analytical weaknesses
- `COUNTER_EVIDENCE` - Contradictory evidence

**Example**:
```bash
python run_task.py debate "Universal basic income would benefit society"
```

### 3. Analysis Mode (`analysis`)
**Use for**: Analyzing topics, exploring implications, critical thinking

**Default prompt**: "Analyze the economic implications of universal basic income"

**Signal types**:
- `OBSERVATION` - Key insights and patterns
- `INSIGHT` - Deeper analysis and development
- `CRITIQUE` - Analytical gaps
- `COUNTERPOINT` - Alternative interpretations

**Example**:
```bash
python run_task.py analysis "Examine the psychological effects of remote work"
```

### 4. Problem Solving Mode (`problem_solving`)
**Use for**: Finding solutions, evaluating approaches, practical challenges

**Default prompt**: "How can we reduce urban traffic congestion..."

**Signal types**:
- `SOLUTION` - Proposed approaches
- `IMPLEMENTATION` - Practical details and steps
- `CHALLENGE` - Obstacles and difficulties
- `OBJECTION` - Reasons why solutions won't work

**Example**:
```bash
python run_task.py problem_solving "How to reduce food waste in restaurants?"
```

## How It Works

### Agent Roles (Same across all modes)

1. **Scouts** - Generate initial signals (drafts, claims, observations, solutions)
2. **Foragers (Support)** - Develop and refine strong signals
3. **Foragers (Critique)** - Identify weaknesses and gaps
4. **Critics** - Focused analytical critique
5. **Haters** - Adversarial challenges and alternatives

### Stigmergic Dynamics

- **Signal Strength**: 0.0 to 1.0, indicates quality/importance
- **Amplification**: Strong signals attract more attention and get amplified
- **Decay**: All signals weaken over time (5% per iteration)
- **Pruning**: Weak signals (< 0.15) are removed
- **Visits**: Tracks how many times a signal was used/referenced

### Results

The swarm produces:
- **Top signals by type** - Strongest drafts, refinements, critiques, etc.
- **Final statistics** - Total signals, distribution, average strength
- **Performance metrics** - Cache hit rate, generation success rate

## Output Examples

### Creative Mode Output
```
--- Top DRAFTs ---
1. [Strength: 0.894]
   In silent rooms we sit alone,
   Screens glowing bright, hearts turned to stone...

--- Top REFINEMENTs ---
1. [Strength: 1.000]
   Enhanced with vivid imagery of disconnection,
   Stronger metaphors, deeper emotional resonance...
```

### Problem Solving Output
```
--- Top SOLUTIONs ---
1. [Strength: 0.912]
   Implement congestion pricing in city centers
   during peak hours...

--- Top IMPLEMENTATIONs ---
1. [Strength: 0.876]
   Deploy automated toll systems, gradual rollout
   starting Q1 2025...
```

## Configuration

Edit `swarm/core/config.py` to adjust:
- Number of agents (`NUM_SCOUTS`, `NUM_FORAGERS`, etc.)
- Agent temperatures (exploration vs exploitation)
- Decay rate, thresholds, iterations
- Model selection

## Advanced: Custom Task Types

See `swarm/core/task_config.py` to:
- Modify existing task templates
- Create new task types
- Customize prompt templates
- Define new signal types

## Comparison: Original vs Task-Based

**Before** (debate-only):
```bash
python run_swarm.py  # Only debates
```

**Now** (flexible):
```bash
python run_task.py creative "YOUR PROMPT"
python run_task.py analysis "YOUR QUESTION"
python run_task.py problem_solving "YOUR CHALLENGE"
```

## Technical Details

### Task Configuration Structure
```python
TaskConfig(
    task_type="creative",
    task_prompt="Write a poem...",
    signal_types={
        "initial": "DRAFT",
        "support": "REFINEMENT",
        "critique": "CRITIQUE",
        "counter": "ALTERNATIVE"
    },
    scout_prompt_template="...",
    # ... more templates
)
```

### Agent Adaptation

Agents automatically adapt their behavior based on task type:
- Scouts use task-specific exploration prompts
- Foragers develop signals appropriately (evidence vs refinement)
- Critics provide relevant analytical feedback
- Haters challenge based on task context

## Examples of Usage

### Creative Writing Workshop
```bash
python run_task.py creative "Write a sci-fi opening paragraph"
python run_task.py creative "Describe a futuristic cityscape"
python run_task.py creative "Create a character backstory for a cyberpunk detective"
```

### Research & Analysis
```bash
python run_task.py analysis "Impact of AI on employment"
python run_task.py analysis "Privacy concerns in social media"
python run_task.py analysis "Evolution of remote education"
```

### Innovation & Solutions
```bash
python run_task.py problem_solving "Reduce corporate carbon footprint"
python run_task.py problem_solving "Improve public transportation accessibility"
python run_task.py problem_solving "Combat misinformation online"
```

## Architecture Benefits

✓ **Flexible** - Same swarm intelligence, different applications
✓ **Extensible** - Easy to add new task types
✓ **Template-based** - Consistent structure across modes
✓ **Stigmergic** - Emergent quality through indirect coordination
✓ **Multi-agent** - Diverse perspectives and critique

---

**Original debate mode still available**: `python run_swarm.py`

**New flexible system**: `python run_task.py <mode> "<prompt>"`
