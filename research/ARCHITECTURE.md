# Swarm Debate System - Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         MAIN ENTRY POINT                         │
│                           (main.py)                              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
        ┌───────────────────────────────────────────────┐
        │          CONFIGURATION LAYER                   │
        │                                               │
        │  ┌─────────────┐      ┌──────────────┐       │
        │  │ settings.py │      │ prompts.py   │       │
        │  │  - CONFIG   │      │  - Templates │       │
        │  │  - Defaults │      │  - Temps     │       │
        │  └─────────────┘      └──────────────┘       │
        └───────────────────────────────────────────────┘
                             │
                             ▼
        ┌───────────────────────────────────────────────┐
        │            CORE COMPONENTS                     │
        │                                               │
        │  ┌──────────────┐  ┌──────────────┐          │
        │  │ StateManager │  │ Agent Base   │          │
        │  │              │  │              │          │
        │  │ - Claims     │  │ - Interface  │          │
        │  │ - Evidence   │  │ - Lifecycle  │          │
        │  │ - Critiques  │  │ - Scoring    │          │
        │  │ - Agents     │  └──────────────┘          │
        │  │ - Logs       │                            │
        │  │ [Thread-Safe]│  ┌──────────────┐          │
        │  └──────────────┘  │ Lifecycle    │          │
        │                    │ Manager      │          │
        │                    │              │          │
        │                    │ - Spawn      │          │
        │                    │ - Death      │          │
        │                    └──────────────┘          │
        └───────────────────────────────────────────────┘
                             │
                   ┌─────────┴─────────┐
                   ▼                   ▼
        ┌──────────────────┐  ┌──────────────────┐
        │   AGENT LAYER    │  │    LLM LAYER     │
        │                  │  │                  │
        │ ┌──────────────┐ │  │ ┌──────────────┐ │
        │ │Claim         │ │  │ │Model         │ │
        │ │Generator     │ │  │ │Manager       │ │
        │ └──────────────┘ │  │ │              │ │
        │                  │  │ │- Lazy Load   │ │
        │ ┌──────────────┐ │  │ │- Quantize    │ │
        │ │Evidence      │ │  │ │- Cache       │ │
        │ │Finder        │ │  │ └──────────────┘ │
        │ └──────────────┘ │  │                  │
        │                  │  │ ┌──────────────┐ │
        │ ┌──────────────┐ │  │ │JSON Parser   │ │
        │ │Critic        │ │  │ │              │ │
        │ │              │ │  │ │- 4 Strategies│ │
        │ └──────────────┘ │  │ └──────────────┘ │
        └──────────────────┘  └──────────────────┘
                   │                   │
                   └─────────┬─────────┘
                             ▼
        ┌───────────────────────────────────────────────┐
        │         ORCHESTRATION LAYER                    │
        │                                               │
        │  ┌──────────────┐  ┌──────────────┐          │
        │  │ Scheduler    │  │ Executor     │          │
        │  │              │  │              │          │
        │  │- Select      │  │- Run Action  │          │
        │  │  Agents      │  │- Update Score│          │
        │  └──────────────┘  │- Log Results │          │
        │                    └──────────────┘          │
        │  ┌──────────────┐                            │
        │  │ Termination  │                            │
        │  │              │                            │
        │  │- Time Limit  │                            │
        │  │- Action Limit│                            │
        │  │- Convergence │                            │
        │  └──────────────┘                            │
        └───────────────────────────────────────────────┘
                             │
                             ▼
        ┌───────────────────────────────────────────────┐
        │            SCORING LAYER                       │
        │                                               │
        │  ┌──────────────┐                            │
        │  │ Evaluator    │                            │
        │  │              │                            │
        │  │- Rankings    │                            │
        │  │- Quality     │                            │
        │  │- Metrics     │                            │
        │  └──────────────┘                            │
        └───────────────────────────────────────────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │  OUTPUT FILES  │
                    │                │
                    │ - JSON Results │
                    │ - Statistics   │
                    └────────────────┘
```

## Data Flow

### Main Loop Flow
```
START
  │
  ▼
Initialize State, LLM, Agents
  │
  ▼
┌─────────────────────────────┐
│     MAIN LOOP               │
│                             │
│  1. Check Termination? ────►│── YES ──► END
│      │                      │
│      NO                     │
│      │                      │
│      ▼                      │
│  2. Scheduler.get_active()  │
│      │                      │
│      ▼                      │
│  3. Scheduler.select()      │
│      │                      │
│      ▼                      │
│  4. Executor.execute()      │
│      │                      │
│      ▼                      │
│  5. Calculate Score         │
│      │                      │
│      ▼                      │
│  6. Update State            │
│      │                      │
│      ▼                      │
│  7. Lifecycle Check         │
│      │                      │
│      ▼                      │
│  8. Spawn/Death             │
│      │                      │
│      └──────────────────────┘
│
▼
Save Results
```

### Agent Execution Flow
```
Agent.should_activate()
  │
  ▼
Agent.execute()
  │
  ├─► Prepare Prompt
  │
  ├─► LLM.generate()
  │     │
  │     ├─► Lazy Load Model (if first time)
  │     ├─► Check Cache (if enabled)
  │     ├─► Run Inference
  │     └─► Return Response
  │
  ├─► Parse JSON
  │     │
  │     ├─► Try direct parse
  │     ├─► Extract {...}
  │     ├─► Extract ```json
  │     └─► Search for JSON
  │
  ├─► Update State
  │     │
  │     ├─► Add Claim/Evidence/Critique
  │     └─► Emit Event
  │
  └─► Return Output
       │
       ▼
Agent.calculate_score()
  │
  ▼
Update Agent.score (EMA)
```

## Component Responsibilities

### Configuration Layer
**Owns**: Settings, prompts, constants
**Used By**: All components
**Dependencies**: None

### Core Layer
**Owns**: State management, agent interface, lifecycle
**Used By**: Agents, orchestration
**Dependencies**: Configuration

### Agent Layer
**Owns**: Specific agent implementations
**Used By**: Orchestration
**Dependencies**: Core, LLM, Configuration

### LLM Layer
**Owns**: Model loading, inference, parsing
**Used By**: Agents
**Dependencies**: Configuration

### Orchestration Layer
**Owns**: Execution flow, scheduling, termination
**Used By**: Main entry point
**Dependencies**: Core, Agents, LLM

### Scoring Layer
**Owns**: Evaluation, metrics, rankings
**Used By**: Main entry point, agents
**Dependencies**: Core

## Concurrency Model

### Current (Phase 1)
```
Sequential Execution:
Agent 1 ──► Agent 2 ──► Agent 3 ──► Agent 4 ...
```

### Future (Phase 2 - Async)
```
Parallel Execution:
Agent 1 ──┐
Agent 2 ──├──► Concurrent
Agent 3 ──├──► 5-10 agents
Agent 4 ──┘
```

## State Management

### Thread Safety
```python
class StateManager:
    def __init__(self):
        self._lock = Lock()

    def add_claim(self, claim):
        with self._lock:
            # Atomic operation
            self.claims.append(claim)
```

### State Transitions
```
Empty State
  │
  ▼
Initialize with Thesis
  │
  ├─► Add Claims
  ├─► Add Evidence (linked to claims)
  └─► Add Critiques (linked to claims)
       │
       ▼
Check Convergence
  │
  ├─► NOT CONVERGED ──► Continue
  └─► CONVERGED ──────► Terminate
```

## Lifecycle Management

### Agent Lifecycle
```
SPAWN
  │
  ▼
Initialize (score=0.5)
  │
  ▼
Execute Actions
  │
  ├─► Success ──► Score Up
  └─► Failure ──► Score Down
       │
       ▼
   Check Score
       │
       ├─► score > 0.75 ──► Maybe SPAWN child
       │
       └─► score < 0.25 ──► Maybe DIE
```

### Population Dynamics
```
Initial: 2 of each type (6 agents)
  │
  ▼
Execute & Evaluate
  │
  ├─► High performers spawn
  │   (capped at max_per_type=5)
  │
  └─► Low performers die
      (keep at least 1 per type)
       │
       ▼
Population size oscillates: 3-15 agents
```

## Performance Characteristics

### Time Complexity
- State operations: O(1) with lock contention
- Agent selection: O(n) where n = num_agents
- Claim lookups: O(n) where n = num_claims (can be optimized)
- Evidence/Critique lookup: O(n) (can be indexed)

### Space Complexity
- State: O(claims + evidence + critiques + agents)
- Model: 14GB (full) or 3.5GB (4-bit quantized)
- Cache: O(unique_prompts) if caching enabled

## Extension Points

### Adding New Agent Types
1. Create class inheriting from `Agent`
2. Implement `should_activate()`, `execute()`, `calculate_score()`
3. Add to agent_classes dict in main.py
4. Add prompts to prompts.py

### Adding New Termination Conditions
1. Edit `orchestration/termination.py`
2. Add condition check in `check_termination()`

### Adding New Metrics
1. Edit `scoring/evaluator.py`
2. Add calculation in `calculate_debate_quality()`

### Adding Async Support
1. Convert `ModelManager.generate()` to async
2. Convert `Agent.execute()` to async
3. Use `asyncio.gather()` in orchestration
4. Add semaphore for concurrency limit

## Security Considerations

### Current Mitigations
- No user input directly to LLM (prompts are templates)
- JSON parsing has fallbacks (no eval())
- State mutations are locked
- File output uses safe json.dump()

### Future Considerations
- Prompt injection prevention
- Rate limiting for LLM calls
- Input sanitization if user prompts added
- Model output validation

## Monitoring & Observability

### Current Metrics
- Total actions
- Actions per minute
- Claim/evidence/critique counts
- Agent population
- Individual agent scores
- Average claim confidence

### Logging Points
- Model loading
- Agent spawns/deaths
- Action success/failure
- Convergence detection
- Error conditions

## Comparison: Before vs After

### Before (Monolithic)
- Single 607-line file
- Global state dictionary
- No separation of concerns
- Difficult to test
- Hard to extend

### After (Modular)
- 20 organized modules
- Encapsulated StateManager
- Clear separation of concerns
- Comprehensive test suite
- Easy to extend

## Next Steps

See `IMPLEMENTATION_QUESTIONS.md` for detailed roadmap of Phase 2 (async) and Phase 3 (advanced features).
