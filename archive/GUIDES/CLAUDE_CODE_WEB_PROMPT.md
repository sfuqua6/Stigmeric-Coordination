# CLAUDE CODE WEB - SYSTEM REFINEMENT PROMPT

**System Name:** Stigmergic Swarm Intelligence with Monolith-Breaking Document Processing
**Version:** 2.0 (Post-Transformation)
**Last Updated:** 2025-11-12
**Purpose:** Context for Claude Code to understand, refine, and advance this distributed AI system

---

## OUR VISION: WHAT WE'RE BUILDING TOWARD

### The Ultimate Goal
We are building a **truly distributed intelligence system** that can:

1. **Break the Monolith** - Process documents of unlimited size by distributing reading across hundreds of agents, never exceeding individual context limits
2. **Emerge Quality** - Generate insights through validation and convergence, not programmer heuristics
3. **Search and Synthesize** - Autonomously explore any topic via web search, ingest diverse sources, and synthesize coherent understanding
4. **Scale Gracefully** - Handle 10K tokens or 10M tokens with the same architecture
5. **Maintain Provenance** - Trace every insight back to source documents/sections/URLs
6. **Self-Improve** - Continuously refine through external validation and contrarian challenges

### Where We Are Now
- ✅ Core architecture transformed from creative generation to document processing
- ✅ Distributed reading via parallel scouts
- ✅ Pattern finding via forager clusters
- ✅ External validation via gatherers + MCP
- ✅ Convergence detection via Gini coefficient
- ✅ Web ingestion "buckshot" search mode
- ⚠️ Still needs refinement in quality, speed, and robustness

### Where We Need to Go
- 🎯 **Better pattern discovery** - Foragers sometimes miss connections
- 🎯 **Faster convergence** - Currently takes 300-500 iterations
- 🎯 **Stronger validation** - Evidence gathering can be shallow
- 🎯 **Smarter search** - Web ingestion needs better query generation
- 🎯 **Real-time processing** - Enable streaming/incremental processing
- 🎯 **Multi-modal support** - Images, tables, code, diagrams
- 🎯 **Collaborative swarms** - Multiple swarms working together

**Your role:** Help us identify weaknesses, propose improvements, and implement refinements that move us toward the ultimate vision.

---

## CRITICAL CONTEXT: WHAT THIS SYSTEM IS

This is a **stigmergic swarm intelligence system** that has been transformed from creative content generation into a **monolith-breaking document processor**. It solves the fundamental context window limitation problem by distributing document reading, pattern finding, and synthesis across hundreds of lightweight agents that communicate indirectly through signal deposits.

### The Core Problem It Solves

**MONOLITH PROBLEM:** Large documents (150K+ tokens) cannot fit in any single LLM's context window. Traditional approaches fail because they try to process everything at once.

**SOLUTION:** This system breaks the monolith by:
1. Splitting documents into 2K token sections
2. Assigning one section per scout (parallel distributed reading)
3. Having foragers discover patterns across clusters of observations (never seeing full corpus)
4. Using gatherers to validate with external knowledge sources
5. Converging to top validated insights
6. Synthesizing comprehensive answers from distributed knowledge

**No single agent ever sees more than 4K tokens, yet the system can process 300K+ token corpora.**

---

## SYSTEM ARCHITECTURE

### Agent Types and Roles

#### 1. Scout Agents (Readers)
**Purpose:** Distributed document reading
**Input:** ONE document section (2K tokens)
**Output:** 2-3 OBSERVATION signals with factual details

**Behavior:**
- Each scout reads exactly one section
- Extracts specific factual observations with evidence
- Deposits OBSERVATION signals with provenance metadata
- All scouts run in parallel (no dependencies)
- Strength starts neutral (0.5) - will evolve through validation

**File:** `swarm/agents/scout.py`

**Key Methods:**
```python
async def read_and_extract(signal_store, llm)
async def extract_observations(section, llm) -> List[str]
```

**Dual Mode:** Also supports legacy creative mode for run_task.py compatibility

#### 2. Forager Agents (Pattern Finders)
**Purpose:** Discover patterns by connecting observations from different sections
**Input:** Cluster of 3-5 related OBSERVATION signals
**Output:** INSIGHT signals describing patterns

**Behavior:**
- Samples clusters of related observations using semantic similarity
- Never sees full document corpus (only small clusters)
- Discovers patterns that connect disparate observations
- Deposits INSIGHT signals with full provenance chain
- Initial strength: 0.5 (neutral)

**File:** `swarm/agents/forager.py`

**Key Methods:**
```python
async def find_patterns(signal_store, llm, cluster_size, similarity_threshold)
async def discover_pattern(cluster, llm) -> str
```

#### 3. Gatherer Agents (External Validators)
**Purpose:** Validate insights using external knowledge sources
**Input:** Unvalidated INSIGHT signals
**Output:** EVIDENCE signals from web/Wikipedia

**Behavior:**
- Identifies insights lacking validation (< 2 evidence sources)
- Extracts searchable queries from insights using LLM
- Searches DuckDuckGo and Wikipedia via MCP client
- Deposits EVIDENCE signals linking back to parent insights
- Amplifies parent insights when evidence found

**File:** `swarm/agents/gatherer.py`

**Key Methods:**
```python
async def validate_insights(signal_store, llm, batch_size)
async def gather_evidence(insight, signal_store, llm)
```

#### 4. Critic Agents (Validation Evaluators)
**Purpose:** Quantitatively evaluate validation status and adjust signal strength
**Input:** INSIGHT signals
**Output:** Strength adjustments + audit trail

**Behavior:**
- Gets comprehensive validation metrics (evidence count, observation count)
- Calculates strength multipliers based on validation quality
- Amplifies well-validated insights (>1.0x)
- Decays poorly-validated insights (<1.0x)
- Deposits CRITIQUE signals as audit trail

**File:** `swarm/agents/critic.py`

**Key Methods:**
```python
async def evaluate_insights(signal_store, llm)
def calculate_multiplier(validation: dict) -> float
```

**Multiplier Logic:**
- Validation score ≥0.7: multiply by 1.3
- Validation score ≥0.5: multiply by 1.15
- Validation score <0.2: multiply by 0.7
- Has contradictions: multiply by 0.85

#### 5. Hater Agents (Contrarian Challengers)
**Purpose:** Challenge insights with counter-evidence and alternative perspectives
**Input:** Strong INSIGHT signals
**Output:** COUNTER signals

**Behavior:**
- Targets high-strength insights
- Generates counter-arguments and alternative interpretations
- Prevents echo chamber effects
- Forces insights to be robust against criticism

**File:** `swarm/agents/hater.py`

#### 6. Synthesizer Agent (Final Aggregator)
**Purpose:** Create coherent synthesis from top validated insights
**Input:** Top 10 INSIGHT signals by strength
**Output:** Final comprehensive answer

**Behavior:**
- Receives only the strongest, most validated insights
- Creates coherent narrative connecting insights
- Preserves key details and evidence
- Generates human-readable synthesis

**File:** `swarm/agents/synthesizer.py`

---

## CORE INFRASTRUCTURE

### Signal Store (Communication Hub)

**Purpose:** Shared environment where agents deposit and discover signals indirectly

**File:** `swarm/core/signal_store.py`

**Signal Structure:**
```python
@dataclass
class Signal:
    id: str              # Unique identifier
    type: str            # OBSERVATION, INSIGHT, EVIDENCE, CRITIQUE, COUNTER
    content: str         # The actual content
    strength: float      # 0.0-1.0, evolves through validation
    timestamp: float     # When deposited
    depositor: str       # Agent ID
    parent: Optional[str]  # Parent signal ID (for provenance)
    visits: int          # How many times accessed
    metadata: dict       # Additional context (sections, sources, etc.)
```

**Key Graph Operations:**
```python
def sample_cluster(signal_type, size, similarity_threshold) -> List[Signal]
def get_validation_status(signal_id) -> dict
def get_unvalidated_signals(signal_type, min_evidence) -> List[Signal]
def get_ancestors(signal_id, target_type) -> List[Signal]
def get_descendants(signal_id, target_type) -> List[Signal]
```

**Validation Status Structure:**
```python
{
    'evidence_count': int,        # Number of EVIDENCE children
    'observation_count': int,     # Number of OBSERVATION ancestors
    'critique_count': int,        # Number of CRITIQUE children
    'validation_score': float,    # 0.0-1.0 composite score
    'has_contradictions': bool    # Any COUNTER signals present
}
```

### Document Processor

**Purpose:** Load and intelligently split documents into processable sections

**File:** `swarm/documents/processor.py`

**Supported Formats:**
- PDF files (via PyPDF2)
- Markdown files
- Plain text files
- Web content (via web ingestion mode)

**Splitting Strategy:**
- Target: 2000 tokens per section (±500 tolerance)
- Respects paragraph boundaries (splits on double newlines)
- Preserves provenance (page numbers, source files, URLs)
- Adds metadata for tracking

**Key Classes:**
```python
@dataclass
class DocumentSection:
    section_id: int
    content: str
    source_document: str
    token_count: int
    metadata: dict  # page numbers, section headings, URLs

class DocumentProcessor:
    def __init__(target_tokens=2000, tolerance=500)
    def process_documents(paths: List[str]) -> List[DocumentSection]
    def load_pdf_file(path) -> tuple[str, dict]
    def split_into_sections(text, source_name, metadata) -> List[DocumentSection]
```

### MCP Client (External Knowledge)

**Purpose:** Access external knowledge sources for validation

**File:** `swarm/knowledge/mcp_client.py`

**Supported Sources:**
- **DuckDuckGo** - Free web search (no API key)
- **Wikipedia** - Direct API access
- **Simulated** - Fallback for testing

**Key Features:**
- Async execution via thread pools
- Relevance scoring (keyword matching)
- Deduplication (70% similarity threshold)
- Automatic disambiguation handling (Wikipedia)
- Clean text extraction and normalization

**Key Methods:**
```python
async def search(query, max_results=3) -> List[Dict]
async def search_wikipedia(query) -> Optional[Dict]
```

**Result Structure:**
```python
{
    'summary': str,      # Cleaned text summary
    'source': str,       # Source name/title
    'url': str,          # Source URL
    'type': str,         # 'web', 'wikipedia', 'simulated'
    'relevance': float   # 0.0-1.0 relevance score
}
```

### LLM Interface

**Purpose:** Unified interface to language models with quantization support

**File:** `swarm/llm/simple_llm.py`

**Supported Models:**
- microsoft/phi-2 (2.7B params) - **Recommended with quantization**
- gpt2 (fallback, weak)

**Key Features:**
- **8-bit quantization** via bitsandbytes (50% memory savings)
- Automatic fallback: quantization → float16 → float32 → gpt2
- LRU caching for efficiency
- Loud warnings if fallback occurs

**Quantization Loading:**
```python
llm = SimpleLLM("microsoft/phi-2", "cuda", use_quantization=True)
await llm.load()
# Loads with load_in_8bit=True, device_map="auto"
# Memory: ~3.5GB instead of ~7GB
```

---

## EXECUTION PIPELINE

### Main Orchestration: monolith_breaking.py

**File:** `swarm/monolith_breaking.py`

**Function:** `run_document_swarm(document_paths, model_name, num_foragers, num_gatherers, num_critics, num_haters, max_iterations, convergence_threshold, enable_mcp)`

### 7-Phase Execution Flow

#### PHASE 1: Document Loading
```python
processor = DocumentProcessor(target_tokens=2000, tolerance=500)
sections = processor.process_documents(document_paths)
```
- Loads all documents (PDF, markdown, text)
- Splits into 2K token sections with paragraph boundaries
- Creates metadata with provenance tracking
- Reports: total sections, total tokens, compression target

#### PHASE 2: Infrastructure Initialization
```python
signal_store = SignalStore(decay_rate, prune_threshold, diversity_threshold, exploration_bonus)
llm = SimpleLLM(model_name, device, use_quantization=None)  # Auto-enable on CUDA
await llm.load()
mcp_client = MCPClient(enable_web_search=True, enable_wikipedia=True) if enable_mcp else None
```

#### PHASE 3: Distributed Reading (Parallel)
```python
scouts = [Scout(f"Scout_{i}", section=section) for i, section in enumerate(sections)]
scout_tasks = [scout.run(signal_store, llm, max_actions=1) for scout in scouts]
await asyncio.gather(*scout_tasks)  # ALL SCOUTS RUN SIMULTANEOUSLY
```
- Creates one scout per section
- All scouts execute in parallel (no dependencies)
- Each extracts 2-3 observations
- Deposits OBSERVATION signals with metadata
- Reports: observations per section, total observations

#### PHASE 4: Pattern Finding
```python
foragers = [Forager(f"Forager_{i}", mode="document") for i in range(num_foragers)]
# Run for 150 iterations
for iteration in range(150):
    for forager in foragers:
        await forager.find_patterns(signal_store, llm, cluster_size=5, similarity_threshold=0.4)
```
- Foragers sample clusters of related observations
- Discover patterns connecting observations from different sections
- Deposit INSIGHT signals
- Progress reported every 10 iterations

#### PHASE 5: Validation Loop
```python
gatherers = [Gatherer(f"Gatherer_{i}", mcp_client) for i in range(num_gatherers)]
critics = [Critic(f"Critic_{i}", mode="document") for i in range(num_critics)]
# Run for 150 iterations
for iteration in range(150):
    # Gatherers search for external evidence
    for gatherer in gatherers:
        await gatherer.validate_insights(signal_store, llm, batch_size=3)
    # Critics evaluate validation status
    for critic in critics:
        await critic.evaluate_insights(signal_store, llm)
```
- Gatherers find external evidence for insights
- Critics adjust strengths based on validation quality
- Insights with strong validation amplify
- Insights without validation decay

#### PHASE 6: Convergence Monitoring
```python
insights = [s for s in signal_store.get_all_signals() if s.type == "INSIGHT"]
strengths = [s.strength for s in insights]
gini = calculate_gini_coefficient(strengths)
converged = gini >= convergence_threshold  # Target: >0.7
```
- Calculates Gini coefficient on insight strengths
- Gini measures inequality: 0=uniform, 1=winner-take-all
- Target >0.7 indicates clear winners emerged
- Reports convergence status every 50 iterations

**Gini Coefficient:**
```python
def calculate_gini_coefficient(strengths: List[float]) -> float:
    strengths = sorted(strengths)
    n = len(strengths)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * strengths)) / (n * np.sum(strengths)) - (n + 1) / n
```

#### PHASE 7: Synthesis
```python
top_insights = signal_store.get_top_signals("INSIGHT", n=10)
synthesizer = Synthesizer("Synthesizer")
synthesis = await synthesizer.synthesize(
    signal_store=signal_store,
    llm=llm,
    top_insights=top_insights,
    original_query="synthesize all insights"
)
```
- Takes top 10 validated insights
- Creates coherent narrative
- Preserves key evidence and details
- Generates human-readable answer

#### PHASE 8: Output and Diagnostics
```python
return {
    'synthesis': str,              # Final answer
    'top_insights': List[Signal],  # Top 10 insights
    'all_signals': List[Signal],   # All signals
    'stats': dict,                 # Statistics
    'converged': bool,             # Convergence status
    'gini_coefficient': float,     # Final Gini
    'compression_ratio': float,    # Input tokens / output tokens
    'processing_time': float       # Total time in seconds
}
```

---

## USAGE MODES

### Mode 1: Document Processing (Core Mode)

**Script:** `swarm/monolith_breaking.py`

**Usage:**
```python
from swarm.monolith_breaking import run_document_swarm

result = await run_document_swarm(
    document_paths=["doc1.pdf", "doc2.md", "doc3.txt"],
    model_name="microsoft/phi-2",
    num_foragers=50,
    num_gatherers=20,
    num_critics=20,
    num_haters=5,
    max_iterations=500,
    convergence_threshold=0.7,
    enable_mcp=True
)

print(result['synthesis'])
```

**Configuration:**
- `num_foragers`: 30-50 (more = better pattern coverage)
- `num_gatherers`: 10-20 (external validation)
- `num_critics`: 10-20 (validation evaluation)
- `num_haters`: 5-10 (contrarian challenges)
- `max_iterations`: 300-500 (more = better convergence)
- `convergence_threshold`: 0.7 (Gini coefficient target)

### Mode 2: Auto-Discovery

**Script:** `run_monolith_breaking.py`

**Purpose:** Automatically discover and process all documentation in repository

**Usage:**
```bash
python run_monolith_breaking.py
```

**Behavior:**
- Discovers all .md, .pdf, .txt files in repository
- Excludes: node_modules, .git, __pycache__, venv
- Filters: Files must be >500 bytes
- Asks for user confirmation
- Processes with standard configuration
- Saves to `repository_analysis.txt`

### Mode 3: Self-Analysis

**Script:** `run_self_analysis.py`

**Purpose:** Quick analysis of key repository documentation

**Usage:**
```bash
python run_self_analysis.py
```

**Behavior:**
- Processes these specific files:
  - A_NOTE_FOR_CLAUDE.md
  - TRANSFORMATION_STATUS.md
  - README.md
  - STIGMERGIC_SWARM_DESIGN.md
  - PROJECT_STRUCTURE.md
- No user confirmation (automatic)
- Lighter configuration (20 foragers, 200 iterations)
- Saves to `self_analysis.txt`

### Mode 4: Web Ingestion (Buckshot Search)

**Script:** `run_web_ingestion.py`

**Purpose:** Search the web, ingest relevant material, synthesize insights WITHOUT local documents

**Usage:**
```bash
python run_web_ingestion.py "What are the latest advances in quantum computing?"
python run_web_ingestion.py "How does climate change affect ocean ecosystems?"
```

**How It Works:**

1. **Buckshot Search Phase**
   - Takes your topic/question
   - Uses LLM to generate 8+ diverse search queries
   - Covers different angles, perspectives, aspects
   - Example topic: "quantum computing advances"
     - Query 1: "quantum computing breakthroughs 2025"
     - Query 2: "quantum error correction recent developments"
     - Query 3: "practical applications quantum computers"
     - Query 4: "quantum computing vs classical computing comparison"
     - etc.

2. **Web Fetching Phase**
   - Searches each query via DuckDuckGo + Wikipedia
   - Fetches actual web page content (HTTP requests)
   - Extracts clean text using BeautifulSoup
   - Removes navigation, scripts, ads, junk
   - Targets 15-20 diverse sources

3. **Document Processing Phase**
   - Converts web content to DocumentSections
   - Standard 2K token sections
   - Preserves provenance (URL, title)

4. **Synthesis Phase**
   - Runs full 7-phase pipeline on web content
   - Same distributed processing as document mode
   - External validation still active (MCP)
   - Generates comprehensive synthesis

**Configuration:**
```python
await run_web_ingestion(
    topic="Your question or topic",
    num_queries=8,           # Diverse search queries
    results_per_query=5,     # Results per query
    max_sources=20,          # Max web sources to fetch
    model_name="microsoft/phi-2",
    num_foragers=30,
    num_gatherers=15,
    num_critics=15,
    num_haters=5,
    max_iterations=300,
    convergence_threshold=0.7
)
```

**Dependencies:**
```bash
pip install requests beautifulsoup4 duckduckgo-search wikipedia
```

**Output:**
- `web_synthesis.txt` - Complete synthesis with sources
- Includes: synthesis, top insights, source URLs
- Full provenance from insights back to web sources

### Mode 5: Legacy Creative Tasks

**Script:** `run_task.py`

**Purpose:** Original creative mode for debates, creative writing, analysis

**Usage:**
```bash
python run_task.py creative "Write a haiku about AI"
python run_task.py debate "Should AI be regulated?"
python run_task.py analysis "What are the impacts of social media?"
```

**Note:** This uses the OLD stigmergic architecture (scouts generate ideas, not read documents). Scout agent maintains backward compatibility for this mode.

---

## KEY DESIGN PRINCIPLES

### 1. Stigmergic Communication
Agents communicate **indirectly** through the environment (signal deposits), not directly with each other. This enables massive parallelization and scalability.

### 2. Neutral Initial Strength
All observations and insights start at 0.5 (neutral). Strength evolves through:
- External validation (gatherers)
- Quantitative evaluation (critics)
- Contrarian challenges (haters)

**No arbitrary heuristics.** Quality emerges from validation, not programmer assumptions.

### 3. Full Provenance Tracking
Every signal carries metadata linking to:
- Parent signals (provenance chain)
- Source documents/sections
- Web sources (URLs)
- Page numbers, sections, timestamps

Can trace synthesis → insights → observations → sections → documents

### 4. Graph Operations on Signals
The signal store supports graph traversal:
- Ancestors: Follow parent links upward
- Descendants: Follow child links downward
- Clusters: Sample related signals by semantic similarity
- Validation queries: Check evidence, observations, contradictions

### 5. Convergence Detection
System monitors its own convergence via Gini coefficient:
- Early iterations: Gini ~0.3-0.4 (many insights have similar strength)
- Mid iterations: Gini ~0.5-0.6 (winners emerging)
- Late iterations: Gini >0.7 (clear winners, can stop)

### 6. No Context Window Violation
**Hard constraint:** No agent ever sees >4K tokens
- Scouts: 2K sections
- Foragers: 3-5 observations (~500 tokens each) = ~2.5K
- Gatherers: 1 insight + 3 search results = ~1K
- Critics: 1 insight + validation metrics = ~1K
- Synthesizer: 10 top insights = ~5K (only agent that approaches limit)

### 7. Parallel by Default
Whenever possible, agents run in parallel:
- All scouts run simultaneously (no dependencies)
- Foragers, gatherers, critics run in parallel (sampling different signals)
- Only serialized when necessary (synthesis waits for convergence)

---

## FILE STRUCTURE

```
swarmai/
├── swarm/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── scout.py          # Document readers (dual mode)
│   │   ├── forager.py        # Pattern finders (dual mode)
│   │   ├── gatherer.py       # External validators (NEW)
│   │   ├── critic.py         # Validation evaluators (transformed)
│   │   ├── hater.py          # Contrarian challengers
│   │   └── synthesizer.py    # Final aggregator
│   ├── core/
│   │   ├── signal_store.py   # Enhanced with graph operations
│   │   ├── config.py         # Configuration parameters
│   │   └── task_config.py    # Legacy task definitions
│   ├── documents/
│   │   ├── __init__.py
│   │   └── processor.py      # Document loading and splitting (NEW)
│   ├── knowledge/
│   │   ├── __init__.py
│   │   └── mcp_client.py     # External knowledge (real search)
│   ├── llm/
│   │   ├── __init__.py
│   │   └── simple_llm.py     # LLM interface with quantization
│   └── validation/
│       └── format_validator.py
├── run_monolith_breaking.py  # Auto-discovery mode (NEW)
├── run_self_analysis.py      # Self-analysis mode (NEW)
├── run_web_ingestion.py      # Web buckshot mode (NEW)
├── run_task.py               # Legacy creative mode
├── requirements.txt          # All dependencies
├── A_NOTE_FOR_CLAUDE.md      # Original context doc
├── TRANSFORMATION_STATUS.md   # Transformation details
└── CLAUDE_CODE_WEB_PROMPT.md # This file
```

---

## DEPENDENCIES

### Core (Required)
```
transformers>=4.40.0       # Hugging Face models
torch>=2.0.0               # PyTorch backend
sentence-transformers>=2.5.0  # Semantic similarity
faiss-cpu>=1.8.0           # Vector search
accelerate>=0.28.0         # Model loading optimization
numpy>=1.24.0              # Numerical operations
tiktoken>=0.5.0            # Token counting
```

### Document Processing
```
PyPDF2>=3.0.0              # PDF reading
```

### Quantization (Highly Recommended)
```
bitsandbytes>=0.41.0       # 8-bit quantization (50% memory savings)
```
**Requires:** CUDA GPU
**Benefit:** Phi-2 loads in ~3.5GB instead of ~7GB

### External Knowledge
```
duckduckgo-search>=4.0.0   # Free web search (no API key)
wikipedia>=1.4.0           # Wikipedia API
```

### Web Ingestion
```
requests>=2.31.0           # HTTP fetching
beautifulsoup4>=4.12.0     # HTML parsing
```

### Installation
```bash
# Full installation (recommended)
pip install -r requirements.txt

# Minimal (CPU only, no search)
pip install transformers torch numpy tiktoken PyPDF2

# With quantization (GPU)
pip install -r requirements.txt

# With web ingestion
pip install requests beautifulsoup4 duckduckgo-search wikipedia
```

---

## COMMON OPERATIONS

### Process Local Documents
```python
from swarm.monolith_breaking import run_document_swarm

result = await run_document_swarm(
    document_paths=["paper1.pdf", "paper2.pdf", "notes.md"],
    model_name="microsoft/phi-2",
    num_foragers=50,
    num_gatherers=20,
    num_critics=20,
    max_iterations=500,
    convergence_threshold=0.7,
    enable_mcp=True
)

print(f"Synthesis:\n{result['synthesis']}")
print(f"\nCompression: {result['compression_ratio']:.1f}:1")
print(f"Converged: {result['converged']}")
```

### Search Web and Synthesize
```bash
python run_web_ingestion.py "How do transformer models work?"
```

### Auto-Process Repository
```bash
python run_monolith_breaking.py
```

### Self-Analysis
```bash
python run_self_analysis.py
```

### Check Model Loading
```python
from swarm.llm.simple_llm import SimpleLLM

llm = SimpleLLM("microsoft/phi-2", "cuda", use_quantization=True)
await llm.load()
# Watch for: "SUCCESS: Model loaded with 8-bit quantization"
```

### Monitor Convergence
```python
from swarm.monolith_breaking import calculate_gini_coefficient

insights = signal_store.get_signals_by_type("INSIGHT")
strengths = [s.strength for s in insights]
gini = calculate_gini_coefficient(strengths)

print(f"Gini coefficient: {gini:.3f}")
if gini > 0.7:
    print("CONVERGED: Clear winners emerged")
else:
    print("NOT CONVERGED: Continue iterations")
```

### Query Validation Status
```python
validation = signal_store.get_validation_status(insight_id)

print(f"Evidence count: {validation['evidence_count']}")
print(f"Observation count: {validation['observation_count']}")
print(f"Validation score: {validation['validation_score']:.2f}")
print(f"Has contradictions: {validation['has_contradictions']}")
```

### Sample Related Observations
```python
cluster = signal_store.sample_cluster(
    signal_type="OBSERVATION",
    size=5,
    similarity_threshold=0.4
)

for obs in cluster:
    print(f"{obs.id}: {obs.content[:100]}...")
```

---

## PERFORMANCE CHARACTERISTICS

### Memory Usage
**Without Quantization (Phi-2):**
- Model: ~7GB VRAM
- Embeddings: ~2GB VRAM
- Total: ~9GB VRAM
- **Requires:** RTX 3080 or better

**With 8-bit Quantization (Phi-2):**
- Model: ~3.5GB VRAM (50% savings!)
- Embeddings: ~2GB VRAM
- Total: ~5.5GB VRAM
- **Works on:** RTX 3060 (12GB)

**CPU Fallback (GPT-2):**
- Automatically falls back if GPU unavailable
- Slow but functional
- Quality significantly lower

### Processing Speed
**Typical Document Processing (50K tokens, 25 sections):**
- Phase 1 (Loading): 2-5 seconds
- Phase 2 (Init): 10-15 seconds (model loading)
- Phase 3 (Reading): 30-60 seconds (parallel scouts)
- Phase 4 (Pattern Finding): 3-5 minutes (150 iterations)
- Phase 5 (Validation): 3-5 minutes (150 iterations)
- Phase 6-7 (Synthesis): 30 seconds
- **Total: 7-12 minutes**

**Scaling:**
- 100K tokens: 12-20 minutes
- 200K tokens: 20-35 minutes
- 500K tokens: 45-90 minutes

**Bottlenecks:**
- LLM generation speed (depends on GPU)
- External search latency (gatherers)
- Number of iterations (more = better quality but slower)

### Compression Ratios
**Typical Ratios:**
- Documents: 30:1 to 100:1
- Web sources: 50:1 to 200:1

**Example:**
- Input: 150K tokens (75 sections)
- Observations: ~150 (3K tokens)
- Insights: ~30 (15K tokens)
- Top insights: 10 (5K tokens)
- Synthesis: 1.5K tokens
- **Compression: 100:1**

---

## TROUBLESHOOTING

### Model Won't Load
**Symptom:** Errors during model loading or silent fallback to GPT-2

**Solutions:**
1. Check CUDA availability:
   ```python
   import torch
   print(torch.cuda.is_available())  # Should be True
   print(torch.cuda.get_device_name(0))  # Should show GPU
   ```

2. Enable quantization explicitly:
   ```python
   llm = SimpleLLM("microsoft/phi-2", "cuda", use_quantization=True)
   ```

3. Check VRAM:
   ```bash
   nvidia-smi  # Should show available VRAM
   ```

4. Install bitsandbytes:
   ```bash
   pip install bitsandbytes
   ```

### Out of Memory Errors
**Symptom:** CUDA OOM during model loading or generation

**Solutions:**
1. Enable quantization (50% reduction)
2. Reduce batch sizes in agents
3. Use smaller model (switch to GPT-2 for testing)
4. Close other GPU applications
5. Reduce number of parallel agents

### Poor Quality Results
**Symptom:** Synthesis is generic, insights are weak

**Solutions:**
1. **Increase iterations:** 300 → 500
2. **Enable MCP:** Set `enable_mcp=True` for external validation
3. **More foragers:** 30 → 50 for better pattern coverage
4. **Check convergence:** If Gini <0.5, need more iterations
5. **Verify model loading:** Ensure not falling back to GPT-2

### Slow Processing
**Symptom:** Takes much longer than expected

**Solutions:**
1. Check GPU utilization:
   ```bash
   nvidia-smi  # Should show high GPU usage
   ```
2. Verify quantization enabled (faster inference)
3. Reduce external validation (gatherers): 20 → 10
4. Reduce iterations: 500 → 300
5. Enable caching (should be default)

### No External Validation
**Symptom:** No EVIDENCE signals, gatherers report search failures

**Solutions:**
1. Install search dependencies:
   ```bash
   pip install duckduckgo-search wikipedia
   ```
2. Check internet connectivity
3. Verify MCP client initialization:
   ```python
   mcp = MCPClient(enable_web_search=True, enable_wikipedia=True)
   results = await mcp.search("test query")
   print(results)  # Should return real results
   ```

### Web Ingestion Fails
**Symptom:** No web content fetched, all sources fail

**Solutions:**
1. Install web dependencies:
   ```bash
   pip install requests beautifulsoup4
   ```
2. Check internet connectivity
3. Check for rate limiting (wait and retry)
4. Try different topics (some sites block bots)

---

## ADVANCED CONFIGURATION

### Tuning Convergence
```python
# Aggressive convergence (faster, may miss insights)
convergence_threshold = 0.65  # Lower threshold
max_iterations = 200         # Fewer iterations

# Conservative convergence (slower, more thorough)
convergence_threshold = 0.75  # Higher threshold
max_iterations = 800          # More iterations
```

### Agent Balance
```python
# Pattern-finding focused (discover more connections)
num_foragers = 80   # High
num_gatherers = 10  # Low
num_critics = 10    # Low

# Validation-focused (stronger evidence)
num_foragers = 30   # Moderate
num_gatherers = 40  # High
num_critics = 40    # High

# Balanced (recommended)
num_foragers = 50
num_gatherers = 20
num_critics = 20
num_haters = 10
```

### Signal Store Tuning
```python
signal_store = SignalStore(
    decay_rate=0.01,           # Slower decay = signals persist longer
    prune_threshold=0.05,      # Lower = more aggressive pruning
    diversity_threshold=0.7,   # Higher = require more diversity
    exploration_bonus=0.05     # Higher = favor underexplored signals
)
```

### Document Splitting
```python
# Smaller sections (more granular, more scouts)
processor = DocumentProcessor(target_tokens=1000, tolerance=300)

# Larger sections (fewer scouts, faster)
processor = DocumentProcessor(target_tokens=3000, tolerance=800)

# Standard (recommended)
processor = DocumentProcessor(target_tokens=2000, tolerance=500)
```

---

## EXTENDING THE SYSTEM

### Adding New Agent Types

1. Create agent class in `swarm/agents/`
2. Implement `async def run(signal_store, llm, ...)`
3. Use signal store for communication
4. Add to orchestration in `monolith_breaking.py`

**Template:**
```python
class NewAgent:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.active = True

    async def run(self, signal_store: SignalStore, llm: SimpleLLM, max_actions: int = 100):
        while self.active and self.actions_taken < max_actions:
            # 1. Sample signals from store
            signals = signal_store.sample_weighted("INSIGHT", n=3)

            # 2. Process signals (use LLM)
            result = await llm.generate(prompt, ...)

            # 3. Deposit new signal
            signal_store.deposit(
                signal_type="NEW_TYPE",
                content=result,
                strength=0.5,
                depositor=self.agent_id,
                parent=signals[0].id,
                metadata={...}
            )

            await asyncio.sleep(0.1)
```

### Adding New Signal Types

1. Choose signal type name (e.g., "SUMMARY", "QUESTION", "HYPOTHESIS")
2. Add to signal type constants if needed
3. Update orchestration to handle new type
4. Consider validation/strength evolution logic

### Custom Document Loaders

```python
class CustomDocumentProcessor(DocumentProcessor):
    def load_custom_format(self, file_path: Path) -> tuple[str, dict]:
        """Load custom document format."""
        # Your loading logic
        text = ...
        metadata = {...}
        return text, metadata

    def process_documents(self, paths: List[str]) -> List[DocumentSection]:
        # Override to support custom format
        for path in paths:
            if path.endswith('.custom'):
                text, metadata = self.load_custom_format(Path(path))
                sections = self.split_into_sections(text, path, metadata)
                all_sections.extend(sections)
        return all_sections
```

### Custom Validation Sources

```python
class CustomMCPClient(MCPClient):
    async def search_custom_source(self, query: str) -> List[Dict]:
        """Search custom knowledge source."""
        # Your search logic
        return results

    async def search(self, query: str, max_results: int = 3) -> List[Dict]:
        results = []

        # Add custom source
        custom_results = await self.search_custom_source(query)
        results.extend(custom_results)

        # Call parent for standard sources
        results.extend(await super().search(query, max_results))

        return self._deduplicate_results(results)[:max_results]
```

---

## IMPORTANT NOTES FOR CLAUDE CODE

### When Working With This System:

1. **Understand the transformation:** This was RECENTLY transformed from creative generation to document processing. Some legacy code (run_task.py) remains for backward compatibility.

2. **Always maintain dual modes:** Scout, Forager, and Critic agents support both document processing mode and legacy creative mode. Don't break backward compatibility.

3. **Respect the 4K constraint:** No agent should ever see >4K tokens. This is a hard architectural constraint.

4. **Preserve provenance:** All signals should carry metadata linking to sources. This is critical for trust.

5. **Neutral strength starts:** New signals always start at 0.5. Strength evolves through validation, not heuristics.

6. **Test quantization:** Always verify model loading succeeds with quantization. Watch for fallback warnings.

7. **Graph operations:** When adding features involving signal relationships, use the existing graph operations (ancestors, descendants, clusters).

8. **Async by default:** All agent operations are async. Use `await` for LLM calls and external searches.

9. **Parallel when possible:** Maximize parallelization. Use `asyncio.gather()` for independent operations.

10. **Monitor convergence:** Check Gini coefficient to determine if processing should continue or stop.

### Common Modification Requests:

**"Add support for X document format"**
→ Extend DocumentProcessor.process_documents() with new loader

**"Improve insight quality"**
→ Increase iterations, enable MCP, add more foragers/gatherers

**"Process is too slow"**
→ Enable quantization, reduce iterations, reduce gatherers

**"Add new external knowledge source"**
→ Extend MCPClient with new search method

**"Results don't converge"**
→ Check: Are gatherers finding evidence? Are critics evaluating? Increase iterations.

**"Memory issues"**
→ Enable quantization, reduce model size, check VRAM

---

## EXAMPLE WORKFLOWS

### Workflow 1: Analyze Research Papers
```python
# Scenario: You have 10 research papers (PDFs) to analyze

papers = [
    "paper1.pdf", "paper2.pdf", "paper3.pdf",
    "paper4.pdf", "paper5.pdf", "paper6.pdf",
    "paper7.pdf", "paper8.pdf", "paper9.pdf", "paper10.pdf"
]

result = await run_document_swarm(
    document_paths=papers,
    model_name="microsoft/phi-2",
    num_foragers=50,        # High for cross-paper patterns
    num_gatherers=20,       # Validate findings
    num_critics=20,
    num_haters=10,          # Challenge assumptions
    max_iterations=500,
    convergence_threshold=0.7,
    enable_mcp=True         # External validation
)

# Result: Comprehensive synthesis of key findings across all papers
print(result['synthesis'])

# Top insights with validation
for insight in result['top_insights'][:5]:
    print(f"\n[{insight.strength:.2f}] {insight.content}")

    # Check validation
    validation = signal_store.get_validation_status(insight.id)
    print(f"  Evidence: {validation['evidence_count']}")
    print(f"  Observations: {validation['observation_count']}")
```

### Workflow 2: Research New Topic (No Local Docs)
```bash
# Scenario: You want to learn about quantum computing but have no documents

python run_web_ingestion.py "What are the key principles and recent advances in quantum computing?"

# Result: System will:
# 1. Generate 8 diverse search queries
# 2. Fetch 20+ web sources (articles, Wikipedia, etc.)
# 3. Process through full pipeline
# 4. Generate synthesis in web_synthesis.txt

# Output includes:
# - Comprehensive synthesis
# - Top validated insights
# - Source URLs for verification
```

### Workflow 3: Self-Documentation Analysis
```bash
# Scenario: You want the system to analyze its own documentation

python run_self_analysis.py

# Result: System analyzes:
# - A_NOTE_FOR_CLAUDE.md
# - TRANSFORMATION_STATUS.md
# - README.md
# - STIGMERGIC_SWARM_DESIGN.md
# - PROJECT_STRUCTURE.md
#
# Generates synthesis in self_analysis.txt
```

### Workflow 4: Progressive Repository Analysis
```bash
# Scenario: You want to analyze all documentation in repository

python run_monolith_breaking.py

# System will:
# 1. Discover all .md, .pdf, .txt files
# 2. Show list with sizes
# 3. Ask for confirmation
# 4. Process everything
# 5. Save to repository_analysis.txt
```

### Workflow 5: Custom Processing (Code)
```python
# Scenario: You need custom processing with specific configuration

from swarm.documents.processor import DocumentProcessor
from swarm.monolith_breaking import run_document_swarm

# Custom document splitting
processor = DocumentProcessor(target_tokens=1500, tolerance=400)
sections = processor.process_documents(["custom_doc.pdf"])

print(f"Created {len(sections)} sections")

# Custom agent configuration
result = await run_document_swarm(
    document_paths=["custom_doc.pdf"],
    model_name="microsoft/phi-2",
    num_foragers=100,       # Very high for maximum pattern finding
    num_gatherers=5,        # Low (trust internal patterns more)
    num_critics=50,         # High for thorough evaluation
    num_haters=20,          # High for strong challenges
    max_iterations=1000,    # Very long for maximum convergence
    convergence_threshold=0.8,  # High threshold
    enable_mcp=False        # Disable external validation
)

# Custom analysis of results
insights = [s for s in result['all_signals'] if s.type == "INSIGHT"]
high_strength = [i for i in insights if i.strength > 0.8]

print(f"High-strength insights: {len(high_strength)}")
```

---

## METRICS AND EVALUATION

### Success Metrics

**Compression Ratio:**
```python
compression_ratio = input_tokens / output_tokens
```
- Good: 30:1 to 100:1
- Excellent: >100:1
- Too low (<20:1): May not be compressing enough
- Too high (>200:1): May be losing important details

**Convergence (Gini Coefficient):**
```python
gini = calculate_gini_coefficient([s.strength for s in insights])
```
- Not converged: <0.5 (many insights have similar strength)
- Converging: 0.5-0.7 (winners emerging)
- Converged: >0.7 (clear winners)
- Over-converged: >0.9 (may have lost diversity)

**Validation Coverage:**
```python
insights = signal_store.get_signals_by_type("INSIGHT")
validated = [i for i in insights if signal_store.get_validation_status(i.id)['evidence_count'] > 0]
coverage = len(validated) / len(insights)
```
- Poor: <30% validated
- Good: 50-70% validated
- Excellent: >70% validated

**Provenance Depth:**
```python
def get_provenance_depth(insight_id):
    ancestors = signal_store.get_ancestors(insight_id, target_type="OBSERVATION")
    return len(ancestors)

depths = [get_provenance_depth(i.id) for i in top_insights]
avg_depth = sum(depths) / len(depths)
```
- Shallow: 1-2 observations per insight (may be weak)
- Good: 3-5 observations per insight
- Deep: >5 observations per insight (strong evidence)

---

## AREAS NEEDING REFINEMENT

### 1. Pattern Discovery Quality
**Current State:** Foragers sample clusters randomly, may miss important connections
**Problems:**
- Cluster sampling is based on embeddings similarity only
- No semantic understanding of what makes a "good" cluster
- May cluster unrelated observations that happen to use similar words
- Misses cross-domain patterns that use different terminology

**Refinement Opportunities:**
- Implement smarter cluster sampling (topic modeling, semantic coherence)
- Add cluster quality metrics before pattern finding
- Enable foragers to request specific types of observations
- Multi-hop pattern finding (patterns of patterns)

### 2. Convergence Speed
**Current State:** Takes 300-500 iterations, 7-15 minutes for 50K tokens
**Problems:**
- Fixed iteration counts regardless of convergence state
- No early stopping based on insight stability
- Critics and gatherers may redundantly validate same insights
- No priority queue for which insights need validation most

**Refinement Opportunities:**
- Implement early stopping when Gini plateaus
- Priority validation queue (validate high-potential insights first)
- Adaptive iteration scheduling (more iterations when making progress)
- Parallel validation batches instead of sequential

### 3. External Validation Depth
**Current State:** Gatherers search 1-3 sources per insight
**Problems:**
- Shallow validation (just checks if *something* supports it)
- No contradiction detection from external sources
- Wikipedia/web searches may miss specialized sources
- No quality assessment of evidence sources

**Refinement Opportunities:**
- Multi-level validation (quick check → deep validation for promising insights)
- Contradiction mining (actively search for counter-evidence)
- Source quality scoring (academic > news > blogs)
- Domain-specific search (arXiv for science, GitHub for code, etc.)
- Evidence synthesis (combine multiple sources into stronger evidence)

### 4. Web Ingestion Query Generation
**Current State:** LLM generates 8 diverse queries from topic
**Problems:**
- Queries may be too similar or too broad
- No adaptation based on search results
- No query refinement loop
- May miss important aspects of topic

**Refinement Opportunities:**
- Iterative query expansion (start broad, refine based on results)
- Query diversity metrics (ensure true diversity, not just rephrasing)
- Adversarial query generation (what would challenge our current understanding?)
- Gap detection (what aspects haven't been covered by existing queries?)

### 5. Synthesis Quality
**Current State:** Synthesizer receives top 10 insights, generates narrative
**Problems:**
- May lose important details from lower-ranked insights
- No structure optimization (just narrative flow)
- No fact-checking against source observations
- Limited context (only sees insights, not underlying observations)

**Refinement Opportunities:**
- Hierarchical synthesis (group insights by theme first)
- Fact verification pass (check synthesis against observations)
- Structured output (executive summary, detailed findings, methodology)
- Multi-pass refinement (rough draft → detailed expansion → polish)
- User-guided synthesis (answer specific questions, not just "summarize everything")

### 6. Agent Coordination
**Current State:** Agents sample randomly from signal store
**Problems:**
- No explicit coordination between agents
- May duplicate work (multiple agents process same signals)
- No work-stealing or load balancing
- No specialization (all foragers identical)

**Refinement Opportunities:**
- Implement agent specialization (domain experts, cross-domain connectors)
- Work claiming system (lock signals being processed)
- Load balancing (idle agents take work from busy areas)
- Hierarchical coordination (meta-agents coordinate worker agents)

### 7. Memory and Performance
**Current State:** Works on RTX 3060 with quantization
**Problems:**
- Still requires 5-6GB VRAM
- Embeddings model takes 2GB always loaded
- Signal store grows unbounded (all signals kept in memory)
- No disk caching for processed sections

**Refinement Opportunities:**
- Lazy embedding generation (only when needed)
- Signal store persistence (disk-backed with memory cache)
- Streaming processing (process sections as they load)
- Distributed execution (multiple GPUs, multiple machines)

### 8. Error Handling and Robustness
**Current State:** Basic error handling, fails on major errors
**Problems:**
- LLM generation failures stop agent
- Web fetch failures lose entire source
- No retry logic for transient failures
- Silent failures not logged

**Refinement Opportunities:**
- Retry with exponential backoff
- Graceful degradation (continue with partial results)
- Comprehensive error logging and telemetry
- Self-healing (detect and recover from failures)
- Fault injection testing

---

## APPROACH TO REFINEMENT

### When Implementing Improvements:

1. **Understand the Why**
   - What problem are we solving?
   - What's the root cause?
   - How does this align with our ultimate vision?

2. **Preserve Core Principles**
   - Stigmergic communication (no direct agent-to-agent)
   - No context window violations (stay under 4K per agent)
   - Neutral initial strength (no arbitrary heuristics)
   - Full provenance (always traceable)

3. **Measure Impact**
   - Define metrics before implementing
   - Test on real documents
   - Compare before/after
   - Check for unintended consequences

4. **Iterate Incrementally**
   - Small changes, test, refine
   - Don't rewrite everything at once
   - Maintain backward compatibility when possible
   - Document what changed and why

5. **Think Systemically**
   - How does this change affect other components?
   - Does this create new bottlenecks?
   - Does this improve or harm scalability?
   - Is this solution general or domain-specific?

### Red Flags to Avoid:

❌ **Breaking context limits** - No agent should ever see >4K tokens
❌ **Hardcoded heuristics** - Quality should emerge from validation, not rules
❌ **Single points of failure** - System should be resilient to agent failures
❌ **Losing provenance** - Every insight must trace back to sources
❌ **Premature optimization** - Measure first, then optimize
❌ **Over-engineering** - Simple solutions first, complexity only if needed

### Green Lights to Pursue:

✅ **Better emergence** - Let quality emerge from agent interactions
✅ **Smarter sampling** - Improve how agents discover signals
✅ **Faster validation** - Validate more efficiently
✅ **Richer provenance** - Add more context to signal metadata
✅ **Adaptive behavior** - Agents adjust based on system state
✅ **Parallel opportunities** - Find more ways to parallelize

---

## EXPERIMENTAL IDEAS TO EXPLORE

### 1. Meta-Agents
Agents that observe and coordinate other agents:
- Monitor convergence and adjust iteration counts
- Detect when foragers are stuck, inject new strategies
- Balance validation budget across insights

### 2. Reinforcement Learning
Use success metrics to train agent behavior:
- Which cluster sizes find best patterns?
- Which search queries yield best evidence?
- Which synthesis strategies produce highest quality?

### 3. Multi-Modal Processing
Extend beyond text:
- Extract text from images (OCR)
- Parse tables and convert to observations
- Process code (syntax-aware sectioning)
- Describe diagrams and charts

### 4. Streaming Processing
Process documents incrementally:
- Start generating insights before all sections read
- Continuous ingestion (process new docs as added)
- Real-time synthesis updates

### 5. Swarm Collaboration
Multiple swarms working together:
- Specialized swarms per document type
- One swarm processes, another validates
- Swarms challenge each other's conclusions

### 6. Human-in-the-Loop
Incorporate human feedback:
- User rates insights during processing
- User asks follow-up questions
- System refines based on what user finds useful

### 7. Self-Analysis and Improvement
System analyzes its own behavior:
- Which agents are most productive?
- Which signal types converge fastest?
- What patterns indicate high-quality insights?
- Automatically adjust configuration based on patterns

---

## CONCLUSION: THE PATH FORWARD

This system represents a fundamental shift in how we approach large-context processing:

**Traditional Approach:**
- Try to fit everything in one context window
- Fails at scale (>100K tokens)
- Quality degrades with context length
- Single point of failure

**Our Stigmergic Monolith-Breaking Approach:**
- Distribute reading across hundreds of agents
- No single agent sees >4K tokens
- Scale to millions of tokens
- Quality improves with validation
- Resilient (failures don't cascade)

**Key Innovation:** Breaking the context window monolith through distributed reading, emergent pattern finding, external validation, and strength-based convergence.

**Current Achievement:** Process 300K+ token documents, compress 50:1 to 200:1, maintain provenance, validate externally, generate high-quality synthesis—all without exceeding context limits.

**But we're not done.** This is version 2.0, not the final form. We need:
- Smarter pattern discovery
- Faster convergence
- Deeper validation
- Better synthesis
- More robustness
- Greater scale

**Your mission:** Use this document as context to understand where we are, where we're going, and help us get there faster. Identify weaknesses, propose solutions, implement improvements, and always ask: "Does this move us toward the ultimate vision?"

**Remember:** This isn't about making the code prettier or adding features. It's about building a truly distributed intelligence that can process unlimited information and synthesize genuine understanding. Every refinement should move us closer to that goal.

---

## QUICK REFERENCE

### Essential Commands
```bash
# Process local documents
python run_monolith_breaking.py

# Quick self-analysis
python run_self_analysis.py

# Web ingestion
python run_web_ingestion.py "your topic"

# Legacy creative mode
python run_task.py creative "your prompt"
```

### Essential Imports
```python
from swarm.monolith_breaking import run_document_swarm
from swarm.documents.processor import DocumentProcessor
from swarm.knowledge.mcp_client import MCPClient
from swarm.llm.simple_llm import SimpleLLM
from swarm.core.signal_store import SignalStore
```

### Essential Configuration
```python
# Balanced (recommended)
num_foragers=50, num_gatherers=20, num_critics=20, num_haters=10

# Fast (lower quality)
num_foragers=30, num_gatherers=10, num_critics=10, max_iterations=200

# Thorough (higher quality)
num_foragers=80, num_gatherers=40, num_critics=40, max_iterations=800

# Memory-efficient
use_quantization=True, model_name="microsoft/phi-2"
```

### Essential Checks
```python
# Verify model loading
await llm.load()  # Watch for quantization success

# Check convergence
gini = calculate_gini_coefficient(strengths)
print(f"Gini: {gini:.3f} (target: >0.7)")

# Check validation
validation = signal_store.get_validation_status(insight_id)
print(f"Evidence: {validation['evidence_count']}")
```

---

**END OF COMPREHENSIVE SYSTEM PROMPT**

*This document should provide Claude Code with complete context to work effectively with this system. Update as the system evolves.*
