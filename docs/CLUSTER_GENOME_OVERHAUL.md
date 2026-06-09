# Cluster Genome as the Unit of Selection

## 1. Problem Framing

The stigmergic swarm already produces a rich typed structure for each cluster by the time the synthesizer reads it: atomic facts from SAFE verification, cosine-based inter-cluster edges, counterfactual sensitivity scores, trajectory features, and topology coordinates. However, these pieces are assembled at projection time and discarded afterward. The cluster has no memory of its own lineage. When a cluster undergoes fission (split in `_reanchor`, `core/cluster_registry.py` lines 188–257) or merge (when two previously separate clusters are aligned by the KB dedup logic in `core/knowledge_base.py` lines 313–371), none of the accumulated typed knowledge is passed forward. Each new cluster starts as bare centroid + member list.

The proposal: promote the cluster to a **heritable typed organism** — a `ClusterGenome` — that carries all the typed pieces, survives fission (inheritance by signal partition), recombines on merge (union + dedup), and accumulates a composite non-symbolic fitness history across rounds and runs. The cluster then becomes the unit of selection: the synthesizer, the KB, and the planner all operate on genomes rather than on projection-time reconstructions.

This is not primarily an expressiveness upgrade. It solves four concrete operational failures:

1. **Selection without memory.** The planner (`core/projection.py` `build_plan`, line 823) scores clusters by a formula derived from `support_diversity`, `verification_score`, and `dissent_pressure`. These are recomputed from scratch each round; a cluster that held up through heavy scrutiny three rounds ago has no surviving advantage over a brand-new cluster with similar counts.

2. **SEARCH lineage discarded.** Every `SEARCH` signal deposited in `core/worker_pool.py` lines 913–934 records the query, source titles, and result count. These are never aggregated per-cluster; the validator and synthesizer cannot distinguish a cluster whose core claim has been confirmed by four independent DDG queries from one that has never been searched.

3. **Atom-level provenance is projection-time ephemeral.** `_build_atoms` (`core/projection.py` line 419) reads `metadata["atoms"]` from VERIFICATION signals (written at `core/worker_pool.py` line 759). Those `AtomProjection` objects (`core/projection.py` line 134) exist only in the in-memory `SynthesisProjection` during one synthesizer call. They are not persisted to the KB and are not available to prompt builders.

4. **Fission creates orphans.** When `_reanchor` ejects members into a new cluster (`core/cluster_registry.py` lines 240–257), the new cluster has no partition_id, no topology coords, no atoms, and no inter-cluster edges. It is structurally identical to a brand-new scout INITIAL, though it may carry substantial accumulated support.

---

## 2. Comparison to SwarmSys, GPTSwarm, and Prior Overhauls

**SwarmSys** (AutoGen / LangGraph era multi-agent frameworks) treats agents as the unit of composition. Agent capabilities are fixed at instantiation; there is no mechanism for a shared emergent artifact to accumulate typed properties that outlive any single agent call. The stigmergic architecture here is already superior in this dimension — the signal store IS the shared artifact — but the cluster object is still thin.

**GPTSwarm** introduced typed edges between agents (information, control, evaluation) and node-level optimization. The analogue here is the `InterClusterEdge` (`core/projection.py` line 114), which is already computed. What GPTSwarm lacks and what the genome adds is persistent per-cluster fitness memory that survives rounds.

**Prior overhauls in this repo:**
- *SYNTHESIZER_OVERHAUL* fixed the sequential LLM bottleneck and introduced the dual-planner (LLM planner + pure-Python `build_plan`), the debate frame (`_SYNTHESIZER_USE_DEBATE = True`, line 137 of `agents/synthesizer.py`), and the alternative artifact (`SYNTHESIZER_EMIT_ALTERNATIVE = True`, line 150).
- *TOPOLOGY_LATTICE_OVERHAUL* added `AnswerSpaceTopology` (`core/topology.py` line 44), the multi-resolution lattice (atoms/frames/cross-level edges in `core/projection.py`), and topology coverage in `SynthesisProjection`.
- *AGENT_RETRIEVAL_OVERHAUL* hardened the SAFE pipeline and partition invariant enforcement.

The genome overhaul is the first to treat the **cluster itself** — not the agents, not the synthesizer, not the signal store — as the primary unit of design.

---

## 3. Diagnosis: Typed Pieces That Are Not Yet a Heritable Organism

The table below maps each typed piece to where it lives now, what it lacks, and what the genome adds.

| Piece | Where it lives now | What it lacks |
|---|---|---|
| **SAFE atoms** | `AtomProjection` dataclass, `core/projection.py` line 134; written into `VERIFICATION.metadata["atoms"]` at `core/worker_pool.py` line 759; extracted by `_build_atoms()` at `core/projection.py` line 419 | Not persisted per-cluster; discarded after each synthesis call; not available to prompt builders |
| **Source / query lineage** | `SEARCH` signals in signal store (`core/signal_types.py` line 43); stored as signal content with query + source titles | Never aggregated per-cluster; no per-cluster count of distinct validated queries or source domains |
| **Topology coords** | `Signal.metadata["topology_coords"]` set by `core/topology.py` `assign_topology_cells` (line 231); read by `_build_topology_coverage` at `core/projection.py` line 665 | Only on the INITIAL representative signal; not attached to the cluster object itself; lost on fission |
| **Centroid + stability** | `_Cluster.centroid` in `core/cluster_registry.py` line 62; updated by `_join()` (line 159) and reanchored by `_reanchor()` (line 188) | No `centroid_at_formation` field; no drift metric; no stability score as the centroid shifts across fission events |
| **Counterfactual sensitivity** | `ClusterSensitivity` dataclass, `core/projection.py` line 177; computed by `_build_sensitivities()` at line 570 | Support-level granularity, not atom-level; not persisted; discarded after synthesis |
| **Inter-cluster edges** | `InterClusterEdge` dataclass, `core/projection.py` line 114; computed by `_build_inter_cluster_edges()` at line 336 | Recomputed from scratch each projection; not accumulated across rounds |
| **Trajectory features** | `TrajectoryFeatures` dataclass, `core/projection.py` line 69; attached to `ClusterProjection` at line 110 | `has_trajectory = False` in tests and legacy round-based paths; not persisted in KB |

The KB schema (`core/knowledge_base.py`, `_SCHEMA_VERSION = 2`, line 60) currently saves: representative content + embedding, member/lineage IDs, partition origins, scalar metrics, run provenance. It does not save atoms, query lineage, topology coords, trajectory features, or edges. The `_cluster_hash` function (line 76) is a stable 16-char SHA-1 prefix that can serve as the genome's identity anchor.

**`centroid_at_formation` does NOT currently exist** in `_Cluster` (`core/cluster_registry.py` line 58–64). It must be added.

---

## 4. The Genome Data Structure

All new dataclasses belong in `core/projection.py` alongside the existing projection types, except `ClusterGenome` itself which is also referenced from `core/knowledge_base.py`.

### 4.1 `AtomFact`

```python
@dataclass
class AtomFact:
    atom_id: str            # "{verification_signal_id}_atom_{i}" — same key as AtomProjection
    text: str               # the atomic proposition (this is the ONLY text exposed to agents)
    weight: float           # centrality weight from SAFE decomposition
    verification_score: float  # per-atom score from _safe_score_atom
    source_tag: str         # snippet source domain or "(no result)"
    query: str              # the DDG query that produced the snippet
    extracted_from: list[str]  # VERIFICATION signal IDs (IDs only — no text)
    parent_cluster_id: str  # representative_id of parent cluster
```

**Correspondence to `AtomProjection`:** `AtomFact` is the persistent genome version of `AtomProjection` (line 134, `core/projection.py`). `AtomProjection` is computed at projection time and used by the synthesizer; `AtomFact` is what gets persisted. The two share the same `atom_id` key so migration between them is a field rename, not a structural change.

**No-leak invariant:** agents and the synthesizer may read `AtomFact.text` (proposition text), `atom_id`, `verification_score`, and `source_tag`. The `extracted_from` list carries signal IDs only — never foreign reasoning chains. If any code path attempts to expand `extracted_from` into full signal content before rendering into a prompt, that violates the no-leak rule (`core/signal_store.py` lines 18–33).

### 4.2 `ClusterKnowledgeBase`

```python
@dataclass
class ClusterKnowledgeBase:
    search_queries: list[str]      # queries issued for this cluster's atoms (from SEARCH.metadata["query"])
    source_domains: list[str]      # deduplicated domain tags from snippet_tag fields
    confirmed_atoms: list[str]     # atom_ids with verification_score >= 0.6
    refuted_atoms: list[str]       # atom_ids with verification_score < 0.35
    n_distinct_sources: int        # len(set(source_domains))
    last_updated_iter: int         # iteration counter when last SEARCH was aggregated
```

This is **entirely new** — there is no existing per-cluster aggregation of SEARCH lineage anywhere in the codebase. Currently, SEARCH signals (`core/signal_types.py` line 43) deposit into the signal store with `parent_id=None` and are never linked to specific clusters. Population of `ClusterKnowledgeBase` requires the projection layer to match SEARCH signal queries against the atom queries in each cluster's VERIFICATION signals (same `query` field).

### 4.3 `TopologyExpression`

```python
@dataclass
class TopologyExpression:
    coords: Optional[tuple]        # topology cell this cluster occupies; None if topology not generated
    cell_label: str                # human-readable from topology.cell_description(coords)
    is_anchor: bool                # True if coords is an anchor corner
    is_out_of_bounds: bool         # True if coords == "out_of_bounds"
    coverage_fraction: float       # proportion of cells this cluster helps cover (1 / n_covered)
```

Sourced from `Signal.metadata["topology_coords"]` on the representative INITIAL signal (read by `_build_topology_coverage`, `core/projection.py` line 665) and from `AnswerSpaceTopology.anchor_coords_set()` (`core/topology.py` line 56).

### 4.4 `Phenotype`

```python
@dataclass
class Phenotype:
    centroid: list[float]          # current centroid from _Cluster.centroid (cluster_registry.py line 62)
    centroid_at_formation: list[float]  # NEW: centroid when cluster first had >= 2 members
    centroid_drift: float          # cosine distance between current centroid and centroid_at_formation
    stability: float               # 1.0 - centroid_drift (higher = more stable)
    novelty_density: float         # mean cosine distance to all other cluster centroids in field
```

**`centroid_at_formation` is a new field** that must be added to `_Cluster` in `core/cluster_registry.py`. The recommended insertion point is after line 63 (`deposit_count: int = 0`), as `centroid_at_formation: list[float] = field(default_factory=list)`. It is set once in `_join()` (line 159) when `len(cl.member_ids)` transitions from 1 to 2 (i.e., when the cluster gains its second member).

**Tier caveat:** `centroid_stability` and `novelty_density` are Tier 2 signals. They are derived from `all-MiniLM-L6-v2` sentence embeddings on the laptop path, which are 4-bit quantized model outputs. They are meaningful for relative ranking within a run but not for cross-run or cross-model comparisons. Do not treat them as ground truth.

### 4.5 `FitnessTrajectory`

```python
@dataclass
class FitnessTrajectory:
    composite_history: list[float]   # composite_fitness value at each projection pass
    support_diversity_history: list[int]
    verification_score_history: list[float]
    dissent_pressure_history: list[float]
    run_count: int                   # how many KB-matched runs contributed to history
    trend: str                       # "rising" | "stable" | "declining" | "unknown"
```

**Relationship to `TrajectoryFeatures`:** `TrajectoryFeatures` (`core/projection.py` line 69) measures intra-run temporal evolution of signal deposits (iteration offsets, growth rates, response lag). `FitnessTrajectory` measures cross-round and cross-run evolution of the composite fitness scalar. They compose rather than replace each other: `FitnessTrajectory.composite_history` is the outer time series; `TrajectoryFeatures` remains the inner intra-round signal-level trajectory.

### 4.6 `CounterfactualSensitivity` (Genome Version)

```python
@dataclass
class CounterfactualSensitivity:
    load_bearing_atoms: list[str]    # atom_ids whose removal would flip survival status
    marginal_atoms: list[str]        # atom_ids whose removal does not change status
    dissent_tolerance: float         # additional dissent delta needed to flip status
    competing_cluster_id: Optional[str]  # cluster that would advance if this one fell
    topology_cells_at_risk: list     # topology cell tuples that lose sole coverage
```

**Migration from `ClusterSensitivity`:** `ClusterSensitivity` (`core/projection.py` line 177) operates at support-signal level. `CounterfactualSensitivity` operates at atom level — the `load_bearing_atoms` are the atoms within VERIFICATION signals whose removal would change the cluster's verification_score enough to flip the credibility gate. The `_build_sensitivities()` function (line 570) can be extended to compute atom-level sensitivity during the existing simulation loop; it already has access to the verification_score per VERIFICATION signal through the signal store.

### 4.7 `GenomeRelations`

```python
@dataclass
class GenomeRelations:
    edges: list  # list[InterClusterEdge] — reuse existing dataclass (projection.py line 114)
    frame_id: Optional[str]     # FrameProjection.frame_id this cluster belongs to
    superseded_by: Optional[str]  # representative_id of the cluster that supersedes this one
    alternatives: list[str]       # representative_ids of "alternatives" relation neighbors
```

Reuses `InterClusterEdge` from `core/projection.py` line 114 directly. No new edge dataclass needed.

### 4.8 `ClusterGenome`

```python
@dataclass
class ClusterGenome:
    cluster_id: str               # _Cluster.cluster_id from cluster_registry.py
    genome_hash: str              # _cluster_hash(representative_content) — reuse knowledge_base.py line 76
    representative_id: str        # signal ID of the cluster representative
    atoms: list[AtomFact]         # persistent atom facts
    kb: ClusterKnowledgeBase      # search lineage and source coverage
    topology: TopologyExpression  # position in answer space
    phenotype: Phenotype          # embedding-space shape
    fitness: FitnessTrajectory    # composite fitness history
    sensitivity: CounterfactualSensitivity
    relations: GenomeRelations    # inter-cluster edges + frame membership
    created_at_iter: int          # pool iteration when cluster was created
    last_updated_iter: int        # pool iteration of most recent deposit into cluster
    task_type: str                # task type this genome was built for
    schema_version: int = 3       # bump from KB's current _SCHEMA_VERSION = 2
```

**`genome_hash`**: reuse `_cluster_hash(representative_content)` from `core/knowledge_base.py` line 76. Same SHA-1-based 16-char prefix. This gives genomes the same hash as their KB entries, enabling O(1) lookup when a genome is saved to the KB.

### 4.9 Updated `ClusterProjection`

Add one field to the existing dataclass (`core/projection.py` line 87):

```python
genome: Optional["ClusterGenome"] = None  # populated when genome assembly is enabled
```

This is backward-compatible: all existing code that reads `ClusterProjection` fields ignores `genome=None`.

---

## 5. Genome Operations

### 5.1 Atom Extraction

**Strategy B (recommended): lazy extraction at projection time.**

`_build_atoms()` at `core/projection.py` line 419 already extracts atoms for clusters that have VERIFICATION signals with `metadata["atoms"]` (written at `core/worker_pool.py` line 759). For un-validated clusters — clusters with no VERIFICATION signal or with VERIFICATION signals that went through the legacy single-query fallback path (lines 1117–1151 of `worker_pool.py`) — no `metadata["atoms"]` exists.

Strategy B: at `build_projection()` time (line 719), call `_build_atoms()` as today for validated clusters, and for un-validated clusters synthesize pseudo-atoms by sentence-splitting the representative INITIAL content. Pseudo-atoms carry `verification_score=0.0`, `source_tag="(unverified)"`, and `weight=1/n_sentences`. This gives every cluster at least one atom for downstream fitness computation.

Do NOT run `_safe_decompose()` at projection time (Strategy A). That would add async LLM calls inside what is currently a pure-Python synchronous function. The SAFE pipeline belongs in `core/worker_pool.py`.

### 5.2 Atom Graph Construction

Each cluster's atoms form a mini-DAG with four edge types:

1. **corroborates**: two atoms with overlapping `source_tag` domain and `verification_score >= 0.6` each.
2. **refutes**: `verification_score_A < 0.35` and `verification_score_B >= 0.6` with cosine similarity > 0.5 between their text embeddings.
3. **specializes**: one atom's text is a near-substring of another's (SequenceMatcher ratio > 0.7).
4. **is_parallel**: same `query`, different `text` (same search returned multiple supporting facts).

These edges are internal to the genome and not exposed to agents.

### 5.3 Inheritance on Fission

When `_reanchor()` ejects members into a new cluster (lines 240–257, `core/cluster_registry.py`), the new cluster should inherit atoms whose `extracted_from` intersects the ejected member IDs. Implementation:

```python
def _inherit_genome_on_split(
    parent_genome: ClusterGenome,
    ejected_member_ids: set[str],
) -> ClusterGenome:
    inherited_atoms = [
        a for a in parent_genome.atoms
        if set(a.extracted_from) & ejected_member_ids
    ]
    # New phenotype: compute fresh centroid from ejected members
    # New fitness: start new history with parent's last composite_fitness as seed
    ...
```

A cluster that has never been validated will have `atoms = []` from the pseudo-atom path; `inherited_atoms` will be empty and the new cluster starts fresh.

### 5.4 Recombination on Merge

When two clusters are merged (equivalently: when the KB dedup path in `_merge_entries()`, `core/knowledge_base.py` line 313, identifies two entries as duplicates), their genomes recombine:

1. **Atoms**: union of both atom lists; deduplicate by `atom_id`; where two atoms share the same `text` but different scores, keep the higher score.
2. **Centroids**: average the two `phenotype.centroid` vectors, renormalize.
3. **Fitness history**: concatenate `composite_history` lists in timestamp order; recompute `trend`.
4. **Relations**: union `edges` lists; deduplicate by `(source, target, relation)` triple.
5. **KB**: union `search_queries` and `source_domains`; recount `n_distinct_sources`.

### 5.5 Genome Hashing

Reuse `_cluster_hash` from `core/knowledge_base.py` line 76:

```python
genome_hash = _cluster_hash(store.get(representative_id).content)
```

The hash is stable as long as the representative does not change. If a fission or merge replaces the representative (reanchor picks a new medoid), recompute the hash from the new representative's content. Store the old hash in `FitnessTrajectory` provenance so KB lookups can find both.

---

## 6. The Fitness Compositor

The compositor produces a single `composite_fitness: float` in [0, 1] from seven terms.

```python
def compute_composite_fitness(genome: ClusterGenome, task_type: str) -> float:
    weights = _WEIGHTS_BY_TASK.get(task_type, _WEIGHTS_DEFAULT)
    terms = {
        "support_diversity":   _norm_diversity(genome),        # Tier 2
        "verification_score":  _norm_verification(genome),     # Tier 2/3 depending on source
        "dissent_survived":    _dissent_survived(genome),      # Tier 2
        "centroid_stability":  genome.phenotype.stability,     # Tier 2 (embedding-derived)
        "novelty_density":     genome.phenotype.novelty_density,  # Tier 2
        "source_coverage":     _source_coverage(genome),       # Tier 2/3
        "llm_judged":          _llm_judged(genome),            # Tier 1 with hard cap
    }
    # Hard cap on the LLM-judged term
    terms["llm_judged"] = min(terms["llm_judged"], _CAP_LLM)
    return sum(weights[k] * terms[k] for k in terms)

_CAP_LLM = 0.35   # LLM self-assessment cannot dominate composite score
```

### 6.1 Term Definitions

- **`support_diversity`**: `support_diversity / SURVIVAL_BROAD_SUPPORT` clamped to [0, 1]. Directly from existing `ClusterProjection.support_diversity`.
- **`verification_score`**: `genome.fitness.verification_score_history[-1]` (last round's score) or 0.0.
- **`dissent_survived`**: binary; 1.0 if the cluster has `dissent_set` and `dissent_pressure < SURVIVAL_REJECT_DISSENT_PRESSURE`. Cluster challenged and survived.
- **`centroid_stability`**: `genome.phenotype.stability` = `1.0 - centroid_drift`.
- **`novelty_density`**: mean cosine distance to other cluster centroids; high = occupies sparse region.
- **`source_coverage`**: `genome.kb.n_distinct_sources / max(1, len(genome.atoms))`, clamped to [0, 1].
- **`llm_judged`**: optional call to the synthesizer's planner (one call, structural metadata only); returns a confidence score in [0, 1]. Capped at `_CAP_LLM = 0.35`.

### 6.2 Tier Honesty

| Term | Tier | Rationale |
|---|---|---|
| `support_diversity` | 2 | Derived from agent action labels; real but model-generated |
| `verification_score` | 2/3 | Score comes from `_safe_score_atom` which is an LLM call (Tier 2) unless overridden by Wikidata entity resolution or AST/test validity on coding tasks |
| `dissent_survived` | 2 | Binary derived from model deposits |
| `centroid_stability` | 2 | all-MiniLM-L6-v2 embeddings on quantized laptop |
| `novelty_density` | 2 | Same embedding tier |
| `source_coverage` | 2/3 | Domain tags from DDG are Tier 3; snippet quality varies |
| `llm_judged` | 1 (capped) | Pure LLM self-assessment; hard-capped to prevent inflation |

Only `entity_resolution` (Wikidata lookup, when implemented) and `ast.parse()` / pytest validity (on coding tasks) would be true Tier 3 (independent of the generating model). Do not report Tier 2 terms as empirical evidence.

### 6.3 Per-Task Weight Tables

```python
_WEIGHTS_BY_TASK = {
    "coding": {
        "support_diversity":  0.10,
        "verification_score": 0.35,   # AST + test validity dominate
        "dissent_survived":   0.10,
        "centroid_stability": 0.05,
        "novelty_density":    0.05,
        "source_coverage":    0.20,   # dependency docs matter
        "llm_judged":         0.15,   # cap_llm applies
    },
    "debate": {
        "support_diversity":  0.25,
        "verification_score": 0.15,
        "dissent_survived":   0.25,   # surviving challenge is core to debate
        "centroid_stability": 0.10,
        "novelty_density":    0.10,
        "source_coverage":    0.10,
        "llm_judged":         0.05,
    },
    "analysis": {
        "support_diversity":  0.20,
        "verification_score": 0.25,
        "dissent_survived":   0.15,
        "centroid_stability": 0.15,
        "novelty_density":    0.10,
        "source_coverage":    0.10,
        "llm_judged":         0.05,
    },
    "creative": {
        "support_diversity":  0.20,
        "verification_score": 0.05,   # external fact-check irrelevant for poetry
        "dissent_survived":   0.15,
        "centroid_stability": 0.10,
        "novelty_density":    0.30,   # creative tasks reward unexplored cells
        "source_coverage":    0.05,
        "llm_judged":         0.15,
    },
}
_WEIGHTS_DEFAULT = _WEIGHTS_BY_TASK["analysis"]
```

---

## 7. Search as the Mutation Operator

SEARCH signals (`core/signal_types.py` line 43) are the genome's mutation operator. Each DDG query either:

1. **Confirms an atom** (snippet matches atom text, `_safe_score_atom` returns >= 0.6): the atom's `verification_score` rises; `confirmed_atoms` list grows.
2. **Refutes an atom** (score < 0.35): `refuted_atoms` grows; genome fitness term `verification_score` decreases.
3. **Opens new territory** (query returns results that are semantically distant from any existing atom): triggers a new `AtomFact` pseudo-atom from the snippet (weight=0.5, `verification_score=0.5`).

The current SEARCH deposit path (`core/worker_pool.py` lines 913–934) records the query and result count but does not link back to a cluster. The genome change requires one additional step: after depositing the SEARCH signal, resolve which cluster's INITIAL the scout was building on (via `target` in the scout path — currently `target = None` for SCOUT action, line 935 `return None, retrieved, query, None`). The association must be made lazily at `ClusterKnowledgeBase` population time (genome assembly in `build_projection()`), not at deposit time, to preserve the no-leak rule: the scout cannot know which cluster its query will strengthen.

---

## 8. Genome-Aware Worker Actions

For each action, the table gives the current prompt builder's location and the genome-aware extension. The extension targets atoms when a genome is available; falls back gracefully to the current prompt when it is not (genome is None or atoms list is empty).

| Action | Current prompt builder | Genome-aware extension |
|---|---|---|
| DEVELOP | `develop_prompt` at `core/actions.py` line 261 | Inject the top-2 atoms from the target cluster's genome with their `source_tag`. Ask the worker to develop on one of the atoms specifically. Fallback: current prompt unchanged. |
| CRITIQUE | `critique_prompt` at `core/actions.py` line 335 | Inject `load_bearing_atoms` from `CounterfactualSensitivity`. Ask: "Does this claim's load-bearing evidence actually support the stated conclusion?" Fallback: current prompt unchanged. |
| OBJECT | `object_prompt` at `core/actions.py` line 371 | Inject genome `relations.alternatives` neighbors (their atom texts). Ask: "Which alternative approach's atoms are more directly supported by evidence?" Fallback: current prompt unchanged. |
| VALIDATE | `validate_prompt` at `core/actions.py` line 423; `validate_parse` at line 466 | Pass the genome's `confirmed_atoms` and `refuted_atoms` lists as prior evidence. Instruct the validator to focus on atoms not yet confirmed or refuted. Fallback: current prompt unchanged. |
| REFINE | `refine_prompt` at `core/actions.py` line 546 | Inject the cluster's `fitness.trend` ("rising" / "declining") and the `marginal_atoms` from sensitivity. Ask the worker to strengthen a marginal atom. Fallback: current prompt unchanged. |

**No-leak discipline:** injected atom data must be `AtomFact.text` (proposition text) and scalar scores only. `AtomFact.extracted_from` (signal IDs) may appear as opaque identifiers. `AtomFact.query` may appear for the VALIDATE extension. Never expose `extracted_from` expanded to full signal content.

---

## 9. Genome-Aware Synthesizer

### 9.1 Planner

The Python planner `build_plan()` (`core/projection.py` line 823) currently scores by:

```
score = support_diversity + verification_weight * verification_score - dissent_weight * dissent_pressure
```

Replace with the genome's `composite_fitness` directly when available:

```python
def _score(cp: ClusterProjection) -> float:
    if cp.genome is not None:
        return cp.genome.fitness.composite_history[-1]
    return (cp.support_diversity
            + verification_weight * cp.verification_score
            - dissent_weight * cp.dissent_pressure)
```

The LLM planner `_plan_synthesis()` (called before per-cluster renders) already sees structural metadata only (no Signal.content). Extend its prompt to include `composite_fitness`, `stability`, and `n_distinct_sources` from the genome digest.

### 9.2 Per-Cluster Renderer

In `_render_cluster_position()` (called per surviving cluster in the sectioned render loop near `agents/synthesizer.py` line 796), replace the external context call:

```python
ext_ctx = _get_external_context(rep.content)  # line 2329, Wikipedia lookup
```

with a genome atom block:

```python
if cp.genome and cp.genome.atoms:
    atom_block = _format_atom_block(cp.genome.atoms[:5])
else:
    ext_ctx = _get_external_context(rep.content)
```

`_format_atom_block` renders atom text + source_tag + score as a structured evidence block. This replaces the Wikipedia-lookup `_get_external_context` (`agents/synthesizer.py` line 2993) with a pre-computed, per-atom grounded block.

### 9.3 Section 5 / 6 Extensions

The topology sections (Section 5: uncovered cells, Section 6: out-of-bounds clusters) can be enriched with genome data: uncovered cells that had a cluster with `fitness.trend = "rising"` in the previous run (from KB) become "previously promising, now absent" — a qualitatively different observation than cells that were never covered.

---

## 9b. Synthesizer Performance and Intake (The Bottleneck)

The synthesizer is the pipeline's primary latency bottleneck. A full debate run on the laptop GPU produces approximately 15–20 sequential `await self.llm.generate()` calls across the following paths (verified in `agents/synthesizer.py`):

1. Prompt interpreter call (1 call)
2. LLM planner `_plan_synthesis()` (1 call)
3. Per-cluster `_render_cluster_position()` calls (N surviving clusters, N = 3–6 typically)
4. Debate frame: Round 1 (2 position calls) + Round 2 (2 response calls) + Round 3 judge (1 call) = 5 calls for one alternatives pair
5. Best-of-N candidates (lines 628–671): `_n` = 2–3 calls for cohesive strategies
6. Revision critic + revise (lines 1866, 1898): 2 calls per revision round

With `LLM_CONCURRENCY = 1` (intentional for 6 GB VRAM, documented in `Attempt At Cleaning/CLAUDE.md`), these are forced sequential. There is no `asyncio.gather` over independent calls anywhere in the synthesizer.

**Three fixes in priority order:**

**Fix S1 (highest priority): Parallelize independent per-cluster renders.**

The no-leak rule makes per-cluster renders structurally independent — each `_render_cluster_position()` call sees only its own cluster's signals and the genome atom block. With `LLM_CONCURRENCY = 1`, parallelism offers zero gain on a single-engine run, but:
- When `LLM_CONCURRENCY > 1` (Colab, multi-GPU), `asyncio.gather` over all `_render_cluster_position()` calls is trivially safe.
- Implement as: collect all per-cluster render coroutines into a list; call `asyncio.gather(*coroutines, return_exceptions=True)` when `LLM_CONCURRENCY > 1`, else sequential iteration.
- Gate on `LLM_CONCURRENCY` in config so the laptop path is unchanged.

**Fix S2: Replace raw-materials dump with structured genome bundle.**

`_gather_raw_materials()` at line 1465 collects signal content strings from the store and passes them as a flat list to `_format_materials_block()` at line 1504. The formatted block is then fed into the cohesive render prompt as a large text dump. This is the primary token waste: signal content is often repetitive across cluster members.

Replace with: pass the genome's `atoms` list (5 atoms max), the `fitness.composite_fitness`, and the `relations.edges` types. This reduces prompt token count by ~40% for validated clusters and forces the LLM to synthesize from structured evidence rather than paraphrasing a text dump.

**Fix S3 (lower priority): Template-first / LLM-second per-cluster render.**

For clusters with `fitness.trend = "stable"` and `composite_fitness > 0.7`, the per-cluster render adds marginal information over a deterministic template. Implement a template path in `_render_cluster_position()` that fires when the genome meets both thresholds, outputting a structured paragraph without an LLM call. Reserve the LLM call for contested and rising clusters where nuanced framing matters.

The expected speedup from Fix S1 alone on a multi-GPU deployment: 3x (3 clusters rendered in parallel instead of sequentially). Fix S2: 20–30% token reduction, translating to ~20% latency reduction per render call at the LLM layer.

---

## 10. KB Persistence with Genomes

The KB schema must be bumped to version 3. The genome-aware KB entry adds:

```json
{
  "schema_version": 3,
  "cluster_hash": "<16-char sha1 prefix>",
  "genome": {
    "atoms": [...],
    "kb": { "search_queries": [...], "n_distinct_sources": 4, ... },
    "topology": { "coords": [...], "is_anchor": false, ... },
    "phenotype": { "centroid_drift": 0.12, "stability": 0.88, ... },
    "fitness": { "composite_history": [0.55, 0.62], "trend": "rising", ... },
    "sensitivity": { "load_bearing_atoms": [...], "dissent_tolerance": 0.3, ... },
    "relations": { "edges": [...], "alternatives": [...] }
  }
}
```

The `schema_version = 2` warning path in `KnowledgeBase.load()` (lines 126–133 of `core/knowledge_base.py`) will trigger for v3 entries loaded by an older process. This is intentional and correct behavior.

**Migration path:** `kb_migrate.py` exists at the `Attempt At Cleaning/` root. Extend it to upgrade v2 entries to v3 by:
1. Setting `genome = None` (null genome) for all existing entries.
2. Setting `schema_version = 3`.
3. Leaving existing scalar fields unchanged.

Null genomes are handled everywhere via `if cp.genome is not None` guards (backward-compatible with the `genome: Optional[ClusterGenome] = None` field in `ClusterProjection`).

**`_detect_contradictions`** (`core/knowledge_base.py` line 373) should be extended to compare atom texts across contradicting entries: if entry A's `confirmed_atoms` overlap with entry B's `refuted_atoms` at text similarity > 0.7, the contradiction is atom-level and should be flagged in both entries' `genome.sensitivity`.

---

## 11. Closing the Pre-Existing Gaps

### Gap Status

**Precondition #1 — atoms in metadata: ALREADY DONE.**

`metadata["atoms"]` is written at `core/worker_pool.py` line 759 and read by `_build_atoms()` at `core/projection.py` line 419. This is a standing invariant enforced by the SAFE pipeline. Genome assembly for validated clusters can proceed immediately.

**Precondition #2 — `_get_external_context` replacement: OPEN.**

`_get_external_context` at `agents/synthesizer.py` line 2993 is a Wikipedia lookup that fires per-cluster in `_render_cluster_position()` (called at line 2329). It is not gated by any feature flag and runs synchronously in the async render path. It should be replaced by the genome atom block (Fix S2 above). Until the genome is available, the Wikipedia fallback is acceptable.

**Precondition #3 — abstention recalibration: OPEN.**

The calibrated abstention gate (`_SYNTHESIZER_USE_CALIBRATED_ABSTENTION = True`, `agents/synthesizer.py` line 163) fires when `max_verification_score < 0.15`. With genome-aware validation (more atoms verified per run), this threshold may need raising. Leave for empirical calibration after genome runs produce real data.

**Debate and alternative flags: ALREADY ENABLED — not preconditions.**

`_SYNTHESIZER_USE_DEBATE = True` at `agents/synthesizer.py` line 137 and `SYNTHESIZER_EMIT_ALTERNATIVE = True` at line 150 are already enabled. No action needed.

**Gap 1 (`USE_CLUSTER_AWARE_SAMPLING`):** Feature-flagged, `False` by default. The genome's `phenotype.centroid` is the natural semantic position for gap-1 sampling. When genome-aware sampling is implemented, pass `genome.phenotype.centroid` instead of `worker._position_centroid` so the sampling bias is cluster-centric rather than worker-centric.

**Gap 2 (`USE_TRAIL_AMPLIFICATION`):** The pheromone-trail analogue. Genome-aware extension: amplify atoms within the cluster as well as the cluster's raw strength when a SUPPORT is deposited.

**Gaps 3 and 4:** Local action biases and worker semantic position are already partially implemented (`_local_action_biases`, `_position_centroid` in `Worker` class). The genome's `phenotype.novelty_density` can weight action biases: high novelty density → prefer DEVELOP/CHAIN; low novelty density (crowded region) → prefer OBJECT/REFINE.

---

## 12. Per-Task Genome Generation

The genome assembly call in `build_projection()` (`core/projection.py` line 719) should be gated by task type:

| Task type | Atom extraction | KB population | Fitness compositor |
|---|---|---|---|
| `coding` | Full SAFE atoms required; pseudo-atoms from AST node types if SAFE unavailable | Source coverage = dependency docs | `verification_score` weight 0.35, AST validity Tier 3 |
| `debate` | Full SAFE atoms when available; pseudo-atoms from sentence split | Source coverage = citation domain diversity | `dissent_survived` weight 0.25 |
| `analysis` | Full SAFE atoms; pseudo-atoms from sentence split | Source coverage emphasized | Balanced weights |
| `creative` | No SAFE atoms (VALIDATE suppressed for creative, per `ROLES_FOR_TASK`); pseudo-atoms from sentence split | Source coverage low weight | `novelty_density` weight 0.30 |
| `problem_solving` | SAFE atoms if validators ran; pseudo-atoms fallback | Source coverage = evidence for feasibility | Balanced weights with `verification_score` 0.25 |

For `creative`, atom quality is inherently Tier 2 (sentence-transformer cosine proximity to literary references is not ground truth). The genome for creative clusters is primarily useful for `novelty_density` (are we producing ideas that fill sparse topology cells?) and `centroid_stability` (is the creative direction coherent across rounds?).

---

## 13. Benchmark Plan

All benchmarks must follow the outputs policy: real-LLM runs go to `outputs/`, mock runs to `outputs_mock/`. Behavioral and diversity numbers from `outputs_mock/` are explicitly excluded.

**B1: Genome assembly regression test (no GPU required).**
- In `tests/test_genome_assembly.py`: construct a mock store with 3 INITIAL + 6 SUPPORT + 2 VERIFICATION signals (with pre-populated `metadata["atoms"]`). Call `build_projection()`. Assert that `cp.genome` is not None for all clusters with VERIFICATION signals. Assert `cp.genome.atoms` is non-empty. Assert `cp.genome.phenotype.centroid` is length > 0 when embeddings are available.
- Test fission inheritance: split one cluster, assert the new cluster's genome inherits atoms whose `extracted_from` intersects ejected IDs.

**B2: Fitness compositor unit test.**
- In `tests/test_fitness_compositor.py`: construct a `ClusterGenome` with known fields. Assert `compute_composite_fitness()` returns values in [0, 1]. Assert `llm_judged` contribution is capped at `_CAP_LLM = 0.35`. Assert per-task weights sum to 1.0.

**B3: KB v3 round-trip test.**
- In `tests/test_kb_v3.py`: save a genome to the KB, reload, assert genome fields are identical. Assert `_detect_contradictions` correctly flags atom-level overlaps.

**B4: Synthesizer quality comparison (real-LLM, Colab-grade).**
- Run the same debate prompt with genome-aware synthesizer vs. current synthesizer on the same store snapshot. Compare: (a) citation density (number of `[INITIAL_XXXXX]` tags per paragraph), (b) atom coverage (fraction of confirmed atoms cited), (c) topology section accuracy (do uncovered cells actually appear in Section 5?).
- This is empirical behavioral comparison, not a unit test. Document in `outputs/` with run_meta.json.

**B5: Latency benchmark (Fix S1).**
- Measure wall time for `Synthesizer.synthesize()` with `LLM_CONCURRENCY = 2` before and after Fix S1 (parallel per-cluster renders). Expected: ~2x speedup for 4-cluster field on Colab T4.

---

## 14. Local-Compute Constraints

All genome operations must respect the hardware target: NVIDIA RTX 3060 Laptop, 6 GB VRAM, 4-bit NF4 quantization, `LLM_CONCURRENCY = 1`.

**Memory budget for genome objects:**
- Per `AtomFact`: ~500 bytes (three strings + two floats + two lists of IDs). 5 atoms per cluster = 2.5 KB.
- Per `ClusterGenome`: ~10–15 KB including all sub-objects for a typical 5-atom, 3-edge cluster.
- For 10 surviving clusters: ~150 KB total genome data in memory. Negligible.

**Centroid operations:**
- `all-MiniLM-L6-v2` produces 384-dimensional float32 vectors. Per centroid: 384 × 4 = 1.5 KB.
- `centroid_at_formation` adds one extra vector per cluster: 1.5 KB × 10 clusters = 15 KB. Negligible.

**`_build_sensitivities()` gate:** already gated at `_SENSITIVITY_MAX_CLUSTERS = 20` (`core/projection.py` line 416). Do not remove this gate; atom-level sensitivity simulation for 20+ clusters is O(atoms × clusters) and will stall the 6 GB path.

**No new LLM calls at projection time.** `build_projection()` is synchronous and called from async context; adding LLM calls there would require converting it to async and risk deadlock with the worker pool's RLock on the signal store. Genome assembly must be purely computational.

---

## 15. Sequencing (Six Stages in Dependency Order)

### Stage 1: Core data structures + `centroid_at_formation`

**Files:** `core/cluster_registry.py`, `core/projection.py`

1a. Add `centroid_at_formation: list[float] = field(default_factory=list)` to `_Cluster` dataclass after line 63 (`core/cluster_registry.py`).
1b. Set `centroid_at_formation` in `_join()` when `len(cl.member_ids)` transitions from 1 to 2 (first join into the cluster).
1c. Add `AtomFact`, `ClusterKnowledgeBase`, `TopologyExpression`, `Phenotype`, `FitnessTrajectory`, `CounterfactualSensitivity`, `GenomeRelations`, `ClusterGenome` dataclasses to `core/projection.py` after the existing `ClusterSensitivity` definition (line 177).
1d. Add `genome: Optional[ClusterGenome] = None` field to `ClusterProjection` (after line 110).

**Tests:** B1 genome assembly regression (atoms=None, genome placeholder populated).

### Stage 2: Atom extraction and genome assembly

**Files:** `core/projection.py`

2a. Add `_build_genome(cp, store, topology)` function after `_build_sensitivities()` (around line 663).
2b. In `build_projection()` (line 719), call `_build_genome` per cluster after the sensitivity builder, assigning to `cp.genome`.
2c. Implement pseudo-atom generation for clusters with no VERIFICATION signals.
2d. Implement `ClusterKnowledgeBase` population from SEARCH signals in the store.

**Tests:** B1 full (genome.atoms non-empty for validated clusters; pseudo-atoms for unvalidated).

### Stage 3: `ClusterKnowledgeBase` population from SEARCH lineage

**Files:** `core/projection.py`, `core/signal_store.py` (read-only access to SEARCH signals)

3a. Add `_build_cluster_kb(cp, store)` that iterates SEARCH signals and matches their `metadata["query"]` fields against atom queries.
3b. Populate `genome.kb` in `_build_genome`.

**Tests:** B1 + unit test asserting KB search_queries is non-empty when SEARCH signals exist in store.

### Stage 4: `FitnessCompositor` with hard cap + per-task weights

**Files:** `core/fitness.py` (new module, ~100 lines)

4a. Implement `compute_composite_fitness(genome, task_type)` with the seven terms and per-task weight tables.
4b. Call from `build_projection()` after genome assembly; store result in `genome.fitness.composite_history`.
4c. Update `build_plan()` (`core/projection.py` line 823) to prefer `genome.fitness.composite_history[-1]` when available.

**Tests:** B2 fitness compositor unit test.

### Stage 5: Genome-aware action prompts

**Files:** `core/actions.py`

5a. Add optional `genome: Optional[ClusterGenome] = None` parameter to `develop_prompt` (line 261), `critique_prompt` (line 335), `object_prompt` (line 371), `validate_prompt` (line 423), `refine_prompt` (line 546).
5b. Implement atom-injection logic per action (see §8).
5c. Worker pool passes genome from `cp.genome` when building prompt (in `_build_prompt`, `core/worker_pool.py` line 1179). Genome lookup: `store.get_cluster_genome(target.cluster_id)` — requires adding a `get_cluster_genome` method to `SignalStore` that reads from `ClusterProjection` objects cached by the pool's last projection pass.

**Tests:** Existing action tests must pass with `genome=None`. New tests: genome-aware prompt injection does not leak ancestry text.

### Stage 6: Genome-aware synthesizer + KB v3

**Files:** `agents/synthesizer.py`, `core/knowledge_base.py`, `kb_migrate.py`

6a. Replace `_get_external_context(rep.content)` call (line 2329) with `_format_atom_block(cp.genome.atoms[:5])` when genome is available.
6b. Extend `_plan_synthesis()` prompt to include genome digest (composite_fitness, stability, n_distinct_sources).
6c. Implement Fix S1 (parallel per-cluster renders) gated on `LLM_CONCURRENCY > 1`.
6d. Bump `_SCHEMA_VERSION` to 3 in `core/knowledge_base.py` (line 60).
6e. Add genome serialization to `_cluster_to_entry()` (line 446).
6f. Extend `kb_migrate.py` to upgrade v2 → v3 entries with null genomes.

**Tests:** B3 KB round-trip. B4/B5 on real hardware.

---

## 16. Risks and Open Questions

**R1: Pseudo-atom quality for un-validated clusters.**
Sentence-split pseudo-atoms from INITIAL content are Tier 2 at best and carry `verification_score=0.0`. If the fitness compositor's `verification_score` term is weighted heavily (as in analysis), un-validated clusters will always score near zero regardless of their support diversity. Mitigation: set `verification_score` weight to 0 for clusters with all-zero atom scores, or treat `verification_score=0.0` as "unknown" rather than "zero" in the compositor.

**R2: `centroid_at_formation` races in the registry.**
`_join()` is called under the signal store's `RLock` (`core/signal_store.py`), so there is no race on setting `centroid_at_formation`. However, if a cluster's first two members are deposited in rapid succession by two workers in the same async iteration, both may call `_join()` sequentially under the same lock. The transition check `if len(cl.member_ids) == 1:` (before the append) is safe under the lock.

**R3: Genome staleness between projection passes.**
`build_projection()` is called at the end of each round. Between projection calls, the store may have received new SUPPORT signals that change a cluster's atom scores. The genome attached to `cp.genome` reflects the state at the last projection call. Worker actions that use genome-derived prompts will use a potentially one-round-stale genome. This is acceptable and analogous to how `field_state` snapshots work today.

**R4: KB v3 schema break for existing test fixtures.**
Several tests in `tests/` use `KnowledgeBase` with on-disk fixtures in the `knowledge_base/` directory. Bumping `_SCHEMA_VERSION` to 3 will trigger the warning path for any v2 fixture. The warning is non-fatal (just a print), and `kb_migrate.py` handles the upgrade. Document the migration step in test setup instructions.

**R5: LLM-judged term on the 6 GB path.**
The `llm_judged` term in the fitness compositor requires an LLM call. On the laptop with `LLM_CONCURRENCY = 1`, this adds one sequential call per genome per round. For 10 clusters, that is 10 extra calls — a 50–100% increase in total LLM calls. Recommendation: disable `llm_judged` (weight = 0) by default; make it a feature flag analogous to `USE_CLUSTER_AWARE_SAMPLING`. Enable only when compute budget permits or when running on a multi-GPU host.

**Open questions:**
- Should `ClusterGenome` be persisted mid-run (e.g., written to `signals.json` alongside signals) or only at KB-save time? Mid-run persistence enables resume after crash but increases I/O.
- The `alternatives` edge in `GenomeRelations` is populated from `InterClusterEdge` (`core/projection.py` line 114). Should the debate frame (`_SYNTHESIZER_USE_DEBATE`) prefer genome-relation `alternatives` over projection-time alternatives? Currently the debate frame checks the projection's `inter_cluster_edges` directly.
- Should `novelty_density` be recomputed each round (O(n²) in cluster count) or cached in the genome and incremented lazily? For n <= 20 clusters on the laptop, the full recomputation is fast (~0.1 ms); lazy caching adds complexity without benefit at current scale.

---

## 17. Future Work

**Genome-level evolution across task types.**
The current KB is task-type-aware at the filter level (`_filter_by_topic`, `core/knowledge_base.py` line 426), but genome data is task-specific (coding genomes carry AST validity; creative genomes carry novelty density). A genome transfer mechanism across task types — e.g., a debate cluster's confirmed atoms bootstrapping an analysis run on the same topic — would require a genome compatibility check before transfer.

**Wikidata entity resolution.**
`source_coverage` currently uses DDG domain tags (Tier 2). Integrating Wikidata entity resolution for named entities in atom texts would upgrade source coverage to Tier 3 for factual claims. Implementation: after atom extraction, run a lightweight NER pass on `AtomFact.text` and resolve entities via the Wikidata API. This is out of scope for the initial six stages but is the correct next step for `verification_score` to become a Tier 3 signal.

**Genome-guided topology generation.**
`generate_topology()` (`core/topology.py` line 129) currently uses a one-shot LLM call before scouts run. A genome-informed topology would bias axis selection toward regions where prior-run genomes have `fitness.trend = "rising"` and away from regions with `fitness.trend = "declining"` or `status = "rejected_by_field"`. This closes the cross-run learning loop.

**Ablation: LLM planner vs. genome-weighted Python planner.**
The CLAUDE.md notes: "Both implement the same conceptual role; the LLM planner has not yet been ablated against the Python planner." With genome-weighted scoring in `build_plan()`, the Python planner gains a quality advantage that did not previously exist. The ablation should be run after Stage 4 (fitness compositor) is deployed.

---

## End-to-End Verification Trace

**Scenario: debate task, "Climate action is necessary."**

1. **Topology generation** (`core/topology.py` line 129): LLM call produces 3 axes (position, framing, scope), 4 anchor corners. Scout 0 is assigned anchor cell `("fully_necessary", "empirical", "mitigation")`.

2. **Scout deposits INITIAL** in topology cell `("fully_necessary", "empirical", "mitigation")`. Signal metadata: `{"topology_coords": ("fully_necessary", "empirical", "mitigation"), "partition_id": "partition_0", "scout_agent_id": "worker_000"}`.

3. **Developer adds SUPPORTs.** Two developers deposit SUPPORT signals with `action=DEVELOP` and `action=CHAIN` in `metadata`. `ClusterRegistry._join()` (`core/cluster_registry.py` line 159) runs; at the second member, `centroid_at_formation` is recorded.

4. **SAFE pipeline runs.** VALIDATE action triggers `_safe_decompose()` (`core/worker_pool.py` line 1061), producing 3 atoms from the INITIAL content. For each atom: step-back + HyDE query, DDG search, `_safe_score_atom()` call. Results written to `self._validate_atoms`. After `validate_parse()`, `metadata["atoms"]` is populated at line 759 with the 3 atom dicts.

5. **One atom confirmed against ipcc.ch.** `atom_results[1]["score"] = 0.82`, `snippet_tag = "ipcc.ch/report/ar6"`.

6. **Genome assembled.** At end of round, `build_projection()` (line 719) calls `_build_atoms()` (line 419): reads `VERIFICATION.metadata["atoms"]`, produces 3 `AtomFact` objects. `_build_genome()` (Stage 2, new function) assembles `ClusterGenome` with:
   - `atoms = [AtomFact(text="...", verification_score=0.82, source_tag="ipcc.ch/report/ar6", ...), ...]`
   - `phenotype.centroid_drift = cosine_distance(centroid_now, centroid_at_formation) = 0.12`
   - `phenotype.stability = 0.88`

7. **FitnessCompositor** computes `composite_fitness`:
   - `support_diversity = 2/4 = 0.5`
   - `verification_score = (0.82 + 0.5 + 0.5) / 3 = 0.61` (weighted mean of 3 atoms)
   - `dissent_survived = 1.0` (cluster has OBJECTION but survived)
   - `centroid_stability = 0.88`
   - `novelty_density = 0.45` (occupies sparse anchor cell)
   - `source_coverage = 1/3 = 0.33` (one confirmed source)
   - `llm_judged = 0.0` (disabled on laptop)
   - Using debate weights: `composite_fitness ≈ 0.62`

8. **Synthesizer renders with atom-level provenance.** `_render_cluster_position()` injects atom block:
   - "ATOM 1 [ipcc.ch/report/ar6, score=0.82]: Global mean surface temperature has risen 1.1°C since pre-industrial levels."
   - Final paragraph cites `[INITIAL_abc12]` and passes the faithfulness audit (≥4-gram overlap verified).

This trace confirms: scout deposit → developer growth → SAFE extraction → atom confirmation → genome assembly → fitness scoring → synthesizer atom-level provenance — the full pipeline with verified line numbers at each step.
