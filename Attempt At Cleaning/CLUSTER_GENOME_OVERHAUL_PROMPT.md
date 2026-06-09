# Claude Code prompt — Cluster Genome as the Unit of Selection

Paste everything below the `---` line into Claude Code from the repository root
(`C:\Users\agsse\Downloads\ai_swarm_mechanics-main (4)\ai_swarm_mechanics-main`).

This is the **fourth and consolidating** design memo. It supersedes the prior
`TOPOLOGY_LATTICE_OVERHAUL.md` memo by re-framing topology, multi-resolution
lattice, and counterfactual sensitivity as facets of a single architectural
object: the **cluster genome**. The two earlier memos (`SYNTHESIZER_OVERHAUL`
and `AGENT_RETRIEVAL_OVERHAUL`) remain in force as load-bearing preconditions.

After this memo lands, `docs/TOPOLOGY_LATTICE_OVERHAUL.md` should be moved to
`docs/archive/` with a note pointing to this memo as the canonical design.

The architectural motivation: the swarm currently has features without an
organism. SAFE atoms exist (validator side); the topology exists; sensitivity
scaffolding exists; the cluster registry maintains persistent identity;
search injects external material; the synthesizer reads scattered scalars
from each cluster. None of these subsystems was designed against a unified
object, so each operates on `ClusterProjection` as if it were a flat metrics
blob. **Promote `ClusterProjection` to a typed cluster genome** and the
features become coherent operations on a single structured substrate.

---

You are working in the `Attempt At Cleaning/` folder of a stigmergic
multi-agent swarm codebase. Read the following files end-to-end before
proposing anything:

- `core/cluster_registry.py` (full) — persistent cluster identity, centroid
  tracking, fission via reanchor + split. This is the substrate the genome
  attaches to.
- `core/projection.py` (full) — the existing `ClusterProjection` schema,
  the inter-cluster edge graph, the support_tree and trajectory fields.
- `core/topology.py` (full) — `AnswerSpaceTopology`, `AnchorCorner`,
  cell assignment for scouts.
- `core/signal_store.py` (focus on `Signal` dataclass with `cluster_id`,
  the deposit path, the embedding cache).
- `core/worker_pool.py` (focus on `Worker.iterate`, `_gather_target`, the
  SAFE pipeline at lines 1051–1116, and the SEARCH signal deposits in
  the SCOUT/DEVELOP paths).
- `core/actions.py` (action specs, `validate_parse`, `_format_safe_external`).
- `agents/synthesizer.py` (full) — current rendering paths and the
  pre-existing improvements (revision loop, best-of-N, debate, alternative).
- `core/knowledge_base.py` (full) — current KB persistence; will need
  schema extension.
- `core/signal_types.py` — note that SEARCH is declared but is not
  fitness-bearing in projection or convergence.

Read the three prior design memos as context but do not duplicate their
content here:

- `docs/SYNTHESIZER_OVERHAUL.md`
- `docs/AGENT_RETRIEVAL_OVERHAUL.md`
- `docs/TOPOLOGY_LATTICE_OVERHAUL.md` (this memo supersedes it)

Do NOT modify code on this pass. The deliverable is a single markdown
document at:

    Attempt At Cleaning/docs/CLUSTER_GENOME_OVERHAUL.md

## Task framing

Three pressures motivate this overhaul, each independently sufficient:

**1. Benchmark performance against SwarmSys.** The project will be evaluated
on at least one shared benchmark from SwarmSys's Table 1 (Omni-Math is the
recommended target — 300 samples, public ground truth, comparable compute
to SwarmSys-8). SwarmSys's pheromone-implicit reinforcement is the weakest
part of their architecture: it operates by re-embedding agent profile text
each round, not by explicit decay or quantitative fitness. SwarmSys has no
heritable cluster-level representation; their clusters are emergent in
embedding space and consulted only by their validator's TERMINATE message.

A swarm that operates on **structured, heritable, non-symbolic genome
objects** has architectural moves available to it that SwarmSys does not.
This memo specifies the smallest set of those moves that delivers a
falsifiable benchmark delta.

**2. The Dawkins challenge.** Cumulative selection — the mechanism that
makes "METHINKS IT IS LIKE A WEASEL" emerge in 40 generations instead of
the heat-death of the universe — requires a heritable representation that
mutates and is selected on. The current swarm has:

- Variation (multiple agents producing variants in parallel)
- Selection pressure (decay, amplify, prune)
- Lineage (parent_id pointers, cluster_id persistence)

But it does *not* have a heritable representation of what each cluster
actually claims. Without that representation, mutations and selection are
not commensurable across generations. The swarm runs as parallel sampling
with post-hoc selection — the architectural ceiling of an ensemble, not of
cumulative selection. The genome is the missing primitive.

**3. The symbolic-communication-proxy loop.** Currently ~95% of the swarm's
selection pressure comes from LLM-on-LLM judgment: agents producing scores
on other agents' outputs. Because all agents share the base model's biases,
the loop is closed — errors compound rather than wash out. The fix is
**non-symbolic fitness signals**: geometric (embedding stability, novelty
density), structural (topology coverage, cluster atomic structure),
temporal (trajectory consistency), and external (search-lineage diversity,
entity resolution against Wikidata). The genome is the substrate that holds
all of these as fields, so fitness composition is a structured operation
over the genome, not a black-box scalar.

The Mount Everest framing from the prior topology memo still applies: build
the bounds before the interior. With the genome architecture, the framing
extends: the bounds (topology) are the **expression environment**; the
interior (cluster contents) is the **genome**. Selection operates on the
match between genome and environment, mediated by non-symbolic fitness.

## Diagnosis: features without organism

The codebase has accumulated five pieces of structure that *would naturally
live on a genome* if a genome existed:

| Piece | Where it lives now | What it should be on the genome |
|---|---|---|
| SAFE atoms | Worker instance attribute `_validate_atoms`; dies with the worker | Atom list (the basepairs) |
| Source/query lineage | Scattered across SEARCH signal metadata | Knowledge base / external grounding lineage |
| Topology coords | Signal `metadata["topology_coords"]` | Expression-environment annotation |
| Centroid + stability | `cluster_registry._Cluster.centroid` | Phenotype anchor |
| Counterfactual sensitivity | (planned, not yet implemented) | Load-bearing atom annotation |
| Inter-cluster edges | Projection-level list, accessed via getattr | Genome relationship records |
| Trajectory features | `TrajectoryFeatures` dataclass on ClusterProjection | Genome fitness history |

Each subsystem produces data that fits naturally as a field on a typed
cluster object. Because no such typed object exists, each subsystem maintains
its own structure, projection wires them together as flat scalars, and the
synthesizer reads them as independent metrics. Promoting the cluster to a
typed genome unifies these subsystems into operations on a single substrate
and *enables* operations that no individual subsystem can perform alone
(genome recombination on cluster merge; genome inheritance on cluster fission;
atom-targeted prompting; cross-run atom-level contradiction detection).

## The cluster genome — data structure

The memo must specify the genome with full field semantics. The canonical
schema:

```python
@dataclass(frozen=True)
class AtomFact:
    atom_id: str
    text: str                            # ~10–25 words, one verifiable proposition
    weight: float                        # centrality (1.0 load-bearing, 0.1 incidental)
    verification_score: float            # per-atom score from SAFE pipeline
    source_domain: Optional[str]         # the domain that corroborated, if any
    source_chunk_id: Optional[str]       # link to specific snippet in the SEARCH lineage
    extracted_from: list[str]            # signal IDs this atom was derived from
    iteration_first_seen: int            # iter_at_deposit when atom was first formed

@dataclass
class ClusterKnowledgeBase:
    queries_issued: list[str]            # all queries from members' SEARCH lineage
    source_domains: set[str]             # distinct top-level domains across the lineage
    source_count: int                    # total distinct retrieved chunks
    domain_diversity: float              # normalized Shannon entropy over domain frequencies
    parametric_content_ratio: float      # fraction of cluster members with no SEARCH lineage
    cross_cluster_source_overlap: float  # fraction of sources shared with other clusters

@dataclass
class TopologyExpression:
    coords: Optional[tuple[str, ...]]    # which cell of AnswerSpaceTopology
    cell_label: Optional[str]            # human-readable cell description
    is_anchor: bool                      # True if cluster occupies an anchor corner
    cell_occupancy_rank: int             # 0 = sole occupant; N = N-th cluster in this cell

@dataclass
class Phenotype:
    centroid: list[float]                # current L2-normalized centroid
    centroid_at_formation: list[float]   # centroid when cluster was first created
    centroid_drift: float                # cosine distance between formation and current
    centroid_stability: float            # 1.0 = stable, 0.0 = drifting wildly
    novelty_density: float               # inverse mean distance to other clusters' centroids

@dataclass
class FitnessTrajectory:
    formation_iteration: int
    fitness_history: list[tuple[int, float]]   # (iter, composite_fitness) sampled at intervals
    strength_history: list[tuple[int, float]]
    member_count_history: list[tuple[int, int]]
    monotone_growth: bool                # True if fitness rose monotonically
    consolidation_iteration: Optional[int]  # iter when fitness plateaued, or None

@dataclass
class CounterfactualSensitivity:
    load_bearing_atoms: list[str]        # atom_ids whose removal flips status
    marginal_atoms: list[str]
    support_removal_robustness: float    # min strength of single support whose removal flips
    competing_takeover: Optional[str]    # cluster_id that would advance if this fell
    topology_cells_at_risk: list[tuple]  # cells that lose coverage if this cluster falls

@dataclass
class GenomeRelations:
    parent_genomes: list[str]            # cluster_ids this descended from (fission/merge)
    descendant_genomes: list[str]        # cluster_ids that descended from this (fission)
    inter_cluster_edges: list            # typed edges from prior overhaul: complements,
                                         # alternatives, shared-evidence, co-contested,
                                         # tension, supersedes

@dataclass
class ClusterGenome:
    """The typed organism. Each ClusterProjection carries one of these."""
    cluster_id: str
    genome_hash: str                     # stable hash over the atom set; identical genomes detectable
    formation_iteration: int

    # The basepairs
    atoms: list[AtomFact]
    atom_graph: dict[str, list[str]]     # atom_id → atom_ids it depends on
                                          # (e.g. SUPPORT atom builds on INITIAL atom)

    # Expression
    topology_expression: TopologyExpression

    # Phenotype
    phenotype: Phenotype

    # External grounding (horizontal gene transfer record)
    knowledge_base: ClusterKnowledgeBase

    # Robustness annotation
    sensitivity: CounterfactualSensitivity

    # Evolutionary history
    trajectory: FitnessTrajectory

    # Heritable relationships
    relations: GenomeRelations

    # Composite fitness (computed by FitnessCompositor; see below)
    composite_fitness: float
    fitness_breakdown: dict[str, float]   # per-term contributions for auditability
```

The `ClusterProjection` becomes:

```python
@dataclass
class ClusterProjection:
    # Existing fields preserved for backward compat
    representative_id: str
    member_ids: list[str]
    support_set: list[str]
    dissent_set: list[str]
    verification_set: list[str]
    status: str
    # ... existing scalar metrics preserved ...

    # NEW: the genome
    genome: ClusterGenome
```

## How prior subsystems feed the genome

The memo must include a mapping table showing where each genome field is
populated from. At minimum:

| Genome field | Populated from |
|---|---|
| `atoms` | SAFE pipeline at `core/worker_pool.py:1051–1116` (validator path), extended to extract atoms from non-validator members at projection time |
| `atom_graph` | Signal `parent_id` links + content-overlap inference among atoms |
| `topology_expression` | Signal `metadata["topology_coords"]` + `core/topology.py` cell lookup |
| `phenotype.centroid` | `cluster_registry._Cluster.centroid` directly |
| `phenotype.centroid_at_formation` | New field on `_Cluster`; snapshot at creation |
| `phenotype.novelty_density` | Computed at projection time over all cluster centroids |
| `knowledge_base.source_domains` | Walked from all SEARCH signals in the cluster's transitive lineage |
| `knowledge_base.queries_issued` | Same |
| `knowledge_base.domain_diversity` | Shannon entropy over source domain frequencies |
| `sensitivity.load_bearing_atoms` | Counterfactual analysis: simulate atom removal, recompute survival, flag those that flip status |
| `trajectory.fitness_history` | Sampled at fixed intervals during the run; logged separately and stitched in at projection |
| `relations.inter_cluster_edges` | Existing typed edges from `core/projection.py` |
| `relations.parent_genomes` | Logged by `cluster_registry._reanchor` when fission occurs; logged by planner's `merge_groups` |
| `composite_fitness` | `FitnessCompositor` (see below) |

## Genome operations

The memo must specify each operation with file paths, function signatures,
and integration points.

### Atom extraction (extending SAFE)

Currently SAFE atoms are produced only at the VALIDATE action
(`core/worker_pool.py:1061`). The genome requires atoms for every cluster
regardless of whether VALIDATE has fired. Two extension strategies; the
memo must recommend one:

**Strategy A: Eager extraction at deposit time.** Every INITIAL/SUPPORT
deposit triggers `_safe_decompose` on its content. Atoms are stored in
`Signal.metadata["atoms"]`. Projection collates per-cluster atoms by union.
Cost: one extra LLM call per deposit. On laptop with NUM_FORAGERS=4 and 16
iters/round this is 64 extra calls per round — expensive.

**Strategy B: Lazy extraction at projection time.** When `build_projection`
runs, decompose each cluster's representative + top-K supports in one batch
call. Cost: one call per cluster at projection time. With ~10–40 clusters
per run, this is 10–40 calls per projection; far cheaper.

Strategy B is recommended for laptop runs (6 GB VRAM, small models). The
memo should specify how atoms produced lazily get associated back to
specific deposits (via content-overlap matching or via storing an
"extracted_from" map). Strategy A is preferred when compute permits because
it gives atom-resolution at every iteration; the memo should mark it as
future work.

### Atom graph construction

For each cluster, derive `atom_graph: dict[atom_id → list[atom_id]]`. An
edge `a → b` means atom `b` is a dependency of atom `a` (removing `b`
would invalidate or weaken `a`). Construction rules:

1. Atoms extracted from a SUPPORT signal depend on the atoms extracted
   from its parent INITIAL.
2. Atoms extracted from a REFINE deposit depend on the dissent atom they
   address.
3. Atoms whose `extracted_from` overlap with another atom's
   `extracted_from` are siblings (no edge).
4. Atoms whose content overlap exceeds a threshold (cosine ≥ 0.85) are
   collapsed into one atom; the surviving atom records both source
   signals in `extracted_from`.

The graph is a DAG over atoms within a cluster. The memo must specify how
it's computed and stored.

### Inheritance on fission

When `cluster_registry._reanchor` ejects members into a new cluster
(`core/cluster_registry.py:241–257`), the new cluster's genome must inherit
a *subset* of the parent's atoms. Specifically: atoms whose `extracted_from`
intersects the ejected members' IDs are inherited; atoms whose
`extracted_from` does not are excluded. The new cluster's
`relations.parent_genomes` records the parent's cluster_id. The parent's
`relations.descendant_genomes` adds the new cluster_id.

This is the architecturally important moment: fission produces descent with
modification. The daughter cluster carries some of its parent's DNA; the
mutations come from the ejected members being a non-random subset (their
ejection was driven by centroid drift).

### Recombination on merge

When the synthesizer's planner produces `merge_groups` declaring two
clusters as the same position, their genomes recombine: union the atoms,
deduplicate by content overlap, average the centroids, union the
knowledge_bases, concatenate the trajectories. Both parent genomes are
recorded in `relations.parent_genomes`. The memo should specify whether
recombined genomes form a *new* cluster_id or whether one of the merged
clusters becomes canonical and absorbs the other.

### Genome hashing

`genome_hash` is a stable hash over the sorted atom set's content. Two
clusters with identical genomes (same atoms, same dependencies) hash
identically — useful for KB persistence (a previously-seen genome doesn't
need to be re-stored) and for detecting accidental duplication.

## The fitness compositor

The composite fitness replaces single-strength scalar in survival decisions.
Specification:

```python
def composite_fitness(genome: ClusterGenome,
                       all_genomes: list[ClusterGenome],
                       weights: dict[str, float],
                       cap_llm: float = 0.35) -> float:
    """Composite fitness from seven terms, with LLM-judged contribution capped."""
    terms = {
        "semantic_strength":     genome.fitness_breakdown.get("llm_judged", 0.0),
        "grounding":             _grounding_score(genome.knowledge_base),
        "topology":              _topology_contribution(genome.topology_expression, all_genomes),
        "centroid_stability":    genome.phenotype.centroid_stability,
        "novelty_density":       genome.phenotype.novelty_density,
        "trajectory":            _trajectory_score(genome.trajectory),
        "entity_resolution":     _entity_resolution_score(genome.atoms),
    }
    # Hard cap on LLM-judged contribution
    terms["semantic_strength"] = min(terms["semantic_strength"], cap_llm)
    # Weighted sum
    return sum(weights.get(k, 0.0) * v for k, v in terms.items())
```

The memo must specify:

1. **The hard cap on `semantic_strength`.** Default `cap_llm = 0.35`. The
   architectural reason: closed-loop pathology cannot dominate selection
   when the LLM-judged term is capped at less than half. If LLM judgment
   disagrees with non-symbolic signals, the non-symbolic signals win. This
   is the single most important configuration choice in this overhaul.

2. **Per-task weight tables.** For factual tasks (coding) weight `grounding`
   and `entity_resolution` heavily; the LLM-judged `semantic_strength`
   matters less because tests and AST checks are authoritative. For
   non-factual tasks (debate, analysis, creative) weight `topology`,
   `novelty_density`, and `centroid_stability` more heavily because
   external grounding is harder to come by. Provide a default weight
   table per task type.

3. **Computation procedure for each non-LLM term**:
   - `_grounding_score`: weighted combination of `domain_diversity` and
     `(1 - parametric_content_ratio)`. Normalized to [0, 1].
   - `_topology_contribution`: 1.0 if this cluster is the sole occupant
     of its cell; lower if multiple clusters occupy the cell. Anchor cells
     get a multiplier of 1.5.
   - `centroid_stability`: 1.0 if `centroid_drift < 0.1`, decreasing
     linearly to 0.0 at `centroid_drift > 0.5`. Drift measured between
     `centroid_at_formation` and current `centroid`.
   - `novelty_density`: mean cosine distance from this cluster's centroid
     to all others' centroids, normalized to [0, 1].
   - `_trajectory_score`: 1.0 if `monotone_growth` is True; 0.5 if
     monotonic but slow; 0.2 if oscillating. Weights `consolidation` as
     a positive signal.
   - `_entity_resolution_score`: fraction of atom-text entities that
     resolve in Wikidata. Cached per-entity to avoid repeated lookups
     within a run.

4. **Fitness breakdown auditability.** Every cluster's `fitness_breakdown`
   dict records the contribution of each term. This is essential for
   debugging and for the synthesizer to render "this cluster's fitness
   comes mostly from grounding (β=0.7) and centroid stability (δ=0.6);
   LLM-judged contribution is at the cap (α=0.35)." The non-LLM-judged
   sources of fitness are surfaceable to the user.

## Search as the mutation operator — explicit treatment

SEARCH signals are currently inert in projection and convergence. They
exist as traces but don't participate in fitness. This overhaul promotes
them to first-class fitness contributors:

1. **Walk SEARCH signal lineage at projection time.** For each cluster,
   collect all SEARCH signals in its transitive ancestry (scout SEARCH
   that produced its INITIAL; developer SEARCH triggered by sparse-cluster
   hook; validator SEARCH per atom). Each SEARCH has metadata: `query`,
   `n_results`, source URLs in the chunks.

2. **Materialize `ClusterKnowledgeBase`.** Aggregate the SEARCH lineage
   into the structured record specified above. This is set arithmetic;
   no LLM calls.

3. **Add `grounding_score` as a non-LLM fitness term.** A cluster grounded
   in five distinct domains scores higher than one with no external
   grounding. The mutation operator (search) now has measurable
   contribution to fitness selection.

4. **Penalize purely parametric clusters.** A cluster with
   `parametric_content_ratio = 1.0` (no member triggered any search)
   gets a fitness penalty proportional to how much its task type prefers
   grounded claims. For coding tasks, parametric-only clusters are
   strongly penalized (we want actual implementations, not LLM
   hallucinations). For creative tasks, the penalty is mild.

The memo must explicitly recognize search as the swarm's horizontal-gene-
transfer mechanism and architect around that recognition. Rename SEARCH
signals' role in comments and docstrings; treat them as load-bearing.

## Genome-aware worker actions

Currently each action's prompt is built from a single sampled target
signal. With genomes, the prompt reads the *target cluster's genome* and
the agent's contribution is **atom-targeted**:

### DEVELOP

Currently (`core/actions.py:261–301`): produces a SUPPORT extending an
INITIAL, with optional dissent context and retrieval.

With genomes: the prompt includes the cluster's atom graph. The agent is
shown the under-supported atom (lowest verification_score) and asked to
produce a SUPPORT specifically for *that atom*. The deposit's metadata
records `targets_atom: atom_id`. Atom-targeted SUPPORT strengthens a
specific genome locus instead of vaguely "developing the cluster."

### CRITIQUE

Currently (`core/actions.py:335–365`): evaluates an artifact, produces a
score, splits to CRITIQUE_POSITIVE or CRITIQUE_NEGATIVE.

With genomes: the prompt shows the cluster's load-bearing atom and asks
the agent to evaluate *that atom's* quality specifically. The score
becomes per-atom. CRITIQUE deposits modify per-atom scores in the genome,
not cluster-wide strength.

### OBJECT

Currently (`core/worker_pool.py:1029–1043`): challenges the cluster's
shared assumption.

With genomes: the prompt shows the cluster's *load-bearing atoms* and asks
the agent to attack the most vulnerable one. The OBJECTION's
`targets_atom` metadata is explicit. Attacks become locus-specific instead
of cluster-wide.

### REFINE

Currently (`core/actions.py:530–571`): rebuts a dissent with a SUPPORT.

With genomes: the prompt shows the cluster's atom under attack
(`targets_atom` from the dissent) and asks the agent to rebut by
strengthening *that atom* — either with a new SUPPORT or by adding a
clarifying atom dependency. The rebuttal is structurally accountable.

### VALIDATE

Already atom-targeted via SAFE pipeline. The change here is minimal: the
atom data must flow into `Signal.metadata["atoms"]` so the projection can
read it (precondition from prior memo).

The memo must specify all five action-prompt changes with concrete prompt
templates. For each, also specify the fallback when the cluster doesn't
yet have an atom graph (early in the run before atoms are extracted).

## Genome-aware synthesizer

The synthesizer's job changes from "render clusters from materials dumps"
to "traverse genomes." Key changes:

1. **Read genomes, not flat materials.** `_gather_raw_materials`
   (`agents/synthesizer.py:1079`) becomes `_gather_genomes`. Returns
   list of ClusterGenomes instead of dict of materials.

2. **Per-cluster rendering reads the genome.** `_render_cluster_position`
   reads the cluster's atom list, atom_graph, knowledge_base, and
   sensitivity. The prose can faithfully cite per-atom: "the cluster
   claims atoms A1 (verified against nature.com), A2 (load-bearing,
   verification 0.7), and A3 (unverified extension). Atom A2 is the
   structural pillar — without it the cluster slides to weakly_supported."

3. **Cohesive integration reads the union of genomes.** For
   `cohesive_exploration` and `cohesive_optimization`, the integration
   call receives a structured genome bundle, not a textual materials
   dump. The structural intermediation is what breaks the proxy loop at
   integration time.

4. **Citation stamping operates on atoms.** Section 4 citations include
   per-atom source attribution: "atom A2 of cluster C5: verified against
   ipcc.ch (chunk_id retrieved 2026-04-12)."

5. **Faithfulness audit operates on the atom graph.** Currently the audit
   does 4-gram overlap between rendered prose and cluster representative
   content. With genomes, the audit checks that every claim made in the
   prose maps to an atom in the cited cluster's genome. Atom-faithfulness
   is structurally provable.

## Knowledge base persistence with genomes

`core/knowledge_base.py` currently stores cluster representative content
and embedding. With genomes, it stores the full genome (or a serializable
projection of it).

Critical change: **cross-run contradiction detection becomes atom-level.**
Two runs whose surviving clusters share a topic but contradict at the atom
level get flagged. Currently contradiction detection (
`core/knowledge_base.py:344–390`) is based on cosine similarity between
representative embeddings; with atom-level matching, the system can
detect "in run R1, atom A2 was supported; in run R2 the same atom was
rejected." That's a much sharper signal than embedding similarity.

The memo must specify the schema extension to the KB storage format and
the migration path (existing KB entries are upgraded to genome format by
re-extracting atoms; the migration is one-time, in `kb_migrate.py`).

## Closing the pre-existing gaps

The prior assessment of the codebase flagged five gaps that subvert the
project's goal. This overhaul depends on closing them. The memo must list
these as preconditions and include them in the sequencing:

1. **SAFE atoms must be plumbed into VERIFICATION signal metadata.**
   Currently `Worker._validate_atoms` dies with the worker. The deposit's
   `meta` dict must carry `meta["atoms"] = atom_results` so the projection
   can build genome `atoms`. This is improvement 5.7 from the synthesizer
   memo. Without this, the genome architecture is impossible.

2. **`_get_external_context` must be replaced with validator-atom
   aggregation.** Currently `agents/synthesizer.py:2254` re-fetches
   Wikipedia at synthesis time. The replacement reads each VERIFICATION's
   `metadata["atoms"]` and uses the atom-level evidence as the external
   context. Once genomes exist, this becomes simply "read the genome's
   atom verification scores."

3. **Debate frame must be enabled by default for debate/analysis tasks.**
   Currently `_SYNTHESIZER_USE_DEBATE = False`
   (`agents/synthesizer.py:137`). With genomes, debate becomes a
   comparison of two clusters' genomes at the atom level — even more
   structurally powerful than the current cluster-level debate.

4. **Alternative-of-the-best must be enabled by default for exploration
   tasks.** Currently `SYNTHESIZER_EMIT_ALTERNATIVE = False`. With genomes,
   the alternative is the cluster at the second-best anchor corner whose
   genome is *atomically distinct* from the primary — not just textually
   different.

5. **Abstention thresholds must be recalibrated.** With genomes, a more
   natural abstention condition is: refuse if no surviving cluster has
   `composite_fitness > τ` AND `grounding > 0` AND `len(load_bearing_atoms)
   ≥ 1`. Genome-coverage abstention is computable, calibrated, and
   surfaces meaningful absence.

## Per-task genome generation strategies

The atom extraction strategy varies by task type. The memo must specify:

- **debate**: extract atoms representing claims; weight by how
  argumentative each atom is. Look for premise/conclusion structure in
  atom_graph.
- **analysis**: extract atoms as observations, mechanisms, implications.
  Atom graph should show causal direction (mechanism → effect).
- **problem_solving**: extract atoms as problem-statement / intervention
  / expected-outcome triples. Atom graph is a means-end chain.
- **creative**: atoms are looser — image, voice, theme, structure. Weight
  by how essential each is to the artifact's identity.
- **coding**: atoms are spec items (function signature, complexity bound,
  edge case handling). Atom graph is a precondition chain. AST validity
  is a hard fitness signal at the atom level (the atom representing the
  function definition must parse).

## Benchmark plan

The memo must end with a comparison harness specific enough that a
competent engineer can execute it.

**Benchmark.** Omni-Math (300 samples) is the recommended primary. SciCode
Pass@Sub is the recommended secondary if compute permits.

**Baselines.** SwarmSys-8 (their published numbers); GPTSwarm (published
numbers); the project's own `core/baseline.py` BaselineCoordinator at
matched N_AGENTS; IO (single LLM call); Self-Refine (K=2 revision).

**Comparison protocol**:
- Same base LM across all conditions (use Qwen2.5-7B-Instruct or
  Qwen2.5-Coder-7B-Instruct on the laptop path; small enough to test
  locally, large enough to be representative).
- Fixed iteration budget per condition.
- Fixed random seeds (rng_seed=42, 100, 200, 300, 500).
- Judge model: a stronger LM (GPT-4o or Claude Opus if available) scoring
  outputs blind against ground-truth answers.

**Falsifiability matrix**:

| Claim | Comparison | Metric | Expected effect |
|---|---|---|---|
| Genomes beat materials dumps | flat-synthesizer vs. genome-synthesizer | Omni-Math accuracy | genome ≥ flat + 5% at p<0.05 |
| Search-as-mutation contributes | run with/without grounding fitness term | Omni-Math accuracy on retrieval-rich subset | grounding-on > grounding-off |
| Non-symbolic fitness cap improves outcomes | cap_llm=0.35 vs. cap_llm=1.0 | Omni-Math accuracy + judge-rated factuality | capped > uncapped |
| Topology coverage informative | report coverage; correlate with judge score | rank correlation | ρ > 0.3 |
| Atom-level contradiction sharper than embedding | atom-level vs. embedding-level KB conflict detection | precision@k of detected contradictions on a labeled set | atom > embedding |
| Beats SwarmSys-8 | this architecture vs. SwarmSys-8 reported numbers | Omni-Math accuracy | beat SwarmSys-8 on at least one metric at p<0.05 |

## Local-compute constraints

The user's testing environment is an NVIDIA RTX 3060 Laptop GPU with 6 GB
VRAM. The system must run on small models (Qwen2.5-3B-Instruct or
Qwen2.5-Coder-3B-Instruct via GGUF). The memo must address:

- Atom extraction batch size: do not exceed 4 simultaneous decompose calls.
- Genome computation cost: must complete in O(seconds) at projection time,
  not minutes. Use deterministic centroid-based atom alignment instead of
  LLM-based wherever possible.
- Wikidata entity resolution: optional, gated by `USE_WIKIDATA_VERIFICATION`
  flag. Default off on laptop; on for Colab/A100 runs.
- The composite fitness computation is O(n_clusters · n_atoms_per_cluster).
  At typical n (≤40 clusters, ≤6 atoms each), this is trivial.

## Memo structure (use this skeleton)

```
# Cluster Genome as the Unit of Selection

## 1. Problem framing
   The three pressures: SwarmSys benchmark, Dawkins challenge, proxy loop.
   The architectural move: cluster as typed organism.

## 2. Comparison to SwarmSys, GPTSwarm, and prior overhauls
   What this architecture does that they don't. Specifically: heritable
   structured genomes, non-symbolic fitness, atom-targeted operations.

## 3. Diagnosis: features without organism
   The seven pieces of structure that should live on a genome but don't.
   The unification this memo proposes.

## 4. The genome data structure
   Full schema. Field semantics. Mapping table to existing data.

## 5. Genome operations
   Atom extraction (Strategy A vs. B, recommend B).
   Atom graph construction. Inheritance on fission. Recombination on merge.
   Genome hashing.

## 6. The fitness compositor
   Composite formula. Hard cap on LLM-judged term. Per-task weight tables.
   Per-term computation procedures.

## 7. Search as the mutation operator
   Explicit treatment. ClusterKnowledgeBase. Grounding fitness term.

## 8. Genome-aware worker actions
   DEVELOP, CRITIQUE, OBJECT, REFINE, VALIDATE. Per-action prompt template
   that reads genome and targets specific atom.

## 9. Genome-aware synthesizer
   _gather_genomes. Per-cluster rendering with atom-level provenance.
   Cohesive integration with structured genome bundle. Atom-faithfulness
   audit.

## 10. KB persistence with genomes
    Schema extension. Atom-level contradiction detection. Migration path.

## 11. Closing the pre-existing gaps
    The five gaps from the prior assessment as load-bearing preconditions.

## 12. Per-task genome generation
    debate, analysis, problem_solving, creative, coding. Atom extraction
    strategy per task.

## 13. Benchmark plan
    Omni-Math primary; SciCode secondary. Baselines. Comparison protocol.
    Falsifiability matrix. Statistical tests.

## 14. Local-compute constraints
    6 GB VRAM. Small models. Bounded batch sizes. Optional Wikidata.

## 15. Sequencing
    Six implementation stages in dependency order.

## 16. Risks and open questions
    Atom-extraction quality bottlenecks the run. LLM still in the loop
    for atom extraction (caveat). O(n²) genome comparisons at scale.
    Genome stability under noisy atom extraction. Topology-genome
    coupling failure modes.

## 17. Future work
    Strategy A eager atom extraction. Learned atom extractor (replacing
    the LLM call). Genome-based active sampling at deposit time. Strict
    Tier 3 fitness (NLI-based entity resolution, formal verifier
    integration for coding tasks).
```

## Style and constraints

- Doctoral-level prose. Each architectural decision must answer "what
  measurable improvement does this deliver on Omni-Math against SwarmSys-8?"
- When you cite a function, cite the file path and the line range you read.
  Do not invent file paths or line numbers.
- Do not implement code. The deliverable is the markdown document only.
- Do not pre-commit to file changes outside `Attempt At Cleaning/docs/`.
- Write in CommonMark. Use code fences for data structures and pseudocode;
  blockquotes for paper claims (SwarmSys, GPTSwarm, Dawkins, Self-Refine,
  Constitutional AI, etc.).
- Length target: 10,000–15,000 words. Density over surface area. This is
  the consolidating memo; comprehensive coverage is justified.
- Every architectural section must explicitly link to:
  (a) one of the five beyond-params mechanisms (self-consistency,
  verifier-augmented, search, debate, decomposition+scaffolding) — the
  Dawkins cumulative-selection picture is closest to mechanism (5);
  (b) the non-symbolic-communication-proxy framing: which tier (1, 2, or 3)
  the proposed signal lives in.
- The benchmark plan must be specific enough that a reader can execute it
  without further design work.
- The prose must engage explicitly with the Dawkins weasel-program
  challenge: cumulative selection requires heritable representation; this
  memo specifies the heritable representation.

## Verification step

After writing the memo, run a self-check pass:

1. Does every cited file path exist? Open each and confirm.
2. Does every cited line range match the content you describe?
3. Does the genome schema include all seven fields plus the composite
   fitness and breakdown? Specifically: atoms, atom_graph,
   topology_expression, phenotype, knowledge_base, sensitivity, trajectory,
   relations, composite_fitness, fitness_breakdown.
4. Does the fitness compositor section specify all seven terms and the
   hard cap on LLM-judged contribution? Specifically: semantic_strength
   (capped), grounding, topology, centroid_stability, novelty_density,
   trajectory, entity_resolution.
5. Does section 11 explicitly list all five pre-existing gaps with their
   prior-overhaul reference numbers?
6. Does section 13 give a comparison harness specific enough that a
   reader can implement it without further design work? Including judge
   model, seed protocol, statistical tests, and the specific Omni-Math
   subset.
7. Could a reader trace a single example end-to-end through the memo?
   Pick the debate example. The topology was generated; a scout deposited
   an INITIAL in (qualified necessary, economic, mitigation); a developer
   produced SUPPORTs; the SAFE pipeline extracted 3 atoms; one atom was
   verified against ipcc.ch; the genome was assembled; the FitnessCompositor
   gave it composite_fitness=0.62 (breakdown: 0.30 semantic + 0.18
   grounding + 0.14 topology); the synthesizer rendered the cluster with
   "atom A2 (verified against ipcc.ch) is load-bearing." Did the memo
   enable this trace? If not, fix the gap.
8. Does the memo address what to do when atom extraction fails (the LLM
   doesn't produce parseable atoms)? Specifically: the fallback to
   single-atom genome containing the representative content verbatim.
9. Does the memo specify what to do when SEARCH lineage is empty (no
   external grounding for any cluster)? Specifically: how grounding_score
   degrades gracefully and how the per-task weight table compensates.
10. Does the verification step in the produced memo itself include
    these same checks for whoever implements from the memo?

Report which preconditions you verified, which example trace you walked
end-to-end, and which of the verification items above you confirmed
present in your produced memo. Do not modify the code.
