# Claude Code prompt — Cluster Genome as the Unit of Selection (CORRECTED)

> **Why this corrected version exists.** The original prompt
> (`CLUSTER_GENOME_OVERHAUL_PROMPT.md`) was written against an older snapshot
> of `Attempt At Cleaning/`. Several of its factual claims about the current
> code are stale, and a number of its cited line numbers are wrong. Most
> importantly, three of its five "load-bearing preconditions" are **already
> implemented** in the current tree, and one genome field it calls "planned,
> not yet implemented" **already exists**. This version re-grounds every
> claim against the code as it stands and corrects the citations. The
> architectural proposal (promote the cluster to a typed, heritable genome)
> is preserved; the false premises are removed.
>
> **Corrections applied (changelog):**
>
> 1. Diagnosis table — `CounterfactualSensitivity` is implemented as
>    `ClusterSensitivity` (`core/projection.py:177`, built by
>    `_build_sensitivities` at `core/projection.py:570`), not "planned."
> 2. Diagnosis table — SAFE atoms are **already** plumbed into VERIFICATION
>    metadata (`core/worker_pool.py:759`) and projected as `AtomProjection`
>    (`core/projection.py:419` `_build_atoms`); they do not die with the worker.
> 3. Precondition #1 (atoms-in-metadata): **already satisfied** — downgraded
>    from blocker to a verify-still-wired check.
> 4. Precondition #3 (debate default): `_SYNTHESIZER_USE_DEBATE = True`
>    already (`agents/synthesizer.py:137`) — removed as a precondition.
> 5. Precondition #4 (alternative default): `SYNTHESIZER_EMIT_ALTERNATIVE = True`
>    already (`agents/synthesizer.py:150`) — removed as a precondition.
> 6. Line-number fixes: `_gather_raw_materials` is at
>    `agents/synthesizer.py:1465` (not 1079); `_get_external_context` is at
>    `agents/synthesizer.py:2993`, called at `:2329` (not 2254);
>    `_detect_contradictions` is at `core/knowledge_base.py:373` (not 344–390);
>    `_format_safe_external` lives in `core/worker_pool.py:540` (not actions.py);
>    `refine_prompt` is at `core/actions.py:546` (the 530–571 range straddled
>    `validate_parse`).
> 7. KB migration — `kb_migrate.py` and a `schema_version` system
>    (`_SCHEMA_VERSION`, currently 2) already exist; the genome migration is a
>    v2→v3 extension, not a from-scratch build.

Paste everything below the `---` line into Claude Code from the repository root
(`C:\Users\agsse\Downloads\ai_swarm_mechanics-main (4)\ai_swarm_mechanics-main`).

This is the **fourth and consolidating** design memo. It supersedes the prior
`TOPOLOGY_LATTICE_OVERHAUL.md` memo by re-framing topology, multi-resolution
lattice, and counterfactual sensitivity as facets of a single architectural
object: the **cluster genome**. The two earlier memos (`SYNTHESIZER_OVERHAUL`
and `AGENT_RETRIEVAL_OVERHAUL`) remain in force as load-bearing preconditions.
After this memo lands, `docs/TOPOLOGY_LATTICE_OVERHAUL.md` should be moved to
`docs/archive/` with a note pointing to this memo as the canonical design.

The architectural motivation: the swarm has built a great deal of typed
structure — SAFE atoms flow into VERIFICATION metadata and are projected as
`AtomProjection`; the topology and multi-resolution lattice exist
(`FrameProjection`, `PropositionProjection`, `CrossLevelEdge`); counterfactual
sensitivity is computed (`ClusterSensitivity`); the cluster registry maintains
persistent identity; search injects external material; the synthesizer reads
these structures. **But these typed objects live at the `SynthesisProjection`
level as parallel lists, keyed back to clusters by `representative_id`, rather
than as fields of a single per-cluster object.** No subsystem owns a unified,
heritable cluster object that survives fission, recombines on merge, and
carries its own fitness history. **Promote the cluster to a typed cluster
genome** — gathering the already-typed pieces into one heritable substrate
attached to each `ClusterProjection` — and operations that no individual
subsystem can perform alone become available: genome recombination on merge,
genome inheritance on fission, atom-targeted prompting, cross-run atom-level
contradiction detection.

The corrected framing is therefore **consolidation and heritability**, not
"features without an organism." The pieces exist and are typed; what is
missing is (a) their unification into one per-cluster object, (b) heritability
of that object across fission/merge, and (c) a composite, non-symbolic fitness
function that operates over it.

---

You are working in the `Attempt At Cleaning/` folder of a stigmergic
multi-agent swarm codebase. Read the following files end-to-end before
proposing anything. **Confirm the current contents and line numbers yourself —
do not trust the line numbers in any prior memo, including this one, without
re-opening the file.**

- `core/cluster_registry.py` (full, 257 lines) — persistent cluster identity,
  running-mean `centroid` (`_Cluster.centroid`, line 62), join/create
  (`try_join` line 88, `create` line 118), and fission via `_reanchor`
  (line 188) + split/eject (lines 220–257). This is the substrate the genome
  attaches to. Note there is **no** `centroid_at_formation` field yet — you
  will propose adding it.
- `core/projection.py` (full, 1442 lines) — the existing `ClusterProjection`
  schema (line 87), `TrajectoryFeatures` (line 68), `InterClusterEdge`
  (line 114), and the already-implemented lattice/sensitivity types:
  `AtomProjection` (line 134), `PropositionProjection` (line 147),
  `FrameProjection` (line 157), `CrossLevelEdge` (line 165),
  `ClusterSensitivity` (line 177). Builders: `_build_inter_cluster_edges`
  (line 336), `_build_atoms` (line 419), `_build_sensitivities` (line 570),
  all wired in `build_projection` (line 719; see lines 798, 808, 816).
- `core/topology.py` (full, 260 lines) — `AnswerSpaceTopology`,
  `AnchorCorner`, cell assignment for scouts.
- `core/signal_store.py` (focus on the `Signal` dataclass with `cluster_id`,
  the deposit path, the embedding cache; 1061 lines).
- `core/worker_pool.py` (1347 lines; focus on `Worker.iterate` at line 665,
  `_gather_target` at line 851, the SAFE pipeline in the VALIDATE branch at
  lines 1045–1151 — `_safe_decompose` is called at line 1061 and atoms are
  written into deposit metadata at line 759 — the OBJECT target-selection
  branch at lines 1029–1043, and `_format_safe_external` at line 540).
- `core/actions.py` (641 lines; action specs and prompt builders:
  `develop_prompt` line 261, `critique_prompt` line 335, `object_prompt`
  line 371, `validate_parse` line 466, `refine_prompt` line 546).
- `agents/synthesizer.py` (full, 3236 lines) — current rendering paths and the
  pre-existing improvements (revision loop, best-of-N, debate, alternative).
  Note: `_gather_raw_materials` is at line 1465; `_get_external_context` is
  defined at line 2993 and called at line 2329; `_SYNTHESIZER_USE_DEBATE = True`
  at line 137 and `SYNTHESIZER_EMIT_ALTERNATIVE = True` at line 150 (both are
  already enabled — see §11).
- `core/knowledge_base.py` (full, 488 lines) — current KB persistence; already
  versioned via `_SCHEMA_VERSION` (currently 2), with `kb_migrate.py` present
  at repo root. `_merge_entries` (line 313) and `_detect_contradictions`
  (line 373) are the dedup and contradiction paths. Will need schema v2→v3
  extension.
- `core/signal_types.py` (56 lines) — note that SEARCH is declared (line ~48)
  and explicitly documented as "Not counted in support_set or dissent_set; the
  projection treats it as metadata" — i.e. it is **not fitness-bearing** in
  projection or convergence.

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
- Lineage (parent_id pointers, cluster_id persistence in `cluster_registry`)
- **And, already, a rich typed representation of what each cluster claims**
  (atoms, propositions, sensitivities) — but that representation is recomputed
  at projection time and is _not heritable_: it does not survive fission, does
  not recombine on merge, and carries no fitness history that persists across
  the run.

The missing primitive is therefore not "a representation of what the cluster
claims" — that exists — but a **heritable** representation: one that descends
with modification through `_reanchor` fission and through planner merges, so
that mutation and selection are commensurable across generations. Without
heritability, the swarm runs as parallel sampling with post-hoc selection —
the architectural ceiling of an ensemble, not of cumulative selection. The
genome is the substrate that makes the already-existing typed representation
heritable.

**3. The symbolic-communication-proxy loop.** Currently the overwhelming
majority of the swarm's selection pressure comes from LLM-on-LLM judgment:
agents producing scores on other agents' outputs. Because all agents share
the base model's biases, the loop is closed — errors compound rather than
wash out. The fix is **non-symbolic fitness signals**: geometric (embedding
stability, novelty density), structural (topology coverage, cluster atomic
structure), temporal (trajectory consistency), and external (search-lineage
diversity, entity resolution against Wikidata). The genome holds all of these
as fields, so fitness composition is a structured operation over the genome,
not a black-box scalar.

> **Honesty caveat to engage with in §6 and §16.** Some of the "non-symbolic"
> terms still depend on the base model. `novelty_density` and
> `centroid_stability` are computed over embeddings produced by the same model
> family that does the judging, so the closed-loop pathology partially leaks
> back in through embedding space. `entity_resolution` against Wikidata is the
> only term fully independent of the base model, and it is gated off by default
> on the laptop path (`USE_WIKIDATA_VERIFICATION=False`). The memo must not
> claim "geometric ⇒ clean"; it must state which terms are genuinely
> model-independent (Tier 3) versus model-derived-but-not-LLM-judged (Tier 2).

The Mount Everest framing from the prior topology memo still applies: build
the bounds before the interior. With the genome architecture, the framing
extends: the bounds (topology) are the **expression environment**; the
interior (cluster contents) is the **genome**. Selection operates on the
match between genome and environment, mediated by non-symbolic fitness.

## Diagnosis: typed pieces that are not yet a heritable organism

The codebase has accumulated typed structure that _already lives near the
cluster_ but is neither unified into one per-cluster object nor heritable. The
column "Where it lives now" reflects the **current** code:

| Piece                      | Where it lives now                                                                                                                                                                                                                                          | What it should be on the genome                                         |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| SAFE atoms                 | Written to VERIFICATION `metadata["atoms"]` (`worker_pool.py:759`); projected as `AtomProjection` list on `SynthesisProjection` (`projection.py:419`, wired at `:808`) — but keyed by `representative_id`, not owned by a cluster object, and not heritable | Atom list (the basepairs), owned by the genome and inherited on fission |
| Source/query lineage       | Scattered across SEARCH signal metadata; **not** aggregated per cluster                                                                                                                                                                                     | Knowledge base / external grounding lineage (`ClusterKnowledgeBase`)    |
| Topology coords            | Signal `metadata["topology_coords"]`; topology coverage on `SynthesisProjection`                                                                                                                                                                            | Expression-environment annotation (`TopologyExpression`)                |
| Centroid + stability       | `cluster_registry._Cluster.centroid` (line 62); running L2 mean. **No** `centroid_at_formation` snapshot exists                                                                                                                                             | Phenotype anchor (add formation snapshot + drift)                       |
| Counterfactual sensitivity | **Implemented** as `ClusterSensitivity` (`projection.py:177`), built by `_build_sensitivities` (`:570`, wired at `:816`) — but recomputed each projection, not stored on a heritable object                                                                 | Load-bearing atom annotation, persisted on the genome                   |
| Inter-cluster edges        | **Implemented** as `InterClusterEdge` (`projection.py:114`), built by `_build_inter_cluster_edges` (`:336`, wired at `:798`)                                                                                                                                | Genome relationship records (`GenomeRelations`)                         |
| Trajectory features        | `TrajectoryFeatures` (`projection.py:68`) on `ClusterProjection`                                                                                                                                                                                            | Genome fitness history (persisted across the run)                       |

The unification this memo proposes is therefore **not** "build typing that
doesn't exist." It is: (1) gather these already-typed pieces into one
per-cluster `ClusterGenome` object; (2) make that object heritable through
`_reanchor` fission and planner merge; (3) add the two pieces that genuinely
do not exist yet — the per-cluster `ClusterKnowledgeBase` (SEARCH-lineage
aggregation) and the composite, capped fitness function.

> **The memo must argue the nesting decision explicitly.** The current design
> keeps atoms/sensitivities/edges as typed lists on `SynthesisProjection`,
> keyed by `representative_id`. Nesting them into a per-cluster `ClusterGenome`
> has a real cost: every consumer of those projection-level lists
> (`agents/synthesizer.py`, `build_plan`, the renderer audit) must be
> rewritten. The memo must state plainly **what nesting buys that
> projection-level typing does not** — the answer is heritability
> (fission/merge), atom-targeted prompting, and genome hashing for KB reuse —
> and must not pretend the current structure is "flat scalars." It is typed;
> it is just not heritable and not per-cluster-owned.

## The cluster genome — data structure

The memo must specify the genome with full field semantics. Where a field
duplicates an existing projection type, the memo must say so and specify
whether the genome **wraps**, **replaces**, or **supersedes** it. Canonical
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
    # NOTE: maps onto the existing AtomProjection (projection.py:134), which
    # already has atom_id, text, weight, verification_score, source_tag, query,
    # parent_cluster_id, parent_verification_id. The memo must specify the
    # field-by-field correspondence and which AtomProjection fields are reused.
```

```python
@dataclass
class ClusterKnowledgeBase:
    queries_issued: list[str]            # all queries from members' SEARCH lineage
    source_domains: set[str]             # distinct top-level domains across the lineage
    source_count: int                    # total distinct retrieved chunks
    domain_diversity: float              # normalized Shannon entropy over domain frequencies
    parametric_content_ratio: float      # fraction of cluster members with no SEARCH lineage
    cross_cluster_source_overlap: float  # fraction of sources shared with other clusters
    # This is genuinely new: SEARCH lineage is currently inert metadata
    # (signal_types.py) and is not aggregated per cluster anywhere.
```

```python
@dataclass
class TopologyExpression:
    coords: Optional[tuple[str, ...]]    # which cell of AnswerSpaceTopology
    cell_label: Optional[str]            # human-readable cell description
    is_anchor: bool                      # True if cluster occupies an anchor corner
    cell_occupancy_rank: int             # 0 = sole occupant; N = N-th cluster in this cell
    # Sourced from signal metadata["topology_coords"] + topology.py cell lookup,
    # and from SynthesisProjection.topology_coverage which already exists.
```

```python
@dataclass
class Phenotype:
    centroid: list[float]                # current L2-normalized centroid (from _Cluster.centroid)
    centroid_at_formation: list[float]   # NEW field on _Cluster; snapshot at creation
    centroid_drift: float                # cosine distance between formation and current
    centroid_stability: float            # 1.0 = stable, 0.0 = drifting wildly
    novelty_density: float               # inverse mean distance to other clusters' centroids
```

```python
@dataclass
class FitnessTrajectory:
    formation_iteration: int
    fitness_history: list[tuple[int, float]]   # (iter, composite_fitness) sampled at intervals
    strength_history: list[tuple[int, float]]
    member_count_history: list[tuple[int, int]]
    monotone_growth: bool                # True if fitness rose monotonically
    consolidation_iteration: Optional[int]  # iter when fitness plateaued, or None
    # Distinct from the existing TrajectoryFeatures (projection.py:68), which
    # records iter_first_support/dissent/verification etc. The memo must say
    # whether FitnessTrajectory subsumes TrajectoryFeatures or composes it.
```

```python
@dataclass
class CounterfactualSensitivity:
    load_bearing_atoms: list[str]        # atom_ids whose removal flips status
    marginal_atoms: list[str]
    support_removal_robustness: float    # min strength of single support whose removal flips
    competing_takeover: Optional[str]    # cluster_id that would advance if this fell
    topology_cells_at_risk: list[tuple]  # cells that lose coverage if this cluster falls
    # IMPORTANT: ClusterSensitivity already exists (projection.py:177) with
    # support_removal_robustness, dissent_amplification_tolerance,
    # load_bearing_supports, marginal_supports, competing_takeover,
    # topology_uncovered_on_removal. The genome version is ATOM-level rather
    # than SUPPORT-level. The memo must specify the migration from
    # support-level (built by _build_sensitivities, projection.py:570) to
    # atom-level, and whether both coexist.
```

```python
@dataclass
class GenomeRelations:
    parent_genomes: list[str]            # cluster_ids this descended from (fission/merge)
    descendant_genomes: list[str]        # cluster_ids that descended from this (fission)
    inter_cluster_edges: list            # reuse existing InterClusterEdge (projection.py:114):
                                         # shared_evidence, co_contested, complements,
                                         # alternatives, supersedes, tension
```

```python
@dataclass
class ClusterGenome:
    """The typed organism. Each ClusterProjection carries one of these."""
    cluster_id: str
    genome_hash: str                     # stable hash over the atom set; identical genomes detectable
                                         # (note: _cluster_hash already exists in knowledge_base.py:76;
                                         #  reuse or extend rather than reinvent)
    formation_iteration: int
    # The basepairs
    atoms: list[AtomFact]
    atom_graph: dict[str, list[str]]     # atom_id → atom_ids it depends on
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
    # Existing fields preserved for backward compat (current schema, projection.py:87):
    representative_id: str
    member_ids: list[str]
    support_set: list[str]
    dissent_set: list[str]
    verification_set: list[str]
    support_diversity: int
    dissent_pressure: float
    verification_score: float
    partition_origins: list[str]
    support_depth: int = 1
    status: str = "unclassified"
    unverified: bool = False
    support_tree: dict = ...
    trajectory: TrajectoryFeatures = ...
    # NEW: the genome
    genome: ClusterGenome = ...
```

## The no-leak rule under genomes (invariant — do not break)

The no-leak rule is documented at `core/actions.py:16–19` ("prompts may only
render `Signal.content` (plus the agent's own prior outputs and retrieved
search chunks). No ancestry text, no other agents' reasoning, no
parent_content fields."). The pool enforces it by sampling a single target
signal in `_gather_target` (`core/worker_pool.py:851`) and the synthesizer
re-states it in `_gather_raw_materials` (`agents/synthesizer.py:1465`: "Reads
only `Signal.content` — no reasoning chains, no metadata that could leak
deposit ordering.").

The genome is **no-leak-safe iff** agents and the synthesizer see only:

- atom **content** (`AtomFact.text`, which is itself proposition text, not
  reasoning), and
- **identifiers** (`atom_id`, `extracted_from` signal IDs, `atom_graph` edges,
  `cluster_id`, `genome_hash`, relation IDs), and
- **scalar fitness fields** (verification scores, centroid_stability, etc.).

The genome must **never** surface another agent's reasoning chain, ancestry
text, or the raw `parent_content` of a signal the current agent did not author.
`extracted_from` carries IDs, not text — that is fine. The memo must add an
explicit "no-leak conformance" clause to the genome spec and to each
genome-aware prompt template in §8, and the §0 verification step must confirm
that no genome-aware prompt renders foreign reasoning text.

This is also the mechanistic statement of the user's framing: the workers are
not "monkeys typing random letters." Each worker performs **cumulative
selection on a heritable genome** — it is shown the target cluster's genome
(its atoms and which atom is weakest) and mutates _that_ locus, the way the
weasel program fixes correct letters and varies the rest. Variation +
selection + heritability is the whole point; the genome is the heritable
representation that makes the selection cumulative rather than i.i.d. sampling.

## How prior subsystems feed the genome

The memo must include a mapping table showing where each genome field is
populated from, **with current line numbers verified against the tree**:

| Genome field                      | Populated from                                                                                                                                                                                                                                                                                                     |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `atoms`                           | `metadata["atoms"]` already written at `core/worker_pool.py:759` (VALIDATE branch, SAFE pipeline `core/worker_pool.py:1045–1151`); already read into `AtomProjection` by `_build_atoms` (`core/projection.py:419`). Extend to also extract atoms from non-validator members at projection time (Strategy B, below) |
| `atom_graph`                      | Signal `parent_id` links + content-overlap inference among atoms                                                                                                                                                                                                                                                   |
| `topology_expression`             | Signal `metadata["topology_coords"]` + `core/topology.py` cell lookup + existing `SynthesisProjection.topology_coverage`                                                                                                                                                                                           |
| `phenotype.centroid`              | `cluster_registry._Cluster.centroid` (`core/cluster_registry.py:62`) directly                                                                                                                                                                                                                                      |
| `phenotype.centroid_at_formation` | NEW field on `_Cluster`; snapshot in `create` (`core/cluster_registry.py:118`)                                                                                                                                                                                                                                     |
| `phenotype.novelty_density`       | Computed at projection time over all cluster centroids                                                                                                                                                                                                                                                             |
| `knowledge_base.source_domains`   | Walked from all SEARCH signals in the cluster's transitive lineage (NEW aggregation)                                                                                                                                                                                                                               |
| `knowledge_base.queries_issued`   | Same                                                                                                                                                                                                                                                                                                               |
| `knowledge_base.domain_diversity` | Shannon entropy over source domain frequencies                                                                                                                                                                                                                                                                     |
| `sensitivity.load_bearing_atoms`  | Atom-level extension of `_build_sensitivities` (`core/projection.py:570`), which currently computes support-level `load_bearing_supports`                                                                                                                                                                          |
| `trajectory.fitness_history`      | Sampled at fixed intervals during the run; logged separately and stitched in at projection                                                                                                                                                                                                                         |
| `relations.inter_cluster_edges`   | Existing `InterClusterEdge` from `_build_inter_cluster_edges` (`core/projection.py:336`)                                                                                                                                                                                                                           |
| `relations.parent_genomes`        | Logged by `cluster_registry._reanchor` (`core/cluster_registry.py:188`, split/eject `220–257`) on fission; logged by planner's `merge_groups`                                                                                                                                                                      |
| `composite_fitness`               | `FitnessCompositor` (see below)                                                                                                                                                                                                                                                                                    |

## Genome operations

The memo must specify each operation with file paths, function signatures,
and integration points.

### Atom extraction (extending SAFE — partially already done)

SAFE atoms are currently produced at the VALIDATE action
(`core/worker_pool.py:1061`, `_safe_decompose`) and **already** written into
`metadata["atoms"]` (`core/worker_pool.py:759`), then read by
`_build_atoms` (`core/projection.py:419`). What is missing is atom coverage
for clusters that **no VALIDATE action reached**. Two extension strategies; the
memo must recommend one:

**Strategy A: Eager extraction at deposit time.** Every INITIAL/SUPPORT
deposit triggers `_safe_decompose` on its content. Atoms are stored in
`Signal.metadata["atoms"]`. Projection collates per-cluster atoms by union.
Cost: one extra LLM call per deposit. On laptop with NUM_FORAGERS=4 and 16
iters/round this is 64 extra calls per round — expensive.

**Strategy B: Lazy extraction at projection time.** When `build_projection`
(`core/projection.py:719`) runs, decompose each cluster's representative +
top-K supports in one batch call **for clusters that lack validator-produced
atoms**. Reuse the atoms already in `metadata["atoms"]` where present. Cost:
one call per un-validated cluster at projection time.

Strategy B is recommended for laptop runs (6 GB VRAM, small models). The memo
should specify how lazily-produced atoms get associated back to specific
deposits (via the existing `extracted_from`/`parent_verification_id` pattern in
`AtomProjection`, or via content-overlap matching). Strategy A is preferred
when compute permits; mark it as future work.

### Atom graph construction

For each cluster, derive `atom_graph: dict[atom_id → list[atom_id]]`. An
edge `a → b` means atom `b` is a dependency of atom `a`. Construction rules:

1. Atoms extracted from a SUPPORT signal depend on the atoms extracted
   from its parent INITIAL.
2. Atoms extracted from a REFINE deposit depend on the dissent atom they
   address.
3. Atoms whose `extracted_from` overlap with another atom's `extracted_from`
   are siblings (no edge).
4. Atoms whose content overlap exceeds a threshold (cosine ≥ 0.85) are
   collapsed into one atom; the surviving atom records both source signals in
   `extracted_from`. (Reuse the existing 0.85 dedup threshold from
   `knowledge_base.py` so the codebase has one duplicate-similarity constant.)

The graph is a DAG over atoms within a cluster. Specify how it is computed and
stored.

### Inheritance on fission

When `cluster_registry._reanchor` ejects members into a new cluster
(`core/cluster_registry.py:220–257`), the new cluster's genome must inherit a
_subset_ of the parent's atoms: atoms whose `extracted_from` intersects the
ejected members' IDs are inherited; others are excluded. The new cluster's
`relations.parent_genomes` records the parent's cluster_id; the parent's
`relations.descendant_genomes` adds the new cluster_id. This is descent with
modification: the daughter carries some parent DNA, and the mutation is that
the ejected members are a non-random subset (ejection was driven by centroid
drift below `CLUSTER_SPLIT_THRESHOLD`).

### Recombination on merge

When the synthesizer's planner produces `merge_groups` declaring two clusters
the same position, their genomes recombine: union the atoms, deduplicate by
content overlap, average the centroids, union the knowledge_bases, concatenate
the trajectories. Both parents are recorded in `relations.parent_genomes`.
Specify whether the recombined genome forms a _new_ cluster_id or whether one
merged cluster becomes canonical and absorbs the other.

### Genome hashing

`genome_hash` is a stable hash over the sorted atom set's content. Reuse or
extend `_cluster_hash` (`core/knowledge_base.py:76`) rather than introducing a
parallel hashing scheme. Two clusters with identical genomes hash identically —
useful for KB reuse (a previously-seen genome need not be re-stored) and for
detecting accidental duplication.

## The fitness compositor

The composite fitness replaces the single-strength scalar in survival
decisions. Specification:

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
    terms["semantic_strength"] = min(terms["semantic_strength"], cap_llm)
    return sum(weights.get(k, 0.0) * v for k, v in terms.items())
```

The memo must specify:

1. **The hard cap on `semantic_strength`.** Default `cap_llm = 0.35`. The
   architectural reason: closed-loop pathology cannot dominate selection when
   the LLM-judged term is capped below half. **But the memo must also state the
   honesty caveat from §1**: `centroid_stability` and `novelty_density` are
   computed over the base model's own embeddings, so capping `semantic_strength`
   alone does not fully sever the closed loop. Classify each term by tier
   (Tier 1 self-consistency / Tier 2 model-derived-but-not-LLM-judged / Tier 3
   model-independent). Only `entity_resolution` (Wikidata) and AST/test
   validity for coding are Tier 3. This is the single most important
   configuration choice in this overhaul; it must be presented with its
   limitation, not as a clean fix.

2. **Per-task weight tables.** For factual tasks (coding) weight `grounding`
   and `entity_resolution` heavily; `semantic_strength` matters less because
   tests and AST checks are authoritative. For non-factual tasks (debate,
   analysis, creative) weight `topology`, `novelty_density`, and
   `centroid_stability` more heavily because external grounding is harder to
   come by. Provide a default weight table per task type.

3. **Computation procedure for each non-LLM term**:
   - `_grounding_score`: weighted combination of `domain_diversity` and
     `(1 - parametric_content_ratio)`. Normalized to [0, 1].
   - `_topology_contribution`: 1.0 if sole occupant of its cell; lower if
     multiple clusters occupy it. Anchor cells get a 1.5 multiplier.
   - `centroid_stability`: 1.0 if `centroid_drift < 0.1`, decreasing linearly
     to 0.0 at `centroid_drift > 0.5`. Drift measured between
     `centroid_at_formation` and current `centroid`.
   - `novelty_density`: mean cosine distance from this centroid to all others,
     normalized to [0, 1].
   - `_trajectory_score`: 1.0 if `monotone_growth`; 0.5 if monotonic but slow;
     0.2 if oscillating. Weights `consolidation` positively.
   - `_entity_resolution_score`: fraction of atom-text entities that resolve in
     Wikidata. Cached per-entity within a run. Gated by
     `USE_WIKIDATA_VERIFICATION` (default off on laptop).

4. **Fitness breakdown auditability.** Every cluster's `fitness_breakdown`
   records each term's contribution, so the synthesizer can render "this
   cluster's fitness comes mostly from grounding (β=0.7) and centroid stability
   (δ=0.6); LLM-judged contribution is at the cap (α=0.35)."

## Search as the mutation operator — explicit treatment

SEARCH signals are currently inert in projection and convergence
(`core/signal_types.py`: "Not counted in support_set or dissent_set; the
projection treats it as metadata"). This overhaul promotes them to first-class
fitness contributors:

1. **Walk SEARCH signal lineage at projection time.** For each cluster, collect
   all SEARCH signals in its transitive ancestry (scout SEARCH that produced
   its INITIAL; developer SEARCH triggered by sparse-cluster hook; validator
   SEARCH per atom). Each SEARCH has metadata: `query`, source titles/URLs.
2. **Materialize `ClusterKnowledgeBase`.** Aggregate the SEARCH lineage into
   the structured record above. Set arithmetic; no LLM calls.
3. **Add `grounding_score` as a non-LLM fitness term.** A cluster grounded in
   five distinct domains scores higher than one with no external grounding.
4. **Penalize purely parametric clusters.** `parametric_content_ratio = 1.0`
   (no member triggered any search) → fitness penalty proportional to how much
   the task type prefers grounded claims. Strong for coding, mild for creative.

The memo must explicitly recognize search as the swarm's horizontal-gene-
transfer mechanism and architect around that. Rename SEARCH's role in comments
and docstrings; treat it as load-bearing.

## Genome-aware worker actions

Each action's prompt is currently built from a single sampled target signal.
With genomes, the prompt reads the _target cluster's genome_ and the agent's
contribution is **atom-targeted**. Cite the current prompt builders precisely:

### DEVELOP

Current builder: `develop_prompt` (`core/actions.py:261`) — produces a SUPPORT
extending an INITIAL, with optional dissent context and retrieval.
With genomes: include the cluster's atom graph; show the under-supported atom
(lowest `verification_score`) and ask for a SUPPORT for _that atom_. Deposit
metadata records `targets_atom: atom_id`.

### CRITIQUE

Current builder: `critique_prompt` (`core/actions.py:335`) — evaluates an
artifact, produces a score, splits to CRITIQUE_POSITIVE / CRITIQUE_NEGATIVE
(see `core/actions.py` around line 359).
With genomes: show the cluster's load-bearing atom; ask for evaluation of
_that atom_. Score becomes per-atom.

### OBJECT

Current builder: `object_prompt` (`core/actions.py:371`); target/representative
selection happens in `core/worker_pool.py:1029–1043`.
With genomes: show the cluster's _load-bearing atoms_; ask the agent to attack
the most vulnerable one. `targets_atom` explicit.

### REFINE

Current builder: `refine_prompt` (`core/actions.py:546`) — rebuts a dissent
with a SUPPORT.
With genomes: show the atom under attack (`targets_atom` from the dissent); ask
the agent to rebut by strengthening _that atom_ — a new SUPPORT or a clarifying
atom dependency.

### VALIDATE

Already atom-targeted via the SAFE pipeline (`core/worker_pool.py:1045–1151`),
and atoms already flow into `Signal.metadata["atoms"]` (`:759`). The change
here is minimal: ensure the atom data continues to reach projection unchanged.

The memo must specify all five action-prompt changes with concrete prompt
templates, and for each, the fallback when the cluster has no atom graph yet
(early in the run, before atoms are extracted): fall back to the current
single-target prompt.

## Genome-aware synthesizer

The synthesizer's job changes from "render clusters from materials dumps" to
"traverse genomes." Cite current code precisely:

1. **Read genomes, not flat materials.** `_gather_raw_materials`
   (`agents/synthesizer.py:1465`) becomes (or is wrapped by) `_gather_genomes`,
   returning a list of `ClusterGenome` instead of a dict of materials.
2. **Per-cluster rendering reads the genome.** The per-cluster renderer reads
   the atom list, atom_graph, knowledge_base, and sensitivity, and can cite
   per-atom: "the cluster claims atoms A1 (verified against nature.com), A2
   (load-bearing, verification 0.7), and A3 (unverified extension)."
3. **Cohesive integration reads the union of genomes.** For
   `cohesive_exploration` / `cohesive_optimization`, the integration call
   receives a structured genome bundle, not a textual dump.
4. **Citation stamping operates on atoms.** Section 4 citations include
   per-atom source attribution.
5. **Faithfulness audit operates on the atom graph.** The current audit does
   n-gram overlap between rendered prose and cluster representative content;
   with genomes, the audit checks that every claim maps to an atom in the cited
   genome.

> Note: `_SYNTHESIZER_USE_DEBATE` (`agents/synthesizer.py:137`) and
> `SYNTHESIZER_EMIT_ALTERNATIVE` (`:150`) are **already `True`** — the
> genome-aware debate/alternative work described here builds on enabled
> features, not disabled ones (see §11).

## Synthesizer performance and information intake (the bottleneck)

The synthesizer is the swarm's serial join, and on the laptop path it is the
dominant wall-clock cost. The memo must address this directly, because the
genome makes the fix natural.

**Current cost profile (verified):** `agents/synthesizer.py` issues ~14
distinct `await self.llm.generate(...)` calls and contains **no
`asyncio.gather`** — every call is sequential. Per synthesis the serial chain
includes: `_interpret_prompt` (1), `_plan_synthesis` (1), up to
`_SECTION_1_MAX_RENDER_FULL = 6` per-cluster `_render_cluster_position` calls
in a serial `for` loop (`:796`), one `_render_debate_frame` per debate group,
`_compose_with_edges` (1), per-contested-cluster dissent renders (Section 2),
`_render_executive_summary` (1), best-of-N cohesive integration
(`_BEST_OF_N` candidates, run sequentially across temperatures, `:628–652`),
and the revision loop (critic + revise = 2, `:1866`, `:1898`). That is
~15–20 sequential small-model calls per run. The render set is correctly
bounded (`_SECTION_1_MAX_RENDER_FULL = 6`, MMR diversity selection), so the
problem is **serial latency and redundant LLM work**, not unbounded fan-out.

The memo must specify three genome-enabled optimizations, in priority order:

**1. Parallelize the independent per-cluster renders.** The no-leak rule makes
per-cluster genome renders mutually independent by construction — no render
may read another cluster's content. Therefore the Section 1 render loop, the
debate frames, and the Section 2 dissent renders can be wrapped in
`asyncio.gather` (bounded by a semaphore sized to the model's
`max_num_seqs` / VRAM, ≤4 concurrent decode streams on the 3060). This alone
collapses the longest serial segment. Specify the semaphore and the gather
sites. This is a pure latency win with no quality change and **does not
violate no-leak** precisely because the genome already partitions the inputs.

**2. Replace the raw-materials dump with a structured genome bundle (intake
optimization).** `_gather_raw_materials` (`:1465`) + `_format_materials_block`
(`:1504`) currently pack each cluster's `rep_content` plus four truncated
support excerpts into a flat text block that one big integration call must
re-read and re-derive structure from. This is token-heavy on a 6 GB model and
keeps the proxy loop intact (the integration call re-judges prose). Replace it
with `_gather_genomes`, which emits a compact **structured** bundle per
cluster: `atoms` (id, text, weight, verification_score), `atom_graph`,
`knowledge_base` summary (domains, grounding), `composite_fitness` +
`fitness_breakdown`, and the typed `relations`. The integration call then
**reduces over structure**, not prose. Fewer tokens, lower VRAM pressure, and
the structural intermediation is what breaks the proxy loop at integration
time. The bundle must still obey no-leak (atom text + IDs + scalars only).

**3. Make per-cluster rendering template-first, LLM-second.** With atoms in
hand, a cluster whose atoms are all high-verification can be rendered by a
deterministic template ("the cluster claims A1 …; A2 (load-bearing, verified
against {domain}) …") with **no LLM call**, reserving `llm.generate` for
clusters that genuinely need prose smoothing or for the single cohesive
integration. Specify the template and the threshold at which a cluster
escalates to an LLM render. On the laptop this can remove most of the six
Section-1 calls.

The user-facing shape the memo should target is an explicit **map-reduce**:
_map_ = per-cluster, parallel, mostly template-driven from the genome's atoms;
_reduce_ = one structured integration call over the genome bundle that
cohesively assembles the surviving clusters. This is the architectural
expression of "many monkeys each finishing their own sentence, the synthesizer
assembling the paragraph." Also specify: reduce `_BEST_OF_N` on the laptop
path (or make best-of-N scoring partly deterministic via genome atom-coverage
in `_score_cohesive_candidate`, `:1752`, so fewer full candidates are needed),
and replace the live Wikipedia fetch in `_get_external_context` (`:2993`,
called `:2329`) with a read of the genome's atom verification scores
(precondition §11.2) — this also removes a synchronous network round-trip from
the serial chain.

## Knowledge base persistence with genomes

`core/knowledge_base.py` is **already versioned** (`_SCHEMA_VERSION`, currently
2; loader warns on newer versions at `:115`) and a migration tool already
exists (`kb_migrate.py` at repo root, which performs the v1→v2 migration,
computes `_cluster_hash`, and writes `schema_version.json`). The genome work is
therefore a **v2→v3** extension, not a new system:

- Store the full genome (or a serializable projection of it) alongside the
  existing cluster entry.
- **Cross-run contradiction detection becomes atom-level.** Current detection,
  `_detect_contradictions` (`core/knowledge_base.py:373`), flags cross-status
  pairs by cosine similarity between representative embeddings (`_cosine_sim`,
  `:72`). With atom-level matching, the system can detect "in run R1 atom A2
  was supported; in run R2 the same atom was rejected" — sharper than embedding
  similarity.
- Specify the v3 schema additions and extend `kb_migrate.py` to upgrade v2
  entries to v3 by re-extracting atoms (one-time). Do not write a new migration
  script; extend the existing one and bump `_SCHEMA_VERSION` to 3.

## Closing the pre-existing gaps

The prior assessment flagged gaps. **Re-verify each against the current tree
before listing it as a blocker** — three of the five originally listed are
already closed:

1. **SAFE atoms plumbed into VERIFICATION metadata — ALREADY DONE.** Confirm
   `core/worker_pool.py:759` still writes `metadata["atoms"]` and
   `_build_atoms` (`core/projection.py:419`) still reads it. This was
   improvement 5.7 from the synthesizer memo and is in place. List it as a
   **standing invariant to protect**, not a task. The genome architecture
   depends on it remaining true.
2. **`_get_external_context` should be replaced with validator-atom
   aggregation — OPEN.** `_get_external_context` (`agents/synthesizer.py:2993`,
   called at `:2329`) still does a live Wikipedia lookup at synthesis time
   (`_wiki_lookup`). Replace it with a read of each VERIFICATION's
   `metadata["atoms"]` (i.e. read the genome's atom verification scores). This
   is a genuine precondition.
3. **Abstention thresholds recalibration — OPEN (proposal).** With genomes, a
   natural abstention condition is: refuse if no surviving cluster has
   `composite_fitness > τ` AND `grounding > 0` AND `len(load_bearing_atoms) ≥
1`. Specify τ and where the gate lives.
4. **Composite fitness must actually drive selection — OPEN.** Survival
   classification (`_apply_survival_filter`, `core/projection.py:1367`) and
   ranking (`_cluster_priority`, `agents/synthesizer.py:227`, formula
   `support_diversity × verification_score / dissent_pressure`; planner
   `build_plan`, `core/projection.py:823`; `_cp_priority`,
   `core/projection.py:313`) currently select on a three-term LLM-adjacent
   heuristic, **not** on `composite_fitness`. If the genome computes a capped,
   non-symbolic composite but selection still runs on the old heuristic, the
   fitness compositor is decorative. The memo must specify exactly where
   `composite_fitness` replaces or augments these scorers, with a fallback for
   early iterations before genomes are populated.
5. **Workers operate on a single signal, not the cluster — OPEN.** Today
   DEVELOP samples one underserved INITIAL (`_sample_underserved_initial`, in
   the DEVELOP branch of `core/worker_pool.py:_gather_target`) and renders that
   one signal. For "work on clusters and cohesively assemble," the genome-aware
   actions in §8 must show the worker the _target cluster's genome_ (atoms +
   weakest atom) and target that locus — rendering only atom content + IDs
   (no-leak). This converts per-signal mutation into per-genome cumulative
   selection.

> The original memo also listed "enable debate by default" and "enable
> alternative-of-the-best by default" as preconditions. Both are **already
> enabled** (`_SYNTHESIZER_USE_DEBATE = True`, `:137`;
> `SYNTHESIZER_EMIT_ALTERNATIVE = True`, `:150`). They are therefore **not**
> preconditions. The genome work _extends_ them (debate becomes an atom-level
> comparison of two genomes; the alternative becomes the second-best anchor
> corner whose genome is _atomically_ distinct), but it does not need to flip
> any flag. The memo must state this correctly and not instruct the
> implementer to "enable" already-enabled features.

## Per-task genome generation strategies

The atom extraction strategy varies by task type. Specify:

- **debate**: atoms = claims; weight by argumentativeness; look for
  premise/conclusion structure in `atom_graph`.
- **analysis**: atoms = observations, mechanisms, implications; atom graph
  shows causal direction.
- **problem_solving**: atoms = problem-statement / intervention /
  expected-outcome triples; atom graph is a means-end chain.
- **creative**: atoms are looser — image, voice, theme, structure; weight by
  essentiality to the artifact's identity.
- **coding**: atoms = spec items (signature, complexity bound, edge case);
  atom graph is a precondition chain; AST validity is a Tier-3 hard fitness
  signal at the atom level.

## Benchmark plan

End with a comparison harness specific enough to execute.

**Benchmark.** Omni-Math (300 samples) primary. SciCode Pass@Sub secondary if
compute permits.

**Baselines.** SwarmSys-8 (published numbers); GPTSwarm (published numbers);
the project's own `core/baseline.py` `BaselineCoordinator` at matched
`N_AGENTS`; IO (single LLM call); Self-Refine (K=2 revision).

**Comparison protocol**:

- Same base LM across all conditions (Qwen2.5-7B-Instruct or
  Qwen2.5-Coder-7B-Instruct on the laptop path).
- Fixed iteration budget per condition.
- Fixed random seeds (rng_seed=42, 100, 200, 300, 500).
- Judge model: a stronger LM (GPT-4o or Claude Opus if available) scoring
  outputs blind against ground-truth answers.

**Falsifiability matrix**:

| Claim                                           | Comparison                                           | Metric                                      | Expected effect              |
| ----------------------------------------------- | ---------------------------------------------------- | ------------------------------------------- | ---------------------------- |
| Genomes beat materials dumps                    | flat-synthesizer vs. genome-synthesizer              | Omni-Math accuracy                          | genome ≥ flat + 5% at p<0.05 |
| Search-as-mutation contributes                  | with/without grounding fitness term                  | Omni-Math accuracy on retrieval-rich subset | grounding-on > grounding-off |
| Non-symbolic fitness cap improves outcomes      | cap_llm=0.35 vs. cap_llm=1.0                         | Omni-Math accuracy + judge-rated factuality | capped > uncapped            |
| Topology coverage informative                   | report coverage; correlate with judge score          | rank correlation                            | ρ > 0.3                      |
| Atom-level contradiction sharper than embedding | atom-level vs. embedding-level KB conflict detection | precision@k on a labeled set                | atom > embedding             |
| Beats SwarmSys-8                                | this architecture vs. SwarmSys-8 reported numbers    | Omni-Math accuracy                          | beat on ≥1 metric at p<0.05  |

## Local-compute constraints

NVIDIA RTX 3060 Laptop, 6 GB VRAM. Small models (Qwen2.5-3B-Instruct or
Qwen2.5-Coder-3B-Instruct via GGUF). The memo must address:

- Atom extraction batch size: ≤ 4 simultaneous decompose calls.
- Genome computation cost: O(seconds) at projection time. Use deterministic
  centroid-based atom alignment over LLM-based wherever possible.
- Wikidata entity resolution: optional, gated by `USE_WIKIDATA_VERIFICATION`.
  Default off on laptop; on for Colab/A100.
- Composite fitness is O(n_clusters · n_atoms_per_cluster); trivial at n ≤ 40
  clusters, ≤ 6 atoms each.

## Memo structure (use this skeleton)

```
# Cluster Genome as the Unit of Selection
## 1. Problem framing
   Three pressures: SwarmSys benchmark, Dawkins challenge, proxy loop.
   The move: consolidate already-typed pieces into a HERITABLE per-cluster organism.
## 2. Comparison to SwarmSys, GPTSwarm, and prior overhauls
   Heritable structured genomes, non-symbolic fitness, atom-targeted operations.
## 3. Diagnosis: typed pieces that are not yet a heritable organism
   What already exists (AtomProjection, ClusterSensitivity, InterClusterEdge,
   TrajectoryFeatures) vs. what is missing (heritability, ClusterKnowledgeBase,
   composite fitness). Explicit nesting-vs-projection-level argument.
## 4. The genome data structure
   Full schema. Field semantics. Correspondence to existing projection types.
## 5. Genome operations
   Atom extraction (Strategy A vs. B, recommend B; note partial existing impl).
   Atom graph construction. Inheritance on fission. Recombination on merge.
   Genome hashing (reuse _cluster_hash).
## 6. The fitness compositor
   Composite formula. Hard cap on LLM-judged term + tier-honesty caveat.
   Per-task weight tables. Per-term computation procedures.
## 7. Search as the mutation operator
   ClusterKnowledgeBase. Grounding fitness term.
## 8. Genome-aware worker actions
   DEVELOP/CRITIQUE/OBJECT/REFINE/VALIDATE with correct current builders + atom targeting.
## 9. Genome-aware synthesizer
   _gather_genomes. Per-cluster atom-level provenance. Structured genome bundle.
   Atom-faithfulness audit. (Debate/alternative already enabled.)
## 9b. Synthesizer performance & intake (the bottleneck)
   Serial chain (~15–20 sequential generate calls, no asyncio.gather).
   Parallelize independent per-cluster genome renders (no-leak makes them
   independent). Replace raw-materials dump with structured genome bundle.
   Template-first / LLM-second per-cluster render. Map-reduce shape.
## 10. KB persistence with genomes
    v2→v3 schema extension. Atom-level contradiction detection. Extend kb_migrate.py.
## 11. Closing the pre-existing gaps
    Re-verified: #1 atoms-in-metadata already DONE (standing invariant);
    #2 _get_external_context replacement OPEN; #3 abstention recalibration OPEN.
    Debate/alternative already enabled — NOT preconditions.
## 12. Per-task genome generation
## 13. Benchmark plan
## 14. Local-compute constraints
## 15. Sequencing (six stages in dependency order)
## 16. Risks and open questions
    Atom-extraction quality. LLM still in the loop for extraction. Embedding-
    space leakage of "non-symbolic" terms. O(n²) genome comparisons. Cost of
    rewriting projection-level consumers to per-cluster nesting.
## 17. Future work
    Strategy A eager extraction. Learned atom extractor. Genome-based active
    sampling. Strict Tier 3 fitness (NLI entity resolution, formal verifier).
```

## Scope discipline for Claude Code (CLI) — keep only what changes the code

This memo is consumed by Claude Code from the repo root. CLI Claude produces
the best result when the prompt is anchored to concrete files/lines and a tight
deliverable, and it is _diluted_ by academic apparatus that does not change a
line of code. Treat the following as **optional / trim-first** if the goal is a
buildable design rather than a publishable paper:

- The 10,000–15,000 word floor and the "doctoral prose" mandate. Length is not
  the goal; an implementer-ready spec of the data structure + the six
  integration points + the no-leak and performance constraints is. If the spec
  is complete at 6,000 words, stop.
- The requirement to link every section to one of the "five beyond-params
  mechanisms" and to blockquote SwarmSys / GPTSwarm / Dawkins / Self-Refine.
  Keep one short framing paragraph; drop the per-section taxonomy bookkeeping.
- The full benchmark essay (§13) and SwarmSys-8 falsifiability matrix — keep
  these **only if** benchmarking is a near-term goal. Otherwise reduce to a
  one-paragraph "how we'll know it worked" and defer the harness.

**Keep (these are what enable the goal):** the genome dataclasses with
field→existing-type correspondence; the no-leak conformance clause; the
populated-from mapping table with verified line numbers; the five genome-aware
action templates with the no-leak fallback; the synthesizer parallelization +
structured-bundle + map-reduce spec; the precondition list (§11) with the
already-done items correctly marked; the sequencing (§15); and the verification
checklist (§0–10), which CLI Claude should actually execute against the live
tree. A focused, file-anchored, self-verifying spec is strictly better here
than a long one.

## Style and constraints

- Doctoral-level prose. Each architectural decision must answer "what
  measurable improvement does this deliver on Omni-Math against SwarmSys-8?"
- When you cite a function, cite the file path and the line range **you
  actually read in the current tree**. Do not copy line numbers from any prior
  memo (including this corrected one) without re-opening the file — line
  numbers drift as code changes.
- Do not implement code. The deliverable is the markdown document only.
- Do not pre-commit to file changes outside `Attempt At Cleaning/docs/`.
- Write in CommonMark. Code fences for data structures/pseudocode; blockquotes
  for paper claims (SwarmSys, GPTSwarm, Dawkins, Self-Refine, Constitutional AI).
- Length target: 10,000–15,000 words. Density over surface area.
- Every architectural section must link to: (a) one of the five beyond-params
  mechanisms (self-consistency, verifier-augmented, search, debate,
  decomposition+scaffolding) — the Dawkins picture is closest to mechanism (5);
  (b) the non-symbolic-communication-proxy framing: which tier (1, 2, or 3) the
  proposed signal lives in. Be precise that `centroid_stability` and
  `novelty_density` are Tier 2 (model-derived), not Tier 3.
- The benchmark plan must be executable without further design work.
- Engage explicitly with the Dawkins weasel-program challenge: cumulative
  selection requires a _heritable_ representation; this memo specifies the
  heritability (fission inheritance + merge recombination) on top of the
  already-existing typed representation.

## Verification step

After writing the memo, run a self-check pass. **Crucially, item 0 is new:
verify current state before asserting any "currently X" claim.**

0. For every sentence of the form "currently the code does X" or "X is not yet
   implemented," open the cited file and confirm it at the current line. In
   particular re-confirm: `metadata["atoms"]` write at `worker_pool.py:759`;
   `_build_atoms` at `projection.py:419`; `ClusterSensitivity` /
   `_build_sensitivities` at `projection.py:177`/`570`; `_SYNTHESIZER_USE_DEBATE`
   and `SYNTHESIZER_EMIT_ALTERNATIVE` values at `synthesizer.py:137`/`150`;
   `_get_external_context` at `synthesizer.py:2993`; `_detect_contradictions`
   at `knowledge_base.py:373`; `_SCHEMA_VERSION` and `kb_migrate.py` existence.
1. Does every cited file path exist? Open each and confirm.
2. Does every cited line range match the content you describe?
3. Does the genome schema include all fields plus composite fitness and
   breakdown? (atoms, atom_graph, topology_expression, phenotype,
   knowledge_base, sensitivity, trajectory, relations, composite_fitness,
   fitness_breakdown.)
4. Does the fitness compositor specify all seven terms and the hard cap, AND
   the tier-honesty caveat (which terms are Tier 2 vs Tier 3)?
5. Does §11 list only the genuinely-open gaps as preconditions (#2
   `_get_external_context`, #3 abstention), correctly mark #1 as a standing
   invariant already satisfied, and correctly note debate/alternative are
   already enabled and therefore not preconditions?
6. Does §13 give a harness implementable without further design — judge model,
   seed protocol, statistical tests, specific Omni-Math subset?
7. Could a reader trace one example end-to-end? Walk the debate example: a
   scout deposits an INITIAL in (qualified necessary, economic, mitigation); a
   developer produces SUPPORTs; the SAFE pipeline extracts 3 atoms (already
   written to `metadata["atoms"]` and projected by `_build_atoms`); one atom is
   verified against ipcc.ch; the genome is assembled by nesting the existing
   AtomProjection/ClusterSensitivity/InterClusterEdge plus the new
   ClusterKnowledgeBase; the FitnessCompositor gives composite_fitness=0.62
   (0.30 semantic capped at 0.35 → contributes 0.30, + 0.18 grounding + 0.14
   topology); the synthesizer renders "atom A2 (verified against ipcc.ch) is
   load-bearing." Did the memo enable this trace? If not, fix the gap.
8. Does the memo address atom-extraction failure (LLM produces no parseable
   atoms)? Specify the fallback to a single-atom genome containing the
   representative content verbatim — and note this matches the existing SAFE
   failure path at `worker_pool.py:1118–1121`.
9. Does the memo specify behavior when SEARCH lineage is empty (no grounding)?
   How `grounding_score` degrades gracefully and how the per-task weight table
   compensates.
10. Does the produced memo's own verification step include item 0 (re-verify
    "currently X" claims) for whoever implements from it?

Report which preconditions you verified against current code, which example
trace you walked end-to-end, and which verification items you confirmed
present. Do not modify the code.
