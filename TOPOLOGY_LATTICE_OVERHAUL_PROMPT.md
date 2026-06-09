# Claude Code prompt — Topology + Lattice + Sensitivity overhaul

Paste everything below the `---` line into Claude Code from the repository root
(`C:\Users\agsse\Downloads\ai_swarm_mechanics-main (4)\ai_swarm_mechanics-main`).

This is the **third** design memo in a sequence. Companion documents:

- `Attempt At Cleaning/docs/SYNTHESIZER_OVERHAUL.md` (composition mechanisms)
- `Attempt At Cleaning/docs/AGENT_RETRIEVAL_OVERHAUL.md` (SAFE/HyDE/Step-Back/FLARE)

This memo is not a replacement. It is the structural axis those two memos did
not cover. The synthesizer needs to read three orthogonal structures, not one:

```
topology   × resolution × sensitivity
(bounds)     (depth)       (robustness)
```

The prior two memos addressed retrieval-side and composition-side mechanisms.
This memo addresses the **data structure the synthesizer parses**. The thesis
of this memo: the synthesizer's failure mode is doing simultaneous *bounding*
and *traversing* inside one set of LLM calls. The fix is to construct the
bounds before the swarm runs, let the swarm populate the bounded space, then
let the synthesizer traverse a pre-bounded topology along a structured path.

---

You are working in the `Attempt At Cleaning/` folder of a stigmergic
multi-agent swarm codebase. Read the following files end-to-end before
proposing anything:

- `agents/synthesizer.py` (full)
- `core/projection.py` (full)
- `core/worker_pool.py` (focus on `Worker.iterate`, `_gather_target`, and
  the SAFE pipeline at lines 1051–1116)
- `core/signal_store.py` (Signal dataclass + metadata schema)
- `core/actions.py` (action specs, especially `validate_parse`)
- `run_swarm.py` (`run_pipeline` and `run_continuous_pipeline`)
- The two prior memos in `Attempt At Cleaning/docs/`

Do NOT modify code on this pass. The deliverable is a single markdown
document at:

    Attempt At Cleaning/docs/TOPOLOGY_LATTICE_OVERHAUL.md

## Task framing

The driving framing is **benchmark performance against SwarmSys**
(arxiv:2510.10047). The project will be evaluated on at least one shared
benchmark from SwarmSys's Table 1 (Omni-Math is the recommended target — 300
samples, well-defined ground truth, reasonable compute). The architectural
question is what to add that gives a measurable delta over SwarmSys-8 at
matched compute, *not* a generic claim that swarms work.

SwarmSys does not have, and cannot easily add, a pre-exploration topology
with coverage tracking. SwarmSys's strongest mechanism (embedding-based
ε-greedy task allocation) is post-hoc adaptive; it does not bound the answer
space in advance. The structural delta this memo proposes is **bounds-first
exploration with topology-aware synthesis**, which yields three falsifiable
benchmark claims:

1. **Coverage at fixed compute**: a topology-driven swarm fills more
   answer-space cells per iteration than free exploration.
2. **Calibrated absence**: the synthesizer can state which regions of the
   answer space were considered and found empty — information SwarmSys
   cannot produce because it has no topology.
3. **Position-of-answer in the space**: the synthesizer can locate its
   answer within the topology (corner / interior / edge) — a meta-
   structural specificity novel against every system in SwarmSys's
   comparison table.

The Mount Everest framing: build the puzzle's edges first (topology bounds),
then the corners (anchor positions), then fill the interior. The synthesizer
becomes a traversal engine over a bounded space, not a summarizer of a flat
list.

## The three structural axes

### Axis 1 — Topology (bounds-first exploration)

Before scouts run, one LLM call generates a task-conditional
`AnswerSpaceTopology`. The topology declares: the dimensions valid answers
vary along; the anchor corners that span the answer space; the boundary
exclusions. Per-task templates differ in *content*, not *procedure* — exactly
the architectural invariance the user flagged.

Data structure:

```python
@dataclass(frozen=True)
class AxisSpec:
    name: str
    values: tuple[str, ...]   # discrete categorical values

@dataclass(frozen=True)
class AnchorCorner:
    coords: tuple[str, ...]   # one value per axis, in axis order
    label: str
    rationale: str            # one-sentence justification of why this is an anchor

@dataclass
class AnswerSpaceTopology:
    task_type: str
    axes: list[AxisSpec]              # primary axis at index 0
    anchor_corners: list[AnchorCorner]
    boundary_exclusions: list[str]    # claim shapes explicitly out of scope
    generation_prompt: str            # the LLM call that produced this
    audit_log: list[dict]             # post-exploration topology-extension events
```

Example for `debate("Climate action is necessary")`:

```json
{
  "axes": [
    {"name": "necessity", "values": ["fully necessary", "qualified necessary", "qualified not necessary", "not necessary"]},
    {"name": "framing", "values": ["empirical", "ethical", "economic", "historical"]},
    {"name": "scope", "values": ["mitigation", "adaptation", "innovation"]}
  ],
  "anchor_corners": [
    {"coords": ["fully necessary", "ethical", "mitigation"], "label": "moral imperative",
     "rationale": "frames necessity as a duty owed to future generations"},
    {"coords": ["not necessary", "economic", "adaptation"], "label": "adaptive market",
     "rationale": "frames adaptation as more cost-effective than mitigation"},
    {"coords": ["fully necessary", "empirical", "mitigation"], "label": "scientific mandate",
     "rationale": "treats necessity as an empirical consequence of warming projections"},
    {"coords": ["not necessary", "historical", "innovation"], "label": "techno-optimist",
     "rationale": "treats prior technological transitions as evidence against mandatory action"}
  ],
  "boundary_exclusions": [
    "denial of climate change as physical phenomenon (out of scope: not a debate position)",
    "specific policy implementations (out of scope: too granular for thesis-level debate)"
  ]
}
```

Example for `coding("Implement a thread-safe LRU cache")`:

```json
{
  "axes": [
    {"name": "correctness_completeness", "values": ["minimal viable", "production hardened"]},
    {"name": "concurrency_model", "values": ["lock-based", "lock-free", "actor"]},
    {"name": "eviction_semantics", "values": ["strict-LRU", "approximate-LRU"]},
    {"name": "memory_model", "values": ["bounded-by-count", "bounded-by-bytes"]}
  ],
  "anchor_corners": [
    {"coords": ["minimal viable", "lock-based", "strict-LRU", "bounded-by-count"], "label": "textbook",
     "rationale": "simplest correct implementation; baseline for comparison"},
    {"coords": ["production hardened", "lock-free", "approximate-LRU", "bounded-by-bytes"], "label": "high-throughput cache",
     "rationale": "production-grade Redis-style implementation"}
  ],
  "boundary_exclusions": [
    "distributed cache (out of scope: single-process LRU)",
    "persistence (out of scope: in-memory only)"
  ]
}
```

The memo must specify:

1. **Topology generation procedure**. One LLM call ahead of scouts. Prompt
   template, max tokens, temperature, parse and validate routine, fallback
   on parse failure. Per-task-type variations the template handles. Where
   in the pipeline this fits (after task_prompt is built, before scouts
   are instantiated).

2. **Scout binding to topology cells**. Each scout is assigned a cell
   (coords tuple) from a coverage-balanced assignment. The scout's prompt
   adds: "your assigned region of the answer space is (coords); generate
   an INITIAL that lives in this region." Anchor-corner cells are
   prioritized in early rounds; interior cells are filled in later rounds.
   The corpus partition (existing mechanism) still provides evidence; the
   topology cell tells the scout where in the answer space to deploy it.

3. **Signal metadata extension**. Every INITIAL deposited under a topology
   cell carries `metadata["topology_coords"] = (...)`. Downstream
   SUPPORT/CRITIQUE/OBJECTION/VERIFICATION signals inherit the topology
   coords of the cluster they target (computed at projection time;
   metadata at deposit may be sparse). Document the canonical metadata
   schema.

4. **Topology-extension on out-of-bounds claims**. A scout that cannot
   produce a claim in its assigned cell, OR a developer that produces a
   SUPPORT whose content does not fit the topology, can trigger a
   topology-extension proposal. The proposal goes to a topology-audit
   LLM call (one per round at most) that either (a) accepts the
   extension and adds a new axis value or anchor corner, or (b) rejects
   it and the claim is filed under `out_of_bounds_clusters` for the
   synthesizer to surface. Audit decisions are logged.

5. **Coverage tracking**. The projection tracks
   `topology_coverage: dict[cell_coords, list[cluster_id]]` and
   `uncovered_cells: list[cell_coords]`. Coverage is a first-class
   metric in summary.json — not derived post-hoc.

### Axis 2 — Multi-resolution lattice

The synthesizer currently reads at one resolution: clusters of INITIAL
signals. A multi-resolution lattice gives it four levels with explicit
cross-level edges:

```
frames (worldviews / thematic groupings)
  ╱│
clusters (your current ClusterProjections)
  │
propositions (developer-derived sub-claims, ~25–50 chars each)
  │
atoms (SAFE atomic facts with per-atom verification)
```

The lattice composes naturally with SAFE — atoms are the bottom layer of
the lattice, not a separate side-structure. This memo proposes the full
lattice as a first-class data structure with cross-level edges typed as
`{supports, refines, contradicts, contextualizes}` and propagation weights.

Data structure (extends, does not replace, existing `ClusterProjection`):

```python
@dataclass
class AtomProjection:
    atom_id: str
    text: str
    weight: float                    # centrality from SAFE decomposition
    verification_score: float        # per-atom score from worker_pool SAFE pipeline
    source_tag: str                  # which external source corroborated
    parent_cluster_id: str
    parent_proposition_id: Optional[str]

@dataclass
class PropositionProjection:
    proposition_id: str
    text: str                        # ~25–50 chars
    atom_ids: list[str]
    parent_cluster_id: str
    verification_score: float        # weighted mean over child atoms

@dataclass
class FrameProjection:
    frame_id: str
    label: str                       # one-phrase thematic name
    cluster_ids: list[str]
    coverage_topology_cells: list[tuple]   # which topology cells this frame spans

@dataclass(frozen=True)
class CrossLevelEdge:
    src_id: str
    dst_id: str
    src_level: str   # one of: "frame", "cluster", "proposition", "atom"
    dst_level: str
    relation: str    # one of: "supports", "refines", "contradicts", "contextualizes"
    weight: float
```

The memo must specify:

1. **Atom projection from existing SAFE data**. The SAFE pipeline in
   `core/worker_pool.py:1051–1116` already produces `atom_results` with
   text, weight, query, score, snippet_tag. The data is currently
   stashed on the worker and discarded. The memo must specify how
   atoms get into `Signal.metadata` so the projection can surface
   them. This is improvement 5.7 from the prior synthesizer memo, and
   it is a load-bearing precondition for this overhaul.

2. **Proposition derivation**. Propositions are sub-claims grouping 1–3
   atoms that share a logical clause. They are derived by an LLM call
   over a cluster's atom set, similar in shape to `_safe_decompose` but
   running over the union of atom texts under a cluster. Provide a
   recommendation for whether to derive propositions per cluster as
   part of projection or as an on-demand step the synthesizer triggers.

3. **Frame derivation**. Frames cluster `ClusterProjection`s at a coarser
   threshold than the existing cluster clustering (e.g.,
   `CLUSTER_SIM_THRESHOLD` is 0.55–0.72; frame threshold could be
   0.35–0.45). Frames may also be derived from topology coverage —
   clusters sharing a topology cell or adjacent cells naturally form a
   frame. Recommend whether to derive frames purely structurally or
   with an LLM labeling pass.

4. **Cross-level edge construction**. Atoms supporting their parent
   cluster's claim get `supports` edges. Atoms contradicting their
   parent cluster (low verification_score) get `contradicts` edges.
   Propositions that refine a cluster's claim narrow scope get
   `refines` edges. Frames that contextualize a cluster get
   `contextualizes` edges. Each edge carries a weight in [0, 1].

5. **Resolution-choice policy in the synthesizer**. The synthesizer's
   first decision is which resolution to render at. Specify a policy:
   haiku → frame-level (one frame, no breakdown); debate → cluster-level
   with proposition support; factual analysis → atom-level with
   per-atom citation; coding → atom-level for spec items + cluster-level
   for approaches. The policy is per-task-type and lives in a
   constant map analogous to `_OUTPUT_STRATEGY_BY_TASK`.

### Axis 3 — Counterfactual sensitivity

For each surviving cluster, compute a sensitivity vector at projection
time: what would the field look like under perturbation? The base LM
cannot infer counterfactual structure from final-state data because
counterfactuals require re-running the survival filter.

Data structure:

```python
@dataclass
class ClusterSensitivity:
    cluster_id: str
    support_removal_robustness: float     # min strength of single support that, if removed, drops status
    dissent_amplification_tolerance: float # how much dissent could grow before status flips
    load_bearing_supports: list[str]      # supports whose removal would change status
    marginal_supports: list[str]          # supports whose removal would not
    competing_takeover: Optional[str]     # cluster that would rise if this one fell
    topology_uncovered_on_removal: list[tuple]  # cells that would lose coverage
```

The memo must specify:

1. **Computation procedure**. Iterate each support in `cluster.support_set`;
   simulate its removal (subtract its contribution from
   `weighted_support`); re-apply the survival filter; record whether
   status changed. O(n_clusters · mean_supports_per_cluster). Manageable
   at typical n.

2. **Synthesizer consumption**. Sensitivity becomes a rendering
   annotation. Prose can say "this claim survives but rests on a single
   piece of evidence ([SUPPORT_00042]); remove that and the claim falls
   below threshold." Specify which renderer paths consume sensitivity
   and how. At minimum: the `_render_cluster_position` path and the
   `_render_alternative_artifact` path.

3. **Coverage-loss propagation**. When a cluster's load-bearing supports
   are removed, the topology cells that cluster occupied may become
   uncovered. Sensitivity vector records which cells. This is novel
   feedback: the renderer can describe not just "if you take cluster X
   away, position Y rises" but "if you take cluster X away, the
   adaptive-market corner of the topology becomes empty."

## The integration: how the synthesizer reads all three

The synthesizer's `SynthesisProjection` becomes:

```python
@dataclass
class SynthesisProjection:
    # Existing fields preserved
    surviving: list[ClusterProjection]
    contested: list[ClusterProjection]
    weakly_supported: list[ClusterProjection]
    rejected_by_field: list[ClusterProjection]
    unverified: list[ClusterProjection]
    partition_coverage: dict
    no_consensus: bool
    inter_cluster_edges: list[InterClusterEdge]   # from prior overhaul

    # NEW: topology axis
    topology: AnswerSpaceTopology
    topology_coverage: dict[tuple, list[str]]
    uncovered_cells: list[tuple]
    out_of_bounds_clusters: list[str]

    # NEW: lattice axis
    frames: list[FrameProjection]
    propositions: list[PropositionProjection]
    atoms: list[AtomProjection]
    cross_level_edges: list[CrossLevelEdge]

    # NEW: sensitivity axis
    cluster_sensitivities: dict[str, ClusterSensitivity]
```

The renderer's traversal policy walks all three axes. The memo must specify
the traversal as a state machine or pseudocode. Reference traversal:

```
1. Edges-first prose: describe the topology bounds.
2. Anchor-corner rendering: for each anchor_corner with coverage,
   render the strongest cluster at that corner.
3. Interior rendering: for each non-corner topology cell with coverage,
   render its dominant cluster with topology coordinates explicit.
4. Sensitivity annotation: inline each cluster's load-bearing or
   marginal support note.
5. Coverage gaps: list uncovered topology cells with one-sentence
   rationale (no LLM call needed — deterministic).
6. Out-of-bounds: list out_of_bounds_clusters as "the swarm proposed
   claims outside the established answer space; these were filed but
   not integrated."
7. Resolution choice: per `_OUTPUT_STRATEGY_BY_TASK`, decide whether
   the renderings above are at frame / cluster / proposition / atom
   resolution.
```

## Pre-existing gaps that must close as part of this overhaul

The prior assessment of the codebase flagged four gaps that subvert the
project's goal. This overhaul cannot deliver benchmark gains until they
close. The memo must list these as preconditions and include them in the
sequencing:

1. **SAFE atoms must be plumbed into VERIFICATION signal metadata.** Currently
   `Worker._validate_atoms` (`core/worker_pool.py:1114`) dies with the
   worker. The deposit's `meta` dict must carry `meta["atoms"] = atom_results`
   so the projection layer can build `AtomProjection`s. Without this,
   the lattice axis is impossible.

2. **`_get_external_context` must be replaced with validator-atom aggregation.**
   Currently `agents/synthesizer.py:2254` re-fetches Wikipedia at
   synthesis time. The replacement reads each VERIFICATION's
   `metadata["atoms"]` and renders the atom-level evidence. This is
   prior synthesizer overhaul Improvement 5.7.

3. **Debate frame must be enabled by default for debate/analysis tasks.**
   Currently `_SYNTHESIZER_USE_DEBATE = False`
   (`agents/synthesizer.py:137`). The debate mechanism is mechanism (4)
   in the beyond-params framework; defaulting it off ships the system
   without the mechanism that distinguishes it from RAG.

4. **Alternative-of-the-best must be enabled by default for exploration tasks.**
   Currently `SYNTHESIZER_EMIT_ALTERNATIVE = False`
   (`agents/synthesizer.py:150`). On the topology axis, the alternative
   artifact maps onto rendering the strongest cluster at a non-primary
   anchor corner — the multi-modal lattice exposure the synthesizer
   overhaul prescribed.

5. **Abstention thresholds must be recalibrated.** The current triple-AND
   (max_ver < 0.15 AND max_div < 2 AND max_dis > 1.2) is nearly
   impossible to satisfy. With the topology axis, a more natural
   abstention is: refuse if `len(uncovered_cells) / len(all_cells) >
   0.5` AND no anchor corner has coverage. Topology-coverage abstention
   is computable, calibrated, and surfaces meaningful absence.

## Per-task-type topology templates

The memo must specify a template per task type registered in
`run_swarm.py:TASK_PROMPTS`. Provide for each: number of axes, default
axis names, anchor-corner count, boundary-exclusion examples. At a
minimum, cover:

- `debate`: 3 axes (position, framing, scope); 4 anchor corners minimum
- `analysis`: 3 axes (descriptive/predictive, micro/macro,
  established/speculative); 4 anchor corners
- `problem_solving`: 3 axes (intervention type, time horizon,
  cost/benefit profile); 3–4 anchor corners
- `creative`: 2 axes (form, voice); 2–3 anchor corners (creative tasks
  have lower-dimensional answer spaces)
- `coding`: variable, derived from prompt's spec keywords; topology
  generation procedure must extract spec items as axes

## Falsifiability matrix

The memo must end with a measurement plan that lets the team report
results comparable to SwarmSys's Table 1. At minimum:

| Claim | Comparison | Metric | Expected effect |
|---|---|---|---|
| Topology improves coverage | free exploration vs. topology-bound | covered_cells / total_cells at fixed iter budget | topology ≥ free + 15% |
| Calibrated absence is informative | abstention gate based on dissent_pressure vs. topology-coverage | judge-rated specificity of refusal | topology coverage path higher specificity |
| Position-of-answer adds value | flat synthesizer vs. topology-aware synthesizer | judge-rated structural specificity of the rendered answer | topology-aware > flat at p < 0.05 |
| Lattice resolution choice matches task | one-resolution vs. task-conditional resolution | Omni-Math accuracy at A=4 | task-conditional ≥ one-resolution |
| Counterfactual sensitivity catches fragile answers | annotated vs. unannotated | judge-rated robustness of presented answers | annotated > unannotated |
| The integration outperforms SwarmSys-8 | SwarmSys-8 vs. topology+lattice+sensitivity at matched compute | Omni-Math accuracy, SciCode Pass@Sub | beat SwarmSys-8 on at least one metric at p < 0.05 |

Specify the judge model (recommend a model strictly stronger than the
swarm's base LM, e.g., Claude Opus or GPT-5 if available). Specify the
random-seed protocol so cross-method differences aren't noise.

## Memo structure (use this skeleton)

```
# Topology + Lattice + Sensitivity: Three-Axis Data Structure for the Synthesizer

## 1. Problem framing
   Why the synthesizer's current data structure cannot deliver beyond-params
   output against SwarmSys. The Mount Everest framing: bounds first, then
   corners, then interior.

## 2. Comparison to SwarmSys
   What SwarmSys does (Explorer/Worker/Validator, embedding-based matching,
   pheromone-implicit reinforcement). What it does not do (no topology,
   no resolution choice, no counterfactual annotation). Where this overhaul
   targets the gap.

## 3. The three axes
   3.1 Topology — answer-space bounds
   3.2 Multi-resolution lattice — abstraction depth
   3.3 Counterfactual sensitivity — robustness annotation

## 4. Data structure
   Full extended SynthesisProjection schema. Each dataclass with field
   semantics, computation procedure, and consumer in the synthesizer.

## 5. The traversal renderer
   Pseudocode of the synthesizer's walk across topology × resolution ×
   sensitivity. State machine. Per-task-type policy.

## 6. Topology generation procedure
   The LLM call. Prompt template. Parse and validate. Fallback. Per-task
   variations.

## 7. Topology-extension audit
   How out-of-bounds claims are surfaced or absorbed. The audit LLM call.
   When it fires.

## 8. Pre-existing gaps closed by this overhaul
   The four flag-off / plumbing items from the prior assessment. Why each
   is load-bearing for this overhaul.

## 9. Per-task-type templates
   debate, analysis, problem_solving, creative, coding. Axis specs and
   anchor-corner counts for each.

## 10. Falsifiability and the SwarmSys comparison
    The benchmark plan. Omni-Math at minimum; SciCode if feasible. Judge
    model, random-seed protocol, A/B regime table.

## 11. Sequencing
    Implementation order. Atom plumbing first (precondition for lattice).
    Topology generation second (precondition for coverage tracking).
    Sensitivity third (small, depends on stable projection). Default-flag
    flips fourth (cheap, immediate effect). Integration in the renderer
    fifth.

## 12. Risks and open questions
    Topology quality bottlenecks runs. Topology generation cost.
    Resolution choice on tasks the templates don't anticipate. Sensitivity
    O(n²) explosion if support_sets get large. What to do when topology
    extension is contentious.
```

## Style and constraints

- Doctoral-level prose. Skeptical, precise, benchmark-anchored.
- Each architectural proposal must answer: "what does this let the
  synthesizer parse that no single forward pass — or any system in
  SwarmSys's Table 1 — could parse?"
- When you cite a file or function, cite the file path and the line
  range you read. Do not invent file paths or line numbers.
- Do not implement code. The deliverable is the markdown document only.
- Do not pre-commit to file changes outside `Attempt At Cleaning/docs/`.
- Write in CommonMark. Use code fences for data structures and
  pseudocode; blockquotes for paper claims (SwarmSys, GPTSwarm,
  Self-Refine, etc.).
- Length target: 7,000–12,000 words. Density over surface area.
- Every section that proposes a structure must explicitly link to the
  beyond-params framework: which of the five mechanisms (self-consistency,
  verifier-augmented, search, debate, decomposition+scaffolding) it
  invokes, and why this implementation is not collapsible into a single
  conditioning.
- The benchmark plan in section 10 must be specific enough that a
  competent engineer can execute it without further design work.

## Verification step

After writing the memo, run a self-check:

1. Does every cited file path exist? Open each and confirm. Do not skip
   this — the prior memos cited correct paths but the discipline must
   continue.
2. Does every cited line range match the content you describe?
3. Does each of the three axes have all four required parts (data
   structure, computation, synthesizer consumption, beyond-params
   argument)?
4. Does section 8 explicitly list all five pre-existing gaps with their
   prior-overhaul reference numbers (5.7, 5.5, 5.9, 5.8 from the
   synthesizer memo; and the SAFE atom plumbing precondition from the
   retrieval memo)?
5. Does section 10 give a comparison harness specific enough that a
   reader can implement it without further design work? Including the
   judge model, seed protocol, and per-claim statistical test.
6. Could a reader trace a single example end-to-end through the memo?
   Pick the debate example. The topology was generated; scout-12 was
   bound to (qualified necessary, economic, mitigation); it deposited
   INITIAL_00023; supports were added; sensitivity was computed; the
   renderer traversed and surfaced this cluster at the
   "qualified-necessary economic-mitigation" interior position with
   the sensitivity note "rests on SUPPORT_00041; without it the
   cluster slides to weakly_supported." Did the memo enable this
   trace? If not, fix the gap.

Report which preconditions you verified and which example trace you
walked end-to-end. Do not modify the code.
