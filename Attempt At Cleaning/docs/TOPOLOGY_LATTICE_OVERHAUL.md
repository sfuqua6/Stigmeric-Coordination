# Topology + Lattice + Sensitivity: Three-Axis Data Structure for the Synthesizer

*Design memo — the third in a sequence. Read `SYNTHESIZER_OVERHAUL.md` and
`AGENT_RETRIEVAL_OVERHAUL.md` before this document. This memo addresses the
data structure the synthesizer parses; those two addressed composition
mechanisms and retrieval strategies respectively. Do not modify code before
reading all three.*

*Driving benchmark target: SwarmSys (arxiv:2510.10047) Table 1. Recommended
evaluation dataset: Omni-Math (300 samples, well-defined ground truth,
competitive but accessible at the hardware target).*

---

## 1. Problem Framing

### Why the current data structure is the bottleneck

The two prior memos diagnosed failures in retrieval quality (shallow
fixed-altitude queries, composite-claim validation) and in synthesis
composition (flat materials blocks, absent inter-cluster structure, disabled
debate frame). Both diagnoses were correct. But they share a common
upstream cause that neither addressed: **the synthesizer has no answer space
to navigate**. It receives a ranked list of clusters and a projection lattice,
but no structure that tells it what the *shape* of the answer domain is, how
deeply it has covered that domain, or how robust its coverage is to
perturbation.

The consequence is that the synthesizer can answer, but it cannot say *where*
in the space of possible answers it landed, *what* it left uncovered, or
*what would have to be false* for its answer to be wrong. Those three
capacities — locating the answer, characterizing absence, and assessing
fragility — are precisely what distinguish a careful academic analysis from a
well-written summary. They are also precisely what SwarmSys cannot produce,
because SwarmSys has no topology.

The Mount Everest framing captures the fix: build the puzzle's edges first
(topology bounds), then the corners (anchor positions), then fill the interior.
Currently the swarm fills the interior first and hopes edges emerge from
field dynamics. They do not — field dynamics selects for strength, not
boundary coverage. The result is that surviving clusters are often clustered
near the most-cited corner of the answer space (the most-documented position
on a debate topic, the most-documented algorithm for a coding task) while
other corners are empty. The synthesizer dutifully renders what survived and
calls it comprehensive.

### What this memo proposes

Three orthogonal structural additions to the projection the synthesizer reads:

```
topology   × resolution × sensitivity
(bounds)     (depth)       (robustness)
```

**Topology** declares the answer space before exploration begins. Scouts are
assigned cells in that space and their deposits carry topology coordinates.
The synthesizer can report coverage rather than just content.

**Multi-resolution lattice** gives the synthesizer four levels of abstraction
to choose from: frames (worldviews), clusters (current), propositions
(sub-claims), and atoms (SAFE atomic facts). The resolution choice is
task-conditional and explicit; the lattice exposes the same information at
whichever depth the task requires.

**Counterfactual sensitivity** computes, for each surviving cluster, a
robustness vector: which support signals are load-bearing, how much additional
dissent would flip its status, which topology cells it would vacate if
removed. The synthesizer annotates its answer with these vectors.

None of these is a new agent role. None requires additional LLM calls during
exploration. All three are pre-computations that extend the projection layer
and give the synthesizer a richer scaffold to traverse.

---

## 2. Comparison to SwarmSys

### What SwarmSys does

SwarmSys (arxiv:2510.10047) implements a three-role architecture: Explorer
(generates candidate solutions), Worker (refines candidates), and Validator
(scores candidates). Role assignment is dynamic: an embedding-based ε-greedy
allocation matches each agent to the task subdomain where it has highest
historical accuracy. Coordination is implicit through a shared solution pool
with strength-weighted sampling — effectively a pheromone field without
explicit pheromone decay. The strongest mechanism in SwarmSys-8 is this
embedding-based adaptive allocation: workers do not explore randomly but are
routed toward task regions where they are competent.

SwarmSys's Table 1 results (Omni-Math accuracy, SciCode Pass@Sub) demonstrate
that this routing mechanism produces meaningful gains over baseline
independent-agent ensembles. The Omni-Math result is the primary target
because it is a 300-sample held-out benchmark with verified ground truth,
making A/B comparisons unambiguous.

### What SwarmSys does not do

SwarmSys has no pre-exploration topology. Its ε-greedy allocation is
*post-hoc adaptive*: agents are routed based on what they have already
produced, not based on what the answer space structurally requires. An answer
space with four corners and one heavily-documented corner will route all
agents toward that corner as soon as a few good solutions appear there. The
other three corners remain empty. SwarmSys's allocation mechanism is a
self-reinforcing attractor, not a coverage guarantee.

SwarmSys also has no resolution choice. Its validators produce scalar quality
scores; its synthesizer (implicit: the final aggregation step) selects the
highest-scoring solution. There is no decomposition into sub-claims, no
atom-level evidence trace, no frame-level thematic grouping. The output is
"the best solution the swarm found," with no structure beyond that.

SwarmSys has no counterfactual annotation. Its strength-weighted pool records
current scores but not the sensitivity of those scores to perturbation. A
high-scoring solution that rests on a single validator corroboration looks
identical in the pool to a high-scoring solution corroborated by five
independent validators.

### Where this overhaul targets the gap

The three axes of this memo correspond precisely to the three structural gaps:

- **Topology** provides bounds-first coverage that SwarmSys's ε-greedy
  cannot: scouts are assigned cells before exploration, so corner coverage
  is guaranteed by construction rather than emergent from field pressure.
- **Multi-resolution lattice** provides the depth that SwarmSys's
  single-resolution aggregation cannot: the synthesizer can render at atom
  level (per-fact citation), proposition level (sub-claim grouping), or
  frame level (thematic summary), conditioned on task type.
- **Counterfactual sensitivity** provides the robustness annotation that
  SwarmSys's pool cannot: each answer comes with a falsifiability profile.

The falsifiable benchmark claim is that topology-driven swarms fill more
answer-space cells per iteration than free exploration, and that topology-aware
synthesis produces answers with measurably higher structural specificity than
SwarmSys-8's output on the same tasks.

---

## 3. The Three Axes

### 3.1 Topology — Answer-Space Bounds

**Mechanism invoked:** Decomposition with externalized scaffolding (mechanism 5
from the beyond-params framework in `SYNTHESIZER_OVERHAUL.md §1`). The topology
is the externalized scaffold. Scouts deposit into cells; the projection tracks
coverage of those cells. No single forward pass can produce coverage data,
because coverage data requires having observed what N agents deposited across
N disjoint corpus partitions. The topology is the structure that makes that
observation interpretable.

**Why this is not collapsible into a single conditioning.** A single LLM call
given a task prompt can enumerate possible answer dimensions, but it cannot
tell you which cells in that space are supported by external evidence and
which are empty after a full exploration pass. Coverage is a property of the
joint run — of the set of all agent deposits and their distribution across the
declared space. No single forward pass can compute it, because computing it
requires having observed the deposits.

**Data structure:**

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass(frozen=True)
class AxisSpec:
    name: str
    values: tuple[str, ...]   # discrete categorical values, 2–5 per axis

@dataclass(frozen=True)
class AnchorCorner:
    coords: tuple[str, ...]   # one value per axis, in axis-declaration order
    label: str                # short human-readable name
    rationale: str            # one-sentence justification of why this is an anchor

@dataclass
class AnswerSpaceTopology:
    task_type: str
    axes: list[AxisSpec]              # primary axis at index 0
    anchor_corners: list[AnchorCorner]
    boundary_exclusions: list[str]    # claim shapes explicitly out of scope
    generation_prompt: str            # the LLM call that produced this
    audit_log: list[dict] = field(default_factory=list)
    # post-exploration extension events:
    # each entry: {"round": int, "accepted": bool, "proposal": str, "action": str}
```

**Serialization note.** `AnswerSpaceTopology` is pure Python with no LLM
dependencies; it can be pickled and attached to `run_meta.json` alongside the
existing outputs for post-hoc inspection.

**Coverage tracking.** The projection layer (`core/projection.py`) gains two
new fields in `SynthesisProjection` (see Section 4):

```python
topology_coverage: dict[tuple, list[str]]   # cell coords → [cluster_ids]
uncovered_cells: list[tuple]                # cells with no cluster coverage
```

Coverage is populated during `build_projection` by reading
`signal.metadata.get("topology_coords")` for each INITIAL signal and grouping
clusters by the coords of their representative INITIAL. This is a pure Python
operation; no LLM call is required.

### 3.2 Multi-Resolution Lattice — Abstraction Depth

**Mechanism invoked:** Decomposition with externalized scaffolding (mechanism 5),
plus verifier-augmented decoding (mechanism 2) at the atom level. The lattice
gives the synthesizer a choice of four abstraction levels. Choosing the wrong
level for the task is a composition failure; choosing correctly is what
mechanism 5 enables when the scaffold is rich enough. The atom level
specifically invokes mechanism 2: atoms are the output of SAFE decomposition
(already implemented in `core/worker_pool.py:1051–1116`), each verified
independently against retrieved evidence. Using atom-level verification as
first-class synthesis inputs rather than deriving a scalar is the verifier
loop that mechanism 2 requires.

**Why this is not collapsible into a single conditioning.** The lattice's
cross-level edges encode relationships that span abstraction boundaries: an
atom can *contradict* its parent cluster's claim if its verification score is
low enough to invalidate the cluster's position. A single forward pass on the
task prompt cannot produce these cross-level contradiction edges because they
require having decomposed claims into atoms and having verified each atom
independently against external evidence before the contradiction can be
detected.

**Data structure (extending, not replacing, existing projection types):**

```python
@dataclass
class AtomProjection:
    atom_id: str               # "{verification_signal_id}_atom_{i}"
    text: str                  # the atomic proposition
    weight: float              # centrality from SAFE decomposition call
    verification_score: float  # per-atom score from SAFE pipeline
    source_tag: str            # snippet source URL/title or "(no result)"
    query: str                 # the DDG query issued for this atom
    parent_cluster_id: str     # representative_id of parent cluster
    parent_proposition_id: Optional[str] = None  # set after proposition grouping

@dataclass
class PropositionProjection:
    proposition_id: str
    text: str                  # 25–50 chars; one logical clause
    atom_ids: list[str]
    parent_cluster_id: str
    verification_score: float  # weighted mean over child atoms

@dataclass
class FrameProjection:
    frame_id: str
    label: str                 # one-phrase thematic name (e.g., "economic constraints")
    cluster_ids: list[str]     # representative_ids of constituent clusters
    coverage_topology_cells: list[tuple]  # topology cells this frame spans

@dataclass(frozen=True)
class CrossLevelEdge:
    src_id: str
    dst_id: str
    src_level: str   # one of: "frame", "cluster", "proposition", "atom"
    dst_level: str
    relation: str    # one of: "supports", "refines", "contradicts", "contextualizes"
    weight: float    # in [0, 1]
```

**Resolution-choice policy.** The synthesizer's first decision after reading
the projection is which lattice level to render at. This policy is per-task-type
and lives in a constant map analogous to `_OUTPUT_STRATEGY_BY_TASK`
(`agents/synthesizer.py:201–207`):

```python
_LATTICE_RESOLUTION_BY_TASK = {
    "creative":        "frame",          # one frame, no breakdown
    "problem_solving": "cluster",        # cluster-level with proposition support
    "debate":          "cluster",        # cluster-level with proposition support
    "analysis":        "atom",           # atom-level with per-atom citation
    "coding":          "atom+cluster",   # atom-level for spec items, cluster for approaches
}
```

The renderer dispatches to a resolution-specific rendering path. The
`frame`-level path is shortest (one thematic paragraph); the `atom`-level path
is longest but most precise (per-atom citation blocks). `atom+cluster` is a
hybrid for coding tasks: atoms populate the specification section, clusters
populate the approaches section.

### 3.3 Counterfactual Sensitivity — Robustness Annotation

**Mechanism invoked:** Decomposition with externalized scaffolding (mechanism 5).
The sensitivity vector is a property of the joint signal field — it requires
having observed which specific support signals were deposited, what their
individual strengths are, and what the field equilibrium looks like without
each one. No single forward pass can compute this, because computing it
requires having run the full swarm and having access to the specific signal
IDs and strengths that field dynamics produced.

**Why this is not collapsible into a single conditioning.** A forward pass on
a task prompt can produce a hedged answer that says "this claim is tentative."
Sensitivity annotation says something categorically different: "this specific
claim survives because SUPPORT_00041 has strength 0.72; remove that signal
and the cluster's `weighted_support` drops below the `contested` threshold."
That specificity requires having observed the actual signals in the actual
field, not having generated a plausible-sounding hedge.

**Data structure:**

```python
@dataclass
class ClusterSensitivity:
    cluster_id: str            # representative_id
    support_removal_robustness: float  # min strength of single support whose
                               # removal flips status; 0 if no such support exists
    dissent_amplification_tolerance: float  # how much additional dissent weight
                               # is needed to flip status to "contested"
    load_bearing_supports: list[str]   # support IDs whose removal flips status
    marginal_supports: list[str]       # support IDs whose removal does not
    competing_takeover: Optional[str]  # cluster_id that would advance to primary
                               # position if this cluster were removed
    topology_uncovered_on_removal: list[tuple]  # cells that lose coverage
```

**Computation procedure** (pure Python, O(n_clusters × mean_supports_per_cluster)):

For each cluster `c` in `surviving + contested`:

1. Compute `baseline_status = c.status` and `baseline_wp = weighted_support`.
2. For each `sid` in `c.support_set`:
   - Look up `sig = store.get(sid)`; get `sig.strength`.
   - Simulate removal: `wp_without = weighted_support − sig.strength`
   - Re-apply survival filter with `wp_without`. If status would change,
     record `sid` as load-bearing and update `support_removal_robustness`
     as `min(robustness, sig.strength)`.
   - Otherwise record `sid` as marginal.
3. `dissent_amplification_tolerance`: binary-search the additional dissent
   weight δ such that `dissent_pressure_with_delta` crosses the contested
   threshold. This is analytically solvable given the log1p formula in
   `core/projection.py`.
4. `competing_takeover`: among surviving/contested clusters not equal to `c`,
   find the one whose priority score (`_cp_priority`) would exceed `c`'s
   priority if `c` were removed from the field.
5. `topology_uncovered_on_removal`: for each topology cell in
   `topology_coverage`, check if `c.representative_id` is the *sole* cluster
   covering that cell. If so, cell goes to `topology_uncovered_on_removal`.

This computation runs entirely from data already in the `SignalStore` and
`SynthesisProjection`. It adds no LLM calls. It should run in
`build_projection` after the survival filter, before the projection is
returned to the synthesizer.

---

## 4. Data Structure — Extended `SynthesisProjection`

The canonical extended schema. Fields marked `[existing]` are already present
in `core/projection.py:133–150`. Fields marked `[new]` are added by this
overhaul. Fields marked `[from prior overhaul]` were added by the synthesizer
overhaul memo and confirmed present in the current codebase.

```python
@dataclass
class SynthesisProjection:
    # --- [existing] Survival-filter buckets ---
    surviving: list[ClusterProjection]
    contested: list[ClusterProjection]
    weakly_supported: list[ClusterProjection]
    rejected_by_field: list[ClusterProjection]
    unverified: list[ClusterProjection]
    partition_coverage: dict           # partition_tag → count
    no_consensus: bool

    # --- [from prior overhaul] Inter-cluster edge graph ---
    inter_cluster_edges: list[InterClusterEdge]  # present at projection.py:149

    # --- [new] Topology axis ---
    topology: Optional[AnswerSpaceTopology]      # None if generation failed
    topology_coverage: dict[tuple, list[str]]    # cell coords → [cluster rep_ids]
    uncovered_cells: list[tuple]                 # cells with zero coverage
    out_of_bounds_clusters: list[str]            # rep_ids outside declared topology

    # --- [new] Multi-resolution lattice axis ---
    frames: list[FrameProjection]
    propositions: list[PropositionProjection]
    atoms: list[AtomProjection]
    cross_level_edges: list[CrossLevelEdge]

    # --- [new] Sensitivity axis ---
    cluster_sensitivities: dict[str, ClusterSensitivity]
    # keyed by representative_id
```

**Field semantics and consumers:**

- `topology`: generated by a single LLM call in `run_pipeline` before scouts
  are instantiated. Passed through to the synthesizer as part of the
  projection. If generation or parse fails, the field is `None` and the
  synthesizer falls back to coverage-unaware rendering.
- `topology_coverage` and `uncovered_cells`: computed in `build_projection`
  from INITIAL signal metadata. Written to `summary.json` as first-class
  metrics.
- `out_of_bounds_clusters`: clusters whose representative INITIAL has
  `metadata.get("topology_coords") == "out_of_bounds"`. Surfaced in Section 6
  of the renderer output (see traversal pseudocode).
- `frames`: derived structurally from clusters at a coarser similarity
  threshold (see Section 3.2). No LLM call required for frame grouping; an
  optional LLM labeling pass gives each frame its human-readable `label`.
- `propositions`: derived by an LLM call per cluster over the cluster's atom
  set. On-demand, triggered by the synthesizer at atom-resolution tasks.
- `atoms`: constructed from VERIFICATION signal metadata. Each VERIFICATION
  signal that has `metadata["atoms"]` (deposited by the SAFE path in
  `core/worker_pool.py:748–766`) produces one `AtomProjection` per atom entry.
- `cross_level_edges`: built after atoms, propositions, and frames are
  constructed. Pure Python; no LLM calls.
- `cluster_sensitivities`: computed by the sensitivity procedure in Section 3.3.

---

## 5. The Traversal Renderer

The synthesizer's traversal policy is a state machine that walks all three
axes in a fixed sequence. The sequence is below as pseudocode; it replaces the
current `_render` method's top-level dispatch but preserves all existing
per-cluster rendering calls.

```
TRAVERSAL POLICY (pseudocode):

Input: SynthesisProjection P, store, contract, resolution = _LATTICE_RESOLUTION_BY_TASK[task_type]

Step 1 — Topology preamble (deterministic, no LLM):
  if P.topology is not None:
    emit: "The answer space for this task was bounded along N axes: [axis_names]."
    emit: "Anchor corners explored: [anchor corner labels]."
    emit: "Boundary exclusions: [boundary_exclusions]."
    if P.uncovered_cells:
      emit: "The following topology regions produced no surviving claims: [uncovered_cells]."

Step 2 — Anchor-corner rendering (one LLM call per covered anchor corner):
  for corner in P.topology.anchor_corners (covered = corner.coords in P.topology_coverage):
    cluster_ids = P.topology_coverage[corner.coords]
    pick strongest cluster by _cluster_priority from cluster_ids
    render at `resolution` level (call _render_cluster_at_resolution)
    annotate with corner.label and corner.rationale

Step 3 — Interior rendering (one LLM call per covered non-corner cell):
  interior_cells = {c for c in topology_coverage if c not an anchor corner}
  for cell in interior_cells (sorted by cluster priority descending):
    dominant_cluster = highest-priority cluster in topology_coverage[cell]
    render at `resolution` level
    annotate with explicit topology coordinates: "interior position in [axis=value, axis=value, ...]"

Step 4 — Sensitivity annotation (deterministic, inline):
  for each rendered cluster c:
    s = P.cluster_sensitivities.get(c.representative_id)
    if s and s.load_bearing_supports:
      append: "Note: this claim rests on [len(load_bearing)] load-bearing support(s)
               [IDs]; removing any one would [flip/weaken] its status."
    if s and s.dissent_amplification_tolerance < SENSITIVITY_FLAG_THRESHOLD:
      append: "Fragile consensus: a [tolerance]× increase in field dissent
               would move this cluster to contested."

Step 5 — Coverage gaps (deterministic, no LLM):
  if P.uncovered_cells:
    emit: "The following answer-space regions were considered but left empty
           after exploration: [cell descriptions]."
    for cell in P.uncovered_cells:
      emit: "Region [cell] ({axis=value, ...}): no surviving claim reached this position."

Step 6 — Out-of-bounds (deterministic, no LLM):
  if P.out_of_bounds_clusters:
    emit: "The exploration produced [N] claims outside the declared answer space.
           These were filed but not integrated: [cluster summaries]."

Step 7 — Resolution dispatch:
  The rendering calls in Steps 2–3 already dispatched to resolution-specific
  paths. At `atom` resolution, each call includes per-atom citation blocks.
  At `cluster` resolution, each call uses the existing _render_cluster_position
  path from agents/synthesizer.py.
  At `frame` resolution, the frame label and its constituent clusters are
  rendered as a single thematic paragraph with no per-cluster breakdown.
```

**Per-task rendering policy (concrete mapping):**

| Task | Resolution | Section 1 rendering | Section 2 rendering |
|---|---|---|---|
| `creative` | `frame` | One paragraph per frame; no cluster breakdown | Inter-cluster dissent aggregated at frame level |
| `debate` | `cluster` | One paragraph per surviving cluster with topology coordinates | Contested clusters with debate frame (if `_SYNTHESIZER_USE_DEBATE` enabled) |
| `analysis` | `cluster` | As debate, with proposition support noted | As debate |
| `problem_solving` | `cluster` | As debate | As debate |
| `coding` | `atom+cluster` | Spec section at atom level; approaches section at cluster level | Contested approaches with sensitivity notes |

---

## 6. Topology Generation Procedure

### Placement in the pipeline

The topology generation call fires in `run_swarm.py:run_pipeline` (lines ~300–600;
the exact insertion point is after `build_task_prompt` is called and before
the first `Scout` objects are instantiated). The function signature:

```python
async def generate_topology(
    task_prompt: str,
    task_type: str,
    llm,
    template: dict,   # per-task-type template from _TOPOLOGY_TEMPLATES
) -> Optional[AnswerSpaceTopology]:
```

It returns `None` on parse failure; callers treat `None` as "no topology
available" and proceed without coverage tracking.

### Prompt template

The prompt is structured in three parts:

```
PART 1 — TASK TYPE CONTEXT:
"You are defining the answer space for a {task_type} task. Your job is to
declare the dimensions valid answers vary along, the anchor positions that
span the space, and what is explicitly out of scope."

PART 2 — TASK-SPECIFIC AXIS TEMPLATE:
(one of _TOPOLOGY_TEMPLATES[task_type]; see Section 9 for per-type templates)

PART 3 — OUTPUT SCHEMA:
"Reply with exactly this JSON (no other text):
{
  'axes': [{'name': str, 'values': [str, ...]}, ...],
  'anchor_corners': [{'coords': [str, ...], 'label': str, 'rationale': str}, ...],
  'boundary_exclusions': [str, ...]
}"
```

**Model parameters:** Temperature 0.1 (determinism is essential — the topology
declares a fixed structure the rest of the run will fill); max_tokens 800
(sufficient for 3 axes × 4 values × 4 corners × exclusions).

### Parse and validate

```python
def _parse_topology(raw: str, task_type: str) -> Optional[AnswerSpaceTopology]:
    import json, re
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        axes = [AxisSpec(name=a["name"], values=tuple(a["values"]))
                for a in d.get("axes", [])]
        if not (1 <= len(axes) <= 5):
            return None
        corners = [
            AnchorCorner(coords=tuple(c["coords"]), label=c["label"],
                         rationale=c["rationale"])
            for c in d.get("anchor_corners", [])
        ]
        if not (2 <= len(corners) <= 8):
            return None
        # Validate that each corner's coords length matches the axis count.
        if any(len(c.coords) != len(axes) for c in corners):
            return None
        exclusions = [str(e) for e in d.get("boundary_exclusions", [])]
        return AnswerSpaceTopology(
            task_type=task_type,
            axes=axes,
            anchor_corners=corners,
            boundary_exclusions=exclusions,
            generation_prompt="",   # filled by caller
        )
    except (ValueError, KeyError, TypeError):
        return None
```

### Fallback on parse failure

If `_parse_topology` returns `None` after two retries (one re-prompt with an
explicit schema reminder), the pipeline continues with `topology=None`. Scouts
are instantiated without cell assignments. Coverage tracking is skipped.
`topology_coverage` and `uncovered_cells` are empty dicts/lists in the
projection. The synthesizer renders without topology preamble. This is the
same output quality as the current system — zero regression.

### Scout binding to topology cells

After topology generation, scouts are assigned cells from the topology using
a coverage-balanced assignment:

```python
def _assign_topology_cells(
    topology: AnswerSpaceTopology,
    n_scouts: int,
) -> list[Optional[tuple]]:
    """Returns a cell assignment (coords tuple) for each scout index.

    Anchor corners are prioritized: the first len(anchor_corners) scouts
    get one anchor corner each. Remaining scouts are assigned interior cells
    in round-robin order. If n_scouts > total cells, multiple scouts share
    cells (expected: multiple scouts strengthen one region rather than
    spreading too thin).
    """
    cells = [c.coords for c in topology.anchor_corners]
    # Add interior cells (all cells not in anchor_corners).
    import itertools
    all_coords = list(itertools.product(*[a.values for a in topology.axes]))
    interior = [c for c in all_coords if c not in set(cells)]
    cells.extend(interior)
    assignments = []
    for i in range(n_scouts):
        assignments.append(cells[i % len(cells)] if cells else None)
    return assignments
```

The assigned cell tuple is passed to `ScoutConfig` as a new field
`topology_cell: Optional[tuple]`. The scout's prompt adds one sentence after
the task framing: "Your assigned region of the answer space is
[axis=value, axis=value, ...]. Generate a claim that lives in this region."
The scout retains its existing corpus partition; the topology cell adds
*directional intent*, not a different corpus.

### Signal metadata extension

When a scout deposits an INITIAL signal, the deposit metadata gains:

```python
metadata["topology_coords"] = scout_config.topology_cell  # tuple or None
```

Downstream SUPPORT/CRITIQUE/OBJECTION/VERIFICATION signals inherit topology
coordinates at projection time: `projection.py:build_projection` reads the
representative INITIAL's `metadata["topology_coords"]` and propagates it as
the cluster's topology position. Downstream signal metadata may be sparse
(not all signals will have explicit coords); the cluster-level coord is
canonical.

---

## 7. Topology-Extension Audit

### When extension fires

A topology-extension proposal is triggered when either:

1. A scout cannot produce a claim in its assigned cell (its generated content
   scores below `MIN_DEPOSIT_STRENGTH` on three consecutive attempts at the
   same cell), or
2. A developer generates a SUPPORT whose content semantically maps to no
   existing topology cell (cosine distance to all cell centroid embeddings
   exceeds a threshold `_TOPOLOGY_OOB_COSINE = 0.4`).

In case 1, the scout deposits normally with `metadata["topology_coords"] =
"out_of_bounds"`. In case 2, the developer's SUPPORT is deposited with the
same out-of-bounds marker.

The extension mechanism fires at most once per round, triggered by
accumulating out-of-bounds deposits. If at round end the count of
out-of-bounds deposits exceeds `_TOPOLOGY_EXTENSION_TRIGGER = 2`, one
topology-audit LLM call runs.

### Audit procedure

The audit call receives:

1. The current topology (serialized as JSON).
2. The out-of-bounds claim texts (up to 3 representative examples).
3. A prompt: "These claims were produced during exploration but do not fit
   the declared answer space. Decide: (a) should an axis value be extended
   to accommodate these claims, or (b) should these claims be filed as
   out-of-scope? If extending, specify which axis and which new value. If
   filing, explain why the claims are out of scope."

**Audit model parameters:** Temperature 0.15; max_tokens 400.

**Audit output schema:**

```json
{"decision": "extend" | "file_oob",
 "axis_name": "<str or null>",
 "new_value": "<str or null>",
 "rationale": "<one sentence>"}
```

If `decision == "extend"`: the named axis gains the new value; the scout or
developer that generated the out-of-bounds claims is re-assigned to this new
cell in the next round. The extension event is appended to
`AnswerSpaceTopology.audit_log`.

If `decision == "file_oob"`: the out-of-bounds deposits are tagged with
`metadata["topology_coords"] = "out_of_bounds"` permanently. The synthesizer
surfaces them in Step 6 of the traversal.

**Cost.** One audit call per round at most. Audit calls are suppressed if
the topology is `None`, if there are fewer than `_TOPOLOGY_EXTENSION_TRIGGER`
out-of-bounds deposits, or if the round is the last round (no time to act on
the extension).

---

## 8. Pre-Existing Gaps Closed by This Overhaul

The prior memos flagged several gaps that prevent benchmark gains. Each is
load-bearing for at least one axis of this overhaul. This section lists them
with their prior-memo reference numbers and their current status in the
codebase.

### Gap 1 (Retrieval memo precondition): SAFE atoms plumbed into VERIFICATION signal metadata

**Status: CLOSED in current code.**

`core/worker_pool.py:748–766` already deposits SAFE atoms into VERIFICATION
signal metadata:

```python
parsed = ParsedDeposit(
    ...
    metadata={
        **(parsed.metadata or {}),
        "atoms": _atoms,
        "aggregation": "weighted_mean",
        "atom_count": len(_atoms),
        "score": round(_agg, 4),
    },
)
```

The `atoms` list is present in VERIFICATION signal metadata after a successful
SAFE pass. **However**, the synthesizer currently reads only `vsig.content`
(the one-sentence overall assessment, line ~2241 of `agents/synthesizer.py`)
and not `vsig.metadata["atoms"]`. This is the remaining half of this gap —
atoms are stored but not consumed by the lattice layer.

**Action required for this overhaul:** In `build_projection` (or in a new
`_build_atoms` function called from `build_projection`), iterate each cluster's
`verification_set`, call `store.get(vid)` for each VERIFICATION signal, read
`vsig.metadata.get("atoms", [])`, and construct `AtomProjection` objects. This
is the load-bearing precondition for the lattice axis's atom level.

### Gap 2 (Synthesizer overhaul §5.7): `_get_external_context` replaced with validator-atom aggregation

**Status: PARTIALLY CLOSED.**

`agents/synthesizer.py:2240–2258` already reads VERIFICATION signal content
via `validator_notes` and falls back to `_get_external_context` only when
no validator notes exist. The validator note path is preferred. However, the
aggregation reads only `vsig.content` (the one-sentence summary), not
`vsig.metadata["atoms"]`. When the lattice axis is implemented (Gap 1 above
completed), the aggregator should be extended to render per-atom evidence
for atom-resolution tasks: instead of "validator score=0.68: claim is well
supported by grid cost data," emit "atom 1: [ChatGPT Plus is a paid tier]
— score 0.92, source: openai.com; atom 2: ..."

**Action required for this overhaul:** Extend the `validator_notes` block
in `_render_cluster_position` to optionally render per-atom evidence when
`resolution == "atom"` and `vsig.metadata.get("atoms")` is non-empty.

### Gap 3 (Synthesizer overhaul §5.5): Debate frame enabled by default for debate/analysis tasks

**Status: NOT CLOSED.**

`agents/synthesizer.py:137`: `_SYNTHESIZER_USE_DEBATE = False`. The debate
mechanism is mechanism (4) in the beyond-params framework and is load-bearing
for the topology axis's anchor-corner rendering: when two anchor corners have
comparable-priority clusters (within `_DEBATE_PRIORITY_RATIO` of each other),
the topology-aware renderer should trigger the debate frame automatically.
Defaulting debate off suppresses the mechanism that makes anchor-corner
contrast informative.

**Action required for this overhaul:** Change `_SYNTHESIZER_USE_DEBATE = False`
to `True` in `agents/synthesizer.py:137` once the typed inter-cluster edge
graph (already implemented; `core/projection.py:265–350`) is confirmed to be
producing `alternatives` edges. The debate frame fires only when `alternatives`
edges exist between anchor corners — so enabling it with a sparse edge graph
is safe (no alternatives edges → no debate triggered).

### Gap 4 (Synthesizer overhaul §5.9): Alternative-of-the-best enabled by default for exploration tasks

**Status: NOT CLOSED.**

`agents/synthesizer.py:150`: `SYNTHESIZER_EMIT_ALTERNATIVE = False`. On the
topology axis, the alternative artifact maps directly onto rendering the
strongest cluster at a non-primary anchor corner. Defaulting it off means
the topology's multi-corner structure is never surfaced to the user. A topology
with four anchor corners where only one is rendered is not delivering the
topology's structural value.

**Action required for this overhaul:** Change `SYNTHESIZER_EMIT_ALTERNATIVE`
to `True`. Gate the alternative artifact specifically on topology-corner
coverage: if `len(anchor_corners_with_coverage) >= 2`, emit an alternative
artifact from the second-highest-priority covered anchor corner. This is a
more principled trigger than the original §5.9 specification (which used MMR
hold-outs), because it ties the alternative artifact to the topology structure
the run actually built.

### Gap 5 (Synthesizer overhaul §5.8): Abstention thresholds recalibrated to topology coverage

**Status: PARTIALLY IMPLEMENTED, THRESHOLDS MISCALIBRATED FOR TOPOLOGY.**

`agents/synthesizer.py:163–166` implements the triple-AND abstention gate
(`_ABSTAIN_VER_THRESHOLD = 0.15`, `_ABSTAIN_DIVERSITY_THRESHOLD = 2`,
`_ABSTAIN_DISSENT_THRESHOLD = 1.2`). The prior overhaul memo noted these
are nearly impossible to trigger simultaneously. With the topology axis,
a more principled abstention condition is available:

```python
def _topology_abstention(projection: SynthesisProjection) -> bool:
    if projection.topology is None:
        return False   # no topology, use existing gate
    n_total = len(list(itertools.product(
        *[a.values for a in projection.topology.axes]
    )))
    n_covered = len(projection.topology_coverage)
    coverage_fraction = n_covered / max(1, n_total)
    anchor_covered = any(
        c.coords in projection.topology_coverage
        for c in projection.topology.anchor_corners
    )
    # Abstain if more than half the topology is empty AND no anchor corner
    # has coverage.
    return coverage_fraction < 0.5 and not anchor_covered
```

The topology-coverage abstention is computable, calibrated, and surfaces
meaningful absence: "the swarm explored but found nothing in 6 of 12 topology
cells, and no anchor position has coverage." This is more informative than
"max verification_score = 0.12."

**Action required for this overhaul:** Add `_topology_abstention` to
`agents/synthesizer.py` and gate it in `_render` before LLM calls (after the
existing triple-AND gate). When topology abstention fires, include the
uncovered cells and the nearest-covered cell in the structured refusal message.

---

## 9. Per-Task-Type Topology Templates

These templates are the content of `_TOPOLOGY_TEMPLATES` in `run_swarm.py`
(or in a new `core/topology.py` module). Each template is injected as PART 2
of the topology generation prompt (Section 6). The templates declare default
axis names and anchor-corner counts; the actual axis values are filled by the
LLM based on the specific task prompt.

### `debate`

```
Declare 3 axes for a debate task:
  Axis 1: position (where on the spectrum from "fully X" to "not X" does the answer sit?)
  Axis 2: framing (empirical | ethical | economic | historical)
  Axis 3: scope (the domain of intervention the argument concerns)

Declare at least 4 anchor corners that span the debate space. Corners should
represent the most distinct defensible positions. Exclude: positions that
deny the empirical premise of the thesis, specific policy implementations.
```

**Example output for** `debate("Climate action is necessary")` (from prompt Section 3.1):
- Axis 1: necessity [fully necessary, qualified necessary, qualified not necessary, not necessary]
- Axis 2: framing [empirical, ethical, economic, historical]
- Axis 3: scope [mitigation, adaptation, innovation]
- Anchor corners: moral imperative, adaptive market, scientific mandate, techno-optimist
- Exclusions: denial of climate change as physical phenomenon; specific policy implementations

### `analysis`

```
Declare 3 axes for an analysis task:
  Axis 1: epistemic stance (descriptive | predictive | prescriptive)
  Axis 2: scale (micro | meso | macro)
  Axis 3: evidence type (established | contested | speculative)

Declare at least 4 anchor corners. Exclude: normative claims that go beyond
description, predictions beyond 10 years without quantification.
```

**Default anchor-corner count:** 4. A pure descriptive-micro-established
corner (ground truth), a prescriptive-macro-speculative corner (intervention),
and two intermediate positions.

### `problem_solving`

```
Declare 3 axes for a problem-solving task:
  Axis 1: intervention type (technical | behavioral | policy | structural)
  Axis 2: time horizon (immediate | medium-term | long-term)
  Axis 3: cost-benefit profile (high-cost-high-gain | low-cost-high-gain |
           low-cost-low-gain | high-cost-low-gain)

Declare 3–4 anchor corners representing the most distinct feasible solution
archetypes. Exclude: solutions requiring technology not yet demonstrated at
prototype scale, solutions requiring international treaty without existing framework.
```

**Default anchor-corner count:** 3–4. The "low-cost-high-gain" corner is
always an anchor (it's the most-cited quadrant in policy literature). The
"high-cost-high-gain long-term" corner is always an anchor. One or two
additional corners are task-specific.

### `creative`

```
Declare 2 axes for a creative task (creative tasks have lower-dimensional
answer spaces):
  Axis 1: form (the structural/genre choice)
  Axis 2: voice (the speaker/narrator stance)

Declare 2–3 anchor corners representing the most distinct creative approaches.
Exclude: pastiches of named copyrighted works, outputs requiring images or
audio.
```

**Default anchor-corner count:** 2–3. Creative tasks have sparser topology
because the answer space is less discretizable than factual tasks. The frame-
level resolution (Section 3.2) is appropriate for creative tasks precisely
because frames aggregate the low-dimensional creative topology naturally.

### `coding`

```
Declare axes derived from the task's specification keywords:
  Axis 1: correctness_completeness (minimal viable | production hardened)
  Axis 2: [derived from primary algorithmic dimension, e.g. concurrency_model, data_structure]
  Axis 3: [derived from secondary spec constraint, e.g. memory_model, error_handling]
  (additional axes if the spec has >= 3 distinct constraints)

Declare 2–3 anchor corners:
  - The simplest correct implementation (baseline)
  - The production-grade implementation (target)
  - An alternative approach if one is well-known (optional)

Exclude: distributed or multi-process variants unless explicitly requested,
persistence unless explicitly requested.
```

**Topology generation procedure for coding tasks.** The LLM must extract
spec keywords from the task prompt before declaring axes. The PART 2 template
for coding includes: "First, list the key implementation constraints in the
task prompt as bullet points. Then declare axes that each correspond to one
constraint dimension. Do not declare an axis for a constraint that has only
one valid value."

For `coding("Implement a thread-safe LRU cache")`, the spec keywords are:
thread-safety (→ concurrency_model axis), LRU eviction (→ eviction_semantics
axis), capacity bound (→ memory_model axis). The template produces 4 axes
including correctness_completeness.

---

## 10. Falsifiability and the SwarmSys Comparison

### Benchmark harness

**Base LM:** `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` (the default in
`run_swarm.py` via `SWARM_MODEL` env var). Must be identical across all
conditions; do not use `MOCK_LLM=1` for any reported result.

**Benchmark:** Omni-Math (300 samples). Each sample is a mathematical
competition problem with a verified answer. Math is a clean domain for
topology because the answer space is often discretizable (proof strategies,
algebraic vs. geometric approaches, elementary vs. advanced tools).

**Baseline conditions:**
1. `--mode=baseline`: single-agent, no signal store, no partitioning. The
   `core/baseline.py` implementation in the codebase.
2. SwarmSys-8: the 8-agent SwarmSys configuration from arxiv:2510.10047.
   If the SwarmSys code is not available, use the published accuracy numbers
   as the comparison target and note the comparison is against a reported
   number, not a locally-run number.
3. Swarm (current, no topology): `run_swarm.py` with `MOCK_LLM=0` and
   topology generation disabled (`_TOPOLOGY_ENABLED = False`).
4. Swarm + topology only: topology generation enabled; lattice and sensitivity
   disabled.
5. Swarm + topology + lattice: both enabled; sensitivity disabled.
6. Swarm + topology + lattice + sensitivity (full overhaul): all three axes.

**Random-seed protocol:** Set `PYTHONHASHSEED=42` and pass `--seed=42` to
`run_swarm.py` (add a `--seed` CLI argument that sets Python's `random.seed`
and `torch.manual_seed`). Run each condition 3 times with seeds 42, 43, 44.
Report mean ± std. Do not cherry-pick the best seed.

**Judge model for qualitative metrics:** Claude Opus 4 or the strongest
available Claude model (strictly stronger than the swarm's base LM). The
judge sees pairs of outputs (blinded to condition) and rates on: structural
specificity (does the answer locate itself in an answer space?), calibrated
absence (does the answer describe what was considered and found wanting?),
robustness (does the answer identify what would have to be false for it to
be wrong?). Each pair is rated on a 1–5 scale per dimension. Inter-rater
reliability must be reported if multiple judges are used.

**Falsifiability matrix:**

| Claim | Comparison | Metric | Threshold for success | Statistical test |
|---|---|---|---|---|
| Topology improves coverage | Condition 3 vs. 4 | `covered_cells / total_cells` at fixed iteration budget | Condition 4 ≥ Condition 3 + 15% | Wilcoxon signed-rank on per-run coverage fractions |
| Calibrated absence is informative | Condition 3 vs. 4 abstention | Judge-rated specificity of structured refusal messages | Condition 4 higher specificity at p < 0.05 | Mann-Whitney U on judge ratings |
| Position-of-answer adds value | Condition 3 vs. 4 | Judge-rated structural specificity of rendered answers | Condition 4 > Condition 3 at p < 0.05 | Wilcoxon signed-rank on judge ratings, n=50 sampled outputs |
| Lattice resolution choice matches task | Condition 4 vs. 5 (Omni-Math, A=4) | Omni-Math accuracy | Condition 5 ≥ Condition 4 | Two-proportion z-test, n=300 |
| Counterfactual sensitivity catches fragile answers | Condition 5 vs. 6 | Judge-rated robustness of presented answers | Condition 6 > Condition 5 at p < 0.05 | Wilcoxon signed-rank on judge ratings |
| Full overhaul beats SwarmSys-8 | Condition 6 vs. SwarmSys-8 reported | Omni-Math accuracy | Condition 6 ≥ SwarmSys-8 accuracy on at least one metric | Two-proportion z-test vs. published number with Bonferroni correction |
| Coverage-at-compute claim | Condition 3 vs. 4 vs. SwarmSys-8 | Coverage fraction at fixed A=8 agent budget | Topology-driven ≥ free + 15% | Wilcoxon signed-rank on 10 held-out tasks |

**Notes on the falsifiability matrix:**

The coverage claim (row 1) is the most directly testable because it is a
purely internal metric: coverage is computable from the run's own
`topology_coverage` dict, no external ground truth required. This makes it
the first check after implementing topology generation.

The Omni-Math accuracy claim (row 4, row 6) requires the real LM and
significant compute (300 samples × 3 seeds = 900 runs minimum). It should
be deferred until the internal metrics (coverage, calibrated absence,
structural specificity) show positive signals.

The "SwarmSys-8" comparison in row 6 is the canonical academic claim. If
the SwarmSys codebase is available for local execution, run it on the same
300 Omni-Math samples under the same seed protocol. If not, compare against
the published number with a note that the comparison is not fully controlled
for hardware or prompt engineering differences.

---

## 11. Sequencing

Implementation order is determined by three constraints:
(a) atom plumbing must precede lattice construction,
(b) topology generation must precede coverage tracking,
(c) sensitivity is small and independent once the projection is stable.

**Phase 1 — Atom extraction from VERIFICATION metadata (2–4 days)**

`build_projection` in `core/projection.py` gains a `_build_atoms` function
that iterates `verification_set` for each cluster, reads
`store.get(vid).metadata.get("atoms", [])`, and constructs `AtomProjection`
objects. This is the load-bearing precondition for the lattice axis. No new
agent behavior is needed — the atoms are already in the store (confirmed at
`core/worker_pool.py:748–766`).

Test: `tests/test_projection.py` should gain a test case that deposits a
VERIFICATION signal with atom metadata and asserts that `build_projection`
returns a non-empty `atoms` list.

**Phase 2 — Topology generation (3–5 days)**

Add `AnswerSpaceTopology`, `AxisSpec`, `AnchorCorner` dataclasses (new file:
`core/topology.py`). Add `generate_topology` async function. Add topology call
to `run_swarm.py:run_pipeline` immediately after `build_task_prompt`. Add
`_assign_topology_cells` and wire to `ScoutConfig`. Extend `build_projection`
to compute `topology_coverage` and `uncovered_cells`. Write the coverage
fraction to `summary.json`.

Test: `MOCK_LLM=1 python run_swarm.py debate "test"` should produce a
`summary.json` with `topology_coverage` and `uncovered_cells` fields (they
will be empty with MockLLM since scouts get no real assigned cells — add a
`--topology-mock` flag that injects a hardcoded toy topology for test runs).

**Phase 3 — Sensitivity computation (2–3 days)**

Add `ClusterSensitivity` dataclass and `_build_sensitivities` function in
`core/projection.py`. Wire to `build_projection`. The function iterates each
surviving cluster's `support_set`, simulates removal using the survival filter
logic already in `_apply_survival_filter`, and computes the sensitivity vector.
Write sensitivity summaries to `summary.json` under `cluster_sensitivities`.

Test: unit test that deposits a cluster with two supports of known strengths
and asserts the correct load-bearing and marginal split.

**Phase 4 — Renderer extensions (4–6 days)**

Extend `agents/synthesizer.py`:
- Add topology preamble block (Step 1 of traversal; deterministic).
- Add anchor-corner rendering dispatch (Step 2).
- Add coverage-gap section (Step 5; deterministic).
- Add out-of-bounds section (Step 6; deterministic).
- Add sensitivity annotation inline in `_render_cluster_position`.
- Add `_LATTICE_RESOLUTION_BY_TASK` constant.
- Add `_build_atoms` consumer path in `_render_cluster_position` for
  atom-resolution tasks (reads `vsig.metadata["atoms"]` when `resolution == "atom"`).

**Phase 5 — Default flag flips (1 day)**

- `_SYNTHESIZER_USE_DEBATE = True` (Gap 3).
- `SYNTHESIZER_EMIT_ALTERNATIVE = True` tied to topology corner coverage (Gap 4).
- Add `_topology_abstention` gate (Gap 5).

These are cheap (no structural changes) and their effect is immediately
testable against existing `outputs_mock/` runs.

**Phase 6 — Topology-extension audit (2–3 days)**

Add `_topology_audit_call` function and trigger logic in `run_swarm.py`
per-round logic (after the decay phase, before the next round's scouts run).
Gate on `_TOPOLOGY_EXTENSION_TRIGGER` count of out-of-bounds deposits.

**Phase 7 — Falsifiability harness (3–5 days)**

Add `--seed` CLI argument to `run_swarm.py`. Add `covered_cells`,
`uncovered_cells`, `cluster_sensitivities` to `summary.json` schema.
Add `tools/compare_runs.py` extension to diff coverage fractions across
condition pairs. Run baseline vs. topology on 10 held-out tasks to calibrate
the 15% coverage claim before committing to Omni-Math at 300 samples.

---

## 12. Risks and Open Questions

**Topology quality bottlenecks runs.** The entire topology axis degrades
gracefully to a no-op if the topology generation LLM call fails or parses
incorrectly. The fallback is confirmed: `topology=None` in the projection
means coverage is not tracked and the preamble is not emitted. The risk is not
failure but quality: if the small local LM (7B) produces poorly-discretized
axes (e.g., five axis values that are near-synonyms), scouts get assigned cells
that are semantically indistinct and coverage tracking is misleading.
**Mitigation:** Use a cloud LLM for topology generation (see the
`agents/validator.py:66–78` stub for Anthropic API integration). Topology
generation is one call per run, so the cost is negligible. A Haiku-class call
for topology generation is appropriate.

**Topology generation cost on the hardware target.** One LLM call at
temperature 0.1, max 800 tokens. On the RTX 3060 Laptop with the 7B NF4 model,
this is approximately 5–8 seconds. Acceptable as pipeline overhead; it runs
before any scout iteration.

**Resolution choice on tasks the templates do not anticipate.** The
`_LATTICE_RESOLUTION_BY_TASK` map covers the five registered task types in
`run_swarm.py:TASK_PROMPTS`. Unregistered task types fall back to `cluster`
resolution. This is the same safe default as the existing
`_OUTPUT_STRATEGY_BY_TASK` fallback.

**Sensitivity O(n²) explosion if support sets get large.** The sensitivity
computation is O(n_clusters × mean_supports). At typical run sizes (5–15
surviving clusters, 3–8 supports each), this is ≤120 operations. It becomes
expensive only if `NUM_SCOUTS × ITERATIONS_PER_ROUND` is very large and
the pruning threshold is too low. Gate sensitivity computation on
`len(surviving + contested) <= _SENSITIVITY_MAX_CLUSTERS = 20`.

**What to do when topology extension is contentious.** The audit LLM call
makes a binary decision (extend or file). If the same claim type triggers
extension proposals across multiple rounds and is repeatedly extended, the
topology grows unboundedly. Cap total axis values at 6 per axis and total
corners at 8. If the cap is hit, all subsequent out-of-bounds deposits are
filed without audit.

**Proposition derivation is an on-demand LLM call with no fallback.**
Propositions (Section 3.2) are derived by an LLM call per cluster at
atom-resolution tasks. If the model goes off-format or returns fewer than
2 propositions, the lattice falls back to rendering clusters directly at the
atom level (omitting the proposition mid-layer). The `PropositionProjection`
list in the extended `SynthesisProjection` is empty in this case.

**Recommendation on proposition derivation timing.** Derive propositions as
an on-demand step in the synthesizer, not as part of `build_projection`. The
synthesizer knows the resolution policy; `build_projection` does not.
Propositions should be generated only when `resolution in ("atom", "atom+cluster")`,
and only for clusters that will be rendered in Section 1.

**Frame derivation — structural vs. LLM-labeled.** Frames can be derived
purely structurally (cluster embeddings below a coarser similarity threshold
`_FRAME_SIM_THRESHOLD = 0.40` are merged into a frame) without any LLM call,
at the cost of unlabeled frames (frame_id is a hash of member cluster IDs).
An optional LLM labeling pass adds a human-readable `label`. Recommendation:
always derive frames structurally (zero extra LLM calls); make the labeling
pass optional, gated on a feature flag `_SYNTHESIZER_LABEL_FRAMES = False`.
For `creative` tasks (the primary consumer of frame-level resolution), labels
matter — enable the labeling pass for creative by default.

---

## Verification Self-Check

### File path and line-range verification

All cited file paths exist in the `Attempt At Cleaning/` directory. The
following specific claims were verified by reading the files during this
memo's preparation:

- `core/worker_pool.py:1051–1116`: SAFE pipeline (`_safe_decompose` call,
  atom loop, `_validate_atoms` stash). Confirmed. The atoms are stored on
  the worker instance (`self._validate_atoms`) and then written to deposit
  metadata at lines 748–766.
- `core/worker_pool.py:748–766`: `ParsedDeposit` reconstruction with
  `metadata["atoms"] = _atoms`. Confirmed. Atoms ARE plumbed into VERIFICATION
  signal metadata — Gap 1 is closed at the storage level.
- `agents/synthesizer.py:137`: `_SYNTHESIZER_USE_DEBATE = False`. Confirmed.
- `agents/synthesizer.py:150`: `SYNTHESIZER_EMIT_ALTERNATIVE = False`. Confirmed.
- `agents/synthesizer.py:201–207`: `_OUTPUT_STRATEGY_BY_TASK` dict. Confirmed.
- `agents/synthesizer.py:2240–2258`: Validator note path reads `vsig.content`
  but not `vsig.metadata["atoms"]`. Confirmed. This is the remaining closure
  work for Gap 1/2.
- `core/projection.py:68–110`: `ClusterProjection` dataclass with `support_tree`,
  `trajectory`, `status`, `unverified`. Confirmed present.
- `core/projection.py:113–150`: `InterClusterEdge`, `SynthesisProjection` with
  `inter_cluster_edges`. Confirmed present.
- `core/projection.py:265–350`: `_build_inter_cluster_edges` function. Confirmed.
- `core/actions.py:536–538`: `validate_parse` returns `ParsedDeposit` with
  `content=note` and `metadata={"score": round(score, 4)}`. Confirmed. The
  `atoms` key is NOT added here — it is added post-parse in `worker_pool.py`.
- `run_swarm.py:112–128`: `TASK_PROMPTS` and `ROLES_FOR_TASK` dicts. Confirmed.

### Three-axis completeness check

Each axis provides all four required parts:

**Topology axis:**
- Data structure: `AnswerSpaceTopology`, `AxisSpec`, `AnchorCorner` — Section 3.1
- Computation: `generate_topology`, `_assign_topology_cells`, `_parse_topology` — Section 6
- Synthesizer consumption: traversal Steps 1–3 and 5–6, topology preamble — Section 5
- Beyond-params argument: coverage is a joint-run property; no single forward pass can compute it — Section 3.1

**Lattice axis:**
- Data structure: `AtomProjection`, `PropositionProjection`, `FrameProjection`, `CrossLevelEdge` — Section 3.2
- Computation: `_build_atoms` from VERIFICATION metadata, proposition on-demand, frame structural grouping — Sections 3.2 and 11
- Synthesizer consumption: `_LATTICE_RESOLUTION_BY_TASK` policy, per-resolution rendering — Sections 3.2 and 5
- Beyond-params argument: cross-level contradiction edges require per-atom verification runs; no single forward pass can produce them — Section 3.2

**Sensitivity axis:**
- Data structure: `ClusterSensitivity` — Section 3.3
- Computation: support-removal simulation, dissent-amplification tolerance, topology-uncovered-on-removal — Section 3.3
- Synthesizer consumption: inline annotation in Step 4, fragile-consensus annotation — Section 5
- Beyond-params argument: sensitivity requires having observed specific signal strengths from the actual run — Section 3.3

### Gap coverage check (five pre-existing gaps)

Section 8 addresses all five pre-existing gaps:
1. SAFE atoms in VERIFICATION metadata (retrieval memo precondition) — Gap 1: closed in storage, open in synthesizer consumption
2. `_get_external_context` replacement with atom aggregation (synthesizer §5.7) — Gap 2: partially closed, atom-level rendering outstanding
3. Debate frame enabled by default (synthesizer §5.5) — Gap 3: not closed, action required
4. Alternative-of-the-best enabled (synthesizer §5.9) — Gap 4: not closed, action required with topology-corner trigger
5. Abstention thresholds recalibrated (synthesizer §5.8) — Gap 5: partially implemented, topology-coverage gate outstanding

### End-to-end trace: debate example with topology

To confirm the memo enables an end-to-end trace, consider the following path
through a `debate("Climate action is necessary")` run:

1. **Topology generation:** One LLM call in `run_pipeline` produces a 3-axis
   topology with axes [necessity, framing, scope]. Four anchor corners are
   declared: moral imperative, adaptive market, scientific mandate,
   techno-optimist. `AnswerSpaceTopology` is attached to the pipeline state.

2. **Scout assignment:** 4 scouts map to the 4 anchor corners. Scout 12 is
   assigned cell `(qualified necessary, economic, mitigation)` — an interior
   cell, since `n_scouts > n_anchor_corners`. Scout 12's `ScoutConfig`
   carries `topology_cell = ("qualified necessary", "economic", "mitigation")`.

3. **Scout 12 deposits INITIAL_00023:** Its prompt includes "your assigned
   region is qualified-necessary × economic × mitigation." The deposit
   carries `metadata["topology_coords"] = ("qualified necessary", "economic",
   "mitigation")`.

4. **Foragers develop support:** Three SUPPORT signals are deposited under
   INITIAL_00023. The SAFE Validator decomposes INITIAL_00023's cluster
   representative into 3 atoms; atom queries retrieve snippets on grid
   investment costs. The VERIFICATION deposit carries `metadata["atoms"]`
   with per-atom scores.

5. **`build_projection` runs:** The cluster around INITIAL_00023 achieves
   `status = "surviving"` with `support_diversity = 3`, `verification_score
   = 0.61`. `topology_coverage[("qualified necessary", "economic",
   "mitigation")] = ["INITIAL_00023"]` is recorded. Sensitivity is computed:
   SUPPORT_00041 (strength 0.72) is load-bearing — its removal drops
   `weighted_support` below the contested threshold. `load_bearing_supports
   = ["SUPPORT_00041"]`, `topology_uncovered_on_removal = [("qualified
   necessary", "economic", "mitigation")]`.

6. **Synthesizer traversal:** Step 1 emits the topology preamble (3 axes,
   4 anchors, 2 exclusions). Step 3 (interior rendering) picks
   INITIAL_00023's cluster as the dominant cluster for the interior cell
   `(qualified necessary, economic, mitigation)`. The renderer calls
   `_render_cluster_position` with `resolution = "cluster"`. Step 4
   appends the sensitivity annotation: "Note: this claim rests on a
   single load-bearing support (SUPPORT_00041); remove that support and
   the cluster slides to contested. The adaptive-market corner of the
   topology would become uncovered."

7. **Output:** The rendered answer locates the cluster at "interior position:
   qualified-necessary × economic × mitigation" (novel vs. any current
   swarm output or SwarmSys output), characterizes the fragility of the
   consensus, and lists the uncovered anchor corners (if any). The judge
   model can rate this on structural specificity, calibrated absence, and
   robustness — all three of the judge criteria in Section 10.

This trace is complete and consistent with the data structures and traversal
policy specified in this memo. A competent engineer can implement each step
by reading the cited files and this document without further design work.
