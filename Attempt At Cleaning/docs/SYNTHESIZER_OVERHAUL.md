# Synthesizer Overhaul: From RAG-Over-Swarm to Beyond-Params

*Design memo — read before touching `agents/synthesizer.py` or `core/projection.py`.*

---

## 1. Problem Framing

### Single forward pass vs. multi-call system

A language model trained on parameters θ produces, for any conditioning context x, a distribution p_θ(y | x). Every single LLM call — regardless of prompt length, few-shot examples, or injected documents — is one evaluation of this distribution. The output y is a sample from p_θ(· | x). There is no mechanism by which a single call can produce an output that p_θ could not, in principle, have produced from some x: the forward pass is the ceiling.

A multi-call system can exceed this ceiling, but only under a specific structural condition. Let {x₁, x₂, …, xₙ} be the conditioning contexts presented to n LLM calls. The final output y* exceeds p_θ iff y* is a function of {p_θ(· | x₁), …, p_θ(· | xₙ)} that no single conditioning x_combined could equivalently produce — i.e., the information-theoretic contribution of the individual call distributions is not collapsible into a single context without loss. If a system merely concatenates agent outputs into a longer x and feeds it to a final LLM call, it has not exceeded p_θ. It has just presented a longer prompt.

### The five mechanisms

The literature identifies five architectural mechanisms that genuinely deliver beyond-params output:

1. **Self-consistency / majority voting** (Wang et al. 2022): run N independent samples, take the modal answer. The voted answer y* is a function of the joint sample distribution; no single call produces the mode directly.
2. **Verifier-augmented decoding** (Madaan et al. 2023 Self-Refine; Bai et al. 2022 Constitutional AI): a generator produces a draft; a critic (the same model with a different conditioning) diagnoses failures; the generator revises. The revision attends to critique content the original call could not have produced, because critique depends on having observed a specific draft.
3. **Search over reasoning paths** (Yao et al. Tree-of-Thought, Hao et al. RAP): explicitly enumerate multiple partial continuations, score them, and prune. The tree-search policy is external scaffolding; no single call sees the whole tree.
4. **Debate** (Irving and Christiano 2018): two or more agents argue opposing positions; a judge call rules on the exchange. The exchange is externally structured; the judge's access to rebuttal content is what exceeds single-call capacity.
5. **Decomposition with externalized scaffolding** (Khot et al. Decomposed Prompting): break the task into subtasks, deposit results in a shared medium, compose. The composition call operates on a scaffold built by subtasks it could not have produced alone.

The swarm's architecture is designed for mechanism (5). Agents operating on disjoint corpus partitions deposit INITIAL, SUPPORT, CRITIQUE, OBJECTION, and VERIFICATION signals into a shared `SignalStore`; strength dynamics (decay, amplify, prune) compute a field equilibrium; `build_projection` in `core/projection.py` converts the equilibrium into a structured lattice; the synthesizer reads the lattice to produce the final artifact. The scaffold — the cluster lattice and its strength field — is exactly the externalized state that mechanism (5) requires.

The central claim of this document is that **the synthesizer barely uses the scaffold**. Mechanisms (2), (3), and (4) are entirely absent at synthesis time. The lattice is partially used; its richest structural features are not exposed to any LLM call. The following sections make this precise.

---

## 2. Current State

### Two-layer architecture

The synthesizer is split into two layers, with a clean API boundary between them.

**Layer 1** is `core/projection.py:125–199` (`build_projection`). It is pure Python with no LLM calls. It reads the live `SignalStore`, clusters INITIAL signals by embedding cosine similarity, aggregates per-cluster metrics, applies KB penalties and boosts, and routes each cluster into one of five buckets: `surviving`, `contested`, `weakly_supported`, `rejected_by_field`, or `unverified`. The result is a `SynthesisProjection` (`core/projection.py:87–98`) carrying five typed lists of `ClusterProjection` objects (`core/projection.py:68–83`). Each `ClusterProjection` carries: `representative_id`, `member_ids`, flat `support_set`, flat `dissent_set`, flat `verification_set`, scalar `support_diversity`, scalar `dissent_pressure`, scalar `verification_score`, list `partition_origins`, scalar `support_depth`, and string `status`.

**Layer 2** is `agents/synthesizer.py:406–682` (`_render`). It is an async orchestrator of structured LLM calls. It consumes the `SynthesisProjection` produced by Layer 1, makes between one and roughly 2×N LLM calls for N surviving/contested clusters, and emits a structured text document.

The design rationale for the split is sound: the Python projection layer does the work that is deterministic and computable; the LLM layer does only what requires language. The problem is that the split has been drawn at the wrong point. Layer 1 loses structural information during projection, and Layer 2 does not receive what Layer 1 does retain.

### Layer 2 call graph

The `_render` method (`agents/synthesizer.py:406`) drives execution in the following order:

1. **`_interpret_prompt`** (`agents/synthesizer.py:688`): one LLM call. Reads the user's task prompt in isolation and extracts a `contract` dict — `{regime, form, structural, soft, length_hint, audience}`. The contract captures output shape (haiku vs. function vs. action plan) independently of what the swarm discovered. Temperature 0.1; max 600 tokens.

2. **`_plan_synthesis`** (`agents/synthesizer.py:822`): one LLM call. Reads a structural digest of surviving and contested clusters — IDs, counts, scalar scores, and an 80-character preview of each representative signal. The planner is explicitly prohibited from seeing signal content. It returns `{render_full, section3_only, merge_groups, notes}`. Temperature 0.2; max 1500 tokens.

3. **Strategy branch**: dispatches to one of three execution paths based on `_OUTPUT_STRATEGY_BY_TASK` (`agents/synthesizer.py:139–145`):
   - `sectioned` (debate, analysis): runs `_render_executive_summary` (deterministic, no LLM since a prior refactor), then per-cluster `_render_cluster_position` and `_render_cluster_dissent` calls in sequence.
   - `cohesive_exploration` (creative, problem_solving): one integration call via `_render_cohesive_exploration` (`agents/synthesizer.py:1133`).
   - `cohesive_optimization` (coding): one integration call via `_render_cohesive_optimization` (`agents/synthesizer.py:1199`).

4. **`_stamp_citations`** (`agents/synthesizer.py:1452`): deterministic. Appends a `## 4. CITATIONS` block mapping cluster IDs to their support/dissent/verification signal IDs.

5. **`_build_faithfulness_audit`** (`agents/synthesizer.py:1663`) or **`_build_cohesive_audit`** (`agents/synthesizer.py:1553`): post-hoc, after the answer is already assembled. The sectioned path runs a 4-gram overlap check between rendered paragraphs and the cluster content they cite. The cohesive path runs a softer content-word overlap check. Both are read-only and produce a flag list written to `renderer_audit.json`. Neither result feeds back into the rendered answer.

### What the synthesizer currently reads

From the `SignalStore` and `SynthesisProjection`, the Layer 2 renderer reads:

- `Signal.content` for representative signals, up to `_REPRESENTATIVE_CHARS = 800` characters (`agents/synthesizer.py:86`).
- `Signal.content` for up to 3–4 support signals per cluster, truncated to `_SUPPORT_CHARS = 400` characters each (`agents/synthesizer.py:1103`).
- `Signal.content` for the single strongest dissent signal when `dissent_pressure > 0.5` (`agents/synthesizer.py:1317–1326`).
- Per-cluster scalar metrics: `support_diversity`, `dissent_pressure`, `verification_score`, `support_depth`, `status`, `unverified` flag, and `partition_origins`.
- Cluster-count summary statistics for the deterministic executive summary.
- An 80-character preview in the planner digest (`agents/synthesizer.py:858`).

Additionally, `_render_cluster_position` calls `_get_external_context` (`agents/synthesizer.py:1298`, implementation at `agents/synthesizer.py:1864–1877`), which fetches a Wikipedia snippet for the cluster's representative claim at synthesis time.

### What the synthesizer currently ignores

The following information is present in the signal store or projection at synthesis time but is not consumed by any LLM call in the current renderer:

**Inter-cluster edge structure.** `ClusterProjection` (`core/projection.py:68–83`) contains no edges to other clusters. The only inter-cluster operation is `_detect_inter_cluster_contradictions` (`agents/synthesizer.py:319–357`): a heuristic that flags pairs where both clusters have `dissent_pressure > 0.3`, share ≥3 content words, and share at least one dissent signal. This heuristic does not detect complement relationships, shared-evidence relationships, alternative relationships, or supersession. When detected, contradictions are appended to Section 2 as a string template (`agents/synthesizer.py:577–580`), not surfaced to any LLM call.

**Support tree structure.** `_compute_initial_metrics` in `core/projection.py:319–411` computes `support_depth` by a BFS traversal over SUPPORT signals only (lines 336–360). The traversal builds a `(child_id, parent_depth)` queue and tracks `support_depth_max`. But only the scalar `support_depth_max` is returned (line 410); the actual tree is not. `_aggregate_cluster` (`core/projection.py:485–669`) promotes this scalar to `ClusterProjection.support_depth`. The full parent–child lineage among SUPPORT signals — which encodes how claims were iteratively refined through CHAIN and DEVELOP actions — is discarded during projection. No LLM call at synthesis time sees which SUPPORT signals are parents of which.

**Strength trajectories.** `Signal.iter_at_deposit` is recorded on every signal (`core/signal_store.py:136`), stamped from `self._current_iter` at deposit time (`core/signal_store.py:302`). Layer 1 uses this for age-weighted dissent_pressure calculation in `_aggregate_cluster` (`core/projection.py:547–569`), but the trajectory itself — when was the first support, when was the first dissent, did supports arrive after the dissent or before — is collapsed into a single scalar and not preserved in `ClusterProjection`. No renderer prompt mentions trajectory.

**VERIFICATION signal content.** `validate_parse` in `core/actions.py:466–539` deposits a VERIFICATION signal with `content=note` (`core/actions.py:537`), where `note` is the validator's one-sentence reasoning or the JSON `reasoning` field. The projection records VERIFICATION signal IDs in `ClusterProjection.verification_set` and computes `verification_score` as mean strength (`core/projection.py:640–642`). The synthesizer never reads the `content` of VERIFICATION signals. Instead, `_render_cluster_position` calls `_get_external_context` (`agents/synthesizer.py:1298`) to re-fetch Wikipedia — duplicating the validator's work while ignoring the validator's actual reasoning and score decomposition.

**SEARCH signal provenance.** The SEARCH signal type (`core/signal_types.py:43`) deposits the query and top source titles/URLs as a stigmergic trace. The synthesizer imports signal types (`agents/synthesizer.py:57–60`) but SEARCH is not in that import list, and no SEARCH content reaches any renderer prompt.

**`PoolState.action_log`.** `PoolState` (`core/worker_pool.py:154`) maintains a `deque(maxlen=_SHARE_WINDOW)` of action strings (`core/worker_pool.py:158`). This records the action-type distribution across the pool's recent iterations — the mix of SCOUT/DEVELOP/CHAIN/CRITIQUE/VALIDATE/REFINE calls. The synthesizer never receives the `PoolState` object; it receives only the `SignalStore` and `SynthesisProjection`.

---

## 3. Diagnosis

### Two structural failures

**Failure 1: The information partition is inverted at synthesis time.**

The planner (`_plan_synthesis`) sees structure without content. It receives cluster IDs, scalar metrics, and 80-character previews. It cannot reason about content similarity beyond topic-keyword overlap; it cannot detect that two clusters make complementary empirical arguments or that one logically supersedes another. The planner is structurally positioned to make exactly the decisions that require content — merge which clusters, surface which disagreement — but is systematically denied content to make them.

The per-cluster renderer (`_render_cluster_position`) sees content without structure. It receives the representative signal content, up to three support excerpts, and optionally the strongest dissent. It does not see the cluster's position in the inter-cluster graph, which adjacent clusters share evidence with it, whether it logically extends or contradicts a neighboring cluster, or what its strength trajectory looks like relative to others. The renderer is structurally positioned to incorporate relational context — "this cluster was challenged and survived; that one was never challenged" — but receives none.

The no-leak rule, as specified in `CLAUDE.md`, is an **inter-agent exploration constraint**: agents must not see each other's reasoning chains during exploration, to prevent premature convergence and preserve diversity. The rule was designed for the Scout–Forager–Critic pipeline, not for the synthesis integration point. Applying it at synthesis time inverts its purpose: at the integration point, the system's job is precisely to integrate structure with content. The no-leak rule ends at the swarm's exploration phase.

**Failure 2: The integration call is fed a flat materials dump.**

`_format_materials_block` (`agents/synthesizer.py:1118–1131`) serializes the cluster lattice into a labeled prose list: a THREAD header with scalar metrics, followed by the representative content and support excerpts as flat text. This discards everything structural the swarm earned: chain depth (which SUPPORT signals built on which), shared evidence (which support signals appear under multiple cluster members), rebutted dissent (whether the dissent that arrived was subsequently addressed), inter-cluster edges, and trajectory timing.

The integration call — `_render_cohesive_exploration` or `_render_cohesive_optimization` — then receives this flat block and is asked to "integrate" the threads. What the LLM is actually doing is reconstructing structure from prose: inferring which threads complement each other, which contradict, which are the strongest, from the flat text alone. This is exactly the work the swarm's strength field and cluster lattice already computed. The integration call is rebuilding what the projection discarded.

### Why each failure prevents beyond-params output

The beyond-params criterion requires that y* be a function of the joint set of LLM call distributions {p_θ(· | xᵢ)} that no single conditioning x_combined could produce. The current architecture fails this test:

- The integration call receives a flat materials block x. Any x_combined that included those same clusters in flattened prose form could, in principle, be fed to a single forward pass and produce the same y*. The swarm contributed no structural information that x_combined does not already encode.
- The planner call (`_plan_synthesis`) operates on structure (IDs + scalars + 80-char previews). A direct forward pass on the task prompt plus a similarly compact structural digest could produce a similar plan. The planner's reasoning adds no information that could not be produced by conditioning on the digest alone.
- The faithfulness audit runs post-hoc and feeds nothing back to the LLM. It is a monitoring mechanism, not a mechanism for producing beyond-params output.
- There is no verifier-mediated revision loop (mechanism 2). There is no search over compositions (mechanism 3). There is no debate between cluster positions (mechanism 4). The swarm provides mechanism (5) scaffolding, but the scaffold is not used structurally.

---

## 4. Theoretical Model

### Cluster lattice as externalized state

The cluster lattice at synthesis time is a multi-level structure:

- **Level 0**: INITIAL signals — the opening claims deposited by Scouts from disjoint corpus partitions.
- **Level 1**: The cluster grouping — sets of semantically near-duplicate INITIALs grouped by cosine similarity above `_CLUSTER_SIM_THRESHOLD` (`core/projection.py:53`).
- **Level 2**: The support DAG — SUPPORT signals deposited by Foragers/Developers via DEVELOP/CHAIN/REFINE actions, forming a tree (for CHAIN) or flat forest (for DEVELOP/REFINE) rooted at each INITIAL.
- **Level 3**: The critique and verification layer — CRITIQUE_NEGATIVE/OBJECTION signals from Critics and Haters; VERIFICATION signals from Validators. These are the field's responses to Level 0–2 content.
- **Level 4 (implicit)**: Inter-cluster relations — currently not reified into any data structure.

A beyond-params synthesis operates on this full multi-level structure. It does not flatten Level 2–4 into a prose blob and ask the LLM to re-derive them. It passes Level 4 edges as a typed graph, passes Level 3 VERIFICATION signal contents as first-class inputs (not re-fetched Wikipedia), and passes Level 2 trajectory features as conditioning metadata.

### Synthesis as verifier-mediated search over compositions

The formal claim is that a synthesis engine exceeds p_θ when its final output y* satisfies:

> y* = argmax_{compositions C} f(p_θ(y | x_C), audit(y, lattice))

where the composition C is a structural function over the cluster lattice (which clusters to merge, which to foreground, which transitions to use between them), x_C is the per-composition context built from C, and `audit` is a faithfulness/structural-satisfaction score computed deterministically from the lattice. The outer maximization is the search over compositions, which no single forward pass can execute.

This decomposes into three separable architectural changes: (a) reify the composition space (the typed edge graph), (b) instantiate the verifier-critique loop so `audit` is informative, (c) run best-of-N or tree-search over the composition space.

---

## 5. Improvements

### 5.1 Typed Inter-Cluster Edge Graph

**Mechanism**: Decomposition with externalized scaffolding (mechanism 5), plus enabling mechanism 4 (debate) and mechanism 2 (verifier loop) on specific edge types.

**Change**: Extend `build_projection` (`core/projection.py:125`) to compute `G = (C, E)` where C is the set of `ClusterProjection` objects and E is a set of typed edges. Edge types: `complements` (high cosine similarity between support sets from different clusters; both survived field pressure), `alternatives` (moderate cosine between representative embeddings; neither logically implies the other; comparable strength), `shared_evidence` (non-empty intersection of support_set IDs), `co_contested` (non-empty intersection of dissent_set IDs), `tension` (there exists a support signal s of cluster A and an objection/critique_negative signal d of cluster B such that cosine(s.embedding, d.embedding) > τ; i.e. what one cluster builds on, another actively contests), `supersedes` (one cluster's representative is semantically contained in another's support_set). Each predicate is computable from existing DAG signals and cached embeddings; no LLM call is required. Expose G to both the planner and per-cluster renderers. Let the planner use edge types to decide whether to merge (shared_evidence), contrast (alternatives), or foreground a tension. Let the renderer use edge types to orient the cluster's prose relative to its neighbors.

**Beyond-params argument**: The typed edge structure is a function of the joint distribution of all agent forward passes — it encodes which claims were built on the same evidence, which were challenged by the same dissent, and which directly undermine each other. No single forward pass on the raw task prompt could produce this structure, because it requires having seen what N agents deposited in response to N disjoint corpus partitions and what the field's strength dynamics selected.

**Empirical signature**: Section 2 (dissent/open questions) should gain precision — flagged contradictions correspond to actual `tension` edges rather than the current shared-topic heuristic in `_detect_inter_cluster_contradictions` (lines 339–357). The merge_group rate in the planner should increase for `shared_evidence` cluster pairs. Downstream: MT-Bench coherence scores on debate outputs should rise when the debate task's clusters include `alternatives` edges that are explicitly surfaced.

**Failure mode**: If cosine similarity thresholds for edge predicates are miscalibrated, the graph becomes either fully connected (everything shares evidence) or sparse (no edges detected). Calibrate τ on a held-out run where the ground-truth relationships are known from the task prompt.

---

### 5.2 Verifier-Mediated Revision Loop at Synthesis

**Mechanism**: Verifier-augmented decoding (mechanism 2).

**Change**: Convert `_build_faithfulness_audit` from a post-hoc monitor into a loop driver. After the first render pass (either sectioned or cohesive), run the audit. If `audit_flags` is non-empty, make a critic call that diagnoses each flag — "paragraph P cites cluster C but shares no 4-gram overlap; the paragraph likely paraphrased rather than grounded in the cluster content." Then make a revision call that patches the flagged paragraphs using the critic's diagnosis. Re-audit. Run K ∈ {1, 2} rounds; Self-Refine literature shows diminishing returns past K=2. Instantiate the critic with a prompt vocabulary adapted from the Hater's prompt (`agents/hater.py`) — specifically, the craft-focused adversarial framing. A critic that shares the generator's training distribution will have overlapping blind spots; using the Hater's adversarial vocabulary partially breaks this symmetry.

**Beyond-params argument**: The revision call attends to specific failure modes identified by the audit, which in turn depend on having observed a specific first-draft. The revision context `x_revision = (first_draft, audit_flags, critic_diagnosis)` is not constructible from the task prompt alone; it requires having generated a concrete first draft and having run a structural audit against the lattice. No single forward pass on x_task_prompt could have produced x_revision.

**Empirical signature**: `total_flags` in `renderer_audit.json` should decrease from round 0 to round K. For the cohesive audit, `content_word_overlap_low` flags should drop. Downstream: faithfulness scores on TruthfulQA-style benchmarks (where the ground-truth answer is known) should improve as hallucinated content is corrected in revision.

**Failure mode**: The revision call may introduce new errors while correcting old ones (error propagation). Monitor: if `total_flags` after revision > `total_flags` before revision, the loop is harmful and should be skipped. Also: the critic call adds one full LLM call per revision round, which is expensive on a 6 GB laptop GPU. K=1 is the budget-safe setting.

---

### 5.3 Best-of-N Composition with Structural Scoring

**Mechanism**: Self-consistency / majority voting (mechanism 1), adapted from answer-selection to composition-selection.

**Change**: Run the cohesive integration call N ∈ {3, 5} times with diverse temperatures and different cluster-ordering seeds. The temperature sweep is `[0.3, 0.5, 0.7]` for exploration, `[0.15, 0.25, 0.35]` for optimization. The cluster ordering for each run is independently shuffled (the integration call prompt is order-sensitive; different orderings produce qualitatively different integrations). Score each completion deterministically: `score = cluster_coverage_rate + faithfulness_word_overlap − audit_flag_count`, where `cluster_coverage_rate` counts how many of the `render_full` cluster IDs appear in the artifact as content words (not as [ID] tags), and `faithfulness_word_overlap` is the metric already computed in `_build_cohesive_audit` (`agents/synthesizer.py:1553–1660`). Take `argmax`. This is pure Python scoring against the cluster lattice — no additional LLM calls.

**Beyond-params argument**: The argmax over N independently-seeded completions is a function of the joint sample distribution, not of any single sample. The scoring function uses cluster coverage — a structural property of the lattice — that no single forward pass can self-evaluate. A model generating a single completion cannot score itself against the lattice it didn't produce; the external scoring is the structurally non-collapsible step.

**Empirical signature**: Cluster coverage rate in the selected artifact should be higher than the mean coverage rate across the N candidates. Downstream: BLEU/ROUGE against reference answers on held-out tasks should show non-trivial improvement over single-sample outputs, particularly when N=5 and the task has a clear best-composition structure.

**Failure mode**: If the N completions are high-variance (divergent outputs from the same materials), the argmax may still be mediocre. This signals that the input materials block is insufficiently structured — improvement 1 (typed edge graph) is a prerequisite for best-of-N to yield coherent variation rather than random variation.

---

### 5.4 Decomposed Integration Along the Edge Graph

**Mechanism**: Decomposition with externalized scaffolding (mechanism 5).

**Change**: Replace the single cohesive integration call with two stages. Stage A: for each cluster in `render_full`, generate one `_render_cluster_position` paragraph (already implemented; `agents/synthesizer.py:1259`). Stage B: a composition call that receives the per-cluster position paragraphs (not raw signal traces) plus the typed edge graph from improvement 1, and explicit transitional cues per edge type — `complements` → weave, `alternatives` → contrast and acknowledge both, `tension` → state that cluster A's position is directly challenged by evidence supporting cluster B, `supersedes` → foreground the superseding cluster and footnote the superseded. The composition call operates on structured paragraphs and a labeled graph, not a flat trace dump. This is the correct abstraction: Stage A renders each cluster's voice; Stage B composes voices according to their structural relationships.

**Beyond-params argument**: Stage B's conditioning context includes both the Stage A paragraph outputs (functions of the cluster's internal DAG) and the typed edge graph (a function of the full lattice that required all agent deposits to construct). Neither is producible from a single forward pass on the task prompt. The composition prompt is strictly richer in structural information than `_format_materials_block` currently provides.

**Empirical signature**: Transitions between cluster paragraphs in the final output should lexically reflect edge types — contrast transitions appear near `alternatives` edges, weave transitions near `complements` edges. Human raters blind to the edge graph should be able to reconstruct edge types from transition language at above-chance accuracy.

**Failure mode**: If the per-cluster paragraphs from Stage A are individually too narrow (each paragraph only sees its own cluster's content), Stage B may not have enough material to build meaningful transitions. A partial fix: let Stage A see the 80-character previews of adjacent clusters via the edge graph, not their full content — enough to write a paragraph that anticipates the transition without importing cross-cluster content.

---

### 5.5 Debate Frame for `alternatives` Cluster Sets

**Mechanism**: Debate (mechanism 4).

**Change**: When the edge graph from improvement 1 identifies ≥2 `alternatives` clusters of comparable strength (within 20% of each other in `_cluster_priority` score; `agents/synthesizer.py:150–152`), instantiate a three-round debate: Round 1, each alternative generates a position paragraph (one LLM call per alternative); Round 2, each alternative generates a response to the strongest sibling's Round 1 paragraph; Round 3, a judge call reads both positions and both responses and identifies the unresolved empirical or structural question — what evidence would distinguish the alternatives. Do not collapse the alternatives into a single answer. The Round 3 output replaces Section 1 for the `alternatives` cluster set; both positions are presented, with the judge's unresolved-question framing.

**Beyond-params argument**: The Round 2 responses attend to specific content in Round 1 paragraphs from a sibling cluster. The judge call attends to both positions and both responses. The judge's identification of the unresolved question is conditioned on having seen the full exchange — a context that does not exist at task-prompt time. A single forward pass on the task prompt could produce a hedged "there are two views" paragraph; it cannot produce a precise characterization of what structural difference between the alternatives is currently unresolved, because that characterization requires having seen the alternatives argue against each other.

**Empirical signature**: Section 2 should show reduced `dissent_pressure` for `alternatives` clusters after debate rendering — the exchange surface the disagreement, which is no longer a diffuse field pressure but a named open question. Downstream: on debate-format tasks where ground-truth "the best answer acknowledges X as unresolved" exists, the debate-rendered outputs should match that criterion more often than sectioned rendering.

**Failure mode**: Round 2 responses may degenerate into repetition rather than genuine rebuttal, particularly with low-capability models. Mitigation: feed the Round 2 prompt an explicit instruction to address the strongest specific claim in the sibling paragraph, not to restate the original position.

---

### 5.6 Strength-Trajectory Features

**Mechanism**: Decomposition with externalized scaffolding (mechanism 5) — exposing a richer scaffold.

**Change**: Use `Signal.iter_at_deposit` (`core/signal_store.py:136`) to compute per-cluster trajectory features before calling any renderer. From the signals already in `ClusterProjection.support_set`, `ClusterProjection.dissent_set`, and `ClusterProjection.verification_set`, compute: `iter_first_support` (min `iter_at_deposit` among support_set), `iter_first_dissent` (min among dissent_set), `iter_first_verification` (min among verification_set), `support_growth_rate` (len(support_set) / (iter_at_deposit_of_last_support − iter_first_support + 1)), `dissent_response_lag` (iter_first_support_after_first_dissent − iter_first_dissent), `objection_survival` (count of OBJECTION signals that arrived but were not subsequently addressed by any SUPPORT signal). These are pure Python computations over the existing signal fields; no new data is required. Pass them into the renderer prompt: "this cluster received its first dissent at iteration 12 and accumulated 4 further support signals between iterations 14 and 19 — render the position as one that survived scrutiny."

**Beyond-params argument**: The trajectory features distinguish three qualitatively different cluster histories: never challenged (high support_growth_rate, objection_survival = 0), challenged and responded to (positive dissent_response_lag, support_growth_rate accelerates after iter_first_dissent), and challenged and weakened (positive dissent_response_lag but support_growth_rate does not recover). A single forward pass on the task prompt cannot make this distinction because the distinction requires having observed the actual sequence of agent responses over time. The trajectory is externalized state that the swarm's iterative dynamics produced.

**Empirical signature**: Renderer output for clusters with positive `dissent_response_lag` and subsequent support growth should contain language like "challenged and supported," while clusters with `objection_survival > 0` should contain hedged language. Downstream: correlation between `objection_survival` and human-rated confidence in the synthesized claim.

**Failure mode**: In the current worker pool, `iter_at_deposit` is only non-zero when `store.set_iteration()` is called by the pool each tick (`core/worker_pool.py:925`). Legacy round-based paths and most unit tests leave `iter_at_deposit = 0` everywhere. The trajectory computation must gate on `current_iter > 0` (the same condition already used in `_aggregate_cluster`; `core/projection.py:548`) and return None / omit trajectory from the prompt when unavailable.

---

### 5.7 Validators as First-Class Synthesis Inputs

**Mechanism**: Verifier-augmented decoding (mechanism 2) — validators as persistent external verifiers, not one-shot depositors.

**Change**: Delete the `_get_external_context` Wikipedia re-fetch at synthesis time (`agents/synthesizer.py:1298`, implementation `agents/synthesizer.py:1864–1906`). Replace with an aggregator over `ClusterProjection.verification_set` that reads each VERIFICATION signal's `content` — which, per `validate_parse` (`core/actions.py:537`), is the validator's one-sentence `reasoning` field. Collect these into a `validator_notes` block and inject it into `_render_cluster_position` in place of `ext_block`. The notes say things like "Wikipedia snippet on binary search: this function signature matches O(log n) standard; one test case may miss negative-value inputs" — which is richer grounding than a bare Wikipedia excerpt, because it reflects the validator's actual reasoning about the specific claim.

**Beyond-params argument**: The validator's VERIFICATION signal is a function of a specific claim (deposited before the validator ran) and specific retrieved source material (what the VALIDATE action found; `core/actions.py:425–465`). Replacing it with a fresh Wikipedia lookup discards the validator's judgment about the match between the claim and the source — the informative part. The validator's one-sentence `reasoning` is a condensed second opinion that the synthesis call can use as evidence; the Wikipedia excerpt alone is not.

**Empirical signature**: When VERIFICATION signals are present and their content is fed to the renderer, `no 4-gram overlap` audit flags for verified clusters should decrease (the renderer now has grounded material to work from). The external-context tag `[External context]` should disappear from outputs, replaced by citations to VERIFICATION signals.

**Failure mode**: VERIFICATION signal content quality depends on the model's compliance with the `validate_prompt` JSON schema (`core/actions.py:450–463`). When the model goes off-format, `validate_parse` sets `note = text` (the raw output), which may be verbose and noisy. The aggregator should apply a length cap (e.g. 150 characters, matching `_SUMMARY_CHARS` from `agents/synthesizer.py:87`) and skip signals whose content begins with the raw prompt text.

---

### 5.8 Calibrated Abstention from Projection State

**Mechanism**: Decomposition with externalized scaffolding — the scaffold provides a no-answer condition that a single forward pass cannot compute.

**Change**: Add a pre-render gate in `_render` (`agents/synthesizer.py:406`) before any LLM calls. Refuse to render a prose artifact iff: `(max(verification_score for surviving) < τ_v AND max(support_diversity for surviving) < τ_s AND max(dissent_pressure for surviving) > τ_d) OR len(surviving) == 0`. On refusal, return a structured "the field did not converge" message that includes the strongest fragments (representative content of the top-3 surviving clusters by `_cluster_priority`) and the specific projection signals that triggered abstention. The abstention message is deterministic Python — no LLM call.

**Beyond-params argument**: A calibrated abstention is a claim about the signal field's epistemic state: "after N agent iterations over M corpus partitions, no cluster cleared the verification/support/dissent threshold jointly." This claim is a function of the full joint run; no single forward pass on the task prompt can produce it because it requires having run the swarm and observed the field's failure to converge. A single LLM call on a hard question will produce a confident hallucinated answer; the swarm with a calibrated abstention gate will produce a structured "I don't know with reasons," which is strictly more informative.

**Empirical signature**: On tasks from TruthfulQA or HaluEval where the correct answer is "insufficient evidence," the abstention rate should be higher for the swarm than for a baseline single-call system, and the abstention messages should correctly identify which threshold was not met. On tasks with clear answers, the abstention rate should be near zero.

**Failure mode**: τ_v, τ_s, τ_d must be calibrated against empirical runs. If τ_v is too high, the system abstains on most outputs. If too low, it never abstains. The thresholds depend on the model and corpus; they are not portable across deployment contexts without recalibration.

---

### 5.9 Alternative-of-the-Best as a Second Artifact

**Mechanism**: Self-consistency (mechanism 1), adapted from single-answer to distribution-characterization.

**Change**: For exploration tasks (regime = "exploration" in the contract), after the primary synthesis, identify the strongest cluster not selected by the planner for `render_full` — specifically, the highest-priority cluster (`_cluster_priority` score; `agents/synthesizer.py:150–152`) in the `section3_only` list or among surviving clusters that were held by MMR diversity filtering. Generate a second artifact from this cluster's thread alone (one additional integration call), framed explicitly as "the strongest alternative direction the swarm explored but did not select as primary." Append this as a `## STRONGEST ALTERNATIVE` section with a brief note on what structural feature (comparable support_diversity, competing partition coverage, different cluster trajectory) makes it the strongest non-selected option.

**Beyond-params argument**: The cluster lattice is multi-modal: after field dynamics, multiple clusters have survived with comparable strength. `_select_diverse_clusters` (`agents/synthesizer.py:179–225`) and the MMR selection in `build_plan` (`core/projection.py:244–268`) explicitly choose the highest-priority diverse subset, leaving a tail of near-equally-supported alternatives. A single forward pass produces one mode; the lattice preserves the second mode. Exposing the second mode is information about the response distribution that no single forward pass produces.

**Empirical signature**: On creative tasks with high output diversity (as measured by `centroid_cosine_distance` in `core/output_diversity.py`), the primary and alternative artifacts should have lower pairwise similarity than two independently sampled single-call outputs. On problem-solving tasks, the alternative should represent a different strategy class, identifiable by its `partition_origins`.

**Failure mode**: The second integration call adds one full LLM call to the cohesive path. On a 6 GB laptop GPU with `LLM_CONCURRENCY = 1`, this doubles synthesis wall time. Gate the second artifact on a flag (`SYNTHESIZER_EMIT_ALTERNATIVE = False` by default) and only enable it when the compute budget permits.

---

### 5.10 Relax the Planner/Renderer Information Partition

**Mechanism**: Decomposition with externalized scaffolding — correcting the partition point.

**Change**: Provide the planner with truncated cluster representatives (200 characters, matching `_SUMMARY_CHARS`; `agents/synthesizer.py:87`) plus the typed edge graph from improvement 1, so it can merge clusters by content rather than by 80-character SHA-style preview. The current 80-character preview (`agents/synthesizer.py:858`) is insufficient for content-based merging: "Stigmergic coordination enable..." and "Pheromone-based routing allow..." look different in 80 characters but may be the same claim. Provide each per-cluster renderer with the structural metrics (but not content) of adjacent clusters in the edge graph — specifically, the `support_diversity`, `verification_score`, and trajectory summary of `complements` and `alternatives` neighbors — so the rendered paragraph can characterize the cluster's position relative to its neighbors without importing their content.

**Beyond-params argument**: The current planner's 80-character preview constraint makes it unable to distinguish content-equivalent clusters that use different surface language. The result is that semantically redundant clusters are rendered as separate positions, diluting the synthesis. Extending the planner's context to 200 characters plus edge graph structure gives it enough information to make merge decisions that are functionally correct, not just topically plausible. The merge decision itself — "these two clusters are saying the same thing from different corpus partitions" — is a structural claim about the signal field that requires having seen what both partitions produced.

**Empirical signature**: The merge_group rate in planner output should increase. The Section 1 paragraph count should decrease for runs with many near-duplicate clusters, while `support_diversity` of the merged positions should increase (combining the diversity of both clusters). The planner's `notes` field should reference specific content when justifying merges, not just "similar previews."

**Failure mode**: Giving the planner 200-character representatives rather than 80-character previews increases the planner prompt length proportionally to the number of candidates. At 12 candidates, the digest grows from ~2KB to ~5KB — within `_PLAN_MAX_TOKENS = 1500` (`agents/synthesizer.py:113`) only if the rest of the prompt is trimmed. The planner prompt must be restructured to accommodate the longer representatives without exceeding the token budget.

---

## 6. Preconditions

The following changes to the rest of the codebase are required before the improvements above can pay off. They are listed in dependency order.

**`ClusterProjection.support_tree`**: `_compute_initial_metrics` (`core/projection.py:319–411`) traverses the SUPPORT DAG via BFS but discards the tree structure, recording only the scalar `support_depth_max` (line 410). Add a `support_tree: dict[str, list[str]]` field to `ClusterProjection` (`core/projection.py:68`) mapping each SUPPORT signal ID to its direct SUPPORT children in the lineage. Populate it during the BFS in `_compute_initial_metrics` by recording `(parent_id, child_id)` pairs for `SUPPORT → SUPPORT` edges. `_aggregate_cluster` should merge these per-member trees into a cluster-level tree. Improvements 5.4 and 5.6 both depend on this.

**Validator `note` surfacing**: `validate_parse` already populates `note` from the model's `reasoning` field (`core/actions.py:510–511`, `core/actions.py:520–521`) and deposits it as VERIFICATION `content` (`core/actions.py:537`). The downstream consumers don't read it: `ClusterProjection.verification_set` carries IDs, and the synthesizer reads only the scalar `verification_score`. No code change is needed in `validate_parse`; what is needed is that `_render_cluster_position` read VERIFICATION signal contents (from `ClusterProjection.verification_set`) rather than calling `_get_external_context`. Improvement 5.7 formalizes this.

**Graph-density convergence halt**: The convergence detector in `core/convergence.py` (not read in detail here) uses population-based criteria. The typed edge graph from improvement 1 provides a new halt signal: the graph has reached minimum density when at least one inter-cluster edge of each relation type has been detected. This prevents the pool from halting when clusters exist but have not yet had enough cross-cluster signal activity to produce informative edges. Add `MIN_INTER_CLUSTER_EDGES = 1` as a configuration constant and check it in the convergence detector. Without this, improvement 1's graph may be sparse at halt time, reducing the value of all edge-dependent improvements.

---

## 7. Falsifiability

The claim that the swarm + synthesizer produces beyond-params output is, in its current form, empirically empty. The architecture does not yet implement any of the five beyond-params mechanisms at synthesis time. The claim becomes testable when at least one improvement from Section 5 is implemented. The comparison harness must hold the following constant:

- Base LM: the same model checkpoint (`deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` or equivalent) used in both swarm and baseline conditions.
- Task prompt: the same prompt presented to both conditions.
- Single-call baseline: one forward pass on the task prompt alone, with no retrieved context and no swarm. This is the `--mode=baseline` flag from `core/baseline.py`, run with N=1 agent.

The benchmark must have external ground truth: TruthfulQA (factual accuracy), HaluEval (hallucination rate), MT-Bench (multi-turn coherence, interpretive tasks), or a domain-specific equivalent. Without external ground truth, the comparison is unfalsifiable — one can always argue that the swarm output is "richer" without being able to measure it.

**The beyond-params claim cannot be verified with mock-LLM runs.** `MockLLM` emits SHA1-seeded phrases regardless of input (`CLAUDE.md`, Outputs section). Behavioral claims require the same model in both conditions across multiple runs with the same seeds.

| Improvement | A/B Regime | Expected Effect | Failure Mode |
|-------------|-----------|-----------------|--------------|
| 5.1 Inter-cluster edge graph | Swarm w/ edge graph vs. swarm w/ heuristic contradiction detection | Reduction in false-positive contradiction flags in Section 2; increased merge_group rate on near-duplicate clusters | No edge types detected (τ miscalibrated); graph fully connected or empty |
| 5.2 Verifier revision loop | Swarm w/ K=1 revision vs. swarm w/o revision | Decrease in `total_flags` in `renderer_audit.json`; improvement in TruthfulQA accuracy | Revision introduces new errors; net flag count increases |
| 5.3 Best-of-N composition | Swarm w/ N=3 vs. N=1 integration | Higher cluster coverage rate; improvement in MT-Bench coherence | N candidates are high-variance noise; argmax is not better than N=1 mean |
| 5.4 Decomposed integration | Swarm w/ Stage A+B vs. single integration call | Human raters correctly identify edge types from transition language; reduced Section 2 size (disagreement resolved structurally) | Stage B ignores Stage A paragraphs; produces independent text |
| 5.5 Debate frame | Swarm w/ debate on alternatives vs. sectioned | Human judges prefer debate output on genuinely open questions; MT-Bench multi-turn appropriately acknowledges openness | Rounds 1–2 degenerate to repetition; Round 3 fails to identify unresolved question |
| 5.6 Trajectory features | Swarm w/ trajectory context vs. w/o | "Survived scrutiny" language correlates with positive `dissent_response_lag` in audit log | `iter_at_deposit` = 0 on all signals (set_iteration not wired); trajectory features uniformly absent |
| 5.7 Validators as inputs | Swarm w/ VERIFICATION content vs. Wikipedia re-fetch | Decrease in `no 4-gram overlap` flags for verified clusters | VERIFICATION note content is off-format; worse than Wikipedia excerpt |
| 5.8 Calibrated abstention | Swarm w/ abstention gate vs. w/o | Higher abstention rate on HaluEval unanswerable instances; near-zero abstention rate on answerable TruthfulQA | τ miscalibrated: either never abstains or always abstains |
| 5.9 Alternative artifact | Swarm w/ second artifact vs. w/o | Lower pairwise similarity between primary and alternative than between two N=2 single-call samples | Second artifact is near-duplicate of primary (MMR selection failed to diversify) |
| 5.10 Relaxed planner partition | Swarm w/ 200-char reps + edge graph in planner vs. 80-char preview only | Increase in accurate merge_group detection; decrease in redundant Section 1 paragraphs | Planner prompt exceeds token budget; truncation corrupts JSON output |

---

## 8. Risks and Open Questions

**Loop amplification.** Improvement 5.2 (revision loop) has a documented failure mode in the Self-Refine literature: the generator and critic share the same training distribution, so the critic cannot reliably diagnose errors that are systematic artifacts of p_θ. Using the Hater prompt vocabulary partially breaks this symmetry but does not eliminate it. The loop may converge to a revised artifact that passes the audit by adopting the critique's exact language — "passing" via lexical overlap rather than epistemic correction.

**Cost explosion.** Improvements 5.2, 5.3, 5.4, and 5.5 each add between 1 and 2N additional LLM calls to the synthesis path. On a 6 GB laptop GPU with `LLM_CONCURRENCY = 1`, the synthesis time is already the wall-clock bottleneck. Implementing all improvements simultaneously would be prohibitive. The sequencing section below prioritizes by independence and compute cost.

**Faithfulness regression.** Improvement 5.10 (relaxed planner partition) gives the planner content beyond 80-character previews. There is a real risk that extending planner context causes the planner to reason from content rather than structure — producing a plan that is "good for the previews I saw" rather than "good for the full cluster content I will read in the render pass." The planner was structurally constrained to structure-only reasoning for a reason. The 200-character extension is a measured relaxation, not an invitation to full content ingestion.

**No-leak ambiguity at integration time.** The no-leak rule (`CLAUDE.md`: "Agents only ever observe signals as artifacts") was written for the Scout–Forager–Critic exploration pipeline. The synthesizer is not a pipeline agent; it is the integration point. But the rule is stated as a global architectural constraint, not qualified to the exploration phase. Improvements 5.4 and 5.10 explicitly relax the rule at integration time. This requires an architectural decision: the no-leak rule should be formally scoped to inter-agent exploration, with an explicit exception for the synthesizer. Failing to document this exception will result in future changes inadvertently re-introducing the restriction.

---

## 9. Sequencing

**Stage 0 — Preconditions (independent; no improvement depends on Stage 1+):**
1. Wire `support_tree` into `ClusterProjection` and populate it in `_compute_initial_metrics`. Required by improvements 5.4 and 5.6.
2. Confirm VERIFICATION signal `note` content is accessible — it already is, no code change needed; document this explicitly so future developers don't re-introduce Wikipedia re-fetch.
3. Add `MIN_INTER_CLUSTER_EDGES` to convergence logic. Required by improvement 5.1.

**Stage 1 — Structural scaffold (improvements that add data to the projection):**
4. Implement typed inter-cluster edge graph (improvement 5.1). This is the dependency for improvements 5.4, 5.5, and parts of 5.10. It is pure Python; no additional LLM calls.
5. Implement strength-trajectory features (improvement 5.6). Pure Python; no additional LLM calls. Low compute cost; immediately improves renderer prompts.
6. Implement validators as first-class inputs (improvement 5.7). Replaces one external network call (`_get_external_context` Wikipedia fetch) with a pure Python read from existing VERIFICATION signals. Net compute savings.

**Stage 2 — Verifier mechanisms (improvements that add LLM calls at synthesis time):**
7. Implement calibrated abstention (improvement 5.8). One Python gate; adds no LLM calls. Ships before any additional LLM calls are added so baseline cost is established.
8. Implement verifier revision loop (improvement 5.2), K=1. One additional LLM call per synthesis. Validates the loop mechanism before adding more.
9. Implement decomposed integration (improvement 5.4). Restructures existing per-cluster calls; net LLM count change is minimal for sectioned output, moderate for cohesive.

**Stage 3 — Search and diversity (improvements that multiply LLM call count):**
10. Implement best-of-N (improvement 5.3), N=3 initially. Validate quality lift before raising N.
11. Implement debate frame for alternatives (improvement 5.5). Gate on `SYNTHESIZER_USE_DEBATE = False` default; enable only when compute budget is available.
12. Implement alternative artifact (improvement 5.9). Gate on `SYNTHESIZER_EMIT_ALTERNATIVE = False`. Lowest priority; most useful for exploration tasks with high cluster diversity.

**Stage 4 — Planner partition relaxation (improvement 5.10):**
13. Last because it touches the planner prompt structure, which interacts with all other improvements, and because it requires Stage 1 (edge graph) to be in place before the extended planner context is meaningful.

---

## Verification Self-Check

**File paths verified:**
- `agents/synthesizer.py` — read in full (lines 1–2108). All cited line ranges confirmed.
- `core/projection.py` — read in full (lines 1–791). All cited line ranges confirmed.
- `core/signal_store.py` — read lines 1–220 (Signal dataclass through store init). `iter_at_deposit` at line 136 confirmed.
- `core/signal_types.py` — read in full. VERIFICATION, SEARCH, CONTRARIAN_TYPES confirmed.
- `core/actions.py:466–539` (`validate_parse`) — read. `content=note` at line 537 confirmed.
- `core/worker_pool.py:154–208` (`PoolState`) — read. `action_log` as `deque` at line 158 confirmed.

**Improvements with strongest empirical case (in order):**
1. **Improvement 5.7** (validators as first-class inputs): zero compute cost increase; validators already produce the reasoning sentence; the only change is reading it instead of re-fetching Wikipedia. Measurable via existing `renderer_audit.json` metrics.
2. **Improvement 5.6** (trajectory features): pure Python; `iter_at_deposit` is already recorded; the trajectory computation adds no LLM calls; the empirical signature (correlation between trajectory and rendered confidence language) is directly testable.
3. **Improvement 5.8** (calibrated abstention): the hardest current problem is that the synthesizer always produces prose even when the field has not converged; the no-consensus gate at `agents/synthesizer.py:412–431` handles the extreme case but not the partial-convergence case.

**The central claim stated in falsifiable form:**
If the swarm with improvements 5.1–5.10 fully implemented does not outperform a single-call baseline on TruthfulQA accuracy and MT-Bench coherence when both use the same base LM, then the architectural thesis of this document is wrong. The improvements are individually testable as listed in Section 7; the joint claim requires all improvements to be implemented and the full falsification harness to be in place.

**The project cannot yet prove this claim.** No empirical A/B run against a single-call baseline has been conducted with the real model. `outputs_mock/` proves plumbing, not behavior. Until empirical runs with `--mode=baseline` versus the full swarm are executed on a shared benchmark with external ground truth, the beyond-params claim is a theoretical prediction, not an observed result.
