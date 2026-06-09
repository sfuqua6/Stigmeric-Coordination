# Claude Code prompt — Synthesizer overhaul memo

Paste everything below the `---` line into Claude Code from the repository root
(`C:\Users\agsse\Downloads\ai_swarm_mechanics-main (4)\ai_swarm_mechanics-main`).

---

You are working in the `Attempt At Cleaning/` folder of a stigmergic multi-agent
swarm codebase. Read `Attempt At Cleaning/CLAUDE.md` first if it exists; then
read `agents/synthesizer.py` end-to-end. Treat that file as the subject of this
task. Do NOT modify code yet — your deliverable for this pass is a single
markdown document.

## Task

Write a comprehensive theoretical and architectural design memo for an
**overhaul of the synthesizer** to the path:

    Attempt At Cleaning/docs/SYNTHESIZER_OVERHAUL.md

(create the `docs/` directory if it does not exist).

The driving framing for the overhaul is one sentence: **"enable a system that
goes beyond its base model's parameters."** That phrase is a precise
theoretical claim — that the orchestrated swarm + synthesizer should produce
outputs that no single forward pass of the underlying LM could equivalently
produce. Every proposal in the memo must be evaluated against that claim.

## Context the memo must establish

Before proposing changes, the memo must accurately describe what exists today.
Read the code, do not paraphrase from training data. Specifically capture:

1. The synthesizer's two-layer split: Layer 1 (pure-Python projection in
   `core/projection.py`) and Layer 2 (LLM-mediated renderer in
   `agents/synthesizer.py`).
2. The current Layer 2 call graph: `_interpret_prompt` → `_plan_synthesis` →
   `_render_executive_summary` (deterministic) → strategy branch (sectioned
   vs. cohesive_exploration vs. cohesive_optimization) → `_stamp_citations`
   → `_build_faithfulness_audit`.
3. What the synthesizer currently reads from the orchestration
   (`SignalStore`, `ClusterProjection`, `Signal.content`, the projection's
   scalar metrics).
4. What the synthesizer ignores from the orchestration. Be specific. At
   minimum: inter-cluster edge structure beyond a single ad-hoc
   `_detect_inter_cluster_contradictions` heuristic; the chain structure
   inside clusters (CHAIN was hard-coded to deepen SUPPORT lineage but the
   projection flattens `support_set` into a list); strength trajectories
   over iterations (`Signal.iter_at_deposit` is recorded but unused at
   synthesis); the validators' VERIFICATION deposits (the synthesizer
   re-runs Wikipedia at synthesis time via `_get_external_context`,
   ignoring validator work); the SEARCH-signal provenance to source
   material; and the `PoolState.action_log`.

## Diagnostic frame

Use the following theoretical frame, articulated explicitly in the memo:

> The current architecture is **retrieval-augmented generation where the
> retriever is a swarm of the same base LM**. Every LLM call — including
> the integration call — is still a single forward pass of `p_θ(y | x)` for
> some `x`. The synthesizer is a thin coordinator over independent forward
> passes; it does not exceed `p_θ`.
>
> A multi-call system exceeds `p_θ` iff its final output `y*` is a function
> of `{p_θ(·|x_1), p_θ(·|x_2), …}` that no single conditioning `x_combined`
> could equivalently produce. Five mechanisms in the literature deliver
> this: (1) self-consistency / majority voting, (2) verifier-augmented
> decoding (Self-Refine, Constitutional AI critique loops), (3) search
> over reasoning paths (tree-of-thought, RAP), (4) debate (Irving–
> Christiano), (5) decomposition with externalized scaffolding.
>
> The swarm already builds the externalized scaffold for mechanism (5) —
> the cluster lattice and strength field. The synthesizer barely uses it.
> Mechanisms (2), (3), (4) are absent at synthesis time.

Then state the **two structural failures** that prevent beyond-params output
today:

- **Information partition is inverted.** The planner sees structure without
  content; the renderer sees content without structure. The no-leak rule
  is an *inter-agent exploration* constraint, not a synthesis-time
  constraint, and is doing harm at the integration point.
- **The integration call is fed a flat materials dump.** `_format_materials_block`
  serializes the lattice into a labeled prose list, discarding the structure
  the swarm earned (chain depth, shared evidence, rebutted dissent, etc.).
  The LM is then asked to reconstruct structure from prose — the work the
  swarm exists to avoid.

## Ten improvements to document

For each improvement below, the memo must give:

1. The mechanism (which of (1)–(5) above it invokes).
2. The concrete change to the synthesizer / projection / signal store.
3. Why this lifts capability above `p_θ` (the beyond-params argument).
4. The empirical signature: what should change in the audit metrics or
   downstream benchmark if it works.
5. The failure mode: what would tell us it does not work.

The ten improvements:

1. **Typed inter-cluster edge graph.** Extend the projection from
   clusters-with-internal-structure to `G = (C, E)` with relation set
   `R = {complements, alternatives, shared-evidence, co-contested,
   tension, supersedes}`. Each edge is computable from existing DAG
   signals: cosine similarity bands, support-set overlap, dissent-set
   overlap, and the `tension` predicate `∃ s ∈ support(a), d ∈ dissent(b)
   with cosine(s, d) > τ`. Expose `G` to both planner and renderer; let
   the renderer choose narrative function per cluster based on its edges.

2. **Verifier-mediated revision loop at synthesis.** Wire
   `_build_faithfulness_audit` into a loop instead of running post-hoc.
   Draft → audit → critic call (diagnose each flag) → revise call →
   re-audit. K=1 to 2 rounds; Self-Refine literature shows diminishing
   returns past that. Instantiate the synthesizer critic with the swarm
   Hater's prompt vocabulary so generator and critic do not share blind
   spots.

3. **Best-of-N composition with structural scoring.** Run the cohesive
   integration call N ∈ {3, 5} times with diverse temperatures and
   different cluster-ordering seeds. Score each completion
   deterministically: `cluster_coverage + faithfulness +
   structural_satisfaction − audit_flag_count`. The first two metrics
   already exist in `_build_cohesive_audit`. Pick argmax. Pareto trade
   the increased compute against quality lift.

4. **Decomposed integration along the edge graph.** Replace the single
   cohesive integration call with two stages: Stage A produces one
   per-cluster voice via `_render_cluster_position`; Stage B composes
   the voices using the edge graph + explicit transitional cues per
   edge type ("→complements: weave; →alternatives: contrast;
   →tension: acknowledge and choose"). Stage B operates on structured
   paragraphs + an explicit graph, not raw signal traces.

5. **Debate frame for `alternatives` cluster sets.** When the edge graph
   identifies ≥ 2 alternatives of comparable strength, render the
   disagreement explicitly: Round 1 each alternative gets a position
   paragraph; Round 2 each responds to the strongest sibling; Round 3
   judge call identifies the unresolved empirical or structural
   question. Do not flatten the disagreement into a single confident
   answer when the underlying question is genuinely open.

6. **Strength-trajectory features.** Use `Signal.iter_at_deposit` to
   compute per-cluster trajectory features: `iter_first_support`,
   `iter_first_dissent`, `iter_first_verification`,
   `support_growth_rate`, `dissent_response_lag`, `objection_survival`.
   Pass these into the renderer prompt: "this cluster was challenged
   at iteration N and the field responded with M supports at
   iterations N+k addressing the challenge — render the position as
   one that survived scrutiny." Distinguishes "never challenged" from
   "challenged and survived" in the rendered prose.

7. **Validators as first-class synthesis inputs.** Delete the
   `_get_external_context` re-fetch of Wikipedia at synthesis time.
   Replace with an aggregator over `cp.verification_set` that reads
   each VERIFICATION signal's content (the validator's
   one-sentence reasoning + score). Validators currently produce
   deposits that gate nothing downstream — this turns vestigial role
   into load-bearing input.

8. **Calibrated abstention from projection state.** Add a pre-render
   gate: refuse to render an artifact iff
   `(max verification_score < τ_v AND max support_diversity < τ_s
   AND max dissent_pressure > τ_d) OR n_surviving == 0`. On refusal
   return a structured "the field did not converge; here are the
   strongest fragments and the structural signals that triggered
   abstention." A calibrated "I don't know" is information a single
   forward pass cannot produce.

9. **Alternative-of-the-best as a second artifact.** For exploration
   tasks, present the artifact + the strongest cluster not selected,
   framed as the strongest alternative. Single forward passes
   produce one mode; the lattice has multi-modal structure that the
   synthesizer normally collapses. Exposing the second mode tells the
   user the *shape* of the response distribution, not just its argmax.

10. **Relax the planner/renderer information partition.** Let the
    planner see truncated cluster reps (200 chars) plus the edge
    graph from improvement 1, so it can merge by content rather than
    by SHA. Let the per-cluster renderer see structural metrics of
    *adjacent* clusters in the edge graph. The no-leak rule's job
    ended at exploration time.

## Preconditions section

The memo must include a "Preconditions" section listing what the rest of the
codebase has to expose before the synthesizer changes pay off. At minimum:

- `ClusterProjection` should carry `support_tree: dict[parent_id, list[child_id]]`
  in addition to the flat `support_set`. The CHAIN action built tree
  structure that the current flattening discards.
- The validator role must surface a reasoning sentence on every VERIFICATION
  signal. `validate_parse` populates `note` already; downstream consumers
  don't read it.
- The convergence detector needs a *graph-density* halt floor in addition
  to the current population floor (`MIN_INITIALS_FOR_HALT = 6`). At least
  one inter-cluster edge in the graph from improvement 1.

## Falsifiability section

The memo must end with a table mapping each improvement to a controlled
comparison: A/B regime, expected effect, failure mode. The benchmark needs
an external ground truth (TruthfulQA, HaluEval, AraEval, MT-Bench, or a
domain-specific equivalent), a fixed base LM held constant across regimes,
and a single-call baseline that uses the same `task_prompt` directly. The
beyond-params claim is irrefutable without this harness; state that
plainly.

## Memo structure (use this skeleton)

```
# Synthesizer Overhaul: From RAG-Over-Swarm to Beyond-Params

## 1. Problem framing
   Single forward pass vs. multi-call system. The five mechanisms.

## 2. Current state
   Two-layer architecture, Layer 2 call graph, what's read, what's ignored.

## 3. Diagnosis
   Two structural failures: inverted information partition, flat materials dump.
   Why each prevents beyond-params output.

## 4. Theoretical model
   Cluster lattice as externalized state. Synthesis as verifier-mediated
   search over compositions. The formal beyond-params criterion.

## 5. Improvements
   One subsection per improvement (1–10). Each subsection has: Mechanism,
   Change, Beyond-params argument, Empirical signature, Failure mode.

## 6. Preconditions
   What the rest of the codebase has to expose first.

## 7. Falsifiability
   The comparison harness. Benchmark, baseline, A/B table.

## 8. Risks and open questions
   Loop amplification, cost explosion, faithfulness regression, no-leak
   ambiguity at integration time.

## 9. Sequencing
   Suggested implementation order. Independent preconditions first
   (support_tree, validator surfacing, edge graph). Then mechanisms that
   build on them (verifier loop, decomposition, debate). Best-of-N and
   abstention land last because they are wrappers around stable inner
   stages.
```

## Style and constraints

- Doctoral-level prose. Skeptical, precise, no marketing language.
- When you cite a function, cite the file path and the line range you read.
- Do not invent file paths or line numbers — read the code.
- Do not embellish the swarm's biological metaphor; the metaphor is
  presentation, not derivation, and the memo should treat it that way.
- Do not implement code. The deliverable is the markdown document only.
- Do not pre-commit to file changes outside `Attempt At Cleaning/docs/`.
- Write in CommonMark; use code fences for code references, blockquotes
  for cited claims.
- Length target: 4,000–8,000 words. Density over surface area.
- Every improvement subsection must explicitly answer the beyond-params
  question: *what structural feature of the swarm's output cannot be
  reconstructed from a single forward pass on the same input?*

## Verification step

After writing the memo, run a self-check pass:

1. Does every cited file path exist? Open each one and confirm.
2. Does every cited line range match the content you describe? Re-read
   and correct.
3. Does each improvement subsection contain all five required parts
   (Mechanism, Change, Beyond-params, Empirical signature, Failure mode)?
4. Is there at least one paragraph that names the central claim in falsi-
   fiable form, and one that admits the project cannot yet prove it?

Report which preconditions you verified and which improvements you found
the strongest empirical case for. Do not modify the code.
