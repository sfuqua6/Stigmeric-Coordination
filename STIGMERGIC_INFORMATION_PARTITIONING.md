# Stigmergic Coordination Through Information Partitioning
## A Multi-Agent LLM Architecture Without Deliberation

**Status:** Draft research note, accompanies the `Attempt At Cleaning/` reference implementation.
**Hardware target:** Single consumer GPU (NVIDIA RTX 3060 Laptop, 6GB VRAM).
**Base model assumption:** DeepSeek-R1-Distill-Qwen-7B (4-bit NF4 quantization).

---

## Abstract

We consider the problem of producing diverse, error-decorrelated outputs from a multi-agent system built on a single language model. Existing multi-agent LLM frameworks (Multi-Agent Debate, AutoGen, MAD-style systems) achieve coordination through deliberation: agents read each other's reasoning chains and revise their own outputs in response. We argue that this mechanism systematically defeats its stated purpose. Conditioning on another agent's reasoning chain is, from the model's standpoint, conditioning on a high-likelihood completion of the original prompt; it concentrates the posterior toward the regions of output space that produced the chain, rather than away from them. The result is a conformity cascade: agents converge rapidly, and the consensus is wrong in coordinated ways.

We propose an alternative coordination mechanism grounded in stigmergy — communication strictly via observable environmental traces, never via visible reasoning. We further argue that the diversity such systems are nominally trying to achieve cannot come from temperature variation or role differentiation when the underlying probability distribution is shared. It must come from differential information conditioning at intake. We formalize this as a coin-flipping metaphor: identical probability machines produce different results when given coins with different weights and tilts. The probability machine is the model; the coin is the conditioning information; the weight and tilt are the partitioning of evidence and the sampling strategy over the shared trace store.

We instantiate these principles in a reference implementation designed to run on a single consumer GPU.

---

## 1. Introduction

Multi-agent large language model systems are motivated by the intuition that committees of agents should outperform single-shot generation, in the same way that ensembles of classifiers outperform single classifiers in classical machine learning. The mechanism that makes ensembles work is well-understood: errors of constituent classifiers must be at least partially uncorrelated, so that majority voting can recover the truth even when individual members are wrong. Bagging achieves this by training each constituent on a bootstrap sample of the data; boosting achieves it by reweighting examples that previous learners got wrong.

The contemporary multi-agent LLM literature does not have an analog of bagging. The dominant pattern is to instantiate N copies of the same base model, give them different role prompts ("you are a Critic"; "you are a Skeptic"), vary the sampling temperature across them, and then have them deliberate by exchanging their reasoning chains as conversational turns. The implicit hope is that role variation and temperature noise will produce sufficient output diversity to make the ensemble useful.

We argue this hope is misplaced for two reasons.

**The conformity problem.** When agents read each other's reasoning chains, the resulting posterior over outputs concentrates rather than diversifies. The mechanism is not social pressure or imitation in any anthropomorphic sense — it is conditioning. A reasoning chain is high-likelihood text from the model's perspective; conditioning on it pulls the next agent's output distribution toward the modes that produced it. Adversarial role prompts ("be skeptical") cannot offset this, because they ask the model to *generate text that looks like skepticism* while the underlying conditioning still concentrates around the previously-stated chain. The empirical signature is convergence to a wrong consensus.

**The conditioning problem.** Variation in temperature and role prompt does not produce statistically independent samples in any rigorous sense. It varies the expression of the same posterior P(y | x, θ). To get genuinely different posteriors P(y | x_i, θ), the conditioning information x_i must differ across agents. A single base model on a single piece of evidence will always have correlated errors no matter how many sampling tricks are applied to it.

The implementation in `Attempt At Cleaning/` instantiates an alternative design that takes both problems seriously.

---

## 2. Why Deliberation Defeats Itself

Let M be a base language model with parameters θ. Each agent in a multi-agent system samples from P(y | x_i, θ) where x_i is the agent's context.

In the standard deliberative setup, x_i is constructed by concatenating: a base prompt, a role instruction, and the previous outputs y_j (and often the reasoning chains behind them) of other agents. The composite context conditions the model on a narrative that already justifies the previous agents' conclusions. Treating that narrative as something to extend rather than challenge is the modal completion behavior of any well-trained LLM: the next-token loss the model was trained on rewards coherent continuation, not contrarian critique.

Adversarial role prompts attempt to override this by instructing the model to challenge what it has just read. Two things happen. First, the model produces text that surface-features as challenge: "however," "but," "this overlooks." Second, the substantive content of the challenge typically rephrases or qualifies the prior chain rather than rejecting its premises. This is not a flaw in any particular implementation — it is what conditioning on a coherent prior chain does to a sequence model's posterior.

The behavioral signature is well-attested in practice: deliberative multi-agent systems converge rapidly to consensus, and the consensus is wrong in coordinated ways on tasks where individual agents would have been wrong differently. The committee is no better than its members because the committee shares a distribution.

Stigmergic insect colonies do not have this problem because no ant ever observes what another ant is thinking. An ant deposits a pheromone — a chemical trace — and another ant detects the trace and either reinforces it (by depositing more pheromone there) or moves on. The decision to reinforce is made on the local information available to the second ant: distance to food, fitness of the path, time since last reinforcement. The colony's collective trail-following emerges from many independent local decisions about traces. No ant ever follows another ant's reasoning, because there is no reasoning to follow — only the trace.

We translate this asymmetry to LLM agents directly. Agents observe deposited signals (the trace) but never the reasoning that produced them. A signal carries its meaning as an artifact: a claim, an item of evidence, a critique. Every agent that processes the signal must independently re-reason from it, which forces independent confabulation, independent confirmation, and independent failure. Signals that survive amplification have done so because many independent instantiations of the model arrived at compatible conclusions from different conditioning paths through the evidence.

We name this *behavior-consensus* and contrast it with *argument-consensus*. Argument-consensus is what deliberative systems produce: agents agree because they have followed the same chain. Behavior-consensus is what stigmergic systems produce: agents agree because their independent processes converged on the same trace. Behavior-consensus is informative about the underlying problem; argument-consensus is informative only about the chain that was followed.

---

## 3. Information Partitioning as the Diversity Engine

If agents must not see each other's reasoning, where does the diversity required for ensemble gains come from?

The coin metaphor names the answer. N identical probability machines (the model M) produce N identical distributions when given identical coins. They produce N *different* distributions when given coins with different weights and tilts. The machine is fixed; the conditioning is variable.

For LLM agents this maps onto three intake variables, each of which is compatible with the no-reasoning rule.

**Corpus partitioning at intake.** A retrieved knowledge corpus C is split into K possibly-overlapping partitions {C_1, ..., C_K}. Each scout agent is given a different partition as its grounding context. Scout 1 generates its initial signals from evidence subset C_1; scout 2 from C_2. Their outputs are drawn from genuinely different posteriors P(y | x_corpus_i, θ) because the conditioning information differs. The signals they deposit reflect what each scout could see, not the model's average response to the full corpus.

**Sampling strategy differentiation.** Downstream agents (foragers, critics, haters) draw from the shared signal store using *different* strategies by construction. Forager A uses stratified sampling over the extremes (one weak signal and one strong signal); forager B uses medium-strength only; forager C uses recently-deposited only; forager D uses under-visited only. Each strategy reveals a different slice of the store; each agent acts on its slice without ever observing the others' slices. The same signal store presents itself differently to differently-tuned samplers.

**Role-conditioned prompting.** Compatible with stigmergy because a role prompt is instruction, not another agent's reasoning. A Hater agent is given a consensus-cluster summary — a distributional view of the signal field, "you observe N similar signals with average strength S" — and asked to challenge the cluster. The hater sees the gradient of the pheromone field, never the minds that produced individual deposits.

The strict constraint that distinguishes this design from deliberative multi-agent systems is the *no-leak rule*: an agent's context never contains another agent's reasoning. Signal content is permitted (it is the trace). Signal IDs and provenance graph structure are permitted (they are observable colony state). Rendered ancestry text is forbidden. Dialogue threads are forbidden. Other agents' chains-of-thought are forbidden. The Critic does not see the Forager's working; the Hater does not see the Critic's reasoning; the Synthesizer does not see anyone's reasoning, only the surviving signals as artifacts.

---

## 4. Mathematical Framework

The signal store is a typed directed acyclic graph G = (V, E) where each node v ∈ V is a signal with state (type, content, strength, parent_id, timestamp). Edges encode parent-child provenance.

**Strength dynamics.** Each signal v has a scalar strength s_v ∈ [0, 1]. Per iteration, all signals decay multiplicatively: s_v ← s_v · (1 − λ), with decay rate λ ∈ (0, 1). Signals can be amplified through corroboration: s_v ← min(1, s_v · α), with amplification factor α ≥ 1. Signals below a prune threshold τ are removed. The continuous-time analog is ds/dt = (k · c(t) − λ) · s, where c(t) is the corroboration rate; equilibria are at s = 1 when amplification dominates and s → 0 when decay dominates. The qualitative behavior is winner-take-all in the corroboration-dominated regime and full evaporation otherwise.

**Provenance-aware initialization.** When a child signal is deposited with parent p, the system queries p's ancestry for VERIFICATION signals. If the average verification strength across them is at least 0.7, the child's initial strength is boosted by a factor of 1 + β · avg, with β = 0.2 (yielding 1.14 to 1.20 boost). This propagates trust through the provenance graph: claims grounded in verified ancestry begin with a strength advantage, not because of who deposited them but because of what they descend from.

**Sample weighting with exploration bonus.** When an agent samples k signals of type T, the candidate set is filtered to type-T signals and each receives a weight w_v = (s_v + ε) + γ · (1 − v_visits / max_visits), where ε is a smoothing constant preventing strength-0 starvation, γ is the exploration bonus, and v_visits is incremented on every sample (so popular signals lose exploration weight over time). Sampling is multinomial over normalized weights. This is a linear blending of strength-driven exploitation and visit-driven exploration.

**Diversity metric.** To monitor whether information partitioning is producing the intended differentiation, we compute pairwise Jaccard distance between agents' context sets:

  d(A_i, A_j) = 1 − |X_i ∩ X_j| / |X_i ∪ X_j|

where X_i is the set of (corpus_chunk_ids ∪ sampled_signal_ids) seen by agent i during a round. Pairwise distances near 1 indicate genuine partitioning; distances near 0 indicate echo-chamber convergence. We log this per round as a swarm health metric and treat it as a leading indicator of whether emergent diversity is present, independent of any judgment about output quality.

**Why behavior-consensus is robust.** Suppose two independent instantiations of M each draw signals from the store using different sampling strategies and arrive at compatible signals s_1, s_2 (compatible meaning either deposited similarly enough to dedupe, or amplifying the same parent). The probability of both arriving at this compatibility under uncorrelated evidence conditioning is approximately the product of each event's marginal probability. If the underlying claim were a confabulation, this joint probability would be small. Behavior-consensus therefore acts as a noisy but informative test of whether the converged-on signal reflects something real about the input. Argument-consensus has no such property because the conditioning is shared.

---

## 5. Architecture

The reference implementation in `Attempt At Cleaning/` consists of:

- **`core/signal_store.py`** — Typed DAG with strength dynamics, similarity dedup, stratified and weighted sampling. Exposes only signal *content* and IDs to consumers; never renders parent reasoning chains. Removes the dialogue / response / get_dialogue_thread surface present in the parent project.

- **`core/intake.py`** — Corpus partitioner. Takes a raw text corpus, chunks it, and assigns chunks to scout agents with controlled overlap. Each scout receives an immutable partition reference; partitions are not crossed during a round.

- **`core/diversity.py`** — Pairwise Jaccard distance computation over agents' context sets, plus a simple aggregate score for the swarm.

- **`core/sampling.py`** — Library of sampling strategies (stratified extremes, medium-only, recent-only, under-visited-only). Agents are constructed with a strategy reference, and different agents in the same role get different strategies.

- **`core/llm.py`** — A single LLM wrapper around DeepSeek-R1-Distill-Qwen-7B with 4-bit quantization, behind an asyncio semaphore sized for the 6GB VRAM budget. Includes a `MockLLM` for development and CI without GPU.

- **`agents/`** — One file per role. Every agent constructs prompts using only its own corpus partition (if a scout) plus its own sampled signals (their `content` only — never `metadata['parent_content']`, never rendered provenance). Agent-to-agent communication happens exclusively through `SignalStore.deposit()`.

- **`run_swarm.py`** — Pipeline orchestrator. Per round: research-and-partition phase, then concurrent agent execution, then decay/prune, then diversity logging.

What is *removed* from the parent project, and why:

- The `swarm/colony/` module (PheromoneField, FungusGarden, TrailNetwork, CasteSystem, Stridulation): not wired into the original pipeline, decorative biomimicry. Removed.
- `dialogue_coordinator.py` and Signal.responses / `get_dialogue_thread`: explicitly anti-stigmergic — they let agents reconstruct argument chains. Removed.
- `evaluate_insights_enhanced` and the full provenance-rendering critic mode: dead code in the original, also a stigmergy violation by design. Removed.
- The mode/phase/task-type/signal-type quadruple classification system: collapsed to a single task config and universal signal types. Removed.
- The `deposit_with_context` helper that injected `parent_content` into child metadata: removed. Children may know they have a parent; they may not see what the parent said.
- All explicit `asyncio.sleep(random.uniform(...))` delays: removed. Agents yield via the event loop properly.

---

## 6. Hardware Constraints as Design Discipline

The 6GB VRAM budget is not a limitation we work around. It is a constraint we exploit.

DeepSeek-R1-Distill-Qwen-7B in 4-bit NF4 occupies approximately 4.5 GB, leaving roughly 1.5 GB for KV cache and activations. This caps practical context length per agent at a few thousand tokens. We cannot dump the full 100K-word retrieved corpus into every agent's prompt. We must partition.

This is the design-discipline observation: the hardware constraint *forces* the architectural commitment we wanted on principle. A reference implementation that ran on an H100 could trivially feed every agent the full corpus and still fit; it would have to choose, against expedience, to partition information for diversity reasons. On a 3060 Laptop, partitioning is the only path. The constraint and the principle align.

The same observation applies to the agent population. We cannot run twelve concurrent generations on this GPU; we run one or two at a time behind a semaphore. This naturally pushes the design toward fewer, more differentiated agents rather than larger, redundant populations. A small population of agents conditioned on genuinely disjoint evidence will outperform a large population of agents redundantly conditioned on the same evidence.

---

## 7. What This Implementation Does Differently

Compared to the parent project on the `analyze-repo-architecture` branch, the cleaned implementation:

- Replaces the deliberative dialogue mechanisms with strict signal-as-artifact communication.
- Adds a corpus partitioner that distributes evidence across scouts.
- Adds explicit sampling strategy differentiation across same-role agents.
- Adds a Jaccard diversity metric logged per round.
- Removes ~500 lines of dead code (mode/phase classification, document-mode workflow without entry point, dialogue coordinator, response thread machinery).
- Removes the unused colony/biomimicry module entirely.
- Keeps the strength dynamics, provenance boost, and stratified sampling that the parent project had built but did not fully exploit.

The headline structural change is that the *signal store stops being a context-rendering substrate and becomes a pure trace medium*. An agent never receives a rendering of "here is what your parent said and here is the chain it came from." It receives a list of signals, each carrying only its content and its metadata that does not include reasoning text. The agent re-reasons from scratch every time.

---

## 8. Open Questions

Several things remain unmeasured and should be the subject of empirical work.

**Does behavior-consensus actually produce better outputs than argument-consensus on standard benchmarks?** The theoretical argument is that it should, because behavior-consensus requires independent rediscovery while argument-consensus requires only chain-following. The empirical test requires running this implementation and a deliberative baseline on the same task suite (TruthfulQA, debate evals, an analysis benchmark) and comparing both output quality and inter-agent error correlation. We do not yet have these numbers.

**How sensitive is the behavior to corpus partitioning strategy?** Disjoint partitions, overlapping partitions, semantically clustered partitions, random partitions — these probably differ in their effects. The current implementation uses non-overlapping random chunking as a baseline; controlled comparison is open.

**How small can the agent population go?** The parent project ran with 4+4+2+2+1+1 agents. With genuinely partitioned information, fewer agents per role may suffice. The lower bound is presumably the number of distinct corpus partitions needed to cover the evidence space, which depends on the task.

**Where does the no-leak rule actually fail?** There are edge cases. A Synthesizer needs to combine surviving signals into a coherent final output, which inevitably involves the model performing argument synthesis over them. Is this a stigmergy violation? We treat it as the necessary read-out step of the system rather than as agent-to-agent communication, but the line is not perfectly clean.

**Does the model's pretraining data leak into pseudo-reasoning across agents?** Two agents conditioned on different evidence may still arrive at correlated outputs because both have memorized the same text from training. This is a deeper independence problem than information partitioning solves, and we have no answer to it beyond "use a stronger base model."

**Is the provenance boost actually load-bearing?** It is the most novel single mechanism in the original project, but its empirical contribution to output quality is unmeasured. Ablation testing is straightforward and not yet done.

The reference implementation is intended to make these questions concrete and answerable, not to claim the answers are already known.

---

## References (informal)

- Du, Yilun, et al. "Improving Factuality and Reasoning in Language Models through Multiagent Debate." 2023.
- Microsoft Research. "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation Framework." 2023.
- Hölldobler & Wilson. *The Ants*. Belknap Press, 1990. (For the canonical biological reference on stigmergic coordination in social insects.)
- Theraulaz & Bonabeau. "A Brief History of Stigmergy." *Artificial Life* 5.2 (1999).
- Breiman, Leo. "Bagging Predictors." *Machine Learning* 24, 1996. (For the classical statement of why ensemble diversity matters.)

---

*This document accompanies the `Attempt At Cleaning/` reference implementation. It is intended as the framing for an eventual research paper rather than as a finished one. Empirical evaluation and ablation studies are the next steps.*
