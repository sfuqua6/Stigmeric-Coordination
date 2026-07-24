# Over-context eval — the decisive A-vs-F experiment (implementation spec)

Why this exists: the 2026 compute-matched literature (arXiv:2604.02460 DPI argument; 2606.13003) says multi-agent systems win only where a single call's effective context utilization degrades. That is exactly the swarm's compression thesis (composer input stays O(K×brief) regardless of corpus size). The current prompt set is parametric-answerable, so A can never beat B/E/F on it for structural reasons. This experiment is the first one the architecture can actually win — and if A loses it, the architecture has no niche (critique loop 2026-07-06, §8).

## Design

Three evidence-pack scales per prompt: **1×, 4×, 16×** the condition-F model's usable context (measure in chars ≈ 4×tokens; for llama-3.1-8b-instant 128K ctx use packs of ~120K/500K/2M chars; for local 7B with 8K ctx use ~24K/96K/384K chars — scale off `--model`).

- **Pack construction** (`eval/packs.py`, new): for each prompt, retrieve aggressively (CompositeRetriever + per-facet searches, no MMR cut) and concatenate chunk texts until the target size; persist to `eval/packs/<pid>_<scale>.jsonl` (one chunk per line: text, source_tag, url). Deterministic once built — build once, reuse across conditions and runs (fairness requires A and F see the SAME pack).
- **Condition A (swarm)**: feed the pack as the corpus (bypass live retrieval: new `--corpus=pack:<path>` mode in run_swarm that loads chunks from the pack file and runs partition_for_scouts / the new continuous partition wiring over it; disable per-action web search via existing search-budget env so the pack is the only evidence).
- **Condition F (single-call RAG)**: same pack, naive fill — stuff chunks in pack order into the prompt until the model context is full (this is the honest practitioner baseline; do NOT give F a smart retriever over the pack at first — that's a separate condition F+ later).
- **B/E unchanged** (no evidence) as anchors.

## Prompts (pack-dependent by construction)

Parametric-answerable debate prompts cannot show the effect. Use prompts whose answer quality depends on pack-specific detail, e.g.:
- "Across the provided reports, which three factors most consistently explain X, and where do the sources disagree?" (synthesis-with-conflict)
- "What does the provided evidence say about Y that contradicts the common assumption Z?" (needle-against-prior)
- Construct 8+ prompts; pre-register in eval/prompts.py as `OVERCONTEXT_SET` with `must_include` checklists mined from the packs (objective anchor, judge-free).

## Metrics & reading

- Primary: A-vs-F win-rate (both-orders judge, Wilson) **per scale**. Prediction if the thesis is right: A≈F at 1×, A>F at 4×/16× with the gap growing.
- Secondary: `must_include` checklist recall per condition per scale (judge-free); cost multiple; wall-clock.
- Kill criterion (pre-registered): if A fails to beat F at 16× with Wilson lower bound > 0.5 at n ≥ 8, the compression thesis is falsified in its own best-case regime.

## Order of work
1. `eval/packs.py` + `--corpus=pack:` loader (small).
2. F pack-stuffing variant in ab_harness (`gen_direct_rag` already exists; add pack path + naive fill).
3. OVERCONTEXT_SET prompts + checklists.
4. One MOCK plumbing pass, then a real run at `--mini 8 --conditions ABEF` per scale.
