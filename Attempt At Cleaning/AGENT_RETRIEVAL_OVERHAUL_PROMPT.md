# Claude Code prompt — Agent-role retrieval overhaul memo

Paste everything below the `---` line into Claude Code from the repository root
(`C:\Users\agsse\Downloads\ai_swarm_mechanics-main (4)\ai_swarm_mechanics-main`).

This is the second design memo in a sequence; the first
(`SYNTHESIZER_OVERHAUL.md`) covers post-exploration composition. This one
covers per-role retrieval and factuality. The two memos compose — synthesizer
Improvement 7 ("validators as first-class synthesis inputs") depends on the
validator changes proposed here.

---

You are working in the `Attempt At Cleaning/` folder of a stigmergic multi-agent
swarm codebase. Read the relevant files end-to-end before proposing anything:

- `core/worker_pool.py` (search budget, action gating, query-planner calls)
- `core/query_planner.py` (current `plan_scout_query`, `plan_develop_query`,
  `plan_validate_query`, `find_cached_query`)
- `core/search_tool.py` (the retrieval cascade)
- `core/retrieval.py` (CompositeRetriever, CachedRetriever)
- `agents/scout.py` (`_compose_query`)
- `agents/validator.py` (`_extract_keyphrase`, `_wiki_lookup`, `parse`)
- `agents/developer.py` (developer's sparse-cluster search hook)
- `core/actions.py` (`validate_prompt`, `validate_parse`)

Treat these files as the subject. Do NOT modify code on this pass — your
deliverable is a single markdown document.

## Task

Write a design memo that maps four published retrieval/factuality techniques
onto specific agent roles in this codebase. Save it to:

    Attempt At Cleaning/docs/AGENT_RETRIEVAL_OVERHAUL.md

The driving framing is unchanged from the synthesizer memo: **"enable a system
that goes beyond its base model's parameters."** Retrieval is the load-bearing
mechanism — the swarm becomes more than its parts when the per-role retrieval
strategies bring in evidence the base LM did not see during training. The four
techniques below each lift retrieval quality along a distinct axis.

## The four techniques

The memo must accurately summarize each technique before assigning it to a
role. Read the actual papers if accessible; otherwise use these condensed
descriptions:

### Technique 1 — SAFE (Wei et al., 2024, Google DeepMind)

**Paper:** "Long-form factuality in large language models."

**Core mechanism:** A long-form claim is decomposed into *atomic facts* by an
LLM. Each atomic fact is then converted into a short, targeted search query
(typically 3–6 words). Each query is evaluated independently, and the
factuality of the original long claim is the aggregate factuality across its
atoms.

**Key insight relevant here:** A single cluster representative typically
encodes 3–6 atomic facts. The current Validator extracts one 5-word keyphrase
from the whole rep — far too coarse. SAFE-style decomposition would issue
multiple short queries per cluster, score each atom independently, and
aggregate. This changes verification from "does this snippet confirm the
whole claim" to "how many of the claim's atomic facts are independently
supported."

### Technique 2 — HyDE (Gao et al., 2022, SIGIR-track work)

**Paper:** "Precise Zero-Shot Dense Retrieval without Relevance Labels."

**Core mechanism:** Hypothetical Document Embeddings. Instead of searching
with the claim or question text, an LLM generates a hypothetical document
that would *answer* or *confirm* the claim. The retrieval target is the
hypothetical document (its embedding, in the dense-retrieval setting).

**Key insight relevant here:** Search-engine relevance is asymmetric — a
question and the document that answers it use different vocabularies. HyDE
shifts the query distribution from "what the asker wrote" to "what a
confirming document would say." Counterintuitively, this often outperforms
direct claim-as-query.

**Adaptation note:** This codebase uses web search (DDG / Cohere / Tavily
cascade), not pure dense retrieval. The HyDE adaptation here is: generate a
short hypothetical-document seed (≤ 50 words), then extract its salient
content terms as the search query. For dense-retrieval paths (Cohere
re-rank), use the hypothetical document's embedding directly.

### Technique 3 — Step-Back Prompting (Zheng et al., ICLR 2024)

**Paper:** "Take a Step Back: Evoking Reasoning via Abstraction in Large
Language Models."

**Core mechanism:** Before answering or searching on a specific question,
the model generates a more abstract version of it. The abstraction is
queried; results inform the specific answer. The "step back" exposes
background concepts the specific question takes for granted.

**Key insight relevant here:** A claim like "ChatGPT Plus offers significant
advantages for versatile assistance" steps back to "ChatGPT Plus capabilities
and pricing" — a query a search engine can satisfy. The unabridged claim is
too composite and editorialized to retrieve cleanly. Step-back gives the
retriever the right altitude.

### Technique 4 — FLARE (Jiang et al., EMNLP 2023)

**Paper:** "Active Retrieval Augmented Generation."

**Core mechanism:** Selective retrieval triggered by generation uncertainty.
The model begins generating; when the next-token confidence (or look-ahead
confidence over a span) drops below a threshold, generation halts and
triggers a retrieval, then resumes. Retrieval fires only where the model
needs it.

**Key insight relevant here:** The current pipeline fires retrieval
speculatively on almost every Scout/Developer/Validator iteration —
~100+ searches per run regardless of whether they help. Section 5 of the
prior critique flagged this as a real cost driver. FLARE-style triggering
would reduce search to the subset where external grounding actually
matters.

## The composition

The four techniques compose in a specific order. The memo must articulate
this composition explicitly:

```
1. FLARE       → decides WHETHER to search at all (uncertainty trigger)
2. Step-Back   → chooses the ALTITUDE of the query (abstract vs. specific)
3. HyDE        → shapes the QUERY CONTENT (hypothetical doc → keywords/embedding)
4. SAFE        → structures the VERDICT (atomic decomposition + per-fact judgment)
```

FLARE gates the pipeline; Step-Back and HyDE shape what gets searched when
the gate opens; SAFE structures how the retrieved evidence is judged. Each
agent role uses a subset of these in a specific order.

## Role assignments

For each role below, the memo must specify: which techniques apply, how
they compose, the file-and-function points where changes land, and the
beyond-params argument for that role.

### Scout — Step-Back + HyDE + FLARE (light)

Current behavior (`agents/scout.py:_compose_query`): rotates through a fixed
list of phrasing variants ("X evidence", "X arguments", "counterarguments
to X", "recent research on X", etc.). The rotation is deterministic and
keyed on `(scout_index, iter_idx)`.

Proposed:
- **Step-back**: before querying, the scout generates a more abstract
  version of the user prompt. Example: "Argue both sides of the following
  thesis: Climate action is necessary" → step-back to "climate policy
  scientific consensus" or "climate mitigation economic impact." The
  step-back is a one-call LLM operation.
- **HyDE**: at lower abstraction levels, the scout generates a 40–80
  word hypothetical paragraph "a Wikipedia paragraph addressing [topic]
  would discuss…", extracts its salient noun phrases, and uses those as
  the search query.
- **FLARE-light**: scouts should still issue some speculative queries
  (exploration is their job), but cap them by *novelty* — if the
  step-back abstraction has already been queried by another scout (the
  `pool_state.served_queries` table), skip and pick a different
  abstraction altitude.

Beyond-params argument: a single forward pass of the base LM cannot
choose its own retrieval altitude or generate its own hypothetical
confirming document and use it to retrieve. The compositional step
introduces evidence the base LM would never have seen at inference.

### Developer (Forager) — FLARE (strict) + HyDE on trigger

Current behavior (`core/worker_pool.py:_gather_target` for `DEVELOP`):
fires retrieval whenever the target cluster has < 2 SUPPORT children
(the "sparse-cluster search hook"). The trigger is structural, not
uncertainty-based.

Proposed:
- **FLARE-strict**: the structural trigger (< 2 supports) is necessary
  but not sufficient. Add an uncertainty trigger: the developer first
  generates a draft SUPPORT, computes its self-rated confidence (a
  follow-up LLM call: "rate your confidence in the factual basis of
  the deposit you just produced, 0–1"), and only retrieves if
  confidence < τ. The draft is discarded if retrieval changes the
  story.
- **HyDE on trigger**: when retrieval fires, the query is generated by
  HyDE from the draft SUPPORT itself — "a sentence in a reliable source
  that would corroborate this support would say…". The hypothetical
  sentence's content terms are the query.
- **Step-back fallback**: if HyDE returns nothing useful (zero results),
  step back to the parent INITIAL's abstract topic and retry.

Beyond-params argument: the developer's draft is an output of a single
forward pass; retrieving only when *that pass* signals uncertainty,
shaped by a *second pass* (HyDE), then re-conditioning, is exactly
mechanism (2) from the synthesizer overhaul — verifier-augmented
generation, here at the per-deposit grain.

### Validator — SAFE (primary) + Step-Back + HyDE

Current behavior (`agents/validator.py:_extract_keyphrase` and the
non-factual `validate_parse` branch in `core/actions.py:450-523`):
extracts a single 5-word keyphrase from the whole cluster rep; issues
one query; receives one snippet; judges the whole claim as
`{engages, quality}` or `{supports, confidence}`.

Proposed (this is the heaviest redesign of the four):
- **SAFE atomic decomposition**: a new LLM call decomposes the cluster
  rep into 3–6 atomic facts. Each fact is one verifiable proposition.
  Example: "ChatGPT Plus offers significant advantages for versatile
  assistance" decomposes to: (a) "ChatGPT Plus is a paid tier of
  ChatGPT"; (b) "ChatGPT Plus has advantages over the free tier"; (c)
  "Those advantages relate to versatility / capability." Each atom
  becomes one VERIFICATION signal (or one entry within a single
  VERIFICATION's structured metadata).
- **Step-back per atom**: each atom's query is composed by stepping
  back to the atom's topic, not by paraphrasing the atom.
- **HyDE per atom**: the query is HyDE-generated from the atom — "a
  reliable source confirming [atom] would say…".
- **Aggregation**: cluster-level `verification_score` becomes a mean
  (or weighted mean) over atom-level scores. Convergence detector's
  quality gate consumes this aggregated score; the FLARE-style gate
  for Developer above is informed by it.

The schema of a VERIFICATION signal extends:

```json
{
  "type": "VERIFICATION",
  "content": "<one-sentence overall assessment>",
  "strength": 0.62,
  "metadata": {
    "atoms": [
      {"text": "ChatGPT Plus is a paid tier", "score": 0.95, "query": "ChatGPT Plus pricing", "snippet_tag": "openai.com"},
      {"text": "ChatGPT Plus has advantages over free", "score": 0.70, "query": "ChatGPT Plus features comparison", "snippet_tag": "wired.com"},
      ...
    ],
    "aggregation": "mean"
  }
}
```

This is the synthesizer-side hook for Improvement 7 of the synthesizer
overhaul: the synthesizer's `_get_external_context` is replaced by an
aggregator over each cluster's VERIFICATION atoms.

Beyond-params argument: a single forward pass cannot decompose-search-
aggregate. The base LM judging the whole claim against one snippet has
one shot at the whole composite; SAFE decomposition gives it 3–6 shots
at narrower propositions, where it is empirically more reliable. The
aggregation across atoms is non-trivial and cannot be reproduced by
any single conditioning.

### Worker pool search budget — FLARE replaces static rate limiter

Current behavior (`core/worker_pool.py:try_reserve_search`): a sliding
window of `SEARCH_BUDGET_PER_WINDOW = 6` searches per 5 seconds. This
is a *rate* limit, not an *uncertainty* limit.

Proposed:
- Keep the rate limit as a hard ceiling (cost protection).
- Add a per-iteration uncertainty check: a worker's would-be deposit
  gets a draft + a confidence score from the LM; the search is only
  reserved if confidence < τ AND the rate budget has room.
- The two together: rate cap is the cost wall; uncertainty trigger is
  the value floor. Most attempts skip the search entirely; the ones
  that do search are the ones that need it.

Beyond-params argument: this is FLARE at the system level rather than
the token level. The system is *active* about retrieval rather than
passive; capability scales with the calibration of the confidence
estimate, which is itself a model output.

## Required design choices the memo must make

For each, give the recommendation and the reasoning:

1. **Confidence estimation for FLARE.** Options: (a) self-rating LLM
   call ("rate your confidence 0–1"), (b) token-level log-probs from
   vLLM (`logprobs` API; not available on MockLLM), (c) entropy over
   the top-k tokens, (d) cluster-level structural proxy
   (`support_diversity` and `verification_score` of the cluster the
   deposit is targeting). Recommend an option per role. Justify
   given the latency budget — every FLARE trigger that requires an
   extra LLM call cuts throughput.

2. **Step-back depth.** One step? Two? Recursive until a query of K
   words is reached? Recommend a default and a per-role override.

3. **HyDE document length.** SAFE's queries are 3–6 words; HyDE
   classically generates 100–200 word documents. The codebase's web
   search performs poorly on long queries — recommend a budget. A
   reasonable compromise: HyDE generates 40–80 words; extract top-5
   noun phrases as the literal query; pass the full HyDE text as the
   *embedding* for dense rerank when Cohere is available.

4. **SAFE atom budget per claim.** 3 atoms? 6? Recommend a default
   based on cluster-rep length, and a cap to bound LLM cost.

5. **Atom aggregation function.** Mean is the obvious default but
   gives equal weight to load-bearing and incidental atoms. Consider
   weighting by atom *centrality* to the original claim (LLM-rated
   at decomposition time) or by atom *score variance* (low variance
   = consensus is high). Recommend an aggregation and justify.

6. **Backoff when retrieval returns nothing.** Step-back to a higher
   abstraction? Skip the atom? Fall back to model parametric
   knowledge with a discount? Recommend behavior and its score
   semantics.

## Preconditions

The memo must list what has to change in supporting infrastructure
before these per-role changes pay off:

- `Signal.metadata` schema needs to carry the atom-level structure on
  VERIFICATION signals. Currently `metadata` is freeform dict; document
  the canonical fields the projection should read.
- The `search_tool` cascade needs to surface raw embedding output when
  Cohere re-rank is enabled, so HyDE's dense-mode retrieval has
  something to consume.
- `pool_state.served_queries` should index by *embedding* (or by
  step-back abstraction key) in addition to exact-string match, so two
  scouts asking semantically equivalent queries hit the cache.
- The LLM backends need a `logprobs` accessor for FLARE-strict
  (vLLM has it; GGUF and Mock do not). Recommend a confidence-call
  fallback for backends without logprobs.

## Falsifiability

The memo must end with a table mapping each technique to a controlled
A/B comparison, the expected effect, and the failure mode. The
benchmark needs a factuality target with external ground truth
(LongFact for SAFE, plain TruthfulQA or FactScore for component
checks), a fixed base LM held constant, and a single-call baseline
that uses neither retrieval nor decomposition.

For SAFE specifically, the comparison should report the standard
SAFE metric: precision × recall over supported / unsupported atomic
claims, against a ground-truth atom set.

For FLARE specifically, report search count and quality delta — the
claim is that selective retrieval matches the quality of speculative
retrieval at a fraction of the cost. The cost frontier matters more
than the quality ceiling.

## Memo structure (use this skeleton)

```
# Agent-Role Retrieval Overhaul: SAFE + HyDE + Step-Back + FLARE

## 1. Problem framing
   Current retrieval pattern. Why per-role redesign rather than central refactor.

## 2. The four techniques
   One subsection per technique. Faithful summary, not aspirational.

## 3. The composition
   FLARE gates → Step-Back abstracts → HyDE shapes → SAFE judges.
   Why this order. Where each technique sits in the pipeline.

## 4. Role assignments
   One subsection per role: Scout, Developer, Validator, Worker-pool budget.
   Each subsection: current behavior with file:line citations; proposed
   behavior; beyond-params argument; file changes; risks.

## 5. Required design choices
   Confidence source, step-back depth, HyDE length, SAFE atom budget,
   aggregation function, retrieval-empty backoff. One recommendation each,
   with reasoning.

## 6. Schema and infrastructure preconditions
   VERIFICATION metadata extension; embedding cache; logprobs accessor;
   served_queries semantic indexing.

## 7. Composition with the synthesizer overhaul
   Where Improvement 7 of the synthesizer memo depends on the changes
   here. Where the edge graph (Improvement 1 of the synthesizer memo)
   benefits from richer VERIFICATION metadata.

## 8. Falsifiability
   A/B comparison harness. Per-technique table.

## 9. Sequencing
   Suggested implementation order. SAFE decomposition first (load-bearing
   for downstream improvements). Then Step-Back and HyDE in the query
   planner. FLARE last because it depends on a calibrated confidence
   source that itself needs evaluation.

## 10. Open questions
   The questions you cannot answer from reading the code, that an
   experimental run would have to settle.
```

## Style and constraints

- Doctoral-level prose. Each technique gets summarized in its own terms,
  then mapped to the codebase's specific file points.
- When you cite a function or threshold, cite the file path and the line
  range you read.
- Do not invent file paths, function names, or line numbers — read the
  code.
- Do not implement code. The deliverable is the markdown document only.
- Be honest about adaptation gaps. HyDE was designed for dense retrieval;
  this codebase uses web search. State the adaptation and what it costs.
  Step-back was studied on QA, not on stigmergic exploration. State the
  transfer assumption.
- Write in CommonMark. Use code fences for code references, blockquotes
  for paper claims.
- Length target: 5,000–8,000 words.
- Every role-assignment subsection must explicitly answer: *what does the
  base LM, in a single forward pass, fail to do that this technique
  enables?*

## Verification step

After writing the memo, run a self-check:

1. Does every cited file path exist? Open each and confirm.
2. Does every cited line range match the content you describe?
3. Is each of the four techniques summarized accurately *before* being
   adapted? The adaptation paragraphs must not silently rewrite the
   paper.
4. Does each role assignment have all four parts (current, proposed,
   beyond-params, files)?
5. Does the composition section actually compose? Could a reader trace
   the path of one query (e.g., from a Developer's draft SUPPORT through
   FLARE → Step-Back → HyDE → snippet → SAFE-style atom aggregation)
   end-to-end from the memo alone?

Report which preconditions you verified and which composition path you
traced. Do not modify the code.
