# Agent-Role Retrieval Overhaul: SAFE + HyDE + Step-Back + FLARE

**Document scope:** `Attempt At Cleaning/` codebase only.
**Companion:** `SYNTHESIZER_OVERHAUL.md` (post-exploration composition).
**Purpose:** Map four published retrieval/factuality techniques onto per-role
retrieval strategies. No code changes are made here; this is a design memo.

---

## 1. Problem Framing

### Current retrieval pattern

Every agent role that triggers a search does so through a thin, stateless
query-construction path. The Scout's `_compose_query` method
(`agents/scout.py:192-222`) rotates deterministically through eight fixed
phrasings of the user prompt — `base`, `base evidence`, `base arguments`,
`base criticism`, `base consequences`, `counterarguments to base`,
`recent research on base`, `case studies of base` — keyed on `iter_idx`
with no LLM call involved. The rotation is useful insofar as it avoids
literal duplicates within one worker's history, but the phrasings are
shallow paraphrases at a fixed level of abstraction: they never step back
to a broader topic, nor do they shape the query to look like a confirming
document.

The Developer's sparse-cluster search hook (`agents/developer.py:122-153`)
fires whenever `n_support_children < 2`. Its query is `f"{snippet}
evidence"` where `snippet` is the first eight words of the target INITIAL's
content (line 133). The trigger is purely structural; no uncertainty signal
determines whether the LM actually needs external grounding for this
particular INITIAL.

The Validator's query path (`core/worker_pool.py:783-819`,
`core/query_planner.py:408-410`) issues one call to
`plan_validate_query(target.content)`, which returns
`_extract_sentence_fragment(target_content, max_words=10)` — a 10-word
natural-language fragment representing the entire cluster representative.
That fragment goes to DDG (the only reliably available backend in the
current environment), returns up to three hits, and two of them are
formatted into a single external block that the Validator LM sees alongside
the whole claim. The LM then produces a scalar judgment over the composite
claim against the composite snippet. There is no decomposition: the claim's
multiple atomic propositions are judged all at once, in one forward pass,
against one undifferentiated snippet.

The pool's search budget gate (`core/worker_pool.py:177-187`) is a sliding
window of `SEARCH_BUDGET_PER_WINDOW = 6` live backend calls per
`SEARCH_WINDOW_S = 5.0` seconds. This is a rate limiter, not an uncertainty
selector. It caps DDG hits to prevent response-time blowout but does not
ask whether any particular worker needs external evidence at all.

### Why per-role redesign rather than a central refactor

A single query-planning module shared by all roles would be easier to
maintain, but the four techniques composing this overhaul are not
task-agnostic: FLARE's uncertainty trigger fires at very different rates for
a Scout (high uncertainty; its job is exploration) versus a Validator (low
uncertainty if its target is factually unambiguous); SAFE decomposition is
meaningful for Validators (composite factual claims) but wasteful for Scouts
(first-order observations from raw evidence). The composition therefore must
be role-specific. A unified `query_planner.py` can supply utility
functions; the policy of which utilities to invoke and in what order belongs
in the role's own gather-target path.

### The beyond-params thesis

The framing that motivates all four techniques is the same: a single forward
pass of the base LM operates only on parameters fixed at training time. The
swarm becomes more than the sum of its base-model calls when retrieval
imports evidence the model did not see at training — or imports it at the
right altitude and in the right form that the base model can actually use.
Each technique lifts one axis of that capability:

- SAFE lifts **verdict resolution** (narrow queries per atomic fact beat a
  single wide query against a composite claim).
- HyDE lifts **query vocabulary** (hypothetical confirming documents share
  the vocabulary of the target corpus, not the vocabulary of the question).
- Step-Back lifts **query altitude** (specific composite claims retrieve
  poorly; their abstract topic retrieves cleanly).
- FLARE lifts **query selectivity** (selective retrieval matched to local
  uncertainty beats speculative retrieval across every iteration).

---

## 2. The Four Techniques

### 2.1 SAFE — Long-Form Factuality via Atomic Decomposition (Wei et al., 2024)

> Wei et al., "Long-form factuality in large language models," Google
> DeepMind, 2024.

**Core mechanism.** A long-form model response is passed to a secondary LM
that decomposes it into *atomic facts* — the smallest independently
verifiable propositions. Each atom is then converted into a short search
query (typically 3–6 words). Each query is evaluated independently against
retrieved evidence, and the response's overall factuality is the aggregate
score across its atoms (precision × recall over supported / unsupported
atoms against a ground-truth atom set).

**Why atoms beat wholes.** A composite claim like "ChatGPT Plus offers
significant advantages for versatile assistance" makes at least three
separable assertions: that ChatGPT Plus is a paid tier, that it has
advantages over the free tier, and that those advantages relate to
versatility. A single retrieved snippet is unlikely to address all three.
An LM judging the composite against a partial snippet makes an inference
over the union; an LM judging each atom against a targeted snippet makes
three narrower inferences, where it is empirically more reliable (Wei et al.
show atom-level judgments have higher precision than composite judgments on
the same ground-truth set).

**What is NOT in the paper.** SAFE's decomposition is run by a capable LM
(GPT-4 class in the paper); running it on the small local model that the
swarm uses for everything else adds LM cost without the same decomposition
quality. The design choices section addresses this.

### 2.2 HyDE — Hypothetical Document Embeddings (Gao et al., 2022)

> Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels,"
> SIGIR 2022 (arXiv 2212.10496).

**Core mechanism.** Instead of encoding the query text and retrieving
documents whose embeddings are nearby, HyDE generates a hypothetical
document that *would answer* the query, encodes that hypothetical document,
and retrieves the real documents nearest to the hypothetical document's
embedding. The intuition is that a question and its answer live in different
regions of embedding space; the hypothetical document lives in the same
region as the real confirming documents.

**Empirical result.** On several BEIR benchmarks, HyDE outperforms
BM25+dense retrieval without any labeled data, including for dense
retrievers trained with contrastive objectives. The gain is largest on tasks
where the question vocabulary is distant from the document vocabulary.

**Adaptation note — web search only.** HyDE was designed for dense
retrieval over a fixed corpus. This codebase's primary backend is
DuckDuckGo (DDG), a keyword-based web search engine; Tavily and Cohere
are gated behind environment variables not currently set. The DDG path
does not accept embedding vectors. The adaptation here is therefore
*lexical HyDE*: the LM generates a short hypothetical paragraph (≤50
words), and its salient noun phrases are extracted and passed as the
literal search query string. This loses the embedding-space matching
guarantee but retains the vocabulary-alignment benefit: a confirming
document's vocabulary shapes the query, not the question's vocabulary.
The cost is one extra LLM call per search trigger; on a 2.7B–7B model
this is a few hundred milliseconds on the target hardware.

### 2.3 Step-Back Prompting (Zheng et al., ICLR 2024)

> Zheng et al., "Take a Step Back: Evoking Reasoning via Abstraction in
> Large Language Models," ICLR 2024.

**Core mechanism.** Before answering or searching on a specific question, a
model is prompted to generate a more abstract version of that question —
stepping back to the background principles or broader category the specific
question instantiates. The abstract version is queried; its results inform
the specific answer. The paper reports consistent gains on Physics, Chemistry,
and MMLU benchmarks versus direct-prompting and chain-of-thought baselines.

**Key insight for retrieval.** Web and Wikipedia search engines are tuned
for noun-phrase queries at middle abstraction. A claim like "renewable energy
could reduce dependency on fossil fuels within the current economic
framework" is too composite and too editorialized for DDG to handle cleanly.
Stepping back to "renewable energy economic transition" or "fossil fuel
dependency reduction" gives the retriever a query it can satisfy. The
step-back produces the right altitude.

**Transfer assumption.** Step-back was studied on factual QA tasks (MMLU,
Physics, Chemistry), where the "step back" is genuinely a background
principle (laws of physics, chemical properties). In the swarm's context,
the step-back is applied to an agent's claim or query, not a QA question.
The assumption is that the same abstraction operation — "what broader topic
does this specific claim instantiate?" — is useful for retrieval even when
the input is an exploratory proposition rather than a question. This is a
transfer assumption that has not been empirically validated in the
stigmergic-exploration setting; see Open Questions.

### 2.4 FLARE — Active Retrieval Augmented Generation (Jiang et al., EMNLP 2023)

> Jiang et al., "Active Retrieval Augmented Generation," EMNLP 2023.

**Core mechanism.** The model generates text iteratively. At each step it
computes the token-level confidence over a forward-looking span (or
equivalently, the perplexity of the draft span). When the confidence of the
draft span falls below a threshold τ, generation pauses, a retrieval is
triggered using the draft span as the query, and generation resumes with the
retrieved evidence appended to context. Retrieval fires only when and where
the model's own uncertainty signals that it needs external grounding.

**Key insight for budgeting.** The current pool fires retrieval
speculatively on almost every Scout and Developer iteration, and on every
Validator iteration, regardless of whether the specific claim or target
benefits from it. FLARE's insight is that *most* forward passes are not
uncertain — the model has sufficient parametric knowledge — and retrieval on
those iterations wastes the search budget (rate-limit capacity and wall time)
on low-marginal-value lookups. Selective retrieval matches speculative
retrieval in quality at a fraction of the cost; the FLARE paper reports
comparable ASQA answer quality at roughly 60% of the retrieval count of a
naive RAG baseline.

**Adaptation note — no token-level logprobs.** FLARE classically uses
token-level log-probabilities to compute span confidence. This codebase's
LLM backends vary: vLLM exposes a `logprobs` API; the GGUF path does not;
MockLLM cannot. The adaptation uses self-rated confidence (a follow-up LLM
call) as the primary signal, with structural proxies as fallbacks.

---

## 3. The Composition

The four techniques compose in a specific order that the codebase should
implement as a layered pipeline:

```
1. FLARE      → decides WHETHER to search at all (uncertainty gate)
2. Step-Back  → chooses the ALTITUDE of the query (abstract vs. specific)
3. HyDE       → shapes the QUERY CONTENT (hypothetical doc → keywords)
4. SAFE       → structures the VERDICT (atomic decomposition + per-atom judgment)
```

**Why this order.**

FLARE is the outermost gate because any LLM cost (Step-Back abstraction
call, HyDE generation call) is wasted if the uncertainty check would not
have triggered retrieval anyway. Step-Back comes before HyDE because the
altitude of the query must be chosen before its vocabulary is shaped —
a HyDE call on a too-specific claim produces a hypothetical document at the
wrong altitude and retrieves off-topic snippets. SAFE comes last because
it operates on the verdict stage: once retrieved snippets are in hand,
SAFE decomposes the claim and issues atom-level queries against those
snippets (or follow-up queries if atoms remain unaddressed). SAFE cannot
run before retrieval because it needs something to match atoms against.

**Tracing one query end-to-end.**

A Developer targets an underserved INITIAL: "Renewable energy transitions
are economically constrained by grid infrastructure costs." The developer
generates a draft SUPPORT: "Grid modernisation investment requirements make
rapid renewable deployment economically infeasible in developing economies."
FLARE check: the developer self-rates confidence at 0.38 (below τ=0.6) —
retrieval triggers. Step-Back: "grid infrastructure renewable energy
economics" (abstracted from the specific developing-economies claim).
HyDE: "A report on renewable energy transition costs would note that grid
expansion capital requirements are the primary barrier to solar and wind
adoption in low-income countries, with estimates in the range of $800B
annually through 2050." Salient noun phrases: "grid expansion capital
requirements", "renewable energy transition costs", "solar wind adoption
developing countries". Query issued: "grid expansion capital requirements
renewable energy developing countries". DDG returns 5 snippets. SAFE
(at Validator time, not Developer time — see role assignments): the
cluster representative is decomposed into 3 atoms; each atom is queried
independently and scored against retrieved snippets; the cluster's
verification score is the weighted mean across atoms. The Developer's draft
SUPPORT, now conditioned on the retrieved snippets, is deposited with
strength 0.6.

---

## 4. Role Assignments

### 4.1 Scout — Step-Back + HyDE + FLARE-light

#### Current behavior

`agents/scout.py:_compose_query` (lines 192-222) is the sole query-
construction site. It takes `iter_idx` and `prior_queries` as inputs,
builds a fixed list of eight phrasings, and returns whichever one has not
yet appeared in `prior_queries`. No LLM call is made; no abstraction is
performed; the phrasing rotation is deterministic and keyed only on
`iter_idx`. The resulting queries — e.g., `"{base} evidence"`,
`"counterarguments to {base}"` — are task-agnostic paraphrases of the
original prompt at a fixed, low abstraction level.

In the worker-pool path (`core/worker_pool.py:643-700`), the scout
additionally calls `core/query_planner.plan_scout_query` (lines 328-372),
which adds stance-modified variants and sentence fragments from existing
high-strength INITIALs. This is an improvement over the pure rotation in
`agents/scout.py`, but neither path generates queries at a different
abstraction level or conditions query vocabulary on what a confirming
document would say.

#### Proposed behavior

**Step-back.** Before querying, the scout makes one LLM call that takes the
user prompt and asks: "What is the broader topic or background principle
that this prompt instantiates? Respond with a 4–8 word phrase." The
step-back phrase becomes the top candidate for the scout's query in the
current iteration. On subsequent iterations, the scout tries step-backs at
different altitudes (one step back, two steps back) so it progressively
explores broader context rather than repeating the same abstraction.

**HyDE.** At lower abstraction levels (near-specific queries), the scout
makes a second LLM call: "Write a 40–50 word paragraph that a reliable
Wikipedia article would contain in addressing [step-back phrase]." The
salient noun phrases of this hypothetical paragraph become the literal
query. Because DDG is a keyword engine, the noun phrases are extracted
rather than the raw paragraph text.

**FLARE-light.** Scouts are exploration agents; their job is to search
speculatively. A strict FLARE gate is inappropriate. The FLARE-light
adaptation here is: if the pool's `served_queries` table already contains a
semantically equivalent query at the step-back abstraction level, the scout
skips search for this iteration and picks a different abstraction altitude.
The dedup check should extend `find_cached_query` (`core/query_planner.py:262-286`)
to match by step-back abstraction key in addition to string similarity.

**File changes.** The step-back and HyDE calls are new methods in
`agents/scout.py` (or in `core/query_planner.py` as utilities called by
the Scout's gather path in `core/worker_pool.py:643-700`). The step-back
call modifies the return value of the query planning path before
`_search()` is invoked.

**Beyond-params argument.** A single forward pass of the base LM cannot
choose its own retrieval altitude or generate its own hypothetical
confirming document and then *use* that document's vocabulary as a query
tool. The step-back call produces a query at a level the LM would not have
selected from its phrasing rotation; the HyDE call maps the LM's internal
representation of a confirming document onto DDG's vocabulary space. Both
introduce evidence the base model would not have retrieved through the
current deterministic rotation.

**Risks.** Each scout iteration gains up to two LLM calls before search.
On a 7B model at the hardware target, this could double per-iteration wall
time. The step-back call is short (token budget ≤30); the HyDE call is
larger (token budget ≤80). The FLARE-light dedup check mitigates redundant
search budget expenditure, partially offsetting the added LLM cost.

### 4.2 Developer (Forager) — FLARE-strict + HyDE on trigger

#### Current behavior

`agents/developer.py:sample()` (lines 93-155) sets the search trigger at
a structural threshold: `n_support_children < _SEARCH_TRIGGER_SUPPORT` (= 2,
line 40). When triggered, the query is `f"{snippet} evidence"` where
`snippet` is the first eight words of the target INITIAL's content (line 133).
The same structural trigger is replicated in `core/worker_pool.py:717-757`
for the continuous-pool Developer path, calling `plan_develop_query` instead.
In both paths, no confidence signal determines whether the Developer's own
draft SUPPORT is actually uncertain; the trigger fires whenever any
underserved cluster exists, regardless of whether the LM has strong
parametric knowledge of that cluster's topic.

#### Proposed behavior

**FLARE-strict.** The structural trigger (fewer than 2 SUPPORT children) is
retained as a necessary condition but is no longer sufficient. When the
structural condition fires, the Developer generates a draft SUPPORT first
(one forward pass), then makes a follow-up call to self-rate its confidence:
"Rate your factual confidence in the deposit you just produced, from 0.0 to
1.0. Consider: does this claim depend on specific data, dates, statistics,
or technical facts you are uncertain about?" If confidence ≥ τ (recommended
default: 0.6), the draft is accepted without retrieval. If confidence < τ,
retrieval is triggered, the draft is discarded, and the Developer generates
a new SUPPORT conditioned on the retrieved snippets.

**HyDE on trigger.** When retrieval fires, the query is generated by HyDE
from the *draft SUPPORT*, not from the INITIAL: "A sentence in a reliable
source that would corroborate this support would say…" The hypothetical
sentence's content terms are extracted and used as the DDG query. This
ensures the query vocabulary is aligned with the support's claim rather than
with the INITIAL's framing (which may be at a different level of specificity
than what the developer is about to claim).

**Step-back fallback.** If HyDE returns zero DDG results (query too specific
or too unusual), the Developer falls back to stepping back to the parent
INITIAL's abstract topic and retrying with that simpler query.

**File changes.** The confidence self-rating call is a new method added to
the Developer's gather path in `core/worker_pool.py:703-757` (or in
`agents/developer.py:sample()`). The current sparse-cluster search hook at
`core/worker_pool.py:717-757` is extended with the FLARE-strict condition
before calling `_search`. The query construction at `core/worker_pool.py:726`
(currently `plan_develop_query(target.content, ...)`) is replaced with a
HyDE call using the draft SUPPORT as input.

**Beyond-params argument.** The Developer's draft SUPPORT is the output of
one forward pass; it encodes the LM's parametric knowledge of the INITIAL's
topic. Retrieving only when *that specific pass* signals uncertainty —
shaped by a *second pass* (HyDE) that maps the draft's vocabulary onto
the retrieval corpus — conditions both whether and where the external call
fires. A single forward pass cannot self-assess its own uncertainty and
adaptively trigger retrieval in the same call. The two-pass structure
introduces evidence exactly where the single-pass model would have been
guessing.

**Risks.** The confidence self-rating adds one LLM call per structural
trigger, which fires whenever any INITIAL has fewer than 2 SUPPORT children.
In the cold-start phase (many underserved clusters), this could add many
self-rating calls. The τ threshold is a hyperparameter; if miscalibrated
downward, almost no retrieval fires and the benefit is lost; if calibrated
upward, retrieval fires on every trigger (equivalent to the current
behavior). See Design Choices (§5.1) for the τ recommendation.

### 4.3 Validator — SAFE (primary) + Step-Back + HyDE

#### Current behavior

The Validator's query path in `core/worker_pool.py` (lines 783-819) calls
`core/query_planner.plan_validate_query(target.content)` (lines 408-410),
which returns `_extract_sentence_fragment(target_content, max_words=10)`.
This 10-word fragment represents the *entire* cluster representative in a
single query. The search returns up to three hits; two are formatted into
one external block. The Validator LM then receives the whole cluster
representative alongside that block and produces a scalar `{supports,
confidence}` judgment (factual tasks) or `{engages, quality}` judgment
(non-factual tasks) via the prompts in `core/actions.py:423-463` and the
parser at `core/actions.py:466-539`.

In `agents/validator.py`, the legacy path uses `_extract_keyphrase`
(lines 164-180) to take the first five non-stopword words from the claim —
even coarser than the 10-word fragment in the worker-pool path. The
`_wiki_lookup` function (lines 183-204) calls `search(query, max_results=2)`
and concatenates two snippets into 600 characters.

The combined result is that a cluster representative encoding 3–5 distinct
propositions is verified by a single 10-word (or 5-word) query against one
or two snippets, producing one scalar score for the whole composite.

#### Proposed behavior

This is the heaviest redesign because it transforms the Validator from a
scalar-score producer into a structured, atom-level evidence aggregator.

**SAFE atomic decomposition.** A new LLM call decomposes the cluster
representative into 3–6 atomic facts. The prompt asks: "Break this claim
into individual verifiable propositions. Each proposition should be one
sentence that could be independently checked against a search result. List
at most 6." This call precedes any search and produces the atom list that
drives all subsequent steps.

**Step-back per atom.** Each atom is abstracted by a step-back call: "What
is the broader topic this proposition concerns? Respond with a 4–6 word
phrase." The step-back phrase becomes the basis for query construction for
that atom.

**HyDE per atom.** The query for each atom is generated by HyDE from the
step-back phrase: "A reliable source confirming [step-back phrase] would
contain a sentence like…" The hypothetical sentence's content terms become
the literal DDG query for that atom.

**Per-atom scoring.** Each atom is sent as a separate `validate_prompt` call
(factual or non-factual, per `_NON_FACTUAL_TASKS` gate at
`core/actions.py:420`) with the atom's retrieved snippet. The score is
a `{supports, confidence}` or `{engages, quality}` pair per atom.

**Aggregation.** The cluster-level `verification_score` is the weighted mean
of atom-level scores, where weights are atom centrality ratings produced by
the decomposition call (see Design Choices §5.5). The VERIFICATION signal
deposited in the store carries the full atom structure in its metadata.

**Extended VERIFICATION signal schema:**
```json
{
  "type": "VERIFICATION",
  "content": "<one-sentence overall assessment>",
  "strength": 0.68,
  "metadata": {
    "atoms": [
      {
        "text": "ChatGPT Plus is a paid tier of ChatGPT",
        "score": 0.92,
        "weight": 0.40,
        "query": "ChatGPT Plus pricing subscription",
        "snippet_tag": "openai.com"
      },
      {
        "text": "ChatGPT Plus has advantages over the free tier",
        "score": 0.75,
        "weight": 0.35,
        "query": "ChatGPT Plus vs free tier features",
        "snippet_tag": "techcrunch.com"
      },
      {
        "text": "Those advantages relate to versatility",
        "score": 0.48,
        "weight": 0.25,
        "query": "ChatGPT Plus versatile capabilities",
        "snippet_tag": "(no result)"
      }
    ],
    "aggregation": "weighted_mean",
    "atom_count": 3
  }
}
```

**File changes.** The Validator's gather-target path in
`core/worker_pool.py:783-819` must be refactored from the current single
`plan_validate_query` → `search` → `stash` pattern into a multi-phase loop:
decompose → (step-back + HyDE + search) per atom → aggregate. The
`validate_prompt` and `validate_parse` functions in `core/actions.py:423-539`
must accept an atom-level call mode (single atom vs. whole-claim). The
`Signal.metadata` schema must be documented to carry the `atoms` list.

**Beyond-params argument.** A single forward pass of the base LM cannot
decompose-search-aggregate. Given a composite claim and one retrieved
snippet, the LM makes one inference over the whole composite; it cannot
isolate which sub-propositions are and are not supported by the snippet. SAFE
decomposition gives it 3–6 separate, narrower inferences, each against a
targeted snippet, where the LM is empirically more reliable. The aggregation
across atoms is a non-trivial operation that depends on per-atom scores that
do not exist until after retrieval; a single forward pass cannot produce it
without the prior retrieval loop.

**Risks.** Atom decomposition adds 1 LLM call; step-back and HyDE add 2
calls per atom; each atom requires a separate `validate_prompt` call. A
6-atom claim costs at minimum 1 + 6×2 + 6 = 19 LLM calls per Validator
iteration versus the current 1. On the hardware target this is a
significant throughput hit. The SAFE atom budget cap (§5.4) must be set
conservatively; 3 atoms default with a 5-atom cap is the recommended
starting point.

### 4.4 Worker Pool Search Budget — FLARE at System Level

#### Current behavior

`core/worker_pool.py:try_reserve_search()` (lines 177-187) is a pure rate
limiter: it prunes timestamps older than `SEARCH_WINDOW_S = 5.0` seconds,
checks whether `len(search_timestamps) >= SEARCH_BUDGET_PER_WINDOW = 6`,
and records the current timestamp if the budget has room. This is a cost
ceiling, not a value floor. It prevents DDG rate-limit responses from
extending per-iteration wall time but does not ask whether a given search
call would produce high-marginal-value evidence.

#### Proposed behavior

**FLARE at the pool level.** Retain the rate limit as a hard ceiling (it is
correctly protecting against DDG rate-limit blowout). Add an uncertainty
check as a soft gate: before calling `try_reserve_search()`, any worker
about to trigger retrieval submits its draft output to a confidence check.
Only calls that clear both gates (uncertainty below τ AND budget has room)
proceed to live DDG.

In practice, this means the per-role FLARE-strict additions described for
Scout and Developer (§4.1, §4.2) are the implementation sites. The pool-
level budget remains unchanged; the per-role uncertainty checks are the
FLARE layer that filters the candidates that reach `try_reserve_search()`.

**Decoupled certainty and budget.** The two-layer gate creates a clean
separation of concerns: the rate cap is cost protection (never more than 6
live calls per 5 seconds); the per-role confidence check is value protection
(only retrieve when the LM genuinely needs external grounding). In the
steady state, far fewer calls reach `try_reserve_search()`, so the budget
ceiling is rarely hit and response times stay low without the current pattern
of budget exhaustion forcing skips mid-run.

**File changes.** The `try_reserve_search()` method and `PoolState` class
in `core/worker_pool.py` remain unchanged. The per-role gather paths
(SCOUT: lines 643-700, DEVELOP: lines 703-757, VALIDATE: lines 783-819)
add the confidence-check gate before calling `try_reserve_search()`.

**Beyond-params argument.** This is FLARE at the system level rather than
the token level. The system is *active* about retrieval rather than passive;
the rate of retrieval tracks the calibration of the confidence estimates,
which are themselves model outputs. A system that always retrieves cannot
adapt its search frequency to evidence density; a system with FLARE-style
gating retrieves proportionally to how often the model genuinely needs help.

---

## 5. Required Design Choices

### 5.1 Confidence source for FLARE

Four options:

**(a) Self-rating LLM call** ("rate your confidence in the factual basis of
the deposit you just produced, 0–1"). Cost: one extra call per trigger.
Available on all backends including MockLLM.

**(b) Token-level log-probabilities from vLLM** (`logprobs` API). Cost:
zero extra calls; log-probs are returned alongside generation. Not available
on the GGUF path or MockLLM.

**(c) Entropy over top-k tokens.** Same cost as (b); same availability
constraint. Empirically more stable than raw log-prob of a single token.

**(d) Structural proxy** (cluster's `support_diversity` and
`verification_score`). Zero extra calls; always available; operates at
cluster granularity rather than deposit granularity.

**Recommendation per role:**

- **Scout (FLARE-light):** Use option (d). Scouts should retrieve
  speculatively; the structural proxy is a soft dedup gate, not a strict
  uncertainty check. A full self-rating call per Scout iteration would double
  per-iteration latency without corresponding value.
- **Developer (FLARE-strict):** Use option (a) as primary with option (d)
  as fallback. The structural trigger (`n_support < 2`) already gates which
  iterations even enter the FLARE check; the self-rating call fires only on
  those iterations. Typical run: ~30% of Developer iterations hit the
  structural trigger; self-rating on those 30% adds ~15% total Developer
  LLM calls. Acceptable given the retrieval cost saved on the 70% that are
  already well-served.
- **Validator:** Use option (a). The Validator's FLARE gate is implicit in
  the SAFE decomposition — decomposition is what triggers atom-level queries.
  The "does this claim need decomposition?" question is answered by whether
  the SAFE call finds more than one atom. A single-atom claim does not need
  multi-step retrieval.

**Fallback for backends without logprobs (GGUF, Mock):** Use option (a)
self-rating exclusively. The FLARE paper used logprobs as a proxy for
uncertainty; the authors also note that self-consistency approaches
(sampling multiple generations and checking agreement) can replace logprobs
at higher LLM cost. For the hardware target, a single self-rating call is
the right trade.

### 5.2 Step-back depth

**Recommendation:** One step back as the default for Scout and Developer;
two steps back as the fallback when a one-step query returns zero results.

Recursive step-back risks producing queries so abstract they are useless
("existence", "science", "history"). The paper studies one step back almost
exclusively; two steps is an unvalidated extension. The practical rule: step
back once, try the query; if zero results, step back again; if still zero
results, fall back to the HyDE-extracted noun phrases from the original claim.

**Per-role override:** The Validator operates at the atom level, where each
atom is already more specific than the full cluster representative. One step
back per atom is appropriate. Validators should not recurse further because
atom specificity is the source of their precision.

### 5.3 HyDE document length

**Recommendation:** 40–50 words for the hypothetical paragraph; extract
top-4 noun phrases as the literal query; discard the rest.

The HyDE paper generates 100–200 word hypothetical documents and encodes the
full text as an embedding. DDG keyword search performs poorly on queries
longer than ~8 words. The optimal trade is to generate enough text that the
confirming-document vocabulary is well-represented in the noun phrases, but
extract only the salient terms for the actual query string. Forty to fifty
words produces 4–8 noun phrase candidates; the top 4 by frequency and
specificity are the query.

When Cohere is available (it is not in the current environment; only DDG is
active), the full 40–50 word hypothetical document should be encoded and
passed directly to the Cohere re-ranker's dense path in `search_tool.py`'s
`_diversify` function (lines 288-320). Currently that function computes
`q_emb = embedder.encode(query, ...)` from the raw query string; the HyDE
adaptation replaces the raw query string with the hypothetical document text
for that embedding call, while retaining the noun-phrase extraction for the
DDG keyword call. The two paths (keyword and dense) then use different query
representations — exactly what HyDE intends.

### 5.4 SAFE atom budget per claim

**Recommendation:** Default 3 atoms; cap at 5; minimum 1.

The "minimum 1" case handles very short cluster representatives (fewer than
15 words) that contain only one verifiable proposition. The "default 3" is
appropriate for most two-to-three sentence cluster representatives. The
"cap at 5" prevents a runaway decomposition LM call from producing
10-atom chains for a dense paragraph.

Atom budget should scale with cluster representative length: `min(5,
max(1, len(content.split()) // 15))`. A 30-word representative typically
contains ~2 propositions; a 75-word representative contains ~5. This
heuristic should be validated against actual cluster representative length
distributions from a real run.

### 5.5 Atom aggregation function

**Recommendation:** Weighted mean, where weights are atom centrality scores
produced by the decomposition call.

At decomposition time, the LM is asked both to list atoms and to rate each
atom's centrality to the original claim (0–1). Load-bearing atoms — those
that express the primary assertion of the claim — get high weights; incidental
atoms — those that express background context or secondary implications — get
low weights. The verification score is then the weighted mean of atom-level
scores.

A plain mean treats "ChatGPT Plus is a paid tier" (easily confirmed, low
load-bearing weight for the interesting part of the claim) equally with
"those advantages relate to versatility" (the substantive and contested
assertion). The centrality-weighted mean focuses the verification score on
the propositions that actually matter for the claim's survival outcome.

An alternative weighting by atom *score variance* (high variance = contested
= more informative) was considered but rejected: contested atoms are often
contested because retrieval failed to find relevant evidence, not because
the claim is false. Rewarding variance conflates "we couldn't verify this"
with "this is controversial."

### 5.6 Backoff when retrieval returns nothing

**Recommendation:** For Scout and Developer, step back one level and retry
once. If the stepped-back query also returns nothing, accept the draft output
at a discounted strength (multiply by 0.8) and do not retrieve. For the
Validator's atom-level path, when a specific atom's query returns nothing,
score that atom at 0.5 (abstain rather than false-negative), and flag it in
the VERIFICATION signal metadata with `"snippet_tag": "(no result)"`.

The 0.5 abstain score for missing evidence is appropriate because a zero
score would mean the atom is *contradicted* by external evidence, which is
not the same as *unaddressed* by it. A 0.5 abstain with a flag allows the
synthesizer to distinguish "well-verified" clusters (all atoms have high
scores and snippets) from "partially verified" clusters (some atoms have
no-result flags) without collapsing both to the same verification score.

---

## 6. Schema and Infrastructure Preconditions

### 6.1 VERIFICATION signal metadata extension

The current `validate_parse` in `core/actions.py:466-539` produces a
`ParsedDeposit` with `metadata={"score": round(score, 4)}`. The proposed
SAFE extension requires an `atoms` list in metadata:

```python
{
    "score": 0.68,          # backward-compat scalar (weighted mean of atom scores)
    "atoms": [
        {
            "text": str,    # the atomic proposition
            "score": float, # 0..1
            "weight": float,# centrality weight from decomposition call
            "query": str,   # the DDG query issued for this atom
            "snippet_tag": str,  # source URL/title or "(no result)"
        },
        ...
    ],
    "aggregation": "weighted_mean",
    "atom_count": int,
}
```

The backward-compat scalar `score` must be preserved because
`projection.py` reads `signal.metadata.get("score", signal.strength)` in
its cluster-quality computation. The `atoms` list is additive; code that
does not know about it ignores it.

The `Signal.metadata` field is currently typed as a freeform `dict` (the
`SignalStore.deposit()` call in `core/signal_store.py` accepts any dict).
No schema enforcement change is needed, but a docstring in `core/actions.py`
should document the canonical fields so the synthesizer's aggregator can read
them reliably.

### 6.2 Dense embedding surfacing from search_tool

`core/search_tool.py:_diversify` (lines 288-320) computes query embeddings
and chunk embeddings via `sentence_transformers.SentenceTransformer` when
`HYBRID_RETRIEVAL` is enabled. These embeddings are computed but not
returned to callers: the function returns a reranked list of `CorpusChunk`
objects, discarding the embedding vectors.

For the HyDE dense-mode adaptation (when Cohere re-rank is available), the
hypothetical document's embedding should be passed *into* `_diversify` as an
optional `query_embedding` override, bypassing the `embedder.encode(query,
...)`call with the already-computed HyDE document embedding. This requires a
signature change: `_diversify(chunks, query, ..., query_embedding=None)` where
non-None `query_embedding` replaces the internal encode call.

In the current environment where only DDG is available and
`HYBRID_RETRIEVAL` is therefore effectively false (DDG returns chunks with
no embeddings), this change can be deferred. The HyDE adaptation falls back
to the lexical path (noun-phrase extraction → keyword query) entirely.

### 6.3 Semantic indexing of served_queries

`PoolState.served_queries` (line 170 of `core/worker_pool.py`) maps literal
query strings to result counts. The dedup check in `find_cached_query`
(`core/query_planner.py:262-286`) uses exact string match, fingerprint
match, and `SequenceMatcher` ratio. Two scouts that produce different
step-back phrasings of the same topic ("climate policy economics" and
"economics of climate policy") will not match, even though they are
semantically equivalent and would retrieve overlapping results.

The recommended extension: add a secondary index mapping
`fingerprint(step_back_phrase) → served_query` so that step-back-derived
queries can be deduped against each other without requiring surface-form
similarity. The `_fingerprint` function in `core/query_planner.py:182-193`
is already defined and can serve as the key.

For full semantic dedup (catching "grid expansion capital requirements" vs.
"capital requirements for grid expansion"), an embedding-level index using
the same `SentenceTransformer` that powers the signal store dedup would be
needed. This is a heavier change and should be deferred until the simpler
fingerprint-level index has been evaluated.

### 6.4 LLM backends and the logprobs accessor

vLLM exposes a `logprobs` parameter in its generation API. The GGUF path
(used in the lower-memory fallback) does not expose per-token log-probs.
MockLLM has no log-probs at all.

Recommendation: add a `confidence(draft: str) -> float` method to the LLM
wrapper interface (`swarm/llm/simple_llm.py` or the equivalent in the
Attempt At Cleaning path). On vLLM backends, implement this via the
`logprobs` API (mean log-prob over the draft tokens, exponentiated). On all
other backends, implement as a self-rating call: generate a confidence rating
prompt and parse the scalar response. MockLLM returns a fixed value (e.g.,
0.5) so FLARE-strict never triggers in mock runs — this is the correct
behavior for plumbing tests (mock retrieval is meaningless).

---

## 7. Composition with the Synthesizer Overhaul

The synthesizer overhaul memo proposes Improvement 7: replacing the
synthesizer's `_get_external_context` call (a single Wikipedia lookup per
surviving cluster) with an aggregator over per-cluster VERIFICATION signals.
That improvement is only possible if VERIFICATION signals carry atom-level
evidence: the aggregator needs to know which atomic facts are externally
supported, not just what the scalar verification score is.

The SAFE extension to the Validator (§4.3 above) is the load-bearing
precondition for Improvement 7. Without atom-level VERIFICATION metadata,
the synthesizer's aggregator has nothing richer than a scalar to read, and
the improvement collapses to the current behavior.

The edge graph (Improvement 1 of the synthesizer overhaul — inter-cluster
relationship mapping) also benefits from richer VERIFICATION metadata:
when two clusters share an atom (e.g., both claim "grid infrastructure
costs are the primary barrier"), their VERIFICATION signals will have
overlapping atom texts. A similarity check on atom texts across clusters is
a more precise signal of cluster overlap than embedding similarity on full
cluster representatives.

Specifically, the synthesizer's scoring of inter-cluster edges should weight
by the number of shared atoms (atom text similarity above a threshold, say
0.8 cosine) rather than by representative-level embedding similarity. This
requires the synthesizer to read the `atoms` list from each cluster's
VERIFICATION signals, which only exists after the Validator overhaul is
implemented.

Timing dependency: the synthesizer improvements that depend on atom-level
VERIFICATION metadata (Improvement 7, and the atom-based edge weighting)
cannot be implemented before the Validator redesign is complete. The two
memos compose sequentially: Validator overhaul first, then the synthesizer
aggregator.

---

## 8. Falsifiability

The following A/B comparisons must be runnable with a fixed base LM, a
fixed factuality benchmark, and a single-call baseline (no retrieval, no
decomposition).

| Technique | Benchmark | Metric | Expected effect | Failure mode |
|-----------|-----------|--------|-----------------|--------------|
| SAFE | LongFact (Wei et al.) or a synthetic factual-claim corpus derived from a run | Precision × recall over supported/unsupported atoms vs. ground-truth atom set | Atom-level precision +10–20pp over composite-claim evaluation; recall neutral | If decomposition produces non-atomic "atoms" (compound sentences), atom-level precision equals composite-level precision — decomposition failed |
| HyDE (lexical) | TruthfulQA retrieval sub-task or factual DDG accuracy spot-check | Retrieved-snippet relevance (human or LM judge, binary) for matched vs. mismatched vocabulary topics | +15–25% relevant snippet rate on topics where question/document vocabulary diverges | If the base LM's hypothetical documents consistently use question vocabulary rather than document vocabulary (i.e., the LM cannot generate confirming-document text for topics it doesn't know), HyDE degrades to direct-claim query |
| Step-Back | DDG zero-results rate per query; per-claim retrieval coverage | Reduction in zero-result queries; increase in fraction of claims that retrieve ≥3 relevant snippets | 30–50% fewer zero-result queries on composite claims | If the step-back always produces a query too abstract to retrieve topically relevant snippets (abstraction overshoot), relevance drops and zero-results move from specific to abstract queries |
| FLARE | Search count per 100 iterations vs. verification quality delta | Same or higher VERIFICATION signal mean strength at 40–60% of the search count | If FLARE gates retrieval too aggressively (τ set too high), quality drops and the system degrades to pure parametric; if τ too low, behavior is equivalent to always-retrieve |
| SAFE + Step-Back + HyDE (Validator) | FactScore on generated text (fraction of supported sentences) | +5–15pp FactScore vs. single-call Validator with one keyphrase query | If the atom-level calls are dominated by "no result" atoms (DDG coverage issue), the weighted mean defaults to 0.5 for many atoms and the aggregated score is uninformative |

**FLARE cost frontier note.** The cost claim for FLARE is specific: selective
retrieval should match always-retrieve in quality at a lower search count.
The comparison must report *both* quality and count, not quality alone. A
system that retrieves half as often but also produces lower-quality
verification signals has failed to demonstrate the FLARE claim.

**SAFE metric note.** The SAFE paper reports the metric as:

```
factuality_score = n_supported_atoms / n_total_atoms
```

where atoms are rated S (supported), NS (not supported), or IR (irrelevant
to verifiability). An A/B comparison should report this metric against a
hand-annotated or GPT-4-generated atom-level ground truth for a subset of
outputs. The ground-truth annotation is the most expensive part; a 50-claim
spot-check is the recommended minimum.

---

## 9. Sequencing

Implementation should proceed in three phases, each one enabling the next:

**Phase 1 — SAFE decomposition on Validator (4–6 weeks estimated).**
This is the load-bearing change: it produces atom-level VERIFICATION
metadata that downstream improvements depend on. Implement:
- SAFE decomposition call in Validator gather-target path
- Extended VERIFICATION metadata schema
- Atom-level score aggregation with centrality weighting
- Backward-compat scalar in metadata

SAFE first because the synthesizer Improvement 7 and the atom-based edge
weighting both block on this output.

**Phase 2 — Step-Back and HyDE in query planner (2–4 weeks after Phase 1).**
After SAFE is in place, Step-Back and HyDE improve the quality of the
atom-level queries. These are utilities shared across roles and can be
implemented in `core/query_planner.py` as optional call paths.
- Step-back utility function (LLM call, returns abstracted phrase)
- HyDE utility function (LLM call, returns hypothetical document + noun phrases)
- Plumb into Scout gather path (Step-Back + HyDE)
- Plumb into Developer gather path (HyDE on trigger, Step-Back fallback)
- Plumb into Validator atom queries (Step-Back + HyDE per atom)

**Phase 3 — FLARE gates (2–4 weeks after Phase 2).**
FLARE is last because it depends on a calibrated confidence source. Calibrating
τ requires observing how often retrieval actually changes a deposit — which
can only be evaluated with real-run data from Phase 1 and Phase 2. Without
Phase 2 in place, the FLARE gate might suppress retrieval precisely on the
iterations where the improved HyDE queries would have retrieved useful
evidence.
- Confidence self-rating utility function
- Scout FLARE-light (served_queries dedup extension)
- Developer FLARE-strict (structural + confidence gate)
- Validator implicit FLARE (single-atom short-circuit)
- Calibration run: measure τ=0.5, τ=0.6, τ=0.7 against quality + search-count frontier

---

## 10. Open Questions

These cannot be resolved from reading the code; experimental runs are required.

1. **Does the 7B local model decompose claims into genuinely atomic facts, or does
   it produce compound sentences that defeat the SAFE precision gain?** A 100-claim
   manual audit of decomposition outputs before investing in the full Validator
   redesign would de-risk this assumption. If the local model cannot reliably
   decompose, the decomposition call should use a cloud LLM (Anthropic Claude
   Haiku or similar; see the stub in `agents/validator.py:66-78`).

2. **What is the right atom budget cap for the target hardware?** The recommended
   default of 3 atoms and a cap of 5 was derived from expected claim structure, not
   from timing measurements on the RTX 3060 Laptop. A timed sampling of Validator
   iterations at various atom budgets is needed to confirm the latency budget.

3. **Does the step-back produce queries at a useful altitude for DDG, or does it
   overshoot to abstractions too broad to retrieve relevant snippets?** The paper
   was evaluated on physics and chemistry QA where background principles are
   well-defined; the swarm's claims are more diverse and less structured.
   A 200-query audit comparing zero-result rates for direct queries vs. step-back
   queries on actual swarm outputs would calibrate the depth recommendation.

4. **Is the SequenceMatcher-based dedup in `find_cached_query` sufficient to catch
   HyDE-generated queries that are semantically equivalent but surface-distinct?**
   Two HyDE calls on related claims might produce "grid expansion financing
   renewable" and "renewable energy grid capital" — very different fingerprints,
   similar retrieval results. An embedding-level dedup cache would catch these;
   whether it is necessary depends on how semantically diverse actual HyDE outputs
   are in practice.

5. **What τ threshold for the FLARE confidence gate produces the Pareto-optimal
   quality/cost tradeoff on this codebase's task distribution?** The FLARE paper
   reports τ=0.5 on their benchmarks; different model families and task types may
   warrant different values. Without A/B data, τ=0.6 is the conservative default,
   but it should be treated as a placeholder pending calibration.

6. **Does the centrality-weighted atom aggregation actually produce more accurate
   cluster-level verification scores than the plain mean?** The weighting hypothesis
   is that load-bearing atoms (the central assertions) matter more than incidental
   ones. This is plausible but not validated in this setting. A ground-truth atom
   annotation of 50 claims, with human-rated centrality scores compared against LM-
   rated centrality, would test whether the LM's centrality ratings are calibrated.

---

## Appendix: Verification Step

### File existence and path checks

All cited file paths were read end-to-end before this memo was written:

- `core/worker_pool.py` ✓
- `core/query_planner.py` ✓
- `core/search_tool.py` ✓
- `core/retrieval.py` ✓
- `agents/scout.py` ✓
- `agents/validator.py` ✓
- `agents/developer.py` ✓
- `core/actions.py` ✓

### Line-range spot checks

- `SEARCH_BUDGET_PER_WINDOW = 6` at `core/worker_pool.py:145` — confirmed.
- `PoolState.served_queries` described as "Maps the literal query string →
  n_results" at `core/worker_pool.py:169-170` — confirmed.
- `try_reserve_search()` implementation at lines 177–187 — confirmed as pure
  rate window, no uncertainty check.
- `plan_validate_query` at `core/query_planner.py:408-410` returns
  `_extract_sentence_fragment(target_content or "", max_words=10)` — confirmed.
- `_extract_keyphrase` at `agents/validator.py:164-180` uses `max_words=5` — confirmed.
- Scout `_compose_query` at `agents/scout.py:192-222` has 8 fixed phrasings,
  no LLM call — confirmed.
- Developer search trigger at `agents/developer.py:122-153`, trigger condition
  `n_support_children < _SEARCH_TRIGGER_SUPPORT` (= 2), query built as
  `f"{snippet} evidence"` at line 133 — confirmed.
- `validate_prompt` non-factual task gate at `core/actions.py:420`:
  `_NON_FACTUAL_TASKS = {"debate", "analysis", "problem_solving", "creative"}` — confirmed.
- `_diversify` in `core/search_tool.py:288-320` computes embeddings internally
  but does not surface them to callers — confirmed.

### Composition trace verified

The end-to-end Developer path traced in §3 (draft → FLARE-strict check →
Step-Back → HyDE → DDG → conditioning) is consistent with the actual code:
`_gather_target` (DEVELOP branch, `core/worker_pool.py:703-757`) is the
correct insertion point; `plan_develop_query` (called at line 726) is the
function to be replaced or extended with HyDE; `_search` is called at
line 733 or 737 after budget reservation. The FLARE confidence gate would
be inserted between the draft-generation step (currently not present —
this is a new call) and the existing `try_reserve_search()` call at
line 733.

### DDG-only constraint

Only DuckDuckGo is available in the current environment; Tavily requires
`TAVILY_API_KEY` (not set) and Cohere requires `SWARM_SEARCH_USE_COHERE=1`
(not set, and pulls a ~1 GB index download). All adaptation notes in this
memo that reference Cohere dense-mode (HyDE embedding path in §5.3,
`_diversify` extension in §6.2) should be treated as future work for when
those backends are available. The lexical HyDE path (noun-phrase extraction
→ DDG keyword query) is fully implementable with DDG alone and is the
primary recommended path.
