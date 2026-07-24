# Stigmergic Multi-Agent LLM System
## Mechanisms and Compute Request — UNC Longleaf

**Prepared by:** Seth Fuqua, UNC–Chapel Hill   **Repository:** github.com/sfuqua6/Stigmeric-Coordination

---

> ### The ask
>
> **Workhorse GPU:** one **A100 40 GB (a100-gpu)** or **L40S 48 GB (l40-gpu)** as a single-GPU batch job — `--qos=gpu_access --gres=gpu:1`, 12–16 CPU, 64–128 GB RAM, up to 2 days. ~250 GB on `/work` for the model cache. Outbound network for retrieval (datamover partition or a pre-warmed cache). Stack already present: `cuda/12.9`, Python 3.12, PyTorch, `vLLM<0.20`.
>
> **Google Colab credits:** a top-up of compute units — e.g. reallocated from students who do not use their institutional Colab allotment. Colab's paid tiers are the only available route to an **H100**, which Longleaf does not have; the largest resident multi-model configuration needs it.

---

## 1. System

About 24 LLM workers run concurrently and build a single typed graph of evidence. A worker reads signals already in the graph (each is content plus an ID and a few scalar tags), generates, and deposits a new signal. Workers never read each other's text or reasoning. When the run halts, a deterministic pass projects the graph into one answer. Two mechanisms do the work and are described below: agents are held independent (no-leak, §2), and their inputs are partitioned so that independence produces different claims (§3).

## 2. No-leak rule

The `Signal` record has no field for replies, no back-reference to a parent's text, and its metadata never contains parent content. Queries that walk a claim's history return IDs and typed counts, not the ancestors' prose. A worker's prompt is assembled from exactly two things: the signals it sampled from the graph (their content only) plus its role instruction — for scouts, its corpus partition instead. This is enforced in the data layer and checked by tests.

The consequence is structural: there is no transcript of reasoning for agents to read, so there is nothing for them to converge onto. Any agreement between two workers has to arise from the evidence, not from one copying another's framing.

## 3. Information intake and partitioning

This is where independence is created rather than assumed. The retrieved corpus is split and handed out so that no two scouts see the same evidence.

- **Chunking:** the corpus is cut into overlapping word windows of `CHUNK_WORDS = 600` with `CHUNK_OVERLAP = 80` (stride 520 words).
- **Disjoint assignment:** with S scouts over C chunks, scout *i* receives the contiguous block `chunks[i*b : (i+1)*b]` where `b = min(CHUNKS_PER_SCOUT_MAX=8, ceil(C/S))`. Blocks do not overlap; each is tagged `partition_i`. A scout never receives another scout's chunks.
- **Bounded window:** a scout renders only its block, capped at ~4000 characters, and rotated by iteration index (same chunks, different starting offset each pass) so repeated calls explore different orderings of the same slice.
- **Web partitions:** `WEB_PARTITION_COUNT = 2` scouts instead get web-search partitions drawn from `N_FACETS = 6` facet queries, tagged `web_<facet>`.
- **Propagation:** an INITIAL claim inherits its scout's `partition_id`; a SUPPORT inherits its parent's. Every claim therefore carries the evidence window it came from, which the read-out uses to measure independent corroboration (§5).

Two scouts thus reason from non-overlapping evidence under different size and ordering limits, so even identical model weights produce different claims. Diversity is a property of the inputs, not of prompt wording or temperature.

## 4. Signal store: similarity, clustering, strength

**Similarity.** Text is embedded with a sentence-transformers model; similarity between two signals is the cosine of their L2-normalized vectors (a dot product). With no embedder available, the fallback is a difflib `SequenceMatcher` character ratio.

**Deduplication.** On deposit, the new text is compared by `SequenceMatcher` against the three most-recent same-type signals; a ratio above `0.95` rejects it and instead amplifies the existing twin (no embedding computed). Below that, the signal is embedded and kept — near-duplicates are not discarded, they are routed into a cluster.

**Clustering.** A new signal joins the cluster with the largest margin `cos(emb, centroid) - T(size)`, provided the margin is non-negative. The base threshold is `0.72` on GPU (`0.55` on laptop); `T(size)` rises by `0.03` per doubling of cluster size (capped at `0.97`), so large clusters demand higher similarity and do not absorb everything. A cluster centroid is a running L2-normalized mean, periodically re-anchored to its medoid (the member with the highest mean cosine to the rest); members that fall below `0.55 / 0.42` are ejected to re-cluster.

**Strength dynamics.** Each signal's strength `s` in (0,1) is updated in logit space, `s' = sigmoid(logit(s) + delta)`, with `logit(s) = ln(s/(1-s))`. Additive deltas keep updates order-independent and prevent saturation:

| Event | Logit delta |
|---|---|
| Decay (each tick) | `-0.10` |
| Corroboration (a SUPPORT lands on the claim) | `+0.30` |
| Dedup hit (near-identical re-deposit) | `+0.10` |
| Provenance boost (parent mean VERIFICATION `v >= 0.70`) | `+0.60 * v` |

Signals whose strength falls below `PRUNE_THRESHOLD = 0.30` are removed; members of a well-supported cluster are protected down to `0.15`. **Novelty selection:** a scout proposes several candidate claims per call; the store scores each by its maximum cosine similarity to the last 30 INITIALs and keeps the one with the lowest maximum — the claim least like what the field already holds.

## 5. Read-out: survival classification

When the run halts, a pure-Python pass (no model) labels every cluster from two quantities:

- **Support diversity** = the number of distinct `(partition_id, depositor)` pairs among a cluster's SUPPORT signals. A cluster must reach `>= 3` to survive — corroboration from at least three independent (evidence-window, role) sources, not one window restated three times.
- **Dissent pressure** = `ln(1 + D/S)`, where D and S are the strength-weighted dissent and support against and for the cluster. `>= 1.5` marks it rejected; `0.5–1.5` contested; below that, surviving if it also clears support diversity.

Only surviving and contested clusters are read out. The graph — what was deposited, how signals connected, what held up under support and dissent — is the actual result; the written answer is a projection of it.

## 6. Retrieval

Retrieval currently runs on DuckDuckGo (the `ddgs` package). Tavily is supported but requires an API key that is not set, and Wikipedia is a last-resort fallback used only when DuckDuckGo returns nothing. For each query the results pass through a relevance gate, BM25 plus dense reciprocal-rank fusion, fact-density and domain-prior reranking, MMR diversification, and optional page-text enrichment.

Queries are planned, not copied from the prompt: the question is abstracted one step back and a hypothetical answer is drafted to search against (HyDE), and a search fires only when the model rates itself uncertain. Each query and its results are deposited as a SEARCH signal, so other workers reuse the retrieval instead of repeating it — shared evidence without shared reasoning.

## 7. Evidence-based context compression

Verification is per-fact. A claim is split into at most three atomic facts, each weighted by how load-bearing it is; each atom is searched and scored independently against retrieved sources, and the cluster's score is the weight-weighted mean of its atoms. This lets scrutiny target the atoms that are influential but weakly supported.

The final answer is composed by a single bounded call that receives only one short brief per surviving cluster plus scalar tags — never the full graph or corpus. Input size is `O(K * brief)` for K surviving clusters, independent of how large the run grew; the field, not a long context window, does the compression. A 4-gram audit then checks each rendered sentence against the cluster it cites, and any figure that was never verified is presented as reported rather than established.

## 8. Why the compute is needed

No-leak removes the cheap way to get diversity (a shared transcript that agents are prompted to disagree with). The two remaining sources are structural: partitioned inputs (§3) and different model families acting as different reasoners. Because the inputs are already independent, using genuinely different models — a Llama-family scout, a Qwen-family developer, a third-family adversary — is what supplies reasoning diversity on top of input diversity.

Those heterogeneous models have to run at the same time, reading and writing the same graph, across roughly 24 workers. A rate-limited cloud API does not support this: it throttles the high-volume roles and caps the per-fact verification of §7 to a single atom per claim, which removes the mechanism that makes the answers checkable. A single A100 40 GB or L40S 48 GB runs a strong local model for the high-volume roles and the full multi-atom verification, and anchors the hybrid configuration in which a local family and one contrasting family operate together. The H100-class card needed for the largest fully-resident multi-model configuration is not available on Longleaf, which is why the Colab credit ask above is part of the request.

> **In one line:** No-leak forces real independence; real independence is supplied by partitioned inputs and heterogeneous models cooperating only through the graph; running those models together with full per-fact verification — not a one-atom API cap — is what a single A100/L40S workhorse provides, with Colab credits covering the H100 case Longleaf cannot.

*The system is implemented and tested; the compute produces the first measured comparison of this independent, evidence-coordinated swarm against a single model and against agents that share a transcript.*
