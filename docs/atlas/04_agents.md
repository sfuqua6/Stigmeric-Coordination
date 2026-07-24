# Atlas 04 — Agent Roles & Actions

> Part of the code atlas (see docs/atlas/README.md). Context: CLAUDE.md, docs/PIPELINE_MAP.md.

## Area summary
All agent roles inherit from `BaseAgent` (`agents/base.py`), whose `run()` loop is the single enforcement point for the no-leak rule: it builds a prompt only from sampled-signal `content` (or, for scouts, the corpus partition) plus a role instruction, runs `_assert_no_leak()` over the prompt string, then `strip_reasoning()` over the output before deposit. Roles differ purely by what they **sample** and what signal **type** they **deposit** — diversity comes from differentiated sampling over the shared store, not prompt/temperature tweaks. The five content roles (Scout/Developer/Critic/Validator/Hater) plus their `coding`-task overrides (`agents/coding_roles.py`) are selected per task via `core/role_registry.py`. `agents/forager.py` is a 16-line back-compat alias re-exporting `Developer` as `Forager`. The base class also carries `partition_id` forward into `deposit_meta` (L327–330) as belt-and-suspenders for the store's hard partition invariant.

## Modules

### `agents/base.py`  (394 lines)
- **Purpose:** Abstract agent contract + the shared run loop, no-leak assertion, and reasoning/scratchpad stripping.
- **Samples / Deposits:** N/A (abstract); subclasses set `OUTPUT_TYPE`/`INPUT_TYPE`.
- **Key symbols:**
  - `BaseAgent` (L191) — abstract base; `run()` (L262) is the canonical loop.
  - `strip_reasoning` (L145) — removes `<think>…</think>` blocks + leading scratchpad sentences; called before every deposit.
  - `_SCRATCHPAD_RE` (L124) — CoT-preamble regex (DeepSeek-R1-Distill etc.).
  - `_assert_no_leak` (L377) — fails loudly on forbidden tokens (`parent_content`, `provenance_chain`, `dialogue thread`, …).
  - `parse_type_proposal` (L56) / `parse_parent_proposal` (L69) — Phase 3A dynamic TYPE/PARENT from model output.
  - `type_parent_instruction` (L100) / `_strip_type_parent_lines` (L109) — TYPE/PARENT prompt block + stripper.
  - `AgentRunStats` (L182) — per-run deposit/dup/novelty counters.
  - partition_id carry-forward (L327–330) — pulls `partition_id` off sampled parents into `deposit_meta`.
- **Depends on / Used by:** `core/signal_store`, `core/diversity` (`AgentContextRecord`), `core/filters` (`is_junk_output`), `core/signal_types`. Base of every role.
- **Tests:** `tests/test_no_leak_real_patterns.py`, `tests/test_strip_reasoning.py`.

### `agents/scout.py`  (342 lines)
- **Purpose:** Only role that conditions on raw evidence (corpus partition or agentic search); seeds the field.
- **Samples / Deposits:** Samples NOTHING from the store (`sample()` returns `[]`, L268); deposits `INITIAL` (and `SEARCH` traces, L126). Never reads other agents' signals — its independence source.
- **Key symbols:**
  - `Scout` (L68) — overrides `run()` (L82) for saturation cap + re-seed.
  - `ScoutConfig` (L47) — partition, `use_search`, `topology_cell`/`topology_cell_desc`.
  - `_compose_query` (L234) — deterministic phrasing rotation for search queries.
  - `build_prompt` (L271) — renders retrieved chunks or `partition.render()`; injects re-seed hint (own deposits only) + topology hint.
  - Multi-claim selection (L181–186) — `split_scout_claims` / `select_novel_claim` keep least-similar candidate.
  - SEARCH deposit (L126–139) — carries `partition_id`.
- **Depends on / Used by:** `core/intake.ScoutPartition`, `core/search_tool`, `core/actions` (`split_scout_claims`/`select_novel_claim`), config tunables. Default scout in role_registry.
- **Tests:** `tests/test_scout_multiclaim.py`, `tests/test_search_diversity.py` (search path).

### `agents/developer.py`  (226 lines)
- **Purpose:** Develops `INITIAL` into defended `SUPPORT` (the former "Forager").
- **Samples / Deposits:** Samples `INITIAL` (gap-fill via `signals_with_few_children_of_type`, then strategy, then `sample_weighted` fallback, L97–110); deposits `SUPPORT` (+ `SEARCH` on sparse support, L147).
- **Key symbols:**
  - `Developer` (L71).
  - `sample` (L93) — three-tier sampling with the empty-strategy fallback (L109) the CLAUDE.md flags "do not remove".
  - `build_prompt` (L163) — renders one signal + stashed dissent block + retrieval block; appends `type_parent_instruction()`.
  - `recent_dissent_targeted` (L43) — module-level strategy (also re-exported by forager.py).
  - `_stashed_dissent` / `_stashed_retrieval` (L87, L90) — sample()→build_prompt() hand-off state.
- **Depends on / Used by:** `core/sampling.SamplingStrategy`, `core/search_tool`, config. Subclassed by `CodeDeveloper`.
- **Tests:** `tests/test_partition_propagation.py`, `tests/test_dissent_compose.py` (indirect).

### `agents/forager.py`  (16 lines)
- **Purpose:** Back-compat shim — `from agents.developer import Developer as Forager`.
- **Samples / Deposits:** N/A (alias).
- **Key symbols:** re-exports `Forager` (= `Developer`) and `recent_dissent_targeted` (L14).
- **Depends on / Used by:** `agents/developer`. Kept for legacy imports; marked "do not add behavior here".
- **Tests:** `tests/test_alias_and_kb_novelty.py` (alias).

### `agents/critic.py`  (218 lines)
- **Purpose:** Evaluates one sampled artifact, emits valence-split critique.
- **Samples / Deposits:** Samples `INITIAL` via strategy; deposits `CRITIQUE_POSITIVE` (score ≥ 0.5) or `CRITIQUE_NEGATIVE` (< 0.5), strength = raw score.
- **Key symbols:**
  - `Critic` (L39); `_last_deposit_type` (L57) tracks valence between `parse()` and `run()`.
  - `parse` (L99) — extracts SCORE, sets valence-determined output type.
  - `run` (L116) — **overridden** to deposit the valence-determined type per deposit (duplicates base run loop).
  - `_SCORE_RE` (L36); `build_prompt` (L62).
- **Depends on / Used by:** `core/sampling`, config `MAX_TOKENS_CRITIC`. Subclassed by `StaticCritic`.
- **Tests:** `tests/test_critique_split.py`.

### `agents/validator.py`  (204 lines)
- **Purpose:** External grounding — looks up a snippet (search_tool), deposits `VERIFICATION` that drives the provenance boost.
- **Samples / Deposits:** Samples `INITIAL` preferring those with ≥2 SUPPORT children (L84), else strategy; deposits `VERIFICATION` (child of the claim).
- **Key symbols:**
  - `Validator` (L49); `sample` (L80) high-stakes routing.
  - `build_prompt` (L92) — JSON-output instruction (Phase K); does NOT render external text into other agents' prompts.
  - `parse` (L125) — JSON-first with SCORE-regex fallback.
  - `_extract_keyphrase` (L164) / `_wiki_lookup` (L183) — external snippet helpers.
  - `_maybe_cloud_call` (L66) — anthropic/gemini hooks, both `NotImplementedError` (DEAD-FLAG).
- **Depends on / Used by:** `core/search_tool`, config. Uses base `run()`. Subclassed by `TestValidator`.
- **Tests:** No direct unit test for `Validator` (NONE FOUND for the prose validator; `TestValidator` is covered by `test_coding_roles.py`).

### `agents/hater.py`  (150 lines)
- **Purpose:** Adversarial pressure on the largest *semantic cluster* (not top-K), emits `OBJECTION`.
- **Samples / Deposits:** Samples cluster representatives (ClusterRegistry → DBSCAN → top-K fallback, L54–80); deposits `OBJECTION` capped at 1/round (`MAX_DEPOSITS_PER_ROUND = 1`, L41).
- **Key symbols:**
  - `Hater` (L32); `sample` (L53) three-tier cluster selection.
  - `build_prompt` (L91) — creative vs structural challenge branch (L114).
  - `extra_deposit_metadata` (L85) — `targets_cluster_id`.
  - `parent_id_for_deposit` (L146) — links to strongest representative.
- **Depends on / Used by:** `core/signal_store` clustering helpers, config. Uses base `run()`. Replaced by `EdgeCaseHater` for coding.
- **Tests:** No direct unit test (NONE FOUND); cluster path exercised by `tests/test_cluster_diversity.py` indirectly.

### `agents/coding_roles.py`  (674 lines)
- **Purpose:** `task_type="coding"` overrides — signals are code artifacts (criteria, snippets, AST checks, edge cases, pytest, assembly).
- **Samples / Deposits:** RequirementsScout → `INITIAL`; CodeDeveloper → `SUPPORT` (fenced code); StaticCritic → `CRITIQUE_*` (ast.parse); EdgeCaseHater → `OBJECTION`; TestValidator → `VERIFICATION` (subprocess pytest).
- **Key symbols:**
  - `RequirementsScout` (L108), `CodeDeveloper` (L196), `StaticCritic` (L295), `EdgeCaseHater` (L412), `TestValidator` (L460), `CodeSynthesizer` (L578).
  - `_extract_code` (L49) / `_has_unbound_self_refs` (L55) — fenced-block + AST guards.
  - `_run_test_subprocess` (L547) — sandboxed pytest, `SUBPROC_TIMEOUT_S=5` (L44).
  - `_EDGE_CASES` (L400) — canonical edge-case library.
- **Depends on / Used by:** `agents/developer.Developer`, `agents/synthesizer.Synthesizer`, `ast`/`py_compile`/`subprocess`. Registered lazily by role_registry.
- **Tests:** `tests/test_coding_roles.py`.

### `core/role_registry.py`  (85 lines)
- **Purpose:** Maps `task_type` → role class dict; lazily registers coding overrides.
- **Samples / Deposits:** N/A.
- **Key symbols:** `get_role_classes` (L74), `_DEFAULT_ROLES` (L32), `_register_coding_roles` (L46), `_TASK_ROLE_OVERRIDES` (L43).
- **Depends on / Used by:** all role classes + `agents/synthesizer`. Called by `run_swarm.py`.
- **Tests:** Exercised indirectly via `tests/test_coding_roles.py`.

## Refinement opportunities

1. `agents/critic.py:116`, `coding_roles.py:147/234/333/502` — DUPLICATION — five near-identical re-implementations of `BaseAgent.run()` (the deposit/dup/novelty loop) exist because each role needs a slightly different deposit step (valence type, fenced-code reject, ast deposit, subprocess). The shared scaffolding (iteration cap, `_assert_no_leak`, `strip_reasoning`, junk filter, consecutive-dup break, novelty embedding) is copy-pasted. Costs: any no-leak/strip change must be applied 5×; high drift risk. Candidate: a `BaseAgent.run()` with a `_make_deposit(samples, raw) -> Optional[sid]` extension hook so subclasses override only the deposit step.
2. `agents/validator.py:66` — DEAD-FLAG — `_maybe_cloud_call` wires `anthropic`/`gemini` providers that both raise `NotImplementedError`; `_cloud_provider` is plumbed through `__init__` but never resolves to working code. Costs: dead config surface, misleads readers into thinking cloud validation exists.
3. `agents/validator.py` (whole) + `agents/hater.py` (whole) — TEST-GAP — no dedicated unit test for the prose `Validator` or `Hater` (only coding variants `TestValidator`/`EdgeCaseHater` are tested via `test_coding_roles.py`). Costs: `sample()` routing (high-stakes INITIAL selection; cluster-vs-DBSCAN fallback) and JSON parse fallback are unverified; regressions land silently.
4. `agents/coding_roles.py:578` (`CodeSynthesizer`) — COMPLEXITY/SCOPE — a full synthesizer subclass lives in the agents-roles file rather than alongside `agents/synthesizer.py`; the assemble/compile/fallback logic (L598–669) is non-trivial and duplicates citation/lineage builders by importing private `_build_citations`/`_build_lineage_dot`. Costs: synthesizer logic split across two files; harder navigation. (Note: synthesizer is owned by another atlas section — flagging the split, not the internals.)
5. `agents/base.py:308` — FRAGILE — dynamic TYPE/PARENT parsing (`parse_type_proposal`/`parse_parent_proposal`) runs on the *raw* model output before any stripping and lets the model relabel its own deposit type and reparent arbitrarily. A model emitting `PARENT: <some_id>` can attach a SUPPORT to an unrelated cluster; not a no-leak break (it reads only IDs/own output), but it weakens partition/cluster integrity. Costs: hard-to-trace cross-cluster contamination. Low-severity NO-LEAK-adjacent: verified the prompt-side block renders only IDs, no ancestry text — no leak.

## Token-cost map
- "Change the no-leak enforcement / forbidden tokens" → `base.py:_assert_no_leak` (L377), `_SCRATCHPAD_RE` (L124), `strip_reasoning` (L145).
- "Add/modify a sampling strategy for a role" → role `sample()`: scout L268 (none), developer L93, critic L60, validator L80, hater L53.
- "Add a new task type's roles" → `core/role_registry.py:get_role_classes` (L74) + `_TASK_ROLE_OVERRIDES` (L43); model on `coding_roles.py`.
- "Fix duplicated run loops / deposit logic" → `base.py:run` (L262) vs the overrides at `critic.py:116`, `coding_roles.py:147/234/333/502`.
