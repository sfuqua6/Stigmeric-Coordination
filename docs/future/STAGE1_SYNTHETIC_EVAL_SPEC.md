# Stage 1 synthetic-world eval — implementation spec

**Status:** design spec, not implemented. Implements `THEFUTURE.md` §4 "Stage 1
— The synthetic-world eval (BUILD FIRST; everything gates on it)". Nothing
below exists yet except the files it cites as reused infrastructure.

**Why this exists, in one paragraph:** every A-vs-F run to date used prompts a
frontier-scale model already knows from pretraining (`hey fable.md`, `THEFUTURE.md`
§0: "B factual checklist, zero evidence shown → 100%"). That makes conditions
B/E's parametric answers indistinguishable from a genuinely-grounded answer on
the mechanical checklist, so no comparison built on top of that prompt set can
ever isolate the effect of showing evidence. `docs/OVERCONTEXT_EVAL_PLAN.md`
tried to route around this by making the *prompts* pack-dependent
(synthesis-with-conflict framing) while leaving the *facts* real and
pretraining-knowable — condition B still scores well because "what's in the
provided reports" is a soft instruction a fluent model can satisfy by
paraphrasing what it already believes about innovation/inequality/nuclear
power. Stage 1 removes the escape hatch: the facts themselves do not exist
anywhere the model could have trained on them, because we invented them after
the model's training data was frozen, procedurally, this run.

---

## 1. World generator design (`eval/worlds.py`, proposed)

### 1.1 Order of construction: facts first, prose second

The generator is two strictly ordered stages, and the ordering is the whole
point — get it backwards (write plausible documents, then extract "facts"
from them) and you're back to hoping the LLM didn't hallucinate a number that
now has no ground truth to check against.

```
Stage A: FactRegistry  = build_fact_registry(seed, template)   # pure Python, no LLM
Stage B: WorldDocuments = render_documents(registry, seed, renderer)  # LLM or template
Stage C: verify_render(registry, WorldDocuments)               # mechanical check, Stage B output only
```

`build_fact_registry` is deterministic pure-Python generation from a
`random.Random(seed)` instance — no LLM call, so the ground truth exists
before any model touches the world and cannot be perturbed by the renderer.
`render_documents` turns the registry into narrative text; **it is the only
stage allowed to call an LLM**, and its only job is prose, never fact
invention. `verify_render` is a mechanical (regex/substring) gate on Stage B's
output — see §1.4 — that must pass before a world is usable.

### 1.2 World templates

Three templates, matching THEFUTURE.md §4's examples, each producing a
different document mix so the corpus doesn't just become "a list of facts
with connective tissue":

| Template | Documents | Facts skew toward |
|---|---|---|
| `invented_company` | 10-K-style annual filing, 2-3 quarterly earnings-call transcripts, 4-6 trade-press articles, 2 analyst notes | quantities (revenue, headcount, market share), causal relations (a supply disruption → a margin miss), contradictions between analyst notes |
| `invented_country` | statistical yearbook chapter, 3-5 news-archive articles spanning a date range, 1 government white paper, 1-2 opposition-party rebuttals | quantities (population, GDP, election results), dated events, contradictions between government and opposition sources |
| `case_file` | incident report, 3-5 witness statements, 1-2 internal memos, 1 investigator's timeline | entities + relations (who was where, when), deliberate witness-statement contradictions with a registered ground-truth resolution |

Each template is a `WorldTemplate` dataclass: `name`, `entity_specs` (types
and counts to generate), `document_specs` (doc type, count, which entities/
facts/contradictions it's allowed to draw from), `filler_doc_spec` (used for
corpus-scale padding, §2).

### 1.3 Fact schema

```python
@dataclass(frozen=True)
class Entity:
    entity_id: str          # "ent_0007"
    kind: str                # "company" | "person" | "location" | "product" | "agency"
    name: str                 # "Meridian Analytics" — must pass the collision
                              # check in §4.2 before use
    aliases: list[str]        # ["Meridian", "MDA"] — used by the scorer's
                              # alias-matching (§3.2), not by the renderer

@dataclass(frozen=True)
class Quantity:
    fact_id: str
    subject_id: str           # Entity.entity_id
    metric: str                # "quarterly_revenue_usd_m" | "population_2024" | ...
    value: float
    unit: str
    as_of: str                 # ISO date or fiscal period label
    tolerance_pct: float = 2.0 # scorer matches within +/- this, see SCALE_CHARS-
                                # style override per metric type if needed

@dataclass(frozen=True)
class CausalRelation:
    fact_id: str
    cause_id: str              # fact_id of the cause (Quantity or Event)
    effect_id: str             # fact_id of the effect
    mechanism: str              # one-sentence mechanism, rendered near-verbatim
                                # by the renderer (this is the "buried in
                                # narrative" fact type the coverage thesis needs)

@dataclass(frozen=True)
class Event:
    fact_id: str
    description: str
    date: str
    entity_ids: list[str]

@dataclass(frozen=True)
class Contradiction:
    contradiction_id: str
    fact_id_a: str              # e.g. a Quantity claimed one way in doc X
    fact_id_b: str               # ...and a different way in doc Y
    doc_id_a: str
    doc_id_b: str
    ground_truth: str            # entity_id/fact_id of the CORRECT value, or
                                  # the literal string "genuinely_ambiguous"
                                  # (some contradictions in the real world never
                                  # resolve — the checklist scores "surfaced the
                                  # conflict", not "picked a side", when this
                                  # is set)

@dataclass(frozen=True)
class FactRegistry:
    world_seed: int
    template_name: str
    entities: list[Entity]
    quantities: list[Quantity]
    relations: list[CausalRelation]
    events: list[Event]
    contradictions: list[Contradiction]
    filler_seed_material: list[str]   # non-fact boilerplate sentences for
                                       # corpus-scale padding, §2 — must be
                                       # provably fact-free (no numbers, no
                                       # entity names) so padding cannot
                                       # accidentally inflate the checklist
```

Every `fact_id` is globally unique within a world and is the join key between
the registry, the rendered documents' provenance metadata (§1.4), and the
checklist (§3). Contradictions are generated by taking a subset of
`Quantity`/`Event` facts and cloning them with a perturbed value assigned to a
*different* document — the clone is itself registered (as a second
`Quantity`/`Event` with the same `metric`/subject but a different `as_of`-tagged
value and a `contradiction_id` link), so both the true and the false claim are
first-class, checkable facts, not something inferred after the fact from text
diffing.

### 1.4 Rendering: LLM vs. templates, and the leakage question

**Decision: LLM rendering (cheap Groq model, temperature 0), with templates as
a first-class fallback and a "no LLM available" degraded mode — not an
either/or.**

Justification:
- The whole point of Stage 1 is testing whether an organization can extract
  facts *buried in narrative prose the way real filings and news archives
  bury them*. A template renderer that just concatenates `f"{entity.name}'s
  {metric} was {value} {unit} as of {as_of}."` produces a corpus that is
  trivially easy for a single strong model to skim — no compression
  advantage is even structurally possible, so the experiment can't be
  decisive in the direction Stage 1 needs (THEFUTURE.md §3: "the coverage
  mechanism has value" is the thing being tested; a template corpus doesn't
  exercise coverage vs. skimming at all, it exercises string search).
- LLM rendering costs one Groq call per document (cheap; a 16x corpus is
  maybe 40-80 documents including filler — see §2), and Groq is already a
  load-bearing dependency (`core/llm_groq.py`, `GroqRouter`) so no new infra.
- Temperature 0 does not guarantee bit-identical output across API calls for
  most hosted inference stacks (batching-dependent floating point is a known
  property of e.g. vLLM continuous batching) — **do not rely on model
  determinism for the "same seed → identical corpus" requirement.** Instead,
  follow the pattern already proven in `eval/packs.py:build_pack` (lines
  157-178): the renderer writes to `eval/worlds/<seed>/rendered/<doc_id>.txt`
  and **reuses the file if it exists** rather than re-calling the LLM. Combined
  with the fact registry being pure-Python-deterministic, this makes the whole
  pipeline deterministic in the sense that matters: re-running
  `generate_world(seed=42)` a second time reuses every artifact and produces
  byte-identical output, exactly like `build_pack`'s reuse contract. The LLM's
  own non-determinism only matters on the very first render of a given
  `(seed, doc_id)`, and that's fine — the fact registry, not the prose, is
  what "same seed" needs to guarantee.
- Template fallback (`renderer="template"`) stays available for: (a) unit
  tests, which must not require a live API key or be slow (`test_worlds.py`
  should run under `MOCK_LLM=1`-style constraints — see §6); (b) any world
  where a document type genuinely doesn't need narrative burial (the
  statistical-yearbook table in `invented_country`, which is legitimately a
  table in the real world too).

**Fact-fidelity check (`verify_render`, mechanical, no LLM, runs on every
rendered document before it's accepted into the corpus):**

1. **Verbatim-presence check.** For every `Quantity`/`Event`/`CausalRelation`
   fact whose `extracted_from` (a field added to each rendered doc's metadata
   recording which `fact_id`s the renderer was asked to include) names this
   document, assert the fact's canonical rendering — value+unit for
   quantities (e.g. `"340.2 million"` or any of a small set of registered
   equivalent forms: `"$340.2M"`, `"340.2M"`, generated once as
   `Quantity.surface_forms` alongside the fact so the check doesn't
   over-fit to one exact string) — appears verbatim somewhere in the
   rendered text. A miss means the renderer dropped or mutated the fact:
   **reject the document and re-render** (retry budget: 3 attempts, then
   fall back to the template renderer for that one document — never silently
   ship a doc that failed this check).
2. **Number-leakage scan.** Regex-extract every numeric token
   (`\d[\d,]*\.?\d*`) from the rendered text. Every extracted number must
   normalize (strip commas/currency symbols) to either (a) a registered
   `Quantity.value`, (b) a date component (year/month/day — dates are
   pervasive and legitimately unregistered numbers; whitelist via a date-
   pattern regex, not a blanket exemption), or (c) a small fixed whitelist of
   narrative connective numbers registered per-template (page numbers,
   "first", "second" spelled or numeral in a fixed enumerated list). Any
   number that survives all three checks is a **fabricated figure** — the
   renderer invented a number that isn't in the registry. This is exactly the
   failure mode `core/actions.py`'s `ungrounded_numbers()` gate polices on the
   *swarm's* output (see CLAUDE.md "Number-grounding gate"); the same
   discipline applies here on the corpus's *input* side, and for the same
   reason: an unregistered number in the corpus has no ground truth, so any
   checklist item built from it would be unscoreable and any condition
   "citing" it would be uncheckable.
3. Both checks are pure string/regex operations over Stage B's output only —
   no LLM judge, so `verify_render` itself introduces no new leakage or
   nondeterminism risk.

---

## 2. Corpus sizing — 1x / 4x / 16x, and compatibility with `eval/packs.py`

**Explicit compatibility statement:** the synthetic-world corpus is written in
the **exact same JSONL pack schema** `eval/packs.py` already defines
(`build_pack`, lines 157-248: one line per chunk, `{"text", "source_tag",
"url"}`) and read back by the exact same `load_pack()` (`eval/packs.py:251-273`).
`run_swarm.py --corpus=pack:<path>` (lines 376-394) does not change at all —
`load_pack()` doesn't know or care whether a chunk came from live retrieval or
from a fictional-world renderer. This is the single largest reuse win in this
spec: Stage 1 needs zero changes to the pack-consumption path, only a new
pack-*production* path.

Field-level mapping:

| Pack JSONL field | Synthetic-world source |
|---|---|
| `text` | one document's rendered text, split into chunks at paragraph boundaries (mirror `core/intake.py`'s existing chunking granularity — reuse `chunk_corpus`'s chunk-size constant rather than inventing a new one) |
| `source_tag` | `doc_id` (e.g. `"acme_10k_fy2024q3"`) — NOT a URL, since these documents don't have one; this is a deliberate deviation from `_evidence_from_pack`'s tag rendering (`eval/ab_harness.py:306`, `f"[{tag}]\n{text}"`), which works unchanged because it only ever treats `source_tag` as an opaque label |
| `url` | `""` always. `load_pack()` (`eval/packs.py:268-272`) already handles a blank `url` gracefully (it isn't read at all by `CorpusChunk`) |

Sizing mechanics — a new `build_pack_from_world()` in `eval/packs.py` (or
`eval/worlds.py`; naming decision deferred to implementation, but it must live
where `build_pack()` lives so `eval/ab_harness.py`'s import surface stays a
one-line branch, see §5):

```python
def build_pack_from_world(world: FactRegistry, docs: list[RenderedDoc],
                           scale: str, out_dir: Path, pid: str,
                           target_chars: int | None = None) -> Path:
```

- Reuses `SCALE_CHARS` (`eval/packs.py:45-49`) and `pack_path_for` /
  `_meta_path_for` (`eval/packs.py:73-78`) verbatim — same defaults (24K/96K/
  384K chars), same override via `target_chars`, same reuse-if-exists
  determinism contract (`eval/packs.py:176-178`).
- Unlike `build_pack()`, there is no retrieval loop (`_gen_queries` / live
  `CompositeRetriever` calls, `eval/packs.py:180-222`) — instead, chunks are
  emitted from the rendered `docs` list in a fixed order (document generation
  order, itself deterministic from the seed) until `target_chars` is hit.
- **Filler documents for scale.** A `1x` world's real content (the template's
  core document set, §1.2) may not reach even the 24K floor, and definitely
  won't reach 384K at `16x` — real 10-Ks and news archives are not that long
  per-entity. Scale is hit by generating *more of the same kind of document*
  (more quarterly transcripts, more news-archive articles, more witness
  statements) from the same registry, each new document drawing a **fresh
  sample of already-registered facts** (never inventing new ground truth) plus
  `filler_seed_material` (§1.3) connective prose. This keeps the checklist
  identical across scales (§3 — same facts, same items) while the corpus
  genuinely grows, which is the property the over-context comparison needs:
  1x/4x/16x must vary corpus size, not corpus *information content* (that
  would confound "more chars" with "more facts to find," muddying the F vs. A
  comparison in the same way `OVERCONTEXT_EVAL_PLAN.md`'s live-retrieval packs
  already control for by building once and reusing across conditions).
- The `.meta.json` sidecar (`eval/packs.py:232-244`) gets the same honest
  achieved/target accounting, plus world-specific provenance: `world_seed`,
  `template_name`, `n_entities`, `n_facts`, `n_contradictions`,
  `fact_registry_path` (so `eval/world_score.py`, §3, can find the ground
  truth for any pack without re-deriving it).

---

## 3. Checklist + mechanical scorer (`eval/world_score.py`, proposed)

**No LLM in the headline metric** — this is the load-bearing design
constraint from THEFUTURE.md §3 ("Scoring is grounded accuracy against the
corpus, mechanically checked — not prose quality judged by an LLM") and §5
rule 3. `eval/judge.py`'s pairwise LLM judge stays exactly as-is and keeps
running for secondary prose comparison (§5) — it is not replaced, just no
longer the number the decision gate reads.

### 3.1 Checklist item types

Every item is derived mechanically from a `FactRegistry`, never hand-written,
so there is no experimenter judgment call between "world exists" and
"checklist exists" for the facts that matter (contrast `OVERCONTEXT_SET`'s
`must_include`, which is hand-written generic vocabulary per
`eval/prompts.py:196-200` precisely because there was no ground-truth
generator to mine from — that gap is what Stage 1 closes).

| Item type | Source | Check |
|---|---|---|
| `atomic_fact_present` | any `Entity`/`Event` mentioned in a prompt's scope | normalized substring match against the entity's `name` OR any registered `aliases` (case-insensitive, punctuation-folded) |
| `quantity_correct` | `Quantity` | extract numeric tokens from the answer within a small window of the metric's registered surface-form keywords (metric name / unit / entity name), normalize, compare to `Quantity.value` within `tolerance_pct`. A number present but outside tolerance is a **miss**, not a partial hit — silently rounding correctness would hide exactly the fabrication failure mode this eval exists to catch |
| `contradiction_surfaced` | `Contradiction` | hit requires BOTH: (a) the answer's text contains anchors for both `fact_id_a` and `fact_id_b`'s values (or their source docs/entities), AND (b) a disagreement-marking cue is present nearby (reuse the cue vocabulary already proven in `OVERCONTEXT_SET`'s `must_include`, `eval/prompts.py:208, 220, 231, 240, 254, 263`: disagree/contradict/conflict/dispute/mixed). Scoring "picked the right side" is a SEPARATE, stricter item only emitted when `Contradiction.ground_truth != "genuinely_ambiguous"`: does the answer's stated value match the ground-truth fact rather than the false one? |
| `citation_grounding` | every inline citation the condition emits (only meaningful for condition A, whose citation tags survive into `citations.json`/`renderer_audit.json` before `eval/judge.py:normalize()` strips them — **this scorer must run on the raw pre-normalized answer/citation artifacts, not the judge's normalized text**) | resolve the cited signal/chunk id to its `source_tag` (== `doc_id`), then check the CLAIM attached to that citation actually appears (verbatim-ish, same substring-match discipline as `atomic_fact_present`) in that `doc_id`'s rendered text per the registry's `extracted_from` mapping. This is citation PRECISION (cited support that's actually there) as opposed to the recall-flavored `atomic_fact_present`/`quantity_correct` items |

### 3.2 Normalization and aliasing rules

- Case-insensitive, whitespace-collapsed, punctuation-folded (strip `,`, `$`,
  `%`, hyphens in compound tokens) before any substring match — same
  discipline as `eval/judge.py:factual_score` (`eval/judge.py:186-194`), which
  this scorer's coarse compatibility path (below) directly reuses.
- Numbers: parse via a tolerant float extraction (strip thousands separators
  and currency symbols) before tolerance comparison; `tolerance_pct` is a
  per-`Quantity` field (default 2.0) because some metrics are naturally
  reported rounded (population figures) and some are not (an exact vote
  count) — a single global tolerance would either be too loose for exact
  counts or too strict for rounded ones.
- Aliases: `Entity.aliases` is the alias table; there is no fuzzy/embedding
  matching in the headline scorer (that would reintroduce an LLM-adjacent,
  non-mechanical judgment call) — if a condition uses a paraphrase not in the
  registered alias list, it's a miss. This is deliberately strict; it's also
  why alias lists should be generated generously (3-5 forms per entity: full
  name, short name, ticker/acronym where applicable) at Stage A generation
  time.
- **Compatibility path:** for every world, also emit a `must_include`-shaped
  list (`list[list[str]]`, same alias-form-per-item structure as
  `eval/prompts.py`'s existing `Prompt.must_include`) derived from just the
  `atomic_fact_present` items, so `eval.judge.factual_score()` and
  `eval.judge.factual_table()` keep working unmodified on synthetic prompts
  too (§5) — this is a coarse recall-only cross-check against the richer
  `world_score.py` output, useful for sanity-checking the two scorers agree
  in direction, not a replacement for either.

### 3.3 Scoring output schema

```jsonc
// eval/results/<exp>/world_scores.jsonl — one row per (pid, condition, item)
{"pid": "...", "condition": "A", "item_id": "q_acme_revenue_fy24q3",
 "item_type": "quantity_correct", "hit": true, "detail": "found 340.2M within 2.0% of 341.0M"}
```

```jsonc
// eval/results/<exp>/world_report.md — aggregation, mirrors eval/judge.py's
// report shape (Wilson CIs, per-condition tables) but for MECHANICAL scores:
{
  "world_seed": 42, "template": "invented_company", "scale": "16x",
  "per_condition": {
    "A": {"atomic_fact_present": 0.83, "quantity_correct": 0.71,
          "contradiction_surfaced": 0.60, "citation_grounding": 0.94, "n_items": 40},
    "F": {"atomic_fact_present": 0.55, "quantity_correct": 0.40,
          "contradiction_surfaced": 0.20, "citation_grounding": null, "n_items": 40}
  },
  "paired_win_rate": {"A_vs_F": {"n": 20, "wins": 13, "ties": 3, "losses": 4,
                                  "win_rate": 0.675, "wilson95": [0.48, 0.83],
                                  "real_win": false}}
}
```

`paired_win_rate` reuses `eval/judge.py`'s `wilson()` and `_summarize_pair()`
(`eval/judge.py:201-210, 288-309`) by import rather than reimplementation —
per-prompt "win" = condition's overall mechanical score (mean across its
items) is strictly higher than the opponent's; this keeps the Wilson-CI
math and the "real win = lower bound > 0.5" convention identical across the
mechanical and LLM-judged reports, so a reader doesn't have to learn two
different statistics.

---

## 4. Validity gates (pre-registered, from THEFUTURE.md §5)

### 4.1 Gate 1 — B-from-memory ceiling

**Threshold: reject the world if condition B's mechanical `world_score`
(mean hit rate across ALL item types, or just `atomic_fact_present` +
`quantity_correct` — the two item types a memory-only answer could
conceivably satisfy by chance/generic hedging) is ≥ 0.20.**

This is not a new number invented for this spec — it is THEFUTURE.md §5 rule
2 verbatim ("Every prompt set ships with a B-from-memory validity check; B ≥
20% on the checklist invalidates the set"), applied to a checklist instead of
a prose prompt set. The **target** for an accepted world should be much
lower in practice — ideally ≤ 0.05, since every fact is a freshly-invented
name/number B has structurally never seen — and a world landing in the 5-20%
band is worth inspecting by hand even if it technically passes (likely cause:
a generic hedge like "revenue likely grew" landing inside `tolerance_pct` of
the registered value by coincidence, or an alias colliding with a common
word). 20% is the hard ceiling, not the design target.

Why B can score above 0% at all even on invented facts: base-rate guessing on
directional/qualitative items (a model guessing "revenue increased" has some
chance of matching a `quantity_correct` item's sign, though not its value
within tolerance) and a well-hedged answer sometimes satisfying
`contradiction_surfaced`'s cue-word check generically ("sources may
disagree on this") without engaging any real contradiction. The scorer's
strict tolerance and dual-anchor requirement for `contradiction_surfaced`
(§3.1) are the main defenses; the gate is the backstop.

### 4.2 Gate 2 — name-collision leakage check (cheap, runs BEFORE corpus build)

Distinct from Gate 1 and cheaper: before rendering any documents, take every
`Entity.name` and run a single small/fast model call per batch of entities
("Have you heard of any of these: {names}? Answer only with any that ring a
bell, or 'none'.") — catches the generator accidentally minting a name that
collides with something real (a company template drawing from a small noun
pool could plausibly mint "Apple Analytics" or a country template mint
"Atlantis" as a joke-name that's actually a known fictional referent with its
own pretraining associations). This gate runs on Stage A output alone (no
corpus needed yet), so a failure is caught and re-rolled (regenerate just the
colliding entity's name, not the whole world) before any of the expensive
document-rendering budget (§1.4) is spent.

### 4.3 Where gates run, and what happens on failure

```
generate_world(seed)
  -> Stage A: build_fact_registry(seed, template)
  -> Gate 2: name_collision_check(registry.entities)         # fast, cheap
       fail -> reroll colliding entity name(s), retry (cap 5), else raise
  -> Stage B: render_documents(registry, seed, renderer)
  -> Stage C: verify_render(registry, docs)                  # mechanical, §1.4
       fail on a doc -> re-render that doc (cap 3), else fall back to template
  -> Gate 1: run condition B (zero-evidence direct call) against the
             world's checklist, score mechanically
       fail (>= 0.20) -> log to eval/results/world_gate_log.jsonl with
                          {seed, template, b_score, reason}, regenerate the
                          WHOLE world with seed+1 (facts, not just names,
                          since a passing name-check doesn't guarantee the
                          fact combination as a whole isn't guessable),
                          cap total regeneration attempts (5) before raising
                          — a template that can't pass Gate 1 in 5 tries
                          needs a design fix (probably: entity/metric pools
                          too generic, or too few facts to make guessing hard),
                          not more retries
  -> world accepted: FactRegistry + rendered docs + gate_log entry persisted
     under eval/worlds/<seed>/
```

Gate 1 is themselves an LLM call (condition B) plus a mechanical score — no
new judge dependency, it reuses the exact `DirectModel` + `_STRONG_DIRECT`
path condition B already uses in `eval/ab_harness.py` (`gen_direct`,
`ab_harness.py:317-322`), just pointed at the checklist-scoring function
instead of the pairwise judge.

---

## 5. Harness integration

### 5.1 CLI

```bash
# Build/validate a world once (idempotent — reuses if eval/worlds/<seed>/ exists
# and its gate_log entry shows accepted=true):
python -m eval.worlds --seed 42 --template invented_company

# Real run, gates the plan's Stage 1 experiment:
GROQ_API_KEY=... python -m eval.ab_harness \
    --name synth_16x --prompt-set synthetic --world 42 \
    --pack-scale 16x --mini 8 --conditions ABEF \
    --model llama-3.1-8b-instant

# Mechanical scoring (headline metric — no LLM):
python -m eval.world_score eval/results/synth_16x

# Secondary prose comparison (existing judge, unchanged):
GROQ_API_KEY=... python -m eval.judge eval/results/synth_16x \
    --judge-model openai/gpt-oss-120b
```

### 5.2 What changes in each file (minimal diff, by design)

- **`eval/worlds.py` (new).** Everything in §1 and §4: `FactRegistry` +
  dataclasses, `build_fact_registry`, `render_documents` (LLM + template
  paths), `verify_render`, `name_collision_check`, `generate_world` (the
  orchestrator above), and a `synthetic_prompts_for_world(world) ->
  list[Prompt]` builder (mirrors `eval/prompts.py`'s `Prompt` dataclass
  exactly — reusing the type rather than inventing a parallel one — but is a
  function, not a module-level constant like `OVERCONTEXT_SET`, because the
  set depends on which world/seed is loaded at runtime). A CLI entry point
  (`python -m eval.worlds --seed N --template T`) for building/inspecting a
  world standalone, independent of any harness run.
- **`eval/packs.py`.** One addition: `build_pack_from_world()` (§2), placed
  next to `build_pack()` and reusing `SCALE_CHARS`/`pack_path_for`/
  `_meta_path_for` (`eval/packs.py:45-49, 73-78`) unchanged. `load_pack()`
  (`eval/packs.py:251-273`) is untouched — this is the compatibility point
  emphasized in §2.
- **`eval/ab_harness.py`.** Minimal, localized changes:
  - `build_parser()` (`ab_harness.py:666-720`): extend the `--prompt-set`
    choices tuple at line 698 from `("default", "overcontext")` to add
    `"synthetic"`; add `--world` (int, the seed).
  - `run_experiment()` (`ab_harness.py:522`): the `base_set` selection at
    line 523-524 gains a third branch — `promptset.synthetic_prompts_for_world
    (eval.worlds.load_world(args.world))` when `args.prompt_set ==
    "synthetic"` (raise clearly if `--world` wasn't passed).
  - The pack-building block (`ab_harness.py:542-550`) gains one branch: if
    `args.prompt_set == "synthetic"`, call `build_pack_from_world(world,
    args.pack_scale, packs_dir, pid=p.pid, target_chars=args.target_chars)`
    instead of `build_pack(p.text, ...)`. Everything downstream — condition A
    via `--corpus=pack:<path>` (`gen_swarm`, `ab_harness.py:585`), condition F
    via `_evidence_from_pack` (`gen_direct_rag`, `ab_harness.py:325-344`), the
    model-parity guard (`ab_harness.py:471-501`) — is untouched, because it
    already only knows about pack *paths*, never how a pack was produced.
  - **Enforce `--pack-scale` is required when `--prompt-set synthetic`** (raise
    a clear `ValueError` in `main()` otherwise): without a pack, condition A
    would fall through to live retrieval/placeholder corpus for an invented
    world (nonsensical — there is nothing to retrieve), and condition F's
    `_retrieve_evidence` (`ab_harness.py:259-286`) would hit a real search
    backend with a fictional company name and get zero or garbage results
    silently, corrupting the comparison without an error.
- **`eval/judge.py` — no changes.** It keeps doing exactly what it does now
  (pairwise LLM judge on `normalize()`d prose, `factual_score`/
  `factual_table` on `must_include`), consuming the compatibility
  `must_include` lists synthetic prompts also carry (§3.2). It remains the
  SECONDARY metric per §5.3 below, never modified to understand worlds.
- **`eval/world_score.py` (new).** Standalone: reads
  `eval/results/<exp>/conditions.jsonl` + `run_meta.json` (which must record
  the `world_seed`/`fact_registry_path`, analogous to how it already records
  `pack_paths`, `ab_harness.py:655`) + the world's `FactRegistry`, and writes
  `world_scores.jsonl` + `world_report.md` (§3.3). Does not touch
  `eval/judge.py` at all.

### 5.3 Decision-gate analysis plan (THEFUTURE.md Stage 1)

1. Build/validate one world per template at each of 1x/4x/16x (reuse the same
   `FactRegistry` across scales — only the pack differs, per §2's filler-doc
   mechanism — so scale is the only varying dimension, matching
   `OVERCONTEXT_EVAL_PLAN.md`'s existing "build once, reuse across
   conditions" fairness principle).
2. **Gate check is mandatory before trusting any A-vs-F number**: confirm
   `eval/results/world_gate_log.jsonl` shows this world's B-score < 0.20 (§4.1)
   at the scale being analyzed — B is re-scored per scale since a larger pack
   changes nothing for condition B (it never sees the pack) but the
   analysis should still reference the specific gate-log entry for
   traceability.
3. Run `ABEF` at each scale (`n >= 20` per prompt per THEFUTURE.md §3's win
   condition — an upgrade from `OVERCONTEXT_EVAL_PLAN.md`'s `n >= 8`, which
   THEFUTURE.md's own verdict-triggering run showed was too small to trust:
   see `THEFUTURE.md` §0, "the existing n=2-8 runs prove nothing either way"
   carried over from CLAUDE.md's rules of evidence).
4. Read the **mechanical** `world_score.py` output as the headline
   (`paired_win_rate.A_vs_F`, §3.3), not the LLM judge's pairwise win-rate —
   THEFUTURE.md §3 is explicit that prose-quality judging is not the product
   here. The LLM judge report remains useful as a secondary check (did A's
   answer read worse despite scoring the same facts? that's an honest
   finding too, just not the gating one).
5. **Branch on the result at 16x:**
   - **F ≈ A** (Wilson lower bound on `A_vs_F` win-rate does not clear 0.5):
     the current orchestration adds nothing even in its designed-for-favorable
     regime (large corpus, synthetic facts, no parametric escape hatch).
     Per THEFUTURE.md §4 Stage 1: do not iterate on the swarm; proceed to
     Stage 2 (claims ledger) with the current pipeline demoted to the
     ablation baseline.
   - **A > F** (Wilson lower bound clears 0.5): the coverage mechanism has
     value under the fair-fight conditions; Stage 2 hardens it rather than
     replacing it outright.
   - **Either way, publish the numbers** (THEFUTURE.md §4: "negative results
     from this harness are publishable; that is the point of the harness") —
     write `world_report.md` and the gate log regardless of which branch
     fires, and report the cost multiple alongside any win exactly as
     `eval/judge.py`'s existing report does (THEFUTURE.md §5 rule 4).
6. Repeat at 1x and 4x for the size-curve; the prediction under the
   compression thesis is A≈F at 1x with the gap (if any) growing through
   4x/16x, mirroring `OVERCONTEXT_EVAL_PLAN.md`'s original prediction, now
   on facts that can't be satisfied from memory.

---

## 6. Test plan

### 6.1 Unit tests (`tests/test_worlds.py`, `tests/test_world_score.py`, new)

All of these must run fast and without a live API key — use the template
renderer path (§1.4) and/or `MockLLM` wherever an LLM call is unavoidable
(the collision check, §4.2, and Gate 1's condition-B call, §4.3), consistent
with the existing convention (`MOCK_LLM=1` for plumbing, CLAUDE.md "Mock mode
is for plumbing checks only").

- **`test_world_determinism`**: `generate_world(seed=42, renderer="template")`
  called twice from a clean `eval/worlds/` produces byte-identical
  `FactRegistry` (dataclass equality) and identical rendered document text.
  Called a THIRD time after deleting only the registry (not the rendered
  docs) must still reuse the existing rendered docs unchanged (proves the
  reuse-if-exists contract, §1.4, not just full-pipeline determinism).
- **`test_fact_registry_referential_integrity`**: every `CausalRelation.
  cause_id`/`effect_id`, every `Contradiction.fact_id_a`/`fact_id_b`/
  `doc_id_a`/`doc_id_b`, and every `Event.entity_ids` entry resolves to a
  fact/entity/doc that actually exists in the registry/document set — a
  small hand-built fixture registry (3 entities, 5 facts, 1 contradiction)
  plus a fuzz pass over `build_fact_registry` output at several seeds.
- **`test_verify_render_catches_dropped_fact`**: hand-craft a `RenderedDoc`
  that's missing a fact its `extracted_from` claims to include; assert
  `verify_render` flags it and does NOT silently pass.
- **`test_verify_render_catches_number_leakage`**: hand-craft a `RenderedDoc`
  containing a number that isn't in the registry, isn't a date, isn't on the
  connective whitelist; assert it's flagged as a fabricated figure.
- **`test_scorer_atomic_fact_present`** / **`test_scorer_quantity_tolerance`**
  / **`test_scorer_contradiction_surfaced`** / **`test_scorer_citation_
  grounding`**: hand-built fixture answers (correct, wrong-value-out-of-
  tolerance, missing entirely, hedged-but-not-anchored) run through each
  `world_score.py` item scorer in isolation; assert expected hit/miss per
  the rules in §3.1-3.2, including the tolerance boundary (value at exactly
  `tolerance_pct` should hit; just past should miss) and the
  dual-anchor requirement for `contradiction_surfaced` (a cue word alone,
  with no anchor to either fact's value, must NOT count as a hit).
- **`test_gate1_b_ceiling_fires`**: feed a stubbed condition-B answer scoring
  25% mechanically into the Gate 1 check; assert `generate_world` logs a
  rejection to `world_gate_log.jsonl` and regenerates with an incremented
  seed (assert the retry actually used a different seed, not the same one).
- **`test_gate2_name_collision_fires`**: seed an entity named `"Tesla"` (a
  known real-world collision) into the collision checker (stubbed/MockLLM
  response simulating "yes, I've heard of this"); assert it's flagged and
  rerolled before any document rendering is attempted.
- **`test_pack_from_world_schema_compat`**: `build_pack_from_world()` output,
  round-tripped through `eval.packs.load_pack()`, produces `CorpusChunk`
  objects with non-empty `text`/`source_tag` and the expected `chunk_id`
  format (`pack_%05d`, matching `eval/packs.py:269`) — proves the schema
  compatibility claimed in §2 mechanically, not just by inspection.

### 6.2 One MOCK end-to-end pass

```bash
MOCK_LLM=1 python -m eval.worlds --seed 1 --template invented_company \
    --renderer template
MOCK_LLM=1 python -m eval.ab_harness --name synth_mock --prompt-set synthetic \
    --world 1 --mini 2 --pack-scale 1x --conditions ABEF
python -m eval.world_score eval/results/synth_mock
```

Assert: `conditions.jsonl` has rows for all 4 conditions × 2 prompts;
`world_scores.jsonl` and `world_report.md` are written and every score is in
`[0, 1]`; `run_meta.json` carries `mock: true` and the existing "MOCK run —
plumbing only, NOT empirical (P0.1)" warning field (`ab_harness.py:656-657`)
unchanged, and `world_report.md` must carry the equivalent warning (do not
let the new report format quietly drop the P0.1 guard the existing one
already has).

---

## 7. Effort estimate and build order

| # | Component | Effort | Depends on |
|---|---|---|---|
| 1 | `eval/worlds.py`: dataclasses + `build_fact_registry` (pure Python, template-based, no LLM) | Medium (1-2 days) | — |
| 2 | Template document renderer (deterministic, zero API cost) | Small-Medium (0.5-1 day) | 1 |
| 3 | `verify_render` mechanical checker (§1.4) | Small (0.5 day) | 1, 2 (needed to test against template output first) |
| 4 | LLM-backed renderer (Groq, temp 0, reuse-if-exists caching) | Medium (1 day) | 1, 3 |
| 5 | `build_pack_from_world()` in `eval/packs.py` + filler-doc scaling (§2) | Small (0.5 day) | 1, 2 or 4 |
| 6 | `eval/world_score.py`: 4 item-type scorers + normalization/aliasing + Wilson pairing (import from `eval/judge.py`) | Medium (1-1.5 days) | 1 |
| 7 | Validity gates: Gate 1 (B-ceiling), Gate 2 (name collision), retry/reroll wiring, `world_gate_log.jsonl` (§4) | Small-Medium (0.5-1 day) | 1, 6 (Gate 1 needs the scorer) |
| 8 | `eval/ab_harness.py` wiring: `--prompt-set synthetic`, `--world`, pack-building branch, required-pack-scale enforcement (§5.2) | Small (0.5 day) | 5, 1 |
| 9 | Unit tests (§6.1) | Medium (1 day) | 1-7 |
| 10 | One MOCK end-to-end pass + fix plumbing bugs it surfaces (§6.2) | Small (0.5 day) | 8, 9 |
| — | **Total implementation** | **~6.5-9 days** | |
| 11 | Real run (`GROQ_API_KEY`, 1x/4x/16x, `n >= 20`, gate checks, decision per §5.3) | Not implementation effort — wall-clock/API-cost bound, likely 1-2 days of run time plus judge passes | 1-10 complete |

**Build order follows the table's numbering** — it is already dependency-
ordered. The one deliberate sequencing choice worth calling out: the template
renderer (#2) and `verify_render` (#3) are built and tested against each
other BEFORE the LLM renderer (#4) exists, so the fact-fidelity checker is
proven correct against a renderer that cannot possibly hallucinate (template
substitution has no failure mode to catch) before being pointed at one that
can. Building #4 first would mean debugging two new, interacting failure
surfaces (a nondeterministic renderer and an unproven checker) at once.
