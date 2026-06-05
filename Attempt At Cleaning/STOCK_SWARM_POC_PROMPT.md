# Claude Code prompt — Stock Swarm POC (executable plan)

Paste everything below the `---` line into a fresh Claude Code session from the
repository root
(`C:\Users\agsse\Downloads\ai_swarm_mechanics-main (4)\ai_swarm_mechanics-main`),
then `cd "Attempt At Cleaning"`.

This document is the **single source of truth** for the Stock Swarm POC. It is
written so that *any* new session can open it, jump to the lowest-numbered
unchecked stage in **Part 5**, and execute it without re-deriving context. Each
stage has a **Goal**, the **Files to touch**, an **Acceptance test**, and a
**Done-when**. Do the stages in order — later stages assume earlier ones exist.

---

You are working in the `Attempt At Cleaning/` folder of a stigmergic multi-agent
swarm codebase. **Read `Attempt At Cleaning/CLAUDE.md` end-to-end first.** Then
read this whole document before writing any code. Confirm the exact line numbers
and symbol names referenced below against the live code — the repo moves and
this memo may have drifted by a few lines.

## Part 0 — Orientation and hard invariants

The swarm is a typed DAG of "signals" deposited by role-specialized LLM agents
into a shared `SignalStore`. Diversity comes from **information partitioning**
(what each agent is shown), not prompt/temperature tweaks. A `coding` task type
already demonstrates the full pattern for adding a domain: specialized role
classes registered via `core/role_registry.py`, its own topology template,
survival profile, and a synthesizer subclass. **The Stock Swarm is built exactly
like `coding`.** Study `agents/coding_roles.py` as your template — it is the
closest existing analog (it has a domain data check `ast.parse`, a real
verifier `TestValidator` that runs a subprocess, and a `CodeSynthesizer` that
assembles a structured artifact). The Stock Swarm replaces "does the code
parse/pass tests" with "does the numeric claim match ground-truth market data."

These invariants are **non-negotiable**. Violating any of them is a defect, not
a tradeoff:

1. **No-leak rule.** Agents observe other agents' work only as signal artifacts
   (content + ID + structural metadata). Never another agent's reasoning chain,
   ancestry text, or chain-of-thought. Enforced in `core/signal_store.py` and
   `agents/base.py` (`_assert_no_leak`). Any new stock role must pass the same
   assertion.
2. **Partition invariant.** Every `INITIAL`/`SUPPORT` deposit must carry a
   non-empty `partition_id`. `deposit()` raises `AssertionError` otherwise. See
   CLAUDE.md "Partition invariant (hard enforcement)." Stock scouts get
   `partition_id` from their analytical lens (Part 3); developers inherit it.
3. **Point-in-time discipline (NEW, and the single biggest correctness risk).**
   When the swarm runs *as of* a historical date `T`, **every** input — news,
   fundamentals, prices, *and KB knowledge* — must be restricted to information
   that existed on or before `T`. Any leak of post-`T` information ("look-ahead
   bias") silently inflates backtest results and makes the whole "make money"
   claim worthless. Treat this like the no-leak rule: loud, central, asserted.
4. **Free only.** No paid APIs, no API keys that cost money. `yfinance` (free,
   no key) is the sanctioned market-data source *if it still works at build
   time* — verify it is not deprecated/broken before relying on it (the project
   owner monitors this). All search/news retrieval reuses the existing free
   stack (`core/search_tool.py`, DuckDuckGo) and the refined search system this
   plan builds on top of it.
5. **Don't casually refactor.** `run_swarm.py`, `core/signal_store.py`,
   `core/projection.py`, and `agents/synthesizer.py` are load-bearing for every
   task type. Add branches keyed on `task_type == "stock"`; do not rewrite
   shared paths.

## Part 1 — North Star (what "success" means)

> **The swarm makes money.** Given an as-of date `T`, it sources a promising
> stock from recent news, gathers ground facts, reasons over a 1-week-to-
> 3-month horizon, emits a report, and commits to a **projected return
> (gain or loss, in %)**. It is graded against the **realized** return over the
> same horizon. Aggregated across many historical `T`, it must beat a naive
> baseline.

The **gradable artifact** is a single structured prediction block the
synthesizer emits at the end of every run (full schema in Appendix A):

```json
{
  "ticker": "NVDA",
  "as_of_date": "2024-01-15",
  "horizon_days": 21,
  "direction": "long",            // long | avoid | short(optional)
  "predicted_return_pct": 8.5,
  "confidence": 0.62,             // 0..1
  "rationale_ref": "answer.txt",  // the human report
  "key_claims": ["INITIAL_00042", "..."]   // signal IDs backing the call
}
```

**Primary metric (this is what we optimize):** realized P&L of acting on the
prediction, aggregated over the backtest set, vs. a baseline. Concretely:
directional hit-rate, mean realized return on `long` calls, and a simple
simulated equity curve — all compared against buy-SPY / buy-and-hold / random
(Part 6 specifies the grader exactly).

**Secondary / diagnostic metrics** (explain *why* it wins or loses, keep
reasoning observable — these are the user's "observe reasoning" requirement):
verified-claim ratio (fraction of numeric claims that match ground truth),
calibration (does stated `confidence` track hit-rate), and a reasoning rubric
(did it cite verified numbers, separate fact from inference, name real risks).

## Part 2 — Diagnostic from the `anothergroq.txt` run

`outputs/anothergroq.txt` is a real Groq run (`llama-3.1-8b-instant` et al.) on
the debate thesis *"Cities should ban private cars to fight climate change."*
(`summary.json` at line 4838; final answer at line 4888). It surfaces concrete
quality failures the Stock Swarm must not inherit. Each is paired with the
refinement it implies. **Several of these are general pipeline wins — implement
them as task-agnostic improvements where noted, per the project goal that
refinements generalize to other tasks.**

| # | Observed failure (evidence) | Refinement | Where |
|---|---|---|---|
| D1 | **Truncated sentences** in the final answer — "...suggests that a gradual." / "...emphasizes the importance." Render caps cut mid-sentence. | Make render caps clause-aware (truncate at sentence boundary; if a `[CITATION]` tag is mid-sentence, keep the clause). Add a post-render check that flags dangling sentences. | `agents/synthesizer.py` render caps (`_SECTION_*` consts ~line 96–100); general fix. |
| D2 | **Heavy repetition** — the position synthesis restates "cities should ban private cars…" ~6×; clusters `INITIAL_00118/00178/00141/00248` are near-duplicates of one claim. | Tighten dedup at deposit and de-duplicate clusters at synthesis (merge clusters with centroid cosine ≥ `CLUSTER_JOIN_THRESHOLD` before rendering). For stock, partition by *lens* (Part 3) so scouts can't all pile onto the obvious claim. | `core/signal_store.py` dedup; `core/projection.py` merge; stock lens partitioning. |
| D3 | **Meta-commentary junk clusters** survived — "The artifact presents a clear thesis…", "Unfortunately, the provided evidence does not seem relevant…" (`INITIAL_00375/00036/00367/00249`). The 8B model evaluated the *prompt* instead of arguing it. `filters.py` catches some but not all. | Strengthen `is_junk_output` patterns (already has an "artifact presents" family — extend it). For stock, add a domain gate: **reject any INITIAL/SUPPORT that contains no ticker and no number** (a stock claim must reference a symbol or a metric). | `core/filters.py`; stock roles' deposit guard. |
| D4 | **Verification ≈ 0** — `avg_verification_score=0.0865`, `max=0.49`, `total_atoms=13`. The Wikipedia/DDG validator barely grounds anything; survival is essentially unverified. | This is the crux for stock. Replace soft text-matching verification with **hard numeric verification against yfinance ground facts** (DataValidator, Part 3). A claim "AAPL forward P/E ≈ 28" is checked against the real number; strength = closeness. This makes `verification_score` meaningful and lets the survival gate actually require it. | `agents/stock_roles.py::DataValidator`; `SURVIVAL_TASK_PROFILES["stock"]`. |
| D5 | **Inter-cluster contradiction spam** — a wall of "[INTER-CLUSTER CONTRADICTION] … share dissent signals ()" with **empty** parens (no actual shared dissent). Fires on topic similarity alone. | Gate the contradiction notice on non-empty shared dissent; collapse duplicates. General fix. | `agents/synthesizer.py::_detect_inter_cluster_contradictions`. |
| D6 | **`convergence_reason: cap_time`** at 915.8s / 349 iters — never reached quality convergence; lots of `429` Groq rate-limits burned wall-clock. | Two parts: (a) honor the `[groq] TIP` and run with `SWARM_MAX_TIME_S=1800`; (b) the token-bucket in `llm_groq.py` is correct — keep concurrency modest. For stock, the bounded data-fetch + numeric verification gives a *natural* convergence signal (claims either match ground truth or don't), so set a quality-based stop. | run env; `core/convergence.py`. |
| D7 | **One mega-cluster + dust** — `INITIAL_00178` had 34–58 members while dozens of singletons sat at `members=1`. Winner-take-all blob plus noise. | Lens partitioning (Part 3) structurally prevents one blob: fundamentals/technical/news/macro/risk claims live in different regions of embedding space. Keep peer-relative pruning (`PEER_PRUNE_*`) so rich clusters aren't over-cut. | stock topology + lens scouts. |
| D8 | **`critic_endorsements=0`** everywhere — critics only ever deposited `CRITIQUE_NEGATIVE`. | The stock `ValuationCritic` deposits `CRITIQUE_POSITIVE` when a claim is numerically consistent (mirrors `StaticCritic`'s parse-OK → positive path). Gives clusters a positive credibility channel. | `agents/stock_roles.py::ValuationCritic`. |

Write these findings into `docs/STOCK_SWARM_DIAGNOSIS.md` (create `docs/` if
needed) as you implement, so the rationale is captured next to the design.

## Part 3 — Target architecture

### 3.1 The loop (single-ticker, news-sourced, as-of `T`)

```
as_of_date T  (and optionally a universe/sector, or an explicit --symbol)
   │
   ├─ [Stage B] Idea sourcing:  refined search over news ≤ T  →  candidate tickers
   │                            → pick the most "promising" symbol (or use --symbol)
   │
   ├─ Ground facts:  StockDataProvider.get_snapshot(symbol, as_of=T)   ← yfinance, ≤ T
   │                 (price, mktcap, P/E, fwd P/E, PEG, EPS, rev growth,
   │                  margins, debt/equity, FCF, div, 52w range, SMA50/200,
   │                  analyst target, recent news headlines ≤ T)
   │
   ├─ Topology:  stock answer-space (stance × horizon × driver)   [Part 3.4]
   │
   ├─ SWARM ROUNDS over the SignalStore:
   │     Scouts (one per LENS, disjoint facts)  →  INITIAL claims
   │     DataValidator (numeric check vs snapshot) →  VERIFICATION  [runs early, Phase A]
   │     ThesisDeveloper  →  SUPPORT (bull/bear evidence, anticipates dissent)
   │     ValuationCritic  →  CRITIQUE_POSITIVE/NEGATIVE (numeric consistency)
   │     RiskHater        →  OBJECTION (concrete downside risks)
   │
   └─ EquityBriefSynthesizer:  report (answer.txt) + structured PREDICTION block
                               → graded later by the backtest harness vs realized return
```

`--symbol NVDA --as-of 2024-01-15` skips Stage B and runs the prediction engine
on a given name. **Build this given-symbol mode first** (Stage 3) — it is the
clean, gradable unit of the system. The news→symbol discovery front-end (Stage
6) is layered on after the engine is proven.

### 3.2 Lenses = partitions (the diversity engine, applied to one ticker)

For a single ticker there is no corpus to slice, so partition by **analytical
lens**. Each scout is assigned one lens and is shown *only* the facts for that
lens. This is the stock analog of disjoint corpus partitions and it structurally
prevents the D7 mega-blob:

| Lens (`partition_id`) | Facts shown | Example INITIAL |
|---|---|---|
| `valuation` | P/E, fwd P/E, PEG, P/S, EV/EBITDA, analyst target vs price | "NVDA trades at 34× fwd earnings vs its 5y median 31× → modestly rich." |
| `growth` | revenue/EPS growth, margins trend, guidance | "Revenue grew 22% YoY with expanding gross margin → growth intact." |
| `technical` | price vs SMA50/200, 52w range position, momentum | "Price is 12% above SMA200 and near 52w high → strong uptrend, extended." |
| `news_sentiment` | recent headlines ≤ T, analyst actions | "Three brokers raised targets this week → positive near-term catalyst." |
| `risk_balance_sheet` | debt/equity, FCF, customer/segment concentration | "Net cash positive, FCF margin 28% → low balance-sheet risk." |

These lens names are the scouts' `partition_id`s. `NUM_SCOUTS` should be ≥ the
number of lenses so each is covered (set via `SWARM_NUM_SCOUTS`).

### 3.3 Roles, traits, and habits (`agents/stock_roles.py`)

Subclass the base/`Developer`/`Synthesizer` exactly as `coding_roles.py` does.
The **traits and habits** below are what make this a powerful tool — they are
enforced in the prompts *and* in deposit guards (not left to model whim):

- **`LensScout`** (subclass of `Scout`/`BaseAgent`): one per lens. Habit:
  *every INITIAL must state a number with a unit and reference the ticker.*
  Deposit guard rejects claims with no ticker and no numeric token (fixes D3).
  Trait: separates **fact** ("P/E is 34", from the snapshot) from **inference**
  ("→ modestly rich", the claim).
- **`ThesisDeveloper`** (subclass of `Developer`): develops INITIALs into bull or
  bear SUPPORT, anticipating the strongest dissent (the base `Developer` already
  stashes dissent — keep that). Habit: cite the specific datum it builds on.
- **`ValuationCritic`** (mirror of `StaticCritic`): checks **numeric
  consistency** of a claim against the snapshot and arithmetic (e.g. does the
  cited P/E equal price ÷ EPS within tolerance?). Deposits `CRITIQUE_POSITIVE`
  when consistent (fixes D8), `CRITIQUE_NEGATIVE` when not, strength = degree of
  consistency.
- **`RiskHater`** (mirror of `EdgeCaseHater`): cycles a canonical **risk
  checklist** (valuation risk, growth deceleration, margin compression,
  competition/share loss, balance-sheet/liquidity, regulatory/litigation,
  customer concentration, macro/rate sensitivity, sentiment reversal) and asks
  whether the bull cluster ignores each. Deposits `OBJECTION` naming the
  specific risk.
- **`DataValidator`** (the heart — replaces Wikipedia text-matching, fixes D4):
  extracts the numeric claim from a signal, fetches the **ground-truth** value
  from `StockDataProvider`, and deposits `VERIFICATION` with
  `strength = closeness(claimed, actual)` (e.g. `1.0` within 2%, linearly to
  `0.0` past 25% error; `0.5` when the metric can't be resolved). Store the
  resolved fact in `metadata["atoms"]` so the genome/atom pipeline picks it up.
- **`EquityBriefSynthesizer`** (subclass of `Synthesizer`, mirror of
  `CodeSynthesizer`): renders the human report **and** the structured PREDICTION
  block (Appendix A). Sections: **Verdict** (direction + predicted_return_pct +
  confidence), **Key metrics table**, **Bull case**, **Bear case**,
  **Valuation**, **Catalysts (≤ T)**, **Risks**, **What would change the
  view**. Every numeric statement carries an `[as-of T]` stamp. Ends with a
  one-line **non-advice disclaimer**. The PREDICTION block is written to
  `prediction.json` in the run dir for the grader.

Global habits baked into prompts: ground every claim in a dated number; never
fabricate a figure not in the snapshot (the DataValidator will catch and
penalize it); state a falsifiable verdict; always emit "what would change my
mind." These are the testable, observable-reasoning behaviors.

### 3.4 Topology template (`core/topology.py`)

Add `_TOPOLOGY_TEMPLATES["stock"]` declaring 3 axes so scouts spread across the
answer space:
- Axis 1 `stance`: bullish | neutral | bearish
- Axis 2 `horizon`: 1-week | 1-month | 3-month
- Axis 3 `driver`: valuation | growth | technical | news | risk

Anchor corners: the strongest bull (growth+technical, 1–3mo), the strongest bear
(valuation+risk), and a neutral/range-bound case. Exclude: anything depending on
post-`T` information; options/derivatives strategies; positions requiring
leverage.

### 3.5 Data layer (`core/stock_data.py`) — free, provider-pattern, point-in-time

Define a `StockDataProvider` Protocol and a `YFinanceProvider` behind it so the
refined-search provider can be swapped in later with **zero caller changes**
(this honors the owner's "refined search generalizes to other tasks" goal while
shipping fast):

```python
class StockDataProvider(Protocol):
    def get_snapshot(self, symbol: str, as_of: date) -> Snapshot: ...   # facts ≤ as_of
    def discover_symbols(self, as_of: date, universe: str | None) -> list[str]: ...
    def realized_return(self, symbol: str, start: date, horizon_days: int) -> float: ...  # GRADER ONLY
```

- `get_snapshot` is what the swarm sees. It **must** clamp every field to
  `≤ as_of`. yfinance historical *prices* are genuinely point-in-time; yfinance
  *fundamentals* and *news* are "as of now" and are the look-ahead hazard — for
  true backtests these come from the **historical DB** the owner is building
  (Part 5, Stage 4). For a same-day ("as_of = today") smoke run, current
  yfinance data is fine.
- `realized_return` is the **reality** the grader uses (price at `start+horizon`
  ÷ price at `start` − 1). It is **never** exposed to any agent — keep it in the
  grader only. (Add an assertion / separate module boundary so no role can
  import it.)
- `discover_symbols` powers Stage B: pull tickers mentioned in news ≤ `as_of`
  via the refined search system, resolve company names → symbols.

Cache snapshots on disk keyed by `(symbol, as_of)` exactly like
`search_tool.py` caches queries, so repeated backtest runs hit disk not network.

### 3.6 KB as a growing "community of experts" (`core/knowledge_base.py`)

The cross-run KB (schema v3, genome-aware) is how the swarm "becomes its own
community of experts." Two additions:

1. **Dated entries (point-in-time safe).** Add `learned_from_date` to every KB
   entry. A run as-of `T` may only read KB entries with
   `learned_from_date ≤ T`. **Without this, the KB is a look-ahead leak** — the
   swarm would "remember" facts from the future. Bump schema → v4 and add the
   v3→v4 path to `kb_migrate.py`.
2. **Relationship channel.** Add a `relationships` store: typed edges between
   tickers (`peer_of`, `same_sector`, `competitor_of`, `supplier_of`,
   `substitute_for`). This is the "Walmart and Dollar General are both grocery
   stores" knowledge that later enables comparative reasoning. Populate it from
   verified co-occurrence + sector metadata. Keep it dated too.

## Part 4 — Exact wiring map (verify line numbers against live code)

Adding `task_type == "stock"` touches these and only these shared points (mirror
how `coding` is wired):

1. `run_swarm.py`
   - `TASK_PROMPTS["stock"]` (~line 113) — e.g. `"Assess whether to buy {prompt} over a 1-week-to-3-month horizon and project its return."`
   - `ROLES_FOR_TASK["stock"]` (~line 124) — `{"critic", "hater", "validator"}` (all roles; domain classes).
   - `_build_corpus_partitions` (~line 1442) — add a `task_type == "stock"` branch that builds **lens partitions** + the snapshot (mirrors the `coding` special-case that needs no corpus).
   - `main()` arg parsing (~line 1811+) — add `--symbol` and `--as-of` flags (like the existing `--corpus=`/`--mode=` parsing ~line 1888–1908). `build_task_prompt` (~line 183) already rejects unknown task types, so `"stock"` is valid once added to `TASK_PROMPTS`.
2. `core/role_registry.py` — add `_register_stock_roles()` + `_TASK_ROLE_OVERRIDES["stock"]` mirroring `_register_coding_roles()` (lines 46–85).
3. `core/topology.py` — `_TOPOLOGY_TEMPLATES["stock"]` (~line 69).
4. `core/config.py`
   - `SURVIVAL_TASK_PROFILES["stock"] = {"requires_verification": True, "credibility_chain_depth": 999, "support_diversity_min": 2}` (~line 417) — like `coding`, so numeric verification gates survival (this is what makes D4 bite).
   - Optional: `TASK_TO_BUNDLE["stock"]` / `TASK_TO_BUNDLE_SMALL["stock"]` (~line 808/816) for local model bundles.
5. `core/search_tool.py` — `_FOLLOWUP_MODIFIERS_BY_TASK["stock"]` (~line 47): e.g. `["earnings", "guidance", "analyst rating", "risks", "10-Q"]`.
6. `agents/synthesizer.py` — `_OUTPUT_STRATEGY_BY_TASK["stock"]` (~line 236). Start with `"sectioned"` (the structured-report shape fits) and override `synthesize()` in the `EquityBriefSynthesizer` subclass for the metrics table + PREDICTION block, exactly as `CodeSynthesizer` overrides it (`coding_roles.py` ~line 578).
7. **New files:** `agents/stock_roles.py`, `core/stock_data.py`, `eval/backtest.py` (grader), `eval/datasets/` (historical DB), `docs/STOCK_SWARM_DIAGNOSIS.md`, `docs/STOCK_SWARM_DESIGN.md`.

The Groq backend already routes per-role models (`core/llm_groq.py`); the new
roles reuse role names (`scout`/`developer`/`critic`/`hater`/`validator`/
`synthesizer`) so no router changes are needed.

## Part 5 — Staged executable plan

Do stages in order. Check the box and append a one-line note to the **Build log**
at the bottom when a stage is Done. Every stage must run under `MOCK_LLM=1`
without a GPU (mock for plumbing) and the test suite must stay green
(`MOCK_LLM=1 SWARM_MIN_TIME_S=0 SWARM_MIN_ITERATIONS=5 pytest tests/ -q`).

- [x] **Stage 0 — Spike: prove yfinance + point-in-time prices (½ day).**
  Goal: de-risk the data layer before building anything on it.
  **FINDING (2026-06-04):** yfinance 0.2.52 is installed and network reaches
  Yahoo, but **Yahoo aggressively rate-limits** (`YFRateLimitError`) even single
  requests from this environment, and `curl_cffi` (the standard browser-
  impersonation workaround) is not installed. yfinance is **not deprecated but
  not reliable for ad-hoc/bulk live fetching.** Consequence: do NOT fetch live
  per run — **ingest each (symbol, as_of) once, slowly, with backoff, into the
  frozen historical DB** (FrozenSnapshotProvider) and run the swarm against the
  frozen DB. This validates the plan's cache/freeze design and strengthens the
  case for the refined-search data path long-term. **Action items:**
  `pip install curl_cffi`; build a slow ingestion script (rate-limit aware) that
  writes `eval/datasets/<TICKER>/<as_of>.json`.

- [ ] **Stage 1 — Register the `stock` task type (skeleton).**
  Goal: `python run_swarm.py stock "NVDA" --symbol NVDA --as-of 2024-01-15` runs
  end-to-end under `MOCK_LLM=1` using the **default** roles (no stock roles yet).
  Wire Part 4 items 1, 3, 4, 5, 6 (`_OUTPUT_STRATEGY_BY_TASK["stock"]="sectioned"`).
  **Acceptance:** a run dir is produced with `answer.txt`/`summary.json`;
  `summary.json.task_type == "stock"`. **Done-when:** no crash, suite green.

- [ ] **Stage 2 — `core/stock_data.py` (provider + yfinance).**
  Goal: implement `StockDataProvider` Protocol, `YFinanceProvider`, on-disk cache,
  and the **grader-only** `realized_return` (with an import-boundary guard so
  roles can't reach it). Snapshot clamps to `≤ as_of`. **Acceptance:** unit test
  `tests/test_stock_data.py` covers snapshot shape, cache hit, `as_of` clamping
  (mock the network), and that `realized_return` is not importable from
  `agents/`. **Done-when:** tests pass.

- [ ] **Stage 3 — `agents/stock_roles.py` + given-symbol engine (the core).**
  Goal: implement `LensScout` (lens partitioning), `ThesisDeveloper`,
  `ValuationCritic`, `RiskHater`, `DataValidator`, `EquityBriefSynthesizer`;
  register them (Part 4 item 2). The synthesizer writes `prediction.json`
  (Appendix A). Add the **domain junk gate** (D3) and the **numeric verification**
  (D4). **Acceptance:** a real (non-mock) Groq run
  `GROQ_API_KEY=… SWARM_MAX_TIME_S=1800 python run_swarm.py stock "NVDA" --symbol NVDA --as-of <recent>`
  produces an equity brief with a metrics table, a bull and a bear cluster, a
  non-zero `verification_score`, and a valid `prediction.json`. **Done-when:**
  `prediction.json` validates against the schema and `predicted_return_pct` is a
  finite number with a horizon.

- [ ] **Stage 4 — Historical DB + point-in-time integrity.**
  Goal: define the historical dataset format under `eval/datasets/` (the owner
  populates it): per `(symbol, as_of)` a frozen snapshot + news ≤ `as_of`, plus
  a price series for realized returns. Make `YFinanceProvider` read from the
  frozen DB when `as_of` is historical (so fundamentals/news aren't look-ahead),
  falling back to live yfinance only when `as_of == today`. Add a **look-ahead
  assertion** that fails a run if any snapshot field has a timestamp > `as_of`.
  **Acceptance:** `tests/test_pointintime.py` asserts no field post-dates `as_of`
  and that the KB read-filter (`learned_from_date ≤ T`) holds. **Done-when:**
  tests pass; a historical run uses only ≤ `T` data.

- [ ] **Stage 5 — Backtest grader + "make money" scorecard (`eval/backtest.py`).**
  Goal: run the engine across a list of `(symbol, as_of, horizon)` cases, collect
  each `prediction.json`, compute realized return via `realized_return`, and emit
  a scorecard: directional hit-rate, mean realized return on `long` calls,
  simulated equity curve, **vs. baselines** (buy-SPY, buy-and-hold same names,
  random/coin-flip). Also the secondary metrics (verified-claim ratio,
  calibration). **Acceptance:** `python eval/backtest.py eval/datasets/<set>.json`
  prints a scorecard table and writes `eval/results/<ts>.json`. **Done-when:** the
  scorecard runs over ≥ 20 cases and reports swarm-vs-baseline. *This is the
  moment "does it make money?" becomes answerable.*

- [ ] **Stage 6 — News-driven symbol discovery (Stage B front-end).**
  Goal: implement `discover_symbols(as_of, universe)` over the refined search /
  news layer, and a discovery pre-stage in `run_swarm.py` so
  `python run_swarm.py stock "tech sector" --as-of <T>` (no `--symbol`) sources a
  promising ticker, then runs the Stage-3 engine on it. **Acceptance:** a run
  with no `--symbol` selects a symbol from ≤ `T` news and produces a prediction.
  **Done-when:** end-to-end news→symbol→prediction works on a historical `T`
  without look-ahead.

- [ ] **Stage 7 — KB accumulation + relationships (community of experts).**
  Goal: add `learned_from_date` (schema v4 + `kb_migrate.py` path) and the dated
  `relationships` channel (Part 3.6). Backtest runs in chronological order so the
  KB grows; later runs read only ≤ `T` knowledge. **Acceptance:** running the
  backtest set twice (cold KB vs warm KB) shows the warm run reusing prior
  verified facts (logged in `kb_diff.json`) with **no** post-`T` leakage.
  **Done-when:** v4 migration tested; relationship edges populated and dated.

- [ ] **Stage 8 — Refinements from Part 2 (generalize) + comparative (stretch).**
  Goal: land the task-agnostic fixes D1 (clause-aware caps), D2 (cluster merge),
  D5 (contradiction gating) so all task types benefit; then, as a stretch, add a
  comparative mode (`stock "WMT vs DG"`) that uses the relationship KB for
  side-by-side metrics and a relative verdict. **Done-when:** D1/D2/D5 fixes have
  tests; comparative mode emits a relative prediction (optional for the POC).

## Part 6 — Success criteria and the grader (precise)

The grader (`eval/backtest.py`) is the contract for "make money." Define it
exactly so results are reproducible and honest:

**Inputs:** a dataset of cases `{symbol, as_of, horizon_days}` (horizons in
{7, 21, 63} trading days ≈ 1w/1m/3m). For each case the engine produces
`prediction.json`.

**Per-case scoring:**
- `realized = realized_return(symbol, as_of, horizon_days)` (reality).
- `direction_correct = sign(predicted_return_pct) == sign(realized)` (for `long`
  vs `avoid`, treat `avoid` as predicting ≤ 0).
- `magnitude_error = |predicted_return_pct − realized*100|`.
- `pnl_if_acted`: `long` → `realized`; `avoid` → `0`; `short`(optional) →
  `−realized`.

**Aggregate (the scorecard):**
- **Hit-rate** = mean(`direction_correct`).
- **Mean realized return on `long` calls** and **mean P&L per case**.
- **Simulated equity curve** = compounding `pnl_if_acted` across cases (or
  equal-weight portfolio per `as_of`).
- **Baselines (must beat):** buy-SPY over the same windows; buy-and-hold the same
  tickers (ignore the swarm's direction call); random direction. Report swarm −
  baseline for each.
- **Calibration:** bucket by `confidence`, show hit-rate per bucket.
- **Verified-claim ratio:** mean over runs of (numeric claims passing
  DataValidator / total numeric claims).

**"Good" thresholds for the POC (tune in the dataset README):** hit-rate
materially > 50% and > buy-SPY hit-rate; positive mean P&L per case that exceeds
the buy-and-hold baseline after costs; verified-claim ratio > 0.6. Be ruthless
about look-ahead bias — a too-good result almost always means leakage (Part 0.3,
Stage 4 assertion). Report confidence intervals; ≥ 20 cases minimum, ≥ 100
preferred, spread across different `as_of` dates and market regimes.

## Part 7 — Guardrails, non-goals, risks

- **Look-ahead bias is the existential risk.** If the backtest looks great,
  suspect leakage first (post-`T` fundamentals/news/prices, or KB entries with
  `learned_from_date > T`). The Stage-4 assertion exists for this.
- **Not financial advice.** Every report ends with a one-line disclaimer; the
  PREDICTION block is for backtesting, not live trading.
- **Free-only.** No paid data/keys. If yfinance breaks, the refined-search
  provider becomes the data path (it's already the swap-in target).
- **Don't break invariants** (no-leak, partition, point-in-time). New roles must
  pass `_assert_no_leak` and supply `partition_id`.
- **Mock mode proves plumbing, not skill.** Never report P&L from `MOCK_LLM=1` /
  `outputs_mock/` — MockLLM emits seeded phrases regardless of input.
- **Non-goals for the POC:** intraday/HFT, options, leverage, multi-leg
  portfolios, real-money execution. Single-name, long/avoid, 1w–3mo only.

## Appendix A — `prediction.json` schema

```json
{
  "schema": "stock_prediction_v1",
  "ticker": "string",
  "as_of_date": "YYYY-MM-DD",
  "horizon_days": 21,
  "direction": "long | avoid | short",
  "predicted_return_pct": 8.5,
  "confidence": 0.62,
  "key_metrics": {"price": 0.0, "fwd_pe": 0.0, "rev_growth_yoy": 0.0, "...": 0.0},
  "bull_points": ["string"],
  "bear_points": ["string"],
  "key_risks": ["string"],
  "what_would_change_view": ["string"],
  "key_claim_ids": ["INITIAL_00042", "SUPPORT_00117"],
  "verified_claim_ratio": 0.0,
  "disclaimer": "Not financial advice; backtest artifact."
}
```

## Appendix B — environment & run cheatsheet

```bash
cd "Attempt At Cleaning"

# Plumbing (no GPU/network), suite green:
MOCK_LLM=1 SWARM_MIN_TIME_S=0 SWARM_MIN_ITERATIONS=5 pytest tests/ -q

# Skeleton run (Stage 1+):
MOCK_LLM=1 python run_swarm.py stock "NVDA" --symbol NVDA --as-of 2024-01-15

# Real Groq run (Stage 3+) — extend the wall clock (D6):
GROQ_API_KEY=xxxx SWARM_MAX_TIME_S=1800 SWARM_NUM_SCOUTS=5 \
  python run_swarm.py stock "NVDA" --symbol NVDA --as-of 2024-01-15

# Backtest scorecard (Stage 5+):
python eval/backtest.py eval/datasets/poc_set.json
```

Useful existing env knobs (see CLAUDE.md): `SWARM_MAX_TIME_S`,
`SWARM_NUM_SCOUTS`, `SWARM_NUM_VALIDATORS`, `SWARM_SURVIVAL_VERIFY_MIN`,
`GROQ_ROLE_*` (per-role model override).

## Appendix C — Quickstart (run it now)

```bash
cd "Attempt At Cleaning"

# 0. (once) generate a synthetic sample DB so the whole loop runs with NO network:
python tools/make_sample_dataset.py                 # -> eval/datasets/sample/

# 1. single prediction (MOCK_LLM = plumbing only; predictions will be 'avoid'):
MOCK_LLM=1 python run_stock.py --symbol NVDA --as-of 2024-01-15 --db eval/datasets/sample/snapshots
#    -> outputs/stock_NVDA_2024-01-15/{answer.txt, prediction.json, signals.json}

# 2. news-driven discovery (omit --symbol; reads _discovery/<as_of>.json):
MOCK_LLM=1 python run_stock.py --as-of 2024-01-15 --db eval/datasets/sample/snapshots

# 3. backtest scorecard over the sample:
MOCK_LLM=1 python eval/backtest.py eval/datasets/sample/dataset.json

# 4. REAL run (needs a Groq key + real/frozen data; MOCK proves plumbing, not skill):
GROQ_API_KEY=... SWARM_MAX_TIME_S=1800 python run_stock.py --symbol NVDA      # live yfinance, today

# tests (89 pass, 1 gated-skip):
MOCK_LLM=1 python -m pytest tests/test_stock_*.py tests/test_backtest_*.py \
  tests/test_symbol_discovery.py tests/test_relationships.py tests/test_ground_truth_boundary.py -q
```

To use REAL data: populate `eval/datasets/<your_set>/` in the same layout as the
sample (`tools/make_sample_dataset.py` is the template). yfinance PRICES are
point-in-time safe (the grader); yfinance FUNDAMENTALS are not, so historical
snapshots must come from your own dated source. `pip install curl_cffi` to dodge
Yahoo's rate limiting during ingestion.

## Build log (append one line per completed stage)

- 2026-06-04 (Opus): Built the correctness-critical cores + tests (47 pass), scaffolded the rest. Mechanical wiring checklist in `docs/STOCK_SWARM_WIRING_TODO.md`.
  - REAL: `core/stock_verify.py` (numeric extraction + closeness + verify_claim — the D4 fix); `core/stock_data.py` (`Snapshot`, lens partitioning, `assert_no_lookahead`, `FrozenSnapshotProvider`); `eval/ground_truth.py` (grader-only `realized_return` + import boundary); `eval/backtest.py` (`score_case`/`aggregate` scorecard); `agents/stock_roles.py` (all 6 roles + `build_stock_agents()`; `DataValidator`/`ValuationCritic` verification grounded in `stock_verify`; `EquityBriefSynthesizer.build_prediction` gradable artifact).
  - STILL STUBBED (mechanical / Sonnet): task-type wiring (Part 4 / wiring-todo steps 1–7); historical DB population (Stage 4); `backtest._run_engine` swarm shell-out (Stage 5); `discover_symbols` (Stage 6); KB dating + relationships (Stage 7); LLM-prose enrichment in the synthesizer; the `predicted_return_pct` magnitude mapping needs calibration on the DB.
- 2026-06-04 (Opus): Stage 0 spike done (finding above) + `YFinanceProvider._fetch_raw`/`_raw_to_snapshot` implemented for real (session+backoff, documented unit conversions: margins/growth ×100, debtToEquity ÷100, dividend yield from rate/price). Offline mapping tests lock the unit conversions (`tests/test_stock_yf_mapping.py`); live smoke test gated behind `RUN_YF_LIVE=1`. Remaining yfinance work: `pip install curl_cffi` + a slow ingestion script to the frozen DB (live fetch couldn't be confirmed here due to the rate limit).
- 2026-06-04 (Opus): **END-TO-END RUNNABLE via a standalone path** (no run_swarm.py edits needed). Built `core/stock_pipeline.py` (round-based orchestrator: Phase A scouts+validators / Phase B devs+critics+haters / decay+prune / synth), `run_stock.py` (CLI), and wired `eval/backtest.py::_run_engine` to run the pipeline in-process against the frozen DB. Added `SURVIVAL_TASK_PROFILES["stock"]` (additive; existing projection tests still pass). Fixed a real bug (developer/hater no-op when the field is empty, else partition-leak). New tests: `test_stock_pipeline.py` (full spine in-process with a scripted LLM — verification fires), `test_backtest_e2e.py` (frozen snapshot → pipeline → prediction → realized → scorecard). **Total stock tests: 64 pass, 1 gated-skip.** Verified `MOCK_LLM=1 python run_stock.py --symbol NVDA --as-of 2024-01-15 --db <frozen>` writes answer.txt + prediction.json.
  - **run_swarm.py task-type wiring is now OPTIONAL** — the standalone path covers manual runs + backtests. Wire it only if you want `python run_swarm.py stock …` parity.
- 2026-06-04 (Opus): **Stage 6 symbol discovery done.** `core/symbol_discovery.py` (cashtag/exchange/name extraction + universe gating + acronym stoplist + recency-weighted ranking; point-in-time drop of items after as_of) with 11 tests (`test_symbol_discovery.py`). Wired `YFinanceProvider.discover_symbols` (live DDG, best-effort) and `run_stock.py` now discovers a ticker when `--symbol` is omitted (frozen path via `_discovery/<as_of>.json`). Smoke-verified the full **no-symbol** loop offline: discover NVDA → snapshot → pipeline → prediction. **Total stock tests: 75 pass, 1 gated-skip.** Remaining: historical DB population + `pip install curl_cffi` ingestion (Stage 4); KB dating + relationships (Stage 7); `predicted_return_pct` magnitude calibration; LLM-prose enrichment; optional run_swarm parity.
- 2026-06-04 (Opus): **Comparative mode done (Stage 8 stretch — your WMT/DG example).** `core/stock_compare.py` (`build_comparison` relative verdict + `run_comparison` orchestration: runs the single-ticker pipeline per ticker on independent stores, ranks by buy-score, emits `comparison.json` with prefer/avoid/spread/relationship). Tests (`test_stock_compare.py`) include an integration case where a bull field is correctly preferred over a bear field. **Total stock tests: 79 pass, 1 gated-skip.**
- 2026-06-04 (Opus): **Dated relationship graph (Stage 7 seed).** `core/relationships.py` — typed, DATED peer/sector/competitor/supplier edges; every query is point-in-time (`learned_on <= as_of`), so the graph can't leak future knowledge into a backtest (same discipline as the KB). Wired into `run_comparison(graph=…)` so it auto-supplies "both grocery" for WMT/DG. Standalone (does NOT touch `core/knowledge_base.py`); Stage 7 can fold it into the KB later. Tests: `test_relationships.py` (9) + a compare-wiring test. **Total stock tests: 89 pass, 1 gated-skip.**
- 2026-06-04 (Opus): **Sample dataset + demonstrated the full backtest loop.** `tools/make_sample_dataset.py` writes a synthetic frozen DB (snapshots + prices + `_discovery` + `dataset.json`) at `eval/datasets/sample/` so the whole grader runs with no network. Ran it: scorecard renders (6 cases; buy&hold baseline +1.4%; swarm 0 under MOCK since predictions are 'avoid'). Fixed two real bugs found by running it: (1) `python eval/backtest.py` / `tools/*.py` needed a sys.path bootstrap (script dir shadowed the repo root); (2) the backtest ran `asyncio.run()` per case, which bound the shared MockLLM's semaphore to a dead loop on case 2 — now one event loop grades all cases. See Appendix C for the quickstart.
- 2026-06-04 (Opus): **Full-suite regression check.** Ran the whole repo suite: 388 passed; the 9 "failures" were all in `test_baseline_mode.py`/`test_coding_roles.py` and were caused by an over-aggressive `SWARM_MAX_ITERATIONS=20` env var *I* set for the run (not the documented invocation) — those 25 tests pass cleanly with `MOCK_LLM=1 SWARM_MIN_TIME_S=0 SWARM_MIN_ITERATIONS=5`. **No regressions from the stock work** (only shared-file change is the additive `SURVIVAL_TASK_PROFILES["stock"]`).
