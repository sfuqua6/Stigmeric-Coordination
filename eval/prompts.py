"""Pre-registered prompt set for the swarm-amplification delta experiment.

The whole experiment is only honest if the prompt set is fixed BEFORE anyone
looks at results (peeking / p-hacking is the easiest way to fool yourself —
you add prompts until the swarm wins). So this file is the contract: edit it
deliberately and rarely, never mid-experiment, and never in response to a
result you didn't like.

Each entry is a `Prompt`:
  - `pid`        stable id (used as the artifact dir / row key — keep stable)
  - `task`       run_swarm task type: debate | analysis | problem_solving |
                 creative | coding
  - `text`       the actual question shown to every condition, verbatim
  - `factual`    True for prompts with checkable answers — these also get an
                 objective, judge-free score via `must_include`
  - `must_include`  list of ground-truth checklist items. Each item is a list
                 of acceptable surface forms (synonyms); the item counts as hit
                 if ANY form appears in the (normalized) answer. Score is the
                 fraction of items hit. This is the bias-free anchor that needs
                 no LLM judge (the spirit of eval/ground_truth.py, generalized
                 from stock returns to factual prose).

The historical prompts ("ban private cars", "does god exist", "climate
action") are kept verbatim so runs are comparable to past outputs.

`DEFAULT_SET` is the registered set. `mini(n)` returns the first n for the
"minimal first run you can do today" (8 prompts).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Prompt:
    pid: str
    task: str
    text: str
    factual: bool = False
    must_include: list[list[str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# The registered set: ~24 prompts across debate / analysis / problem_solving,
# with >= 6 factual prompts carrying ground-truth checklists.
# ---------------------------------------------------------------------------

DEFAULT_SET: list[Prompt] = [
    # --- debate (historical — kept verbatim for cross-run comparison) -------
    Prompt("ban_cars", "debate",
           "Cities should ban private cars to fight climate change."),
    Prompt("god_exists", "debate",
           "Does God exist?"),
    Prompt("climate_action", "debate",
           "Climate action is necessary even at significant economic cost."),
    Prompt("ai_regulation", "debate",
           "Advanced AI development should be paused until safety is solved."),
    Prompt("ubi", "debate",
           "A universal basic income would do more good than harm."),
    Prompt("nuclear_power", "debate",
           "Nuclear power is essential to decarbonizing the grid."),

    # --- analysis ----------------------------------------------------------
    Prompt("innovation", "analysis",
           "What causes innovation in societies?"),
    Prompt("inequality_drivers", "analysis",
           "What are the main drivers of rising wealth inequality?"),
    Prompt("remote_work", "analysis",
           "How has remote work changed organizational productivity?"),
    Prompt("social_media_teens", "analysis",
           "How does social media affect adolescent mental health?"),
    Prompt("inflation_causes", "analysis",
           "What were the primary causes of the 2021-2023 inflation surge?"),
    Prompt("language_extinction", "analysis",
           "Why do languages go extinct, and what is lost when they do?"),

    # --- problem_solving ---------------------------------------------------
    Prompt("traffic", "problem_solving",
           "How can a mid-sized city reduce traffic congestion?"),
    Prompt("ocean_plastic", "problem_solving",
           "How can we meaningfully reduce ocean plastic pollution?"),
    Prompt("teacher_shortage", "problem_solving",
           "How should a country address a chronic teacher shortage?"),
    Prompt("antibiotic_resistance", "problem_solving",
           "How can the spread of antibiotic resistance be slowed?"),
    Prompt("housing_affordability", "problem_solving",
           "How can a city make housing more affordable without crashing prices?"),
    Prompt("wildfire", "problem_solving",
           "How can wildfire risk be reduced in the wildland-urban interface?"),

    # --- factual (>= 6) — objective, judge-free ground-truth checklists -----
    Prompt("photosynthesis", "analysis",
           "Explain how photosynthesis works.",
           factual=True,
           must_include=[
               ["chlorophyll"],
               ["sunlight", "light energy", "solar"],
               ["carbon dioxide", "co2"],
               ["water", "h2o"],
               ["glucose", "sugar", "carbohydrate"],
               ["oxygen", "o2"],
               ["chloroplast"],
           ]),
    Prompt("greenhouse_effect", "analysis",
           "What is the greenhouse effect and why does it warm the planet?",
           factual=True,
           must_include=[
               ["infrared", "longwave", "long-wave"],
               ["carbon dioxide", "co2"],
               ["methane", "ch4"],
               ["water vapor", "water vapour"],
               ["atmosphere"],
               ["re-emit", "reradiat", "re-radiat", "trap", "absorb"],
           ]),
    Prompt("french_revolution", "analysis",
           "What were the main causes of the French Revolution?",
           factual=True,
           must_include=[
               ["financial crisis", "debt", "bankrupt", "fiscal"],
               ["taxation", "taxes", "tax"],
               ["estates", "third estate", "estates-general", "estates general"],
               ["famine", "bread", "food", "harvest"],
               ["enlightenment", "rousseau", "voltaire"],
               ["monarchy", "louis xvi", "absolutism"],
           ]),
    Prompt("how_vaccines_work", "analysis",
           "How do vaccines work?",
           factual=True,
           must_include=[
               ["immune system", "immune response", "immunity"],
               ["antibodies", "antibody"],
               ["antigen", "pathogen", "weakened", "inactivated", "mrna"],
               ["memory cells", "memory", "b cells", "t cells"],
           ]),
    Prompt("supply_demand", "analysis",
           "Explain the law of supply and demand.",
           factual=True,
           must_include=[
               ["supply"],
               ["demand"],
               ["price"],
               ["equilibrium", "market clearing", "clears"],
               ["shortage", "surplus", "scarcity"],
           ]),
    Prompt("water_cycle", "analysis",
           "Describe the water cycle.",
           factual=True,
           must_include=[
               ["evaporation", "evaporat"],
               ["condensation", "condens"],
               ["precipitation", "rain", "snow"],
               ["collection", "runoff", "groundwater", "river", "ocean"],
           ]),
    Prompt("tcp_handshake", "coding",
           "Explain the TCP three-way handshake.",
           factual=True,
           must_include=[
               ["syn"],
               ["syn-ack", "syn ack", "synack"],
               ["ack"],
               ["sequence number", "seq", "isn"],
               ["connection", "establish"],
           ]),
]


def mini(n: int = 8, source: list[Prompt] | None = None) -> list[Prompt]:
    """First `n` prompts from `source` (default DEFAULT_SET) — the minimal
    first run (an hour, one model)."""
    return (source if source is not None else DEFAULT_SET)[:n]


def by_id(pid: str, source: list[Prompt] | None = None) -> Prompt | None:
    return next((p for p in (source if source is not None else DEFAULT_SET)
                 if p.pid == pid), None)


def factual_prompts() -> list[Prompt]:
    return [p for p in DEFAULT_SET if p.factual]


# ---------------------------------------------------------------------------
# OVERCONTEXT_SET — the decisive A-vs-F over-context experiment
# (docs/OVERCONTEXT_EVAL_PLAN.md). Parametric-answerable prompts (most of
# DEFAULT_SET) can never show a compression-thesis win: a single call already
# "knows" the answer from pretraining, so bigger evidence packs don't move
# quality. These prompts instead depend on pack-specific detail — synthesis-
# with-conflict ("where do the provided sources disagree") and needle-
# against-prior ("what does the evidence say that contradicts assumption Z")
# — so answer quality can actually track how much of a large evidence pack a
# condition manages to use. Paired with `--pack-scale {1x,4x,16x}` in
# eval/ab_harness.py, which builds/loads the SAME evidence pack for both
# condition A (swarm, --corpus=pack:<path>) and condition F (single-call RAG,
# naive pack-order fill). `must_include` here anchors judge-free scoring on
# generic multi-perspective synthesis vocabulary — real per-pack ground truth
# would need to be mined from the built pack contents (deferred; see plan doc
# order-of-work step 3), so these are deliberately generic "did the answer
# actually engage with disagreement/evidence" checks rather than pack-specific
# facts.
OVERCONTEXT_SET: list[Prompt] = [
    Prompt("oc_innovation_drivers", "analysis",
           "Based on the provided evidence, which factors most consistently "
           "explain what drives innovation in societies, and where do the "
           "sources disagree?",
           factual=True,
           must_include=[
               ["disagree", "disagreement", "conflict", "contradict"],
               ["evidence", "source", "study", "report"],
               ["factor", "driver", "cause"],
           ]),
    Prompt("oc_inequality_conflict", "analysis",
           "Across the provided reports, what does the evidence say about "
           "the main drivers of rising wealth inequality, and which claims "
           "contradict the common assumption that education alone explains "
           "the trend?",
           factual=True,
           must_include=[
               ["education"],
               ["contradict", "disagree", "conflict", "counter"],
               ["inequality"],
               ["evidence", "source", "data"],
           ]),
    Prompt("oc_remote_work_conflict", "analysis",
           "Based on the provided evidence, how has remote work changed "
           "organizational productivity, and where do the sources disagree "
           "about the size or direction of the effect?",
           factual=True,
           must_include=[
               ["productivity"],
               ["disagree", "disagreement", "conflict", "mixed"],
               ["remote work", "remote-work", "work from home", "wfh"],
           ]),
    Prompt("oc_nuclear_debate", "debate",
           "Based on the provided evidence, is nuclear power essential to "
           "decarbonizing the grid? Identify which claims in the evidence "
           "conflict with each other and explain how you weighed them.",
           factual=True,
           must_include=[
               ["nuclear"],
               ["conflict", "disagree", "contradict", "dispute"],
               ["decarbon", "carbon", "emission"],
               ["weigh", "trade-off", "tradeoff", "balance"],
           ]),
    Prompt("oc_antibiotic_resistance_conflict", "problem_solving",
           "Based on the provided evidence, what does the evidence say "
           "about how to slow the spread of antibiotic resistance, "
           "including any recommendations in the evidence that contradict "
           "each other?",
           factual=True,
           must_include=[
               ["antibiotic", "antimicrobial"],
               ["resistance"],
               ["contradict", "conflict", "disagree"],
               ["evidence", "source", "study"],
           ]),
    Prompt("oc_ubi_evidence", "debate",
           "Based on the provided evidence, would a universal basic income "
           "do more good than harm? Point to where the sources disagree "
           "and what evidence would resolve the disagreement.",
           factual=True,
           must_include=[
               ["universal basic income", "ubi"],
               ["disagree", "disagreement", "conflict"],
               ["evidence"],
           ]),
]
