"""Synthetic-world generator — Stage 1 of the Organization Program.

Implements `docs/future/STAGE1_SYNTHETIC_EVAL_SPEC.md` Sec 1 + Sec 4: a
deterministic fact registry (no LLM), a narrative renderer (template
fallback + optional LLM path), a mechanical fact-fidelity checker, and the
two validity gates (name-collision, B-from-memory ceiling).

Order of construction (spec Sec 1.1) is load-bearing and must not be
reordered:

    Stage A: build_fact_registry(seed, template)     # pure Python, no LLM
    Stage B: render_documents(registry, seed, renderer)   # LLM or template
    Stage C: verify_render(registry, docs)           # mechanical, Stage B only

The ground truth (Stage A) exists before any model touches the world, so it
cannot be perturbed by the renderer — the entire point of Stage 1 (see the
spec's opening paragraph: real A-vs-F runs to date used pretraining-knowable
facts, which makes a memory-only answer indistinguishable from a grounded
one on a factual checklist).

CLI
---
    python -m eval.worlds --seed 42 --template invented_company
    python -m eval.worlds --seed 1 --template invented_company --renderer template
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_WORLDS_DIR = _REPO / "eval" / "worlds"
_GATE_LOG_PATH = _REPO / "eval" / "results" / "world_gate_log.jsonl"


def _stable_seed(*parts) -> int:
    """Deterministic integer seed derived from `parts`, stable ACROSS
    process restarts. `random.Random((a, b, ...))` (a tuple) falls through
    to Python's built-in `hash()`, which is randomized per-process for str
    (PYTHONHASHSEED, on by default since Python 3.3) — using it here would
    silently break the "same seed -> byte-identical rendering" guarantee
    every re-run of generate_world() depends on. sha256 has no such
    randomization, so this is safe to use as a random.Random() seed."""
    key = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


# ---------------------------------------------------------------------------
# Fact schema (spec Sec 1.3). frozen dataclasses = facts are immutable once
# minted; every *_id below is the join key between the registry, the
# rendered documents' provenance metadata, and eval/world_score.py's
# checklist items.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Entity:
    entity_id: str
    kind: str                  # "company" | "person" | "location" | "product" | "agency"
    name: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class Quantity:
    fact_id: str
    subject_id: str             # Entity.entity_id
    metric: str
    value: float
    unit: str
    as_of: str
    tolerance_pct: float = 2.0
    surface_forms: tuple[str, ...] = ()   # registered renderings, see _surface_forms()


@dataclass(frozen=True)
class CausalRelation:
    fact_id: str
    cause_id: str
    effect_id: str
    mechanism: str


@dataclass(frozen=True)
class Event:
    fact_id: str
    description: str
    date: str
    entity_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class Contradiction:
    contradiction_id: str
    fact_id_a: str
    fact_id_b: str
    doc_id_a: str
    doc_id_b: str
    ground_truth: str           # fact_id of the correct value, or "genuinely_ambiguous"


# Deviation from spec Sec 1.3: the spec's FactRegistry dataclass does not
# list a `documents` field, but Stage B (render_documents) needs to know
# which document each fact belongs to and which facts a given document is
# supposed to render (`extracted_from`, used by verify_render and by the
# citation_grounding scorer). DocumentSpec is the minimal addition that
# closes that gap without inventing a parallel registry.
@dataclass
class DocumentSpec:
    doc_id: str
    doc_type: str
    entity_ids: list[str] = field(default_factory=list)
    fact_ids: list[str] = field(default_factory=list)   # quantity/event/relation ids


@dataclass
class FactRegistry:
    world_seed: int
    template_name: str
    entities: list[Entity]
    quantities: list[Quantity]
    relations: list[CausalRelation]
    events: list[Event]
    contradictions: list[Contradiction]
    filler_seed_material: list[str]
    documents: list[DocumentSpec] = field(default_factory=list)  # deviation, see above

    def fact_index(self) -> dict[str, object]:
        idx: dict[str, object] = {}
        for coll in (self.quantities, self.relations, self.events):
            for f in coll:
                idx[f.fact_id] = f
        return idx

    def entity_index(self) -> dict[str, Entity]:
        return {e.entity_id: e for e in self.entities}

    def document_index(self) -> dict[str, DocumentSpec]:
        return {d.doc_id: d for d in self.documents}


@dataclass
class RenderedDoc:
    doc_id: str
    doc_type: str
    text: str
    extracted_from: list[str]   # fact_ids this doc's text was asked to include
    renderer: str                # "template" | "llm" | "template_filler"


# ---------------------------------------------------------------------------
# World templates (spec Sec 1.2). Kept as one generic engine parameterized by
# a WorldTemplate config rather than three bespoke generators — the three
# templates differ only in entity kinds, document types, and metric/name
# vocabularies, and a shared engine is much easier to keep referentially
# correct (Sec 4's referential-integrity test) than three near-duplicate ones.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WorldTemplate:
    name: str
    entity_kinds: tuple[tuple[str, int], ...]     # (kind, count)
    doc_types: tuple[tuple[str, int, int], ...]   # (doc_type, min_count, max_count)
    metrics: tuple[tuple[str, str, float, float, float], ...]
    # (metric_name, unit, min_value, max_value, tolerance_pct)
    name_pool: dict[str, tuple[str, ...]]         # kind -> word pool for name-minting
    event_verbs: tuple[str, ...]
    filler_sentences: tuple[str, ...]             # fact-free connective boilerplate


_NAME_ADJ = ("Meridian", "Northgate", "Vantage", "Solace", "Ashworth", "Blackfield",
             "Corvid", "Delvane", "Emberton", "Farlow", "Greyharbor", "Hollowmere")
_NAME_NOUN = ("Analytics", "Dynamics", "Holdings", "Systems", "Ventures", "Materials",
              "Logistics", "Robotics", "Foundry", "Instruments", "Networks", "Works")
_PERSON_FIRST = ("Elena", "Marcus", "Priya", "Tomas", "Ingrid", "Kwame", "Naomi",
                 "Declan", "Yuki", "Farida", "Otto", "Renata")
_PERSON_LAST = ("Voss", "Okafor", "Lindqvist", "Reyes", "Marchetti", "Sundaram",
                "Belanger", "Iwu", "Halvorsen", "Petrescu", "Naidu", "Whitcombe")
_COUNTRY_ADJ = ("Kestrel", "Auroline", "Veridian", "Thornmark", "Calderas",
                "Brindlewood", "Solvane", "Marrowfen")
_COUNTRY_NOUN = ("Republic", "Federation", "Union", "Commonwealth", "Dominion")
_CITY_NAMES = ("Port Halden", "New Averil", "Costrel", "Brambury", "Ilmarin",
               "Suncrest", "Delford", "Marnhaven")
_LOCATION_NAMES = ("Route 14 overpass", "the Millbrook warehouse district",
                    "the Ashgrove transit hub", "Pier 9", "the old cannery lot",
                    "the Kestrel Building lobby", "the north maintenance shed")

_TEMPLATES: dict[str, WorldTemplate] = {
    "invented_company": WorldTemplate(
        name="invented_company",
        entity_kinds=(("company", 1), ("product", 3), ("person", 3), ("agency", 2)),
        doc_types=(("10k_filing", 1, 1), ("earnings_call", 2, 3),
                    ("trade_press", 4, 6), ("analyst_note", 2, 2)),
        metrics=(
            ("quarterly_revenue_usd_m", "USD million", 40.0, 900.0, 2.0),
            ("headcount", "employees", 200.0, 12000.0, 1.0),
            ("gross_margin_pct", "percent", 18.0, 62.0, 3.0),
            ("market_share_pct", "percent", 2.0, 35.0, 5.0),
        ),
        name_pool={"company": _NAME_ADJ, "product": _NAME_NOUN,
                    "person": _PERSON_FIRST, "agency": _NAME_NOUN},
        event_verbs=("announced a supply disruption at", "opened a new facility near",
                     "reported a margin miss attributed to", "completed an acquisition of",
                     "disclosed a leadership change at"),
        filler_sentences=(
            "The following section provides additional background for context.",
            "Analysts note that broader sector conditions remained mixed during the period.",
            "Management reiterated its long-term strategy on the accompanying call.",
            "The disclosures below are presented in the order filed.",
            "Further detail is available in the appendix to this report.",
        ),
    ),
    "invented_country": WorldTemplate(
        name="invented_country",
        entity_kinds=(("location", 1), ("agency", 2), ("person", 3), ("location", 2)),
        doc_types=(("statistical_yearbook", 1, 1), ("news_archive", 3, 5),
                    ("white_paper", 1, 2), ("opposition_rebuttal", 1, 2)),
        metrics=(
            ("population_2024", "people", 900_000.0, 40_000_000.0, 1.0),
            ("gdp_growth_pct", "percent", -3.0, 9.0, 3.0),
            ("election_result_pct", "percent", 20.0, 65.0, 2.0),
            ("unemployment_pct", "percent", 2.0, 22.0, 3.0),
        ),
        name_pool={"location": _COUNTRY_ADJ, "agency": _NAME_NOUN,
                    "person": _PERSON_FIRST},
        event_verbs=("held a contested election in", "enacted a reform package covering",
                     "faced a currency shock centered on", "opened a new port near",
                     "issued a white paper concerning"),
        filler_sentences=(
            "The yearbook presents figures as compiled by the national statistics office.",
            "Independent observers offered mixed assessments of the process.",
            "The rebuttal was issued through the usual opposition channels.",
            "Further tables are reproduced in the appendix.",
            "Coverage of the period was extensive across the domestic press.",
        ),
    ),
    "case_file": WorldTemplate(
        name="case_file",
        entity_kinds=(("person", 5), ("agency", 1), ("location", 2), ("product", 1)),
        doc_types=(("incident_report", 1, 1), ("witness_statement", 3, 5),
                    ("internal_memo", 1, 2), ("investigator_timeline", 1, 1)),
        metrics=(
            ("time_of_incident_minute_offset", "minutes after midnight", 0.0, 1439.0, 5.0),
            ("distance_from_scene_m", "meters", 1.0, 500.0, 8.0),
            ("duration_at_scene_min", "minutes", 1.0, 180.0, 5.0),
        ),
        name_pool={"person": _PERSON_FIRST, "agency": _NAME_NOUN,
                    "location": _LOCATION_NAMES, "product": _NAME_NOUN},
        event_verbs=("was seen near", "reported hearing a disturbance at",
                     "arrived at", "left the vicinity of", "was interviewed about"),
        filler_sentences=(
            "This statement was taken under standard caution.",
            "The timeline below reflects the investigator's consolidated account.",
            "Further exhibits are referenced in the case index.",
            "All times are approximate unless otherwise noted.",
            "The memo was circulated to the case file on the date shown.",
        ),
    ),
}


def available_templates() -> list[str]:
    return sorted(_TEMPLATES)


# ---------------------------------------------------------------------------
# Stage A: build_fact_registry — pure Python, deterministic, no LLM.
# ---------------------------------------------------------------------------

_DATE_RE = re.compile(r"\b(19|20)\d{2}\b")  # whitelisted by verify_render


def _mint_name(kind: str, rng: random.Random, template: WorldTemplate,
               used: set[str], attempt: int = 0) -> str:
    """Deterministic name minting from the template's word pool. `used` is
    checked for in-world collisions (two entities minted the same name);
    Gate 2 (name_collision_check) separately checks for REAL-WORLD collisions."""
    pool = template.name_pool.get(kind, _NAME_ADJ)
    if kind == "company":
        name = f"{rng.choice(_NAME_ADJ)} {rng.choice(_NAME_NOUN)}"
    elif kind == "person":
        name = f"{rng.choice(_PERSON_FIRST)} {rng.choice(_PERSON_LAST)}"
    elif kind == "location" and template.name == "invented_country":
        name = f"{rng.choice(_COUNTRY_ADJ)} {rng.choice(_COUNTRY_NOUN)}"
    elif kind == "location":
        name = rng.choice(pool if pool else _LOCATION_NAMES)
    elif kind == "product":
        name = f"{rng.choice(pool)} {rng.choice(('One', 'Prime', 'Edge', 'Core', 'Max'))}"
    else:  # agency / fallback
        name = f"{rng.choice(('Bureau of', 'Office of', 'Division of'))} {rng.choice(pool)}"
    if name in used and attempt < 20:
        return _mint_name(kind, rng, template, used, attempt + 1)
    return name


def _aliases_for(kind: str, name: str, rng: random.Random) -> tuple[str, ...]:
    """3-5 alias forms per entity (spec Sec 3.2: alias lists should be
    generous). Deterministic given rng state."""
    parts = name.split()
    forms = {name}
    if len(parts) > 1:
        forms.add(parts[0])                      # short/first form
        forms.add("".join(w[0] for w in parts).upper())  # acronym e.g. "MDA"
        forms.add(parts[-1])
    if kind == "company":
        forms.add(name.replace(" ", ""))
    return tuple(sorted(forms))


def _surface_forms(value: float, unit: str) -> tuple[str, ...]:
    """Registered equivalent renderings of a quantity's value+unit, so
    verify_render's verbatim check doesn't over-fit to one exact string
    (spec Sec 1.4.1)."""
    forms: set[str] = set()
    if unit == "percent":
        forms.add(f"{value:.1f}%")
        forms.add(f"{value:.1f} percent")
    elif "USD" in unit:
        if value >= 1000:
            forms.add(f"${value/1000:.2f}B")
            forms.add(f"{value/1000:.2f} billion")
        else:
            forms.add(f"${value:.1f}M")
            forms.add(f"{value:.1f} million")
            forms.add(f"${value:.1f} million")
    elif unit in ("people",):
        forms.add(f"{value:,.0f} {unit}")
        forms.add(f"{value:,.0f}")
    else:
        forms.add(f"{value:.1f} {unit}")
        forms.add(f"{value:.1f}")
    return tuple(sorted(forms))


def _gen_entities(rng: random.Random, template: WorldTemplate) -> list[Entity]:
    entities: list[Entity] = []
    used_names: set[str] = set()
    idx = 0
    for kind, count in template.entity_kinds:
        for _ in range(count):
            name = _mint_name(kind, rng, template, used_names)
            used_names.add(name)
            eid = f"ent_{idx:04d}"
            entities.append(Entity(entity_id=eid, kind=kind, name=name,
                                    aliases=_aliases_for(kind, name, rng)))
            idx += 1
    return entities


def _gen_documents(rng: random.Random, template: WorldTemplate) -> list[DocumentSpec]:
    docs: list[DocumentSpec] = []
    idx = 0
    for doc_type, lo, hi in template.doc_types:
        n = rng.randint(lo, hi)
        for _ in range(n):
            docs.append(DocumentSpec(doc_id=f"{template.name}_{doc_type}_{idx:03d}",
                                      doc_type=doc_type))
            idx += 1
    return docs


def _as_of_labels(n: int) -> list[str]:
    """Fixed, deterministic fiscal/period labels — no rng needed, order only
    matters relative to n."""
    years = list(range(2021, 2021 + max(1, (n // 4) + 1)))
    labels = []
    for y in years:
        for q in range(1, 5):
            labels.append(f"FY{y}Q{q}")
    return labels[:n]


def _gen_quantities(rng: random.Random, template: WorldTemplate,
                     entities: list[Entity], docs: list[DocumentSpec]
                     ) -> list[Quantity]:
    quantities: list[Quantity] = []
    subjects = [e for e in entities if e.kind in ("company", "location", "product")] or entities
    as_of_pool = _as_of_labels(max(4, len(docs)))
    idx = 0
    for metric, unit, lo, hi, tol in template.metrics:
        n_periods = min(len(as_of_pool), max(2, len(docs)))
        for p in range(n_periods):
            subject = subjects[p % len(subjects)]
            value = round(rng.uniform(lo, hi), 1)
            as_of = as_of_pool[p]
            fid = f"q_{idx:04d}"
            quantities.append(Quantity(
                fact_id=fid, subject_id=subject.entity_id, metric=metric,
                value=value, unit=unit, as_of=as_of, tolerance_pct=tol,
                surface_forms=_surface_forms(value, unit)))
            idx += 1
    return quantities


def _gen_events(rng: random.Random, template: WorldTemplate,
                 entities: list[Entity], docs: list[DocumentSpec]) -> list[Event]:
    events: list[Event] = []
    n_events = max(2, len(docs) // 2)
    for i in range(n_events):
        verb = rng.choice(template.event_verbs)
        subj = rng.choice(entities)
        loc = rng.choice(entities)
        year = 2021 + (i % 4)
        month = rng.randint(1, 12)
        day = rng.randint(1, 28)
        desc = f"{subj.name} {verb} {loc.name}"
        events.append(Event(fact_id=f"e_{i:04d}", description=desc,
                             date=f"{year:04d}-{month:02d}-{day:02d}",
                             entity_ids=(subj.entity_id, loc.entity_id)))
    return events


def _gen_relations(rng: random.Random, quantities: list[Quantity],
                    events: list[Event]) -> list[CausalRelation]:
    relations: list[CausalRelation] = []
    n = min(3, len(events) - 1) if len(events) > 1 else 0
    for i in range(n):
        cause = events[i]
        effect = quantities[(i * 3) % len(quantities)] if quantities else None
        if effect is None:
            continue
        mechanism = (f"the disruption described in \"{cause.description}\" is cited "
                     f"as the direct driver behind the {effect.metric.replace('_', ' ')} "
                     f"figure for {effect.as_of}")
        relations.append(CausalRelation(
            fact_id=f"r_{i:04d}", cause_id=cause.fact_id, effect_id=effect.fact_id,
            mechanism=mechanism))
    return relations


def _assign_facts_to_docs(rng: random.Random, docs: list[DocumentSpec],
                           quantities: list[Quantity], events: list[Event],
                           relations: list[CausalRelation]) -> None:
    """Round-robin assignment (deterministic given rng draw order) of every
    non-contradiction fact to exactly one document's `fact_ids`/`entity_ids`."""
    if not docs:
        return
    all_facts: list[tuple[str, tuple[str, ...]]] = (
        [(q.fact_id, (q.subject_id,)) for q in quantities]
        + [(e.fact_id, e.entity_ids) for e in events]
        + [(r.fact_id, ()) for r in relations]
    )
    for i, (fid, ents) in enumerate(all_facts):
        doc = docs[i % len(docs)]
        doc.fact_ids.append(fid)
        for e in ents:
            if e not in doc.entity_ids:
                doc.entity_ids.append(e)


def _gen_contradictions(rng: random.Random, quantities: list[Quantity],
                         events: list[Event], docs: list[DocumentSpec],
                         n_contradictions: int = 3) -> list[Contradiction]:
    """Clone a subset of Quantity facts with a perturbed value assigned to a
    DIFFERENT document (spec Sec 1.3: both the true and false claim are
    first-class checkable facts). Mutates `quantities` and the target
    document's fact_ids in place; returns the Contradiction records."""
    contradictions: list[Contradiction] = []
    if len(docs) < 2 or not quantities:
        return contradictions
    doc_index = {d.doc_id: d for d in docs}
    candidates = [q for q in quantities if any(q.fact_id in d.fact_ids for d in docs)]
    n = min(n_contradictions, len(candidates))
    picked = rng.sample(candidates, k=n) if n else []
    next_idx = len(quantities)
    for i, orig in enumerate(picked):
        doc_a = next(d for d in docs if orig.fact_id in d.fact_ids)
        others = [d for d in docs if d.doc_id != doc_a.doc_id]
        if not others:
            continue
        doc_b = rng.choice(others)
        # Perturb the value meaningfully (>= 2x tolerance so it's a genuine
        # conflict, not rounding noise the tolerance would absorb).
        delta_frac = max(0.10, orig.tolerance_pct / 100.0 * 3)
        sign = 1 if rng.random() < 0.5 else -1
        new_value = round(orig.value * (1 + sign * delta_frac), 1)
        new_as_of = orig.as_of + "_revised"
        clone_fid = f"q_{next_idx:04d}"
        next_idx += 1
        clone = Quantity(fact_id=clone_fid, subject_id=orig.subject_id,
                          metric=orig.metric, value=new_value, unit=orig.unit,
                          as_of=new_as_of, tolerance_pct=orig.tolerance_pct,
                          surface_forms=_surface_forms(new_value, orig.unit))
        quantities.append(clone)
        doc_b.fact_ids.append(clone_fid)
        ground_truth = "genuinely_ambiguous" if (i % 3 == 2) else orig.fact_id
        contradictions.append(Contradiction(
            contradiction_id=f"c_{i:04d}", fact_id_a=orig.fact_id,
            fact_id_b=clone_fid, doc_id_a=doc_a.doc_id, doc_id_b=doc_b.doc_id,
            ground_truth=ground_truth))
    return contradictions


def build_fact_registry(seed: int, template_name: str) -> FactRegistry:
    """Stage A. Deterministic, pure Python — no LLM call. Same (seed,
    template_name) always produces an equal FactRegistry (dataclass
    equality), including nested list order."""
    if template_name not in _TEMPLATES:
        raise ValueError(f"unknown template {template_name!r}; use one of "
                         f"{available_templates()}")
    template = _TEMPLATES[template_name]
    rng = random.Random(seed)
    entities = _gen_entities(rng, template)
    docs = _gen_documents(rng, template)
    quantities = _gen_quantities(rng, template, entities, docs)
    events = _gen_events(rng, template, entities, docs)
    relations = _gen_relations(rng, quantities, events)
    _assign_facts_to_docs(rng, docs, quantities, events, relations)
    contradictions = _gen_contradictions(rng, quantities, events, docs)
    filler = list(template.filler_sentences)
    return FactRegistry(world_seed=seed, template_name=template_name,
                        entities=entities, quantities=quantities,
                        relations=relations, events=events,
                        contradictions=contradictions,
                        filler_seed_material=filler, documents=docs)


# ---------------------------------------------------------------------------
# Stage B: template renderer (deterministic fallback, zero API cost).
# Built and tested FIRST, before the LLM renderer, per spec Sec 7's build
# order: a renderer that cannot hallucinate is what verify_render (Stage C)
# gets proven against before being pointed at one that can.
# ---------------------------------------------------------------------------

def _fact_sentence(fact, entities_by_id: dict[str, Entity]) -> str:
    """One deterministic sentence per fact type, containing its canonical
    surface form / mechanism text verbatim (what verify_render checks for)."""
    if isinstance(fact, Quantity):
        subj = entities_by_id.get(fact.subject_id)
        subj_name = subj.name if subj else fact.subject_id
        metric_label = fact.metric.replace("_", " ")
        sf = fact.surface_forms[0] if fact.surface_forms else f"{fact.value} {fact.unit}"
        return f"{subj_name}'s {metric_label} as of {fact.as_of} was {sf}."
    if isinstance(fact, Event):
        return f"On {fact.date}, {fact.description}."
    if isinstance(fact, CausalRelation):
        return fact.mechanism.capitalize() + "."
    return ""


def _render_doc_template(doc: DocumentSpec, registry: FactRegistry,
                          rng: random.Random) -> RenderedDoc:
    fidx = registry.fact_index()
    eidx = registry.entity_index()
    # Header is a bracketed label only (no doc_id digits, e.g. "10k_filing"'s
    # "10" or the doc's numeric suffix) — verify_render's number-leakage scan
    # would otherwise misfire on it; see _DOC_HEADER_RE, which strips any
    # line matching this shape before scanning for fabricated figures.
    sentences: list[str] = [f"[{doc.doc_type.upper()}]"]
    for fid in doc.fact_ids:
        fact = fidx.get(fid)
        if fact is not None:
            sentences.append(_fact_sentence(fact, eidx))
    # Interleave a couple of fact-free filler sentences, deterministically
    # chosen from the template's registered filler pool.
    if registry.filler_seed_material:
        k = min(2, len(registry.filler_seed_material))
        sentences.extend(rng.sample(registry.filler_seed_material, k=k))
    text = "\n".join(s for s in sentences if s)
    return RenderedDoc(doc_id=doc.doc_id, doc_type=doc.doc_type, text=text,
                       extracted_from=list(doc.fact_ids), renderer="template")


def render_documents_template(registry: FactRegistry) -> list[RenderedDoc]:
    """Stage B, template path. Deterministic given the registry (rng reseeded
    per-doc from world_seed + doc_id so re-renders are byte-identical)."""
    out = []
    for doc in registry.documents:
        rng = random.Random(_stable_seed(registry.world_seed, doc.doc_id))
        out.append(_render_doc_template(doc, registry, rng))
    return out


def _fill_doc(world: FactRegistry, index: int) -> RenderedDoc:
    """Corpus-scale FILLER document (spec Sec 2): draws a FRESH sample of
    already-registered facts (never invents new ground truth) plus
    filler_seed_material connective prose. Deterministic given
    (world_seed, index) so a pack build is reproducible without persisting
    every filler doc to eval/worlds/<seed>/rendered/ (they are pack-build
    artifacts, not part of the canonical 1x document set generate_world()
    persists). Used by eval.packs.build_pack_from_world to manufacture 4x/
    16x scale without changing the checklist (same facts, same items)."""
    rng = random.Random(_stable_seed(world.world_seed, "filler", index))
    doc_id = f"{world.template_name}_filler_{index:04d}"
    all_facts = list(world.quantities) + list(world.events) + list(world.relations)
    eidx = world.entity_index()
    # Header excludes doc_id (its numeric suffix would otherwise misfire the
    # number-leakage scan; see _DOC_HEADER_RE / _strip_header_line).
    sentences = ["[FILLER]"]
    sample = []
    if all_facts:
        k = min(len(all_facts), rng.randint(3, 6))
        sample = rng.sample(all_facts, k=k)
        for f in sample:
            sentences.append(_fact_sentence(f, eidx))
    if world.filler_seed_material:
        k = min(3, len(world.filler_seed_material))
        sentences.extend(rng.sample(world.filler_seed_material, k=k))
    text = "\n".join(s for s in sentences if s)
    extracted_from = [f.fact_id for f in sample]
    return RenderedDoc(doc_id=doc_id, doc_type="filler", text=text,
                       extracted_from=extracted_from, renderer="template_filler")


# ---------------------------------------------------------------------------
# Stage B, LLM path (Groq, temp 0). Cache-and-reuse per (seed, doc_id),
# mirroring eval/packs.py:build_pack's reuse-if-exists contract (spec
# Sec 1.4): the renderer writes eval/worlds/<seed>/rendered/<doc_id>.txt and
# reuses the file if present rather than re-calling the LLM.
# ---------------------------------------------------------------------------

_LLM_RENDER_PROMPT = (
    "Write the body of a {doc_type} document (plain prose, no markdown "
    "headers) that naturally incorporates ALL of the following facts, each "
    "stated using the EXACT number/value given (do not round or rephrase "
    "the numbers) — bury them in ordinary narrative language rather than "
    "listing them:\n\n{facts}\n\n"
    "Do not invent any additional numbers, dates, or figures beyond what is "
    "listed above (dates for narrative flavor are fine). Keep it to 2-4 "
    "short paragraphs.\n\nDOCUMENT BODY:"
)


def _facts_block(doc: DocumentSpec, registry: FactRegistry) -> str:
    fidx = registry.fact_index()
    lines = []
    for fid in doc.fact_ids:
        fact = fidx.get(fid)
        if fact is not None:
            lines.append("- " + _fact_sentence(fact, registry.entity_index()))
    return "\n".join(lines)


def _rendered_path(seed: int, doc_id: str, out_dir: Optional[Path] = None) -> Path:
    base = out_dir or (_WORLDS_DIR / str(seed) / "rendered")
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{doc_id}.txt"


async def _render_doc_llm(doc: DocumentSpec, registry: FactRegistry, llm,
                          out_dir: Optional[Path], max_attempts: int = 3
                          ) -> RenderedDoc:
    """One LLM call per (seed, doc_id), reused thereafter. Retries up to
    `max_attempts` on a verify_render failure, then falls back to the
    template renderer for THIS document only (never ships a doc that failed
    the fact-fidelity check, per spec Sec 1.4)."""
    path = _rendered_path(registry.world_seed, doc.doc_id, out_dir)
    if path.exists():
        text = path.read_text(encoding="utf-8")
        return RenderedDoc(doc_id=doc.doc_id, doc_type=doc.doc_type, text=text,
                           extracted_from=list(doc.fact_ids), renderer="llm")
    prompt = _LLM_RENDER_PROMPT.format(doc_type=doc.doc_type,
                                        facts=_facts_block(doc, registry) or "(none)")
    for attempt in range(max_attempts):
        raw = await llm.generate(prompt, role="synthesizer", max_tokens=600,
                                 temperature=0.0)
        candidate = RenderedDoc(doc_id=doc.doc_id, doc_type=doc.doc_type,
                                text=(raw or "").strip(),
                                extracted_from=list(doc.fact_ids), renderer="llm")
        result = verify_render(registry, [candidate])[0]
        if result.ok:
            path.write_text(candidate.text, encoding="utf-8")
            return candidate
        print(f"[worlds] LLM render of {doc.doc_id} failed verify_render "
              f"(attempt {attempt+1}/{max_attempts}): missing={result.missing_facts} "
              f"fabricated={result.fabricated_numbers}", file=sys.stderr)
    print(f"[worlds] {doc.doc_id}: LLM render exhausted retries, falling back "
          f"to template renderer for this document", file=sys.stderr)
    fallback_rng = random.Random(_stable_seed(registry.world_seed, doc.doc_id, "fallback"))
    fallback = _render_doc_template(doc, registry, fallback_rng)
    path.write_text(fallback.text, encoding="utf-8")
    return fallback


def render_documents(registry: FactRegistry, renderer: str = "template",
                     llm=None, out_dir: Optional[Path] = None
                     ) -> list[RenderedDoc]:
    """Stage B. `renderer="template"` (default, no LLM, deterministic) or
    `renderer="llm"` (needs `llm`, an object with an async .generate(...)
    method — GroqBackend / MockLLM both satisfy this contract)."""
    if renderer == "template":
        return render_documents_template(registry)
    if renderer == "llm":
        if llm is None:
            raise ValueError("renderer='llm' requires an llm= engine")
        async def _run():
            return [await _render_doc_llm(doc, registry, llm, out_dir)
                    for doc in registry.documents]
        return asyncio.run(_run())
    raise ValueError(f"unknown renderer {renderer!r}; use 'template' or 'llm'")


# ---------------------------------------------------------------------------
# Stage C: verify_render — mechanical fact-fidelity check on Stage B output
# only (spec Sec 1.4). No LLM judge here: pure regex/substring.
# ---------------------------------------------------------------------------

@dataclass
class VerifyResult:
    doc_id: str
    ok: bool
    missing_facts: list[str] = field(default_factory=list)
    fabricated_numbers: list[str] = field(default_factory=list)


_NUMBER_TOKEN_RE = re.compile(r"\d[\d,]*\.?\d*")
# Small fixed whitelist of narrative connective numbers (enumeration /
# ordinal digits) — spec Sec 1.4.2 (c). Kept intentionally small: anything
# past this is either a registered value, a date, or a fabrication.
_CONNECTIVE_WHITELIST = {"1", "2", "3", "4", "5", "1st", "2nd", "3rd", "4th", "5th"}

# ISO-date component whitelist — spec Sec 1.4.2 (b): "whitelist via a
# date-pattern regex, not a blanket exemption". Event.date is rendered as
# YYYY-MM-DD; the tokenizer above splits that into three separate numeric
# tokens at the hyphens (year/month/day), none of which are registered
# Quantity values, so without this every event date would misfire as a
# fabricated figure.
_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")


def _date_component_whitelist(text: str) -> set[str]:
    out: set[str] = set()
    for m in _ISO_DATE_RE.finditer(text):
        y, mo, d = m.groups()
        out.add(y)
        out.add(mo)
        out.add(str(int(mo)))
        out.add(d)
        out.add(str(int(d)))
    return out


def _normalize_number(tok: str) -> str:
    # Same discipline as core/actions.py's ungrounded_numbers() normalizer:
    # strip thousands separators and a bare trailing decimal point (a number
    # at the end of a sentence, e.g. "...was 20,659,392.", tokenizes with a
    # trailing "." that isn't part of the value).
    return tok.replace(",", "").rstrip(".")


def _registered_number_strings(registry: FactRegistry) -> set[str]:
    """Every numeric substring that appears in a registered Quantity's
    surface forms, OR in an entity's name/aliases (e.g. "Route 14 overpass",
    "Pier 9" — a location name pool entry that happens to contain a digit is
    not a fabricated figure) — the set of numbers verify_render considers
    grounded."""
    out: set[str] = set()
    for q in registry.quantities:
        for sf in q.surface_forms:
            for tok in _NUMBER_TOKEN_RE.findall(sf):
                out.add(_normalize_number(tok))
        # Also the raw float rendering, in case a renderer used a form not
        # in the registered surface-form list but still matching the value.
        out.add(_normalize_number(f"{q.value:.1f}"))
        out.add(_normalize_number(f"{q.value:g}"))
    for e in registry.entities:
        for name in (e.name, *e.aliases):
            for tok in _NUMBER_TOKEN_RE.findall(name):
                out.add(_normalize_number(tok))
    return out


_DOC_HEADER_RE = re.compile(r"^\[[A-Z0-9_ —-]+\]\s*$")


def _strip_header_line(text: str) -> str:
    """Drop a leading bracketed structural label line (e.g. "[10K_FILING]",
    "[FILLER — doc_003]") before the number-leakage scan — it's a
    document-type tag, not narrative prose, and document-type codes like
    "10-K" legitimately contain digits that have nothing to do with the
    registry's facts."""
    lines = text.splitlines()
    if lines and _DOC_HEADER_RE.match(lines[0].strip()):
        return "\n".join(lines[1:])
    return text


def verify_render(registry: FactRegistry, docs: list[RenderedDoc]) -> list[VerifyResult]:
    """Stage C. For each doc: (1) verbatim-presence check for every fact its
    `extracted_from` claims to include, (2) number-leakage scan for any
    numeric token not registered/dated/whitelisted. Pure string/regex over
    Stage B output only — introduces no new leakage or nondeterminism."""
    fidx = registry.fact_index()
    eidx = registry.entity_index()
    registered_numbers = _registered_number_strings(registry)
    results: list[VerifyResult] = []
    for doc in docs:
        missing: list[str] = []
        for fid in doc.extracted_from:
            fact = fidx.get(fid)
            if fact is None:
                continue
            if isinstance(fact, Quantity):
                forms = fact.surface_forms or (f"{fact.value} {fact.unit}",)
                if not any(sf in doc.text for sf in forms):
                    missing.append(fid)
            elif isinstance(fact, Event):
                if fact.description not in doc.text:
                    missing.append(fid)
            elif isinstance(fact, CausalRelation):
                # near-verbatim: the mechanism sentence (case-insensitive,
                # allow the leading-capital rendering _fact_sentence uses)
                if fact.mechanism.lower() not in doc.text.lower():
                    missing.append(fid)

        scan_text = _strip_header_line(doc.text)
        date_whitelist = _date_component_whitelist(scan_text)
        fabricated: list[str] = []
        for tok in _NUMBER_TOKEN_RE.findall(scan_text):
            norm = _normalize_number(tok)
            if norm in _CONNECTIVE_WHITELIST:
                continue
            if _DATE_RE.search(tok) or re.match(r"^\d{4}$", norm):
                continue  # year
            if norm in date_whitelist:
                continue  # month/day component of a rendered ISO date
            if norm in registered_numbers:
                continue
            # Strip a leading '$' or trailing unit already handled by the
            # tokenizer; if it survives all three checks, it's unregistered.
            fabricated.append(tok)
        results.append(VerifyResult(doc_id=doc.doc_id, ok=not missing and not fabricated,
                                    missing_facts=missing, fabricated_numbers=fabricated))
    return results


# ---------------------------------------------------------------------------
# Gate 2 — name-collision leakage check (cheap, Stage A output only, before
# any document rendering budget is spent). Spec Sec 4.2.
# ---------------------------------------------------------------------------

def name_collision_check(entities: list[Entity], llm, batch_size: int = 8
                         ) -> list[str]:
    """Synchronous wrapper; see `name_collision_check_async`."""
    return asyncio.run(name_collision_check_async(entities, llm, batch_size))


async def name_collision_check_async(entities: list[Entity], llm,
                                     batch_size: int = 8) -> list[str]:
    """Returns the list of entity_ids the model claims to recognize (a real-
    world collision — a company template minting "Apple Analytics", a
    country template minting a name with its own pretraining associations).
    `llm` needs only an async .generate(prompt, ...) method."""
    flagged: list[str] = []
    for i in range(0, len(entities), batch_size):
        batch = entities[i:i + batch_size]
        names = ", ".join(f'"{e.name}"' for e in batch)
        prompt = (f"Have you heard of any of these: {names}? "
                 f"Answer only with any that ring a bell, or 'none'.")
        resp = (await llm.generate(prompt, role="synthesizer", max_tokens=60,
                                   temperature=0.0) or "").lower()
        if "none" in resp and not any(e.name.lower() in resp for e in batch):
            continue
        for e in batch:
            if e.name.lower() in resp:
                flagged.append(e.entity_id)
    return flagged


def _reroll_entity_name(entity: Entity, rng: random.Random,
                        template: WorldTemplate, used_names: set[str]) -> Entity:
    new_name = _mint_name(entity.kind, rng, template, used_names)
    used_names.add(new_name)
    return Entity(entity_id=entity.entity_id, kind=entity.kind, name=new_name,
                 aliases=_aliases_for(entity.kind, new_name, rng))


def run_gate2(registry: FactRegistry, llm, rng: random.Random,
             max_rerolls: int = 5) -> FactRegistry:
    """Runs Gate 2, rerolling any flagged entity's name in place (mutates a
    COPY of the registry's entity list; other facts referencing entity_id
    are untouched since entity_id, not name, is the join key)."""
    template = _TEMPLATES[registry.template_name]
    entities = list(registry.entities)
    used_names = {e.name for e in entities}
    for attempt in range(max_rerolls):
        flagged = name_collision_check(entities, llm)
        if not flagged:
            registry.entities = entities
            return registry
        flagged_set = set(flagged)
        entities = [
            _reroll_entity_name(e, rng, template, used_names) if e.entity_id in flagged_set
            else e
            for e in entities
        ]
        print(f"[worlds] Gate 2: rerolled {len(flagged)} colliding entity name(s), "
              f"attempt {attempt+1}/{max_rerolls}")
    raise RuntimeError(f"Gate 2: could not clear name collisions after "
                       f"{max_rerolls} reroll attempts")


# ---------------------------------------------------------------------------
# Gate 1 — B-from-memory ceiling (spec Sec 4.1). Reuses ab_harness's
# DirectModel + _STRONG_DIRECT path (condition B) exactly, pointed at the
# mechanical checklist scorer instead of the pairwise judge.
# ---------------------------------------------------------------------------

_GATE1_CEILING = 0.20


def _mechanical_b_score(answer: str, registry: "FactRegistry") -> float:
    """Mean hit rate over atomic_fact_present + quantity_correct items only
    (spec Sec 4.1: "the two item types a memory-only answer could
    conceivably satisfy by chance/generic hedging"). Deferred import to
    avoid a hard circular dependency (world_score imports FactRegistry from
    this module)."""
    from eval.world_score import build_checklist_items, score_item
    items = [it for it in build_checklist_items(registry)
            if it["item_type"] in ("atomic_fact_present", "quantity_correct")]
    if not items:
        return 0.0
    hits = sum(1 for it in items if score_item(answer, it, registry)[0])
    return hits / len(items)


async def run_gate1_async(registry: FactRegistry, llm, prompts: list) -> float:
    """Runs condition-B (zero-evidence direct call) against every synthetic
    prompt for this world and returns the mean mechanical B-score. `llm`
    needs only an async .generate(...) method (DirectModel or a stub)."""
    from eval.ab_harness import _STRONG_DIRECT
    scores = []
    for p in prompts:
        prompt_text = _STRONG_DIRECT.format(q=p.text)
        ans = await llm.generate(prompt_text, role="synthesizer", max_tokens=600,
                                 temperature=0.3)
        scores.append(_mechanical_b_score(ans or "", registry))
    return sum(scores) / len(scores) if scores else 0.0


def run_gate1(registry: FactRegistry, llm, prompts: list) -> float:
    return asyncio.run(run_gate1_async(registry, llm, prompts))


def _log_gate1_result(seed: int, template_name: str, b_score: float,
                      accepted: bool, gate_log_path: Optional[Path] = None) -> None:
    path = gate_log_path or _GATE_LOG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"seed": seed, "template": template_name, "b_score": round(b_score, 4),
             "accepted": accepted, "ceiling": _GATE1_CEILING,
             "reason": ("b_score >= ceiling" if not accepted else "ok"),
             "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Persistence (registry.json + rendered/<doc_id>.txt + documents_manifest.json)
# ---------------------------------------------------------------------------

def _world_dir(seed: int, out_dir: Optional[Path] = None) -> Path:
    return out_dir or (_WORLDS_DIR / str(seed))


def _registry_to_dict(registry: FactRegistry) -> dict:
    d = asdict(registry)
    return d


def _registry_from_dict(d: dict) -> FactRegistry:
    entities = [Entity(**{**e, "aliases": tuple(e.get("aliases", ()))})
               for e in d["entities"]]
    quantities = [Quantity(**{**q, "surface_forms": tuple(q.get("surface_forms", ()))})
                 for q in d["quantities"]]
    relations = [CausalRelation(**r) for r in d["relations"]]
    events = [Event(**{**e, "entity_ids": tuple(e.get("entity_ids", ()))})
             for e in d["events"]]
    contradictions = [Contradiction(**c) for c in d["contradictions"]]
    documents = [DocumentSpec(**doc) for doc in d.get("documents", [])]
    return FactRegistry(world_seed=d["world_seed"], template_name=d["template_name"],
                        entities=entities, quantities=quantities, relations=relations,
                        events=events, contradictions=contradictions,
                        filler_seed_material=list(d.get("filler_seed_material", [])),
                        documents=documents)


def save_world(registry: FactRegistry, docs: list[RenderedDoc],
               out_dir: Optional[Path] = None) -> Path:
    wdir = _world_dir(registry.world_seed, out_dir)
    wdir.mkdir(parents=True, exist_ok=True)
    (wdir / "registry.json").write_text(
        json.dumps(_registry_to_dict(registry), indent=2), encoding="utf-8")
    rendered_dir = wdir / "rendered"
    rendered_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for d in docs:
        (rendered_dir / f"{d.doc_id}.txt").write_text(d.text, encoding="utf-8")
        manifest.append({"doc_id": d.doc_id, "doc_type": d.doc_type,
                         "extracted_from": d.extracted_from, "renderer": d.renderer})
    (wdir / "documents_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return wdir


def load_world(seed: int, out_dir: Optional[Path] = None) -> FactRegistry:
    """Loads just the FactRegistry for `seed` (the ground truth) — matches the
    spec Sec 5.2 call pattern `eval.worlds.load_world(args.world)`. Use
    `load_rendered_docs` alongside this for the documents."""
    wdir = _world_dir(seed, out_dir)
    path = wdir / "registry.json"
    if not path.exists():
        raise FileNotFoundError(f"no world at {wdir} - run generate_world({seed}, ...) first")
    return _registry_from_dict(json.loads(path.read_text(encoding="utf-8")))


def load_rendered_docs(seed: int, out_dir: Optional[Path] = None) -> list[RenderedDoc]:
    wdir = _world_dir(seed, out_dir)
    manifest_path = wdir / "documents_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"no rendered docs at {wdir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    docs = []
    for m in manifest:
        text = (wdir / "rendered" / f"{m['doc_id']}.txt").read_text(encoding="utf-8")
        docs.append(RenderedDoc(doc_id=m["doc_id"], doc_type=m["doc_type"], text=text,
                                extracted_from=m["extracted_from"], renderer=m["renderer"]))
    return docs


def _gate_log_accepted(seed: int, template_name: str,
                       gate_log_path: Optional[Path] = None) -> bool:
    path = gate_log_path or _GATE_LOG_PATH
    if not path.exists():
        return False
    accepted = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("seed") == seed and e.get("template") == template_name:
            accepted = bool(e.get("accepted"))
    return accepted


# ---------------------------------------------------------------------------
# Orchestrator (spec Sec 4.3)
# ---------------------------------------------------------------------------

def generate_world(seed: int, template_name: str, renderer: str = "template",
                   llm=None, gate1_llm=None, gate2_llm=None,
                   skip_gate1: bool = False, skip_gate2: bool = False,
                   out_dir: Optional[Path] = None, gate_log_path: Optional[Path] = None,
                   max_world_regen: int = 5
                   ) -> tuple[FactRegistry, list[RenderedDoc]]:
    """generate_world(seed) -> Stage A -> Gate 2 -> Stage B -> Stage C ->
    Gate 1 -> accepted world persisted under eval/worlds/<seed>/ (spec Sec 4.3).

    Idempotent: if eval/worlds/<seed>/ already exists AND the gate log shows
    this (seed, template) accepted, reuses it without regenerating anything
    (including without re-running Gate 1/Gate 2, which would otherwise cost
    an LLM call every invocation).
    """
    wdir = _world_dir(seed, out_dir)
    if (wdir / "registry.json").exists() and _gate_log_accepted(seed, template_name, gate_log_path):
        print(f"[worlds] reuse existing accepted world: {wdir}")
        return load_world(seed, out_dir), load_rendered_docs(seed, out_dir)

    cur_seed = seed
    for regen in range(max_world_regen):
        registry = build_fact_registry(cur_seed, template_name)
        rng = random.Random(cur_seed)

        if not skip_gate2 and gate2_llm is not None:
            registry = run_gate2(registry, gate2_llm, rng)

        docs = render_documents(registry, renderer=renderer, llm=llm,
                                out_dir=(_world_dir(cur_seed, out_dir) / "rendered"))
        verify_results = verify_render(registry, docs)
        for vr, doc in zip(verify_results, docs):
            if not vr.ok:
                print(f"[worlds] WARNING: {doc.doc_id} failed verify_render "
                     f"post-render (missing={vr.missing_facts}, "
                     f"fabricated={vr.fabricated_numbers}) - template renderer "
                     f"should have caught this pre-emptively.", file=sys.stderr)

        if skip_gate1 or gate1_llm is None:
            _log_gate1_result(cur_seed, template_name, 0.0, True, gate_log_path)
            registry.world_seed = cur_seed
            save_world(registry, docs, _world_dir(cur_seed, out_dir))
            return registry, docs

        prompts = synthetic_prompts_for_world(registry)
        b_score = asyncio.run(run_gate1_async(registry, gate1_llm, prompts))
        accepted = b_score < _GATE1_CEILING
        _log_gate1_result(cur_seed, template_name, b_score, accepted, gate_log_path)
        if accepted:
            save_world(registry, docs, _world_dir(cur_seed, out_dir))
            print(f"[worlds] world seed={cur_seed} template={template_name} "
                 f"ACCEPTED (B-score {b_score:.3f} < {_GATE1_CEILING})")
            return registry, docs
        print(f"[worlds] world seed={cur_seed} REJECTED: B-score {b_score:.3f} "
             f">= {_GATE1_CEILING} ceiling; regenerating with seed+1")
        cur_seed += 1

    raise RuntimeError(f"Gate 1: could not produce a passing world for template "
                       f"{template_name!r} after {max_world_regen} regenerations "
                       f"starting at seed {seed} — the template's entity/metric "
                       f"pools are likely too generic or too few facts to make "
                       f"guessing hard (spec Sec 4.3).")


# ---------------------------------------------------------------------------
# synthetic_prompts_for_world — mirrors eval/prompts.py's Prompt exactly, but
# as a function (spec Sec 5.2): the prompt set depends on which world/seed
# is loaded at runtime, so it can't be a module-level constant the way
# OVERCONTEXT_SET is.
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATES = {
    "invented_company": (
        "analysis",
        (
            "task_1",
            "Based on the provided evidence, summarize {subject}'s reported "
            "financial and operational performance, including specific "
            "figures where given, and identify any places where different "
            "sources disagree.",
        ),
        (
            "task_2",
            "Based on the provided evidence, what does the record say about "
            "{subject}'s revenue trajectory, and where do the filings and "
            "trade press conflict on specific reported figures?",
        ),
    ),
    "invented_country": (
        "analysis",
        (
            "task_1",
            "Based on the provided evidence, summarize the key statistics "
            "and developments concerning {subject}, including specific "
            "figures where given, and identify any places where sources "
            "disagree.",
        ),
        (
            "task_2",
            "Based on the provided evidence, what do the yearbook and news "
            "archive say about {subject}, and where does the government "
            "account conflict with the opposition rebuttal on specific "
            "reported figures?",
        ),
    ),
    "case_file": (
        "problem_solving",
        (
            "task_1",
            "Based on the provided evidence, reconstruct the timeline of "
            "events around {subject}, including specific figures where "
            "given, and identify any contradictions between witness "
            "accounts.",
        ),
        (
            "task_2",
            "Based on the provided evidence, what do the witness statements "
            "say happened around {subject}, and where do specific reported "
            "figures (times, distances, durations) conflict between "
            "accounts?",
        ),
    ),
}


def synthetic_prompts_for_world(world: FactRegistry) -> list:
    """Builds a small set (>= 2) of eval.prompts.Prompt objects scoped to
    this world — two angles on the same subject (a summary prompt and a
    conflict-focused prompt) so a --mini N harness run always has more than
    one prompt to exercise conditions ABEF against, matching the pattern of
    every other registered prompt set (DEFAULT_SET / OVERCONTEXT_SET both
    ship several). Also emits the must_include compatibility list (spec Sec
    3.2) so eval.judge.factual_score/factual_table keep working unmodified
    on synthetic prompts."""
    from eval.prompts import Prompt
    task, *variants = _PROMPT_TEMPLATES.get(
        world.template_name, _PROMPT_TEMPLATES["invented_company"])
    main_subject = next((e for e in world.entities
                        if e.kind in ("company", "location")), world.entities[0])
    must_include = [[main_subject.name.lower()] + [a.lower() for a in main_subject.aliases]]
    for q in world.quantities[:6]:
        must_include.append(list(q.surface_forms) or [str(q.value)])
    prompts = []
    for pid_base, template in variants:
        prompts.append(Prompt(
            pid=f"synth_{world.world_seed}_{pid_base}", task=task,
            text=template.format(subject=main_subject.name),
            factual=True, must_include=list(must_include)))
    return prompts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Build/inspect a synthetic Stage-1 world")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--template", choices=available_templates(),
                    default="invented_company")
    ap.add_argument("--renderer", choices=("template", "llm"), default="template")
    ap.add_argument("--skip-gate1", action="store_true",
                    help="skip the B-from-memory ceiling gate (needs an LLM call)")
    ap.add_argument("--skip-gate2", action="store_true",
                    help="skip the name-collision gate (needs an LLM call)")
    return ap


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    llm = None
    gate1_llm = None
    gate2_llm = None
    if args.renderer == "llm" or not args.skip_gate1 or not args.skip_gate2:
        import os
        if os.environ.get("MOCK_LLM", "").strip() not in ("", "0", "false", "False"):
            from core.llm import MockLLM
            engine = MockLLM()
        elif os.environ.get("GROQ_API_KEY"):
            from core.llm_groq import GroqBackend
            engine = GroqBackend(model="llama-3.1-8b-instant",
                                 api_key=os.environ["GROQ_API_KEY"])
        else:
            engine = None
        if args.renderer == "llm":
            llm = engine
        if not args.skip_gate1:
            gate1_llm = engine
        if not args.skip_gate2:
            gate2_llm = engine
    registry, docs = generate_world(
        args.seed, args.template, renderer=args.renderer, llm=llm,
        gate1_llm=gate1_llm, gate2_llm=gate2_llm,
        skip_gate1=args.skip_gate1, skip_gate2=args.skip_gate2)
    print(f"[worlds] world seed={args.seed} template={args.template}: "
         f"{len(registry.entities)} entities, {len(registry.quantities)} quantities, "
         f"{len(registry.events)} events, {len(registry.contradictions)} contradictions, "
         f"{len(docs)} documents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
