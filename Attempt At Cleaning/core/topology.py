"""Answer-space topology: bounds-first exploration scaffolding.

Generates a task-conditional AnswerSpaceTopology before scouts run.
Scouts are assigned cells from this topology and carry topology_coords
in their INITIAL signal metadata so coverage can be tracked in the
projection layer.

Usage in run_swarm.py:
    from core.topology import generate_topology, assign_topology_cells
    topology = await generate_topology(task_prompt, task_type, llm)
    cells = assign_topology_cells(topology, NUM_SCOUTS)

Signals carry metadata["topology_coords"] = cell_tuple_or_None.
build_projection reads this to populate topology_coverage/uncovered_cells.
"""

from __future__ import annotations

import itertools
import json
import re
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AxisSpec:
    name: str
    values: tuple[str, ...]   # 2–5 discrete categorical values


@dataclass(frozen=True)
class AnchorCorner:
    coords: tuple[str, ...]   # one value per axis, in axis-declaration order
    label: str
    rationale: str            # one-sentence justification


@dataclass
class AnswerSpaceTopology:
    task_type: str
    axes: list[AxisSpec]
    anchor_corners: list[AnchorCorner]
    boundary_exclusions: list[str]
    generation_prompt: str = ""
    audit_log: list[dict] = field(default_factory=list)

    def all_cells(self) -> list[tuple[str, ...]]:
        """Return all coordinate tuples in the topology grid."""
        return list(itertools.product(*[a.values for a in self.axes]))

    def anchor_coords_set(self) -> set[tuple[str, ...]]:
        return {c.coords for c in self.anchor_corners}

    def cell_description(self, coords: tuple[str, ...]) -> str:
        """Human-readable description of a cell."""
        parts = [f"{ax.name}={v}" for ax, v in zip(self.axes, coords)]
        return ", ".join(parts)


# ---------------------------------------------------------------------------
# Per-task prompt templates
# ---------------------------------------------------------------------------

_TOPOLOGY_TEMPLATES: dict[str, str] = {
    "debate": (
        "Declare 3 axes for a debate task:\n"
        "  Axis 1: position (where on the spectrum from 'fully X' to 'not X' does the answer sit?)\n"
        "  Axis 2: framing (empirical | ethical | economic | historical)\n"
        "  Axis 3: scope (the domain the argument concerns — e.g. mitigation, adaptation, innovation)\n"
        "Declare at least 4 anchor corners spanning the debate space — the most distinct defensible positions.\n"
        "Exclude: positions that deny the empirical premise of the thesis; specific policy implementations."
    ),
    "analysis": (
        "Declare 3 axes for an analysis task:\n"
        "  Axis 1: epistemic stance (descriptive | predictive | prescriptive)\n"
        "  Axis 2: scale (micro | meso | macro)\n"
        "  Axis 3: evidence type (established | contested | speculative)\n"
        "Declare at least 4 anchor corners. Exclude: normative claims beyond description; "
        "predictions beyond 10 years without quantification."
    ),
    "problem_solving": (
        "Declare 3 axes for a problem-solving task:\n"
        "  Axis 1: intervention type (technical | behavioral | policy | structural)\n"
        "  Axis 2: time horizon (immediate | medium-term | long-term)\n"
        "  Axis 3: cost-benefit profile (low-cost-high-gain | high-cost-high-gain | "
        "low-cost-low-gain | high-cost-low-gain)\n"
        "Declare 3-4 anchor corners representing the most distinct feasible solution archetypes.\n"
        "Exclude: solutions requiring undemonstrated technology; solutions requiring international "
        "treaties without existing framework."
    ),
    "creative": (
        "Declare 2 axes for a creative task (creative tasks have lower-dimensional answer spaces):\n"
        "  Axis 1: form (the structural or genre choice — e.g. narrative, lyric, dramatic)\n"
        "  Axis 2: voice (the speaker stance — e.g. intimate, ironic, detached, celebratory)\n"
        "Declare 2-3 anchor corners representing the most distinct creative approaches.\n"
        "Exclude: pastiches of named copyrighted works; outputs requiring images or audio."
    ),
    "coding": (
        "First, extract key implementation constraints from the task prompt as bullet points.\n"
        "Then declare axes — each axis corresponds to one constraint dimension. "
        "Do not declare an axis for a constraint that has only one valid value.\n"
        "Always include:\n"
        "  Axis 1: correctness_completeness (minimal viable | production hardened)\n"
        "  Additional axes: derived from the primary algorithmic and spec dimensions "
        "(e.g. concurrency_model, eviction_semantics, memory_model, error_handling).\n"
        "Declare 2-3 anchor corners: simplest correct implementation (baseline); "
        "production-grade implementation (target); optional alternative approach.\n"
        "Exclude: distributed or multi-process variants unless explicitly requested; "
        "persistence unless explicitly requested."
    ),
}

_TOPOLOGY_DEFAULT_TEMPLATE = (
    "Declare 3 axes capturing the key dimensions the answer varies along.\n"
    "Declare 3-4 anchor corners representing the most distinct valid positions.\n"
    "Exclude claims that are trivially out of scope for this task."
)


# ---------------------------------------------------------------------------
# Topology generation
# ---------------------------------------------------------------------------

async def generate_topology(
    task_prompt: str,
    task_type: str,
    llm,
) -> Optional[AnswerSpaceTopology]:
    """Generate an AnswerSpaceTopology by one LLM call before scouts run.

    Returns None on parse failure after two retries; callers treat None as
    'no topology available' and skip coverage tracking with zero regression.
    """
    template = _TOPOLOGY_TEMPLATES.get(task_type, _TOPOLOGY_DEFAULT_TEMPLATE)
    schema_instruction = (
        '\nReply with EXACTLY this JSON (no other text, no markdown fences):\n'
        '{"axes": [{"name": "string", "values": ["string", ...]}, ...], '
        '"anchor_corners": [{"coords": ["string", ...], "label": "string", '
        '"rationale": "string"}, ...], '
        '"boundary_exclusions": ["string", ...]}'
    )
    prompt = (
        f"You are defining the answer space for a {task_type} task.\n"
        f"TASK: {task_prompt}\n\n"
        f"Your job: declare the dimensions valid answers vary along, the anchor "
        f"positions that span the space, and what is explicitly out of scope.\n\n"
        f"{template}"
        f"{schema_instruction}"
    )

    for attempt in range(2):
        try:
            raw = await llm.generate(
                prompt,
                role="topology",
                max_tokens=800,
                temperature=0.1,
            )
            topology = _parse_topology(raw or "", task_type, prompt)
            if topology is not None:
                print(
                    f"[topology] generated: {len(topology.axes)} axes, "
                    f"{len(topology.anchor_corners)} anchors, "
                    f"{len(topology.all_cells())} cells"
                )
                return topology
            if attempt == 0:
                # Retry with schema reminder
                prompt = prompt + "\n\nIMPORTANT: reply with only the JSON object, nothing else."
        except Exception as exc:
            print(f"[topology] generation attempt {attempt+1} failed: {type(exc).__name__}: {exc}")
    print("[topology] generation failed after 2 attempts — proceeding without topology")
    return None


def _parse_topology(raw: str, task_type: str, generation_prompt: str) -> Optional[AnswerSpaceTopology]:
    """Parse and validate topology JSON from the LLM's response."""
    text = (raw or "").strip()
    # Strip markdown fences if present
    text = re.sub(r"```(?:json)?", "", text).strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        raw_axes = d.get("axes", [])
        if not isinstance(raw_axes, list) or not (1 <= len(raw_axes) <= 5):
            return None
        axes: list[AxisSpec] = []
        for a in raw_axes:
            name = str(a.get("name", "")).strip()
            values = [str(v).strip() for v in a.get("values", []) if str(v).strip()]
            if not name or not (2 <= len(values) <= 6):
                return None
            axes.append(AxisSpec(name=name, values=tuple(values)))

        raw_corners = d.get("anchor_corners", [])
        if not isinstance(raw_corners, list) or not (2 <= len(raw_corners) <= 8):
            return None
        corners: list[AnchorCorner] = []
        for c in raw_corners:
            coords = tuple(str(v).strip() for v in c.get("coords", []))
            label = str(c.get("label", "")).strip()
            rationale = str(c.get("rationale", "")).strip()
            if len(coords) != len(axes) or not label:
                return None
            corners.append(AnchorCorner(coords=coords, label=label, rationale=rationale or label))

        exclusions = [str(e).strip() for e in d.get("boundary_exclusions", []) if str(e).strip()]

        return AnswerSpaceTopology(
            task_type=task_type,
            axes=axes,
            anchor_corners=corners,
            boundary_exclusions=exclusions,
            generation_prompt=generation_prompt[:500],
        )
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# Cell assignment for scouts
# ---------------------------------------------------------------------------

def assign_topology_cells(
    topology: Optional[AnswerSpaceTopology],
    n_scouts: int,
) -> list[Optional[tuple]]:
    """Return a cell assignment (coords tuple) for each scout index [0, n_scouts).

    Anchor corners get priority: scouts 0..len(anchors)-1 receive one anchor
    corner each. Remaining scouts cycle through interior cells (non-anchor).
    If n_scouts > total cells, multiple scouts share cells — intentional, to
    strengthen coverage in that region.

    Returns list of length n_scouts. None entries mean no topology (pass-through).
    """
    if topology is None:
        return [None] * n_scouts

    anchor_cells = [c.coords for c in topology.anchor_corners]
    all_cells = topology.all_cells()
    interior = [c for c in all_cells if c not in set(anchor_cells)]

    ordered = anchor_cells + interior
    if not ordered:
        return [None] * n_scouts

    return [ordered[i % len(ordered)] for i in range(n_scouts)]


def format_cell_for_prompt(topology: AnswerSpaceTopology, cell: tuple[str, ...]) -> str:
    """Format a topology cell as a human-readable string for the scout prompt."""
    return topology.cell_description(cell)
