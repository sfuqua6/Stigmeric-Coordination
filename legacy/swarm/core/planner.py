"""Pre-flight task planner.

Runs a single LLM call before the swarm starts to set direction across
all agents without altering how they communicate. The planner is NOT a
swarm agent — it never reads or deposits signals.

What it produces:
  - output_format: the form the final answer should take
  - per-role directives: injected into each agent's prompt
  - decomposition: optional sub-problems for complex tasks
  - strength_mode: which scoring heuristic scouts/foragers use
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from ..llm.simple_llm import SimpleLLM
from .logging_config import get_logger

logger = get_logger(__name__)

_VALID_FORMATS = {"code", "prose", "list", "analysis", "creative"}
_VALID_STRENGTH = {"code", "prose"}

_PLANNER_PROMPT = """\
Analyze this task and complete every field below. Keep each answer to one sentence.

Task: "{prompt}"

OUTPUT_FORMAT: [one of: code / prose / list / analysis / creative]
SCOUT_DIRECTIVE: [what should each explorer produce?]
SUPPORT_DIRECTIVE: [how should developers elaborate on those outputs?]
CRITIQUE_DIRECTIVE: [what angle should critics examine from?]
CHALLENGE_DIRECTIVE: [what angle should challengers attack from?]
SYNTHESIS_DIRECTIVE: [what must the final answer look like?]
DECOMPOSITION: [write NONE if simple; or list 2-4 sub-problems separated by |]
STRENGTH_MODE: [one of: code / prose]
"""


@dataclass
class TaskFrame:
    """Structured task direction produced by the Planner.

    Injected into agent prompts to set direction without altering the
    stigmergic signal-flow wiring.
    """
    output_format: str = "prose"
    scout_directive: str = ""
    forager_support_directive: str = ""
    forager_critique_directive: str = ""
    hater_directive: str = ""
    synthesizer_directive: str = ""
    decomposition: List[str] = field(default_factory=list)
    strength_mode: str = "prose"


def _extract(label: str, text: str) -> str:
    m = re.search(rf"^{label}\s*[:\=]\s*(.+)$", text, re.MULTILINE | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def _parse_frame(raw: str) -> TaskFrame:
    frame = TaskFrame()

    fmt = _extract("OUTPUT_FORMAT", raw).lower().strip("[]")
    frame.output_format = fmt if fmt in _VALID_FORMATS else "prose"

    frame.scout_directive = _extract("SCOUT_DIRECTIVE", raw)
    frame.forager_support_directive = _extract("SUPPORT_DIRECTIVE", raw)
    frame.forager_critique_directive = _extract("CRITIQUE_DIRECTIVE", raw)
    frame.hater_directive = _extract("CHALLENGE_DIRECTIVE", raw)
    frame.synthesizer_directive = _extract("SYNTHESIS_DIRECTIVE", raw)

    sub_raw = _extract("DECOMPOSITION", raw)
    if sub_raw and sub_raw.upper().strip("[]") not in ("NONE", ""):
        parts = [p.strip().strip("[]") for p in sub_raw.split("|") if p.strip()]
        frame.decomposition = [p for p in parts if len(p) > 3][:4]

    sm = _extract("STRENGTH_MODE", raw).lower().strip("[]")
    frame.strength_mode = sm if sm in _VALID_STRENGTH else "prose"

    return frame


def _is_usable(frame: TaskFrame) -> bool:
    """Return True if the parsed frame has enough useful content."""
    return bool(frame.scout_directive or frame.synthesizer_directive)


class Planner:
    """Single-call pre-flight task planner. Not a swarm agent."""

    async def plan(self, task_prompt: str, llm: SimpleLLM) -> Optional[TaskFrame]:
        """Analyze task and return a TaskFrame directing the swarm.

        Returns None on failure — callers should fall back to default behavior.
        """
        prompt = _PLANNER_PROMPT.format(prompt=task_prompt[:400])
        try:
            raw = await llm.generate(prompt, max_tokens=250, temperature=0.2, use_cache=False)
            if not raw or len(raw.strip()) < 30:
                logger.warning("[Planner] Response too short — using default frame")
                return None

            frame = _parse_frame(raw)

            if not _is_usable(frame):
                logger.warning("[Planner] Could not parse usable directives — using default frame")
                return None

            logger.info(
                f"[Planner] format={frame.output_format!r} "
                f"strength={frame.strength_mode!r} "
                f"sub-problems={len(frame.decomposition)}"
            )
            return frame

        except Exception as e:
            logger.error(f"[Planner] Failed: {e}", exc_info=True)
            return None
