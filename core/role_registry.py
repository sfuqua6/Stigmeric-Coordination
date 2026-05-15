"""Role registry — maps task types to agent role classes.

The framework is task-adaptive when role implementations are domain-specific.
For coding tasks, agents treat signals as code artifacts rather than prose
claims: fenced code blocks, acceptance criteria, test results, and edge cases
replace the discourse primitives used in debate/analysis modes.

# FUTURE-CLAUDE NOTE: To add a new task type, define its role classes
# (subclassing the relevant BaseAgent specializations), add them to
# ROLES_FOR_TASK_CLASS, and add a task prompt template to TASK_PROMPTS in
# run_swarm.py. The pipeline instantiates agents via get_role_classes() so
# you do not need to touch the main pipeline loop.

Usage:
    from core.role_registry import get_role_classes
    classes = get_role_classes("coding")
    ScoutClass = classes["scout"]
"""

from __future__ import annotations

from typing import Optional

# Default role classes (all task types unless overridden)
from agents.scout import Scout
from agents.developer import Developer
from agents.critic import Critic
from agents.hater import Hater
from agents.validator import Validator
from agents.synthesizer import Synthesizer

_DEFAULT_ROLES = {
    "scout": Scout,
    "developer": Developer,
    "critic": Critic,
    "hater": Hater,
    "validator": Validator,
    "synthesizer": Synthesizer,
}

# Task-specific overrides. Keys are task_type strings; values are dicts
# mapping role name → class. Missing keys fall back to _DEFAULT_ROLES.
_TASK_ROLE_OVERRIDES: dict[str, dict] = {}


def _register_coding_roles() -> None:
    """Lazily register coding roles to avoid circular import at module load."""
    try:
        from agents.coding_roles import (
            RequirementsScout,
            CodeDeveloper,
            StaticCritic,
            EdgeCaseHater,
            TestValidator,
            CodeSynthesizer,
        )
        _TASK_ROLE_OVERRIDES["coding"] = {
            "scout": RequirementsScout,
            "developer": CodeDeveloper,
            "critic": StaticCritic,
            "hater": EdgeCaseHater,
            "validator": TestValidator,
            "synthesizer": CodeSynthesizer,
        }
    except ImportError as exc:
        import warnings
        warnings.warn(f"[role_registry] coding roles not available: {exc}",
                      RuntimeWarning, stacklevel=2)


_coding_registered = False


def get_role_classes(task_type: str) -> dict:
    """Return the role class dict for a given task type.

    Falls back to default roles for any role not overridden by the task type.
    """
    global _coding_registered
    if task_type == "coding" and not _coding_registered:
        _register_coding_roles()
        _coding_registered = True

    overrides = _TASK_ROLE_OVERRIDES.get(task_type, {})
    return {**_DEFAULT_ROLES, **overrides}
