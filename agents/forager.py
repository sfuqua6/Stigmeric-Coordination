"""Backward-compatibility shim: Forager → Developer.

The role was renamed to Developer in §5 of the 2026-05 directive.
The 'Forager' metaphor (ant foraging for food) doesn't describe what
this role actually does (developing existing signals into defended SUPPORT).
The class name Forager is kept as an alias so existing imports don't break,
but all new code should import Developer from agents.developer.

# FUTURE-CLAUDE NOTE: Forager is a deprecated alias for Developer.
# Do not add new behavior here. Add it to agents/developer.py.
# This shim will be removed once all call sites are updated.
"""

from agents.developer import Developer as Forager, recent_dissent_targeted

__all__ = ["Forager", "recent_dissent_targeted"]
