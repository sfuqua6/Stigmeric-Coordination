"""Universal signal type system for stigmergic coordination.

This module defines STRUCTURAL signal types that are independent of task domain.
The types describe the ROLE a signal plays in swarm coordination, not its semantic meaning.

DO NOT add domain-specific types here. Use task_config display_names for semantic labels.
"""


class SignalType:
    """Universal signal types for all swarm coordination modes.

    These types are structural primitives that enable stigmergic coordination.
    They are task-agnostic and mode-agnostic.

    Workflow:
        1. Scouts deposit INITIAL signals (starting points for exploration)
        2. Foragers sample INITIAL, deposit SUPPORT or CRITIQUE
        3. Critics evaluate quality, amplify/dampen strength
        4. Haters sample INITIAL, deposit OBJECTION (adversarial)
        5. Validators check quality gates
        6. Synthesizer aggregates all signals into SYNTHESIS

    Signal Graph Structure:
        INITIAL (root)
        ├── SUPPORT (elaboration)
        │   ├── SUPPORT (nested support)
        │   └── CRITIQUE (challenge to support)
        ├── CRITIQUE (weakness)
        │   └── SUPPORT (counter-critique)
        └── OBJECTION (adversarial)
            └── DEFENSE (forager response)
                └── OBJECTION (hater counter)
    """

    # Core signal types (used by all task modes)
    INITIAL = "INITIAL"          # Scout-generated starting points (was: CLAIM, DRAFT, FINDING, SOLUTION)
    SUPPORT = "SUPPORT"          # Forager elaborations that strengthen (was: EVIDENCE, REFINEMENT, IMPLEMENTATION)
    CRITIQUE = "CRITIQUE"        # Forager challenges that identify weaknesses (was: CRITIQUE, CHALLENGE)
    OBJECTION = "OBJECTION"      # Hater adversarial contradictions (was: COUNTER_EVIDENCE, OBJECTION)
    SYNTHESIS = "SYNTHESIS"      # Synthesizer final output (was: SYNTHESIS, ANSWER)

    # Internal coordination types (not visible in final output)
    OBSERVATION = "OBSERVATION"  # Raw data ingestion (document processing mode)
    INSIGHT = "INSIGHT"          # Forager pattern discoveries (document processing mode)
    DEFENSE = "DEFENSE"          # Forager responses to objections (dialogue mode)

    # Legacy types (deprecated, kept for backward compatibility)
    # TODO: Remove these after migrating all code to universal types
    CLAIM = "INITIAL"            # Alias for backward compatibility
    EVIDENCE = "SUPPORT"         # Alias for backward compatibility
    COUNTER_EVIDENCE = "OBJECTION"  # Alias for backward compatibility
    DRAFT = "INITIAL"            # Alias for backward compatibility
    REFINEMENT = "SUPPORT"       # Alias for backward compatibility
    ALTERNATIVE = "OBJECTION"    # Alias for backward compatibility
    FINDING = "INITIAL"          # Alias for backward compatibility
    SOLUTION = "INITIAL"         # Alias for backward compatibility
    IMPLEMENTATION = "SUPPORT"   # Alias for backward compatibility
    CHALLENGE = "CRITIQUE"       # Alias for backward compatibility

    @classmethod
    def all_types(cls) -> list[str]:
        """Get list of all core signal types (excluding aliases).

        Returns:
            List of core signal type strings
        """
        return [
            cls.INITIAL,
            cls.SUPPORT,
            cls.CRITIQUE,
            cls.OBJECTION,
            cls.SYNTHESIS,
            cls.OBSERVATION,
            cls.INSIGHT,
            cls.DEFENSE
        ]

    @classmethod
    def output_types(cls) -> list[str]:
        """Get types that appear in final synthesis output.

        Returns:
            List of output signal types
        """
        return [
            cls.INITIAL,
            cls.SUPPORT,
            cls.CRITIQUE,
            cls.OBJECTION,
            cls.SYNTHESIS
        ]

    @classmethod
    def is_initial_type(cls, signal_type: str) -> bool:
        """Check if a signal type is an initial/root type.

        Args:
            signal_type: Signal type to check

        Returns:
            True if type is an initial signal (scout-generated)
        """
        # Handle both new and legacy types
        return signal_type in [
            cls.INITIAL,
            "CLAIM", "DRAFT", "FINDING", "SOLUTION"  # Legacy compatibility
        ]

    @classmethod
    def is_support_type(cls, signal_type: str) -> bool:
        """Check if a signal type is a support/elaboration type.

        Args:
            signal_type: Signal type to check

        Returns:
            True if type supports/strengthens parent signal
        """
        return signal_type in [
            cls.SUPPORT,
            "EVIDENCE", "REFINEMENT", "IMPLEMENTATION"  # Legacy compatibility
        ]

    @classmethod
    def is_critique_type(cls, signal_type: str) -> bool:
        """Check if a signal type is a critique/challenge type.

        Args:
            signal_type: Signal type to check

        Returns:
            True if type critiques/challenges parent signal
        """
        return signal_type in [
            cls.CRITIQUE,
            "CHALLENGE"  # Legacy compatibility
        ]

    @classmethod
    def is_objection_type(cls, signal_type: str) -> bool:
        """Check if a signal type is an adversarial objection type.

        Args:
            signal_type: Signal type to check

        Returns:
            True if type is adversarial objection
        """
        return signal_type in [
            cls.OBJECTION,
            "COUNTER_EVIDENCE", "ALTERNATIVE"  # Legacy compatibility
        ]


# Convenience constants for common usage
INITIAL = SignalType.INITIAL
SUPPORT = SignalType.SUPPORT
CRITIQUE = SignalType.CRITIQUE
OBJECTION = SignalType.OBJECTION
SYNTHESIS = SignalType.SYNTHESIS
OBSERVATION = SignalType.OBSERVATION
INSIGHT = SignalType.INSIGHT
DEFENSE = SignalType.DEFENSE
