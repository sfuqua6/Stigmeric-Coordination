"""Universal signal types.

Five content types plus one verification type. The CRITIQUE type is now
split into CRITIQUE_POSITIVE (score >= 0.5, extends the support cluster)
and CRITIQUE_NEGATIVE (score < 0.5, adversarial challenge). This fixes the
valence-collapse bug where a high-quality positive assessment and a damning
rejection both deposited as CRITIQUE with identical structural weight.

    INITIAL           — opening claim/observation/draft from a Scout
    SUPPORT           — evidence or development that builds on an INITIAL or SUPPORT
    CRITIQUE_POSITIVE — evaluative endorsement (score >= 0.5); counted in support_set
    CRITIQUE_NEGATIVE — evaluative challenge (score < 0.5); counted in dissent_set
    OBJECTION         — adversarial counter-claim from a Hater
    VERIFICATION      — external-source check from a Validator (used by provenance boost)

CONTRARIAN_TYPES: signals that receive the slower DELTA_DECAY_CONTRARIAN
per round rather than DELTA_DECAY. Only negative-valence types are
contrarian; CRITIQUE_POSITIVE decays at the normal rate.

Backward-compatibility: the old CRITIQUE constant is preserved as an alias
for CRITIQUE_NEGATIVE so existing signals.json and knowledge_base files
still load. New deposits use CRITIQUE_POSITIVE / CRITIQUE_NEGATIVE. Do not
re-introduce CRITIQUE as a deposit type in new code.

# FUTURE-CLAUDE NOTE: CRITIQUE was split in §5 of the 2026-05 directive.
# CRITIQUE_POSITIVE goes into support_set in projection._compute_initial_metrics.
# CRITIQUE_NEGATIVE and OBJECTION go into dissent_set. DO NOT collapse these
# back into a single CRITIQUE type; that is the bug the split fixed. If you
# see CRITIQUE in the signal store JSON from old runs, it maps to CRITIQUE_NEGATIVE
# for projection purposes (see core/projection.py).
"""

INITIAL = "INITIAL"
SUPPORT = "SUPPORT"
CRITIQUE_POSITIVE = "CRITIQUE_POSITIVE"
CRITIQUE_NEGATIVE = "CRITIQUE_NEGATIVE"
OBJECTION = "OBJECTION"
VERIFICATION = "VERIFICATION"

# Backward compatibility alias: old code deposited CRITIQUE; treat as negative.
CRITIQUE = CRITIQUE_NEGATIVE

CONTENT_TYPES = (INITIAL, SUPPORT, CRITIQUE_POSITIVE, CRITIQUE_NEGATIVE, OBJECTION)
ALL_TYPES = CONTENT_TYPES + (VERIFICATION,)

# Only negative-valence types hold steady against decay (they represent
# genuine disagreement that should persist until actively rebutted).
CONTRARIAN_TYPES = (CRITIQUE_NEGATIVE, OBJECTION)

# Legacy set for code that still references the old CRITIQUE alias.
_LEGACY_CONTRARIAN_TYPES = (CRITIQUE_NEGATIVE, OBJECTION)
