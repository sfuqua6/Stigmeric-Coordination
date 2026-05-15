"""Universal signal types.

Four content types and one verification type. No legacy aliases, no
mode-conditional types, no display-name shadowing.

INITIAL       — opening claim/observation/draft from a Scout
SUPPORT       — evidence or development that builds on an INITIAL or SUPPORT
CRITIQUE      — evaluative challenge from a Critic
OBJECTION     — adversarial counter-claim from a Hater
VERIFICATION  — external-source check from a Validator (used by provenance boost)
"""

INITIAL = "INITIAL"
SUPPORT = "SUPPORT"
CRITIQUE = "CRITIQUE"
OBJECTION = "OBJECTION"
VERIFICATION = "VERIFICATION"

CONTENT_TYPES = (INITIAL, SUPPORT, CRITIQUE, OBJECTION)
ALL_TYPES = CONTENT_TYPES + (VERIFICATION,)
CONTRARIAN_TYPES = (CRITIQUE, OBJECTION)
