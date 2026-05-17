"""Content quality filters shared between the agent layer and the knowledge base.

Used in two places:
  1. agents/base.py and agents/scout.py — reject junk before depositing.
  2. core/knowledge_base.py — refuse to persist scratchpad that slipped through.

Three pattern families, per the post-run audit:
  Family 1: first-person reflexive openers ("This is my", "I need to", etc.)
  Family 2: mid-stream reasoning interjections ("Hmm,", "Maybe I can", etc.)
  Family 3: prompt-template echoes (---SIGNAL, [Insert, signal IDs, Assistant, etc.)
Plus a cheap decoder-repetition check (>=50% identical sentences).
"""

from __future__ import annotations

import re
from collections import Counter


# ---------------------------------------------------------------------------
# FAMILY 1 + 2 (full): first-person reflexive AND mid-stream interjections.
# These match the FIRST line only (they're scratchpad openers).
# ---------------------------------------------------------------------------
_JUNK_MARKER_RE = re.compile(
    r"^(?:"
    # family 1 — reflexive first-person openers
    r"so,?\s+i\s+(?:have\s+to|need\s+to|will|should|am\s+going)"
    r"|this\s+is\s+my\b"
    r"|alright,?"
    r"|okay,?"
    r"|ok,?"
    r"|i'?ll\s+(?:need|present|write|produce|develop|argue|explain|consider|examine|start|begin|focus)"
    r"|i'?m\s+(?:going|trying|asked|supposed|tasked)\s+to"
    r"|since\s+the\s+"
    r"|given\s+the\s+(?:partition|task|signal|claim|prompt)"
    r"|let\s+me\b"
    r"|the\s+task\s+is\b"
    r"|the\s+user\s+(?:wants|asks|is\s+asking|has\s+asked|wants\s+me)"
    r"|my\s+(?:claim|response|answer|initial|task|goal|argument|critique|objection)\s+(?:is|will|would|should)?"
    r"|i\s+need\s+to\b"
    r"|i\s+should\s+(?:produce|write|develop|consider|examine|present|argue|explain|provide|focus|note)"
    r"|i\s+have\s+to\b"
    r"|i\s+must\b"
    r"|i\s+want\s+to\b"
    r"|i\s+think\s+(?:i\b|that\b|the\b|this\b)"
    r"|i\s+can\s+(?:see|argue|present|develop|note|think|consider)"
    r"|to\s+answer\s+this\b"
    r"|i\s+am\s+(?:going\s+to|tasked|supposed|here\s+to)"
    r"|as\s+(?:a\s+)?(?:scout|forager|critic|hater|validator|synthesizer)\b"
    # family 2 — mid-stream interjections at start of line
    r"|hmm,?"
    r"|wait,?"
    r"|actually,?\s+"
    r"|maybe\s+i\s+(?:can|should|could|would)"
    r"|or\s+perhaps\b"
    r"|alternatively,?\s+i\b"
    r"|but\s+i\s+(?:think|need|should|can|must)"
    r")",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# FAMILY 3: template-leakage patterns. These can appear anywhere in the content,
# so we search() rather than match(). ANY hit is grounds for rejection.
# ---------------------------------------------------------------------------
_TEMPLATE_LEAK_PATTERNS = [
    # explicit prompt-scaffolding delimiters echoed back
    re.compile(r"---\s*(?:SIGNAL|ARTIFACT|EVIDENCE|CLAIM|END\s+(?:SIGNAL|ARTIFACT|EVIDENCE|CLAIM))\b", re.IGNORECASE),
    # placeholder syntax we use in prompts that the model regurgitated
    re.compile(r"\[(?:Insert|placeholder|Your|Example|TASK|CLAIM|No\s+claim\s+yet)\b", re.IGNORECASE),
    # signal IDs from our store — agents shouldn't be citing IDs in their deposits
    re.compile(r"\b(?:INITIAL|SUPPORT|CRITIQUE|OBJECTION|VERIFICATION)_\d{4,5}\b"),
    # chat-template token leakage (Qwen / generic)
    re.compile(r"\b(?:Assista(?:nt)?|<\|im_(?:end|start)\|>|<\|assistant\|>|<\|user\|>|<\|system\|>)\b"),
    # literal section labels from our prompts
    re.compile(r"^\s*(?:TASK|CLAIM|OBSERVATION|DEVELOPMENT|CRITIQUE|OBJECTION|SCORE|ANSWER)\s*:", re.MULTILINE | re.IGNORECASE),
    # explicit instructions the model parroted
    re.compile(r"(?:use\s+the\s+following\s+format|produce\s+(?:your|one|the)\s+claim|format\s+your\s+response)", re.IGNORECASE),
    # "Format your response exactly as:" type leakage
    re.compile(r"(?:exactly\s+as|formatted\s+as|response\s+exactly)\s*[:\.]", re.IGNORECASE),
    # reasoning mid-stream interjections that slipped past strip_reasoning
    re.compile(r"\bbut\s+wait[,:\s]", re.IGNORECASE),
    re.compile(r"\bnow\s+it'?s\s+your\s+turn\b", re.IGNORECASE),
    re.compile(r"\bnow\s+i\s+need\s+to\b", re.IGNORECASE),
    re.compile(r"\bthat'?s\s+my\s+(?:first[- ]order\s+)?claim\b", re.IGNORECASE),
    re.compile(r"\bi\s+remember\s+hearing\b", re.IGNORECASE),
    re.compile(r"\bi'?m\s+supposed\s+to\b", re.IGNORECASE),
    re.compile(r"\bthis\s+initial\s+claim\b", re.IGNORECASE),
    re.compile(r"\bthis\s+is\s+my\s+initial\s+claim\b", re.IGNORECASE),
    re.compile(r"\bbased\s+on\s+the\s+evidence\s+assigned\b", re.IGNORECASE),
    # Meta-acknowledgment of empty/sparse input — scouts admitting they have
    # no evidence to ground a claim. These look like substantive prose but
    # are actually structural complaints about the partition; rejecting them
    # prevents the field from treating "I had nothing to work with" as a
    # claim to argue about.
    re.compile(r"\bgiven\s+no\s+specific\s+evidence\b", re.IGNORECASE),
    re.compile(r"\bwithout\s+any\s+specific\s+evidence\b", re.IGNORECASE),
    re.compile(r"\bthe\s+absence\s+of\s+empirical\s+proof\b", re.IGNORECASE),
    re.compile(r"\bin\s+my\s+(?:corpus\s+)?partition\b", re.IGNORECASE),
    re.compile(r"\bconsidering\s+the\s+absence\s+of\b", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Null / refusal / trivially-short content
# ---------------------------------------------------------------------------
_NULL_PATTERNS = [
    re.compile(r"^\s*\(?\s*(?:none|n/?a|null|skip|no\s+claim|none\s+yet|nothing\s+to)\b", re.IGNORECASE),
]

_PRONOUN_RE = re.compile(r"\b(?:i|me|my|myself|we|our|ours)\b", re.IGNORECASE)

_MIN_CONTENT_WORDS = 6  # was 3; bumped because real claims need substance


def _has_decoder_repetition(content: str) -> bool:
    """Cheap check: 3+ sentences, and 50%+ share the same 80-char prefix.

    Catches the paragraph-multiplication pathology where DeepSeek emits the
    same sentence five times because repeat_penalty wasn't set or wasn't high
    enough. Even with repeat_penalty=1.15, some prompts still trigger it.
    """
    sentences = [s.strip() for s in re.split(r"[.!?]\s+", content) if s.strip()]
    if len(sentences) < 3:
        return False
    counts = Counter(s.lower()[:80] for s in sentences)
    most_common, n = counts.most_common(1)[0]
    return n / len(sentences) >= 0.5


def is_junk_output(content: str) -> bool:
    """Return True if content looks like agent scratchpad or template leakage.

    Six checks (any one rejects):
      1. Empty / whitespace-only.
      2. Too short (< _MIN_CONTENT_WORDS words).
      3. First line begins with a Family 1 or Family 2 scratchpad marker.
      4. Any Family 3 template-leakage pattern appears anywhere.
      5. First 50 words are >40% self-referential pronouns.
      6. Decoder repetition: 50%+ of sentences share an 80-char prefix.
    """
    if not content or not content.strip():
        return True

    stripped = content.strip()

    # (2) too short
    if len(stripped.split()) < _MIN_CONTENT_WORDS:
        return True

    # (3) scratchpad markers on the first line
    first_line = stripped.split("\n")[0].strip()
    if _JUNK_MARKER_RE.match(first_line):
        return True

    # (4) template-leakage anywhere in content
    for pat in _TEMPLATE_LEAK_PATTERNS:
        if pat.search(stripped):
            return True

    # null / refusal
    for pat in _NULL_PATTERNS:
        if pat.search(first_line):
            return True

    # (5) pronoun density in first 50 words
    first_words = stripped.split()[:50]
    if first_words:
        pronoun_count = len(_PRONOUN_RE.findall(" ".join(first_words)))
        if pronoun_count / len(first_words) > 0.40:
            return True

    # (6) decoder repetition
    if _has_decoder_repetition(stripped):
        return True

    return False
