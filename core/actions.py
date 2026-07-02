"""Action templates for the continuous worker pool.

Each action exposes:
  - `prompt(field_sample, task_prompt, user_prompt, ...) -> str` — build the
    LLM prompt for a worker about to execute this action
  - `parse(raw) -> ParsedDeposit` — turn the model's response into a deposit
    spec (signal_type, content, strength, parent_id_override)
  - `OUTPUT_TYPE` — the default signal type this action deposits
  - `precondition(field_state) -> bool` — whether the action is available
    given the current field state

Actions replace the old per-role classes (Scout/Developer/Critic/Hater/
Validator). A Worker picks an action per iteration; the action determines
prompt, sampling target, and output type.

No-leak rule still applies: prompts may only render Signal.content (plus
the agent's own prior outputs and retrieved search chunks). No ancestry
text, no other agents' reasoning, no parent_content fields.
"""

from __future__ import annotations

import json
import re as _re
from dataclasses import dataclass
from typing import Optional, Callable

from .signal_store import SignalStore, Signal
from .signal_types import (
    INITIAL, SUPPORT, CRITIQUE_POSITIVE, CRITIQUE_NEGATIVE,
    OBJECTION, VERIFICATION, SEARCH,
)


# Action enum
SCOUT    = "SCOUT"
DEVELOP  = "DEVELOP"
CHAIN    = "CHAIN"
CRITIQUE = "CRITIQUE"
OBJECT   = "OBJECT"
VALIDATE = "VALIDATE"
REFINE   = "REFINE"

ACTIONS = (SCOUT, DEVELOP, CHAIN, CRITIQUE, OBJECT, VALIDATE, REFINE)


# ---------------------------------------------------------------------------
# Field-state snapshot (read-only view passed to action choosers/preconditions)
# ---------------------------------------------------------------------------

@dataclass
class FieldState:
    n_initials: int = 0
    n_supports: int = 0
    n_critique_pos: int = 0
    n_critique_neg: int = 0
    n_objections: int = 0
    n_verifications: int = 0
    n_searches: int = 0
    max_strength: float = 0.0
    avg_strength: float = 0.0
    # Cluster counts (filled from a projection snapshot when available).
    n_clusters_with_support_div_ge_2: int = 0
    n_surviving: int = 0
    n_surviving_unverified: int = 0
    iteration: int = 0
    elapsed_s: float = 0.0

    @classmethod
    def from_store(cls, store: SignalStore, iteration: int = 0,
                   elapsed_s: float = 0.0) -> "FieldState":
        stats = store.stats()
        by_type = stats.get("by_type", {})
        # Cluster gate: count INITIALs with >= 2 SUPPORT children.
        n_div_ge_2 = len(store.signals_with_many_children_of_type(
            INITIAL, SUPPORT, 2
        ))
        return cls(
            n_initials=by_type.get(INITIAL, 0),
            n_supports=by_type.get(SUPPORT, 0),
            n_critique_pos=by_type.get(CRITIQUE_POSITIVE, 0),
            n_critique_neg=by_type.get(CRITIQUE_NEGATIVE, 0),
            n_objections=by_type.get(OBJECTION, 0),
            n_verifications=by_type.get(VERIFICATION, 0),
            n_searches=by_type.get(SEARCH, 0),
            max_strength=float(stats.get("max_strength", 0.0) or 0.0),
            avg_strength=float(stats.get("avg_strength", 0.0) or 0.0),
            n_clusters_with_support_div_ge_2=n_div_ge_2,
            iteration=iteration,
            elapsed_s=elapsed_s,
        )


# ---------------------------------------------------------------------------
# Parsed deposit spec returned by parse()
# ---------------------------------------------------------------------------

@dataclass
class ParsedDeposit:
    signal_type: str
    content: str
    strength: float
    parent_id_override: Optional[str] = None
    metadata: Optional[dict] = None


# ---------------------------------------------------------------------------
# Action preconditions
# ---------------------------------------------------------------------------

ACTION_PRECONDITIONS: dict[str, Callable[[FieldState], bool]] = {
    SCOUT:    lambda f: True,
    DEVELOP:  lambda f: f.n_initials >= 1,
    CHAIN:    lambda f: f.n_supports >= 2,
    # CRITIQUE / OBJECT used to require max_strength >= 0.5 / 0.6. That
    # gated critics out of exactly the regime where they're most needed:
    # weak fields with no signal above 0.5 after decay. Removed in favour
    # of pure structural preconditions — the field-state cluster gate
    # (signals_with_many_children) keeps OBJECT from firing on lone INITIALs.
    CRITIQUE: lambda f: f.n_initials >= 2,
    OBJECT:   lambda f: f.n_initials >= 3,
    # VALIDATE used to require an INITIAL with >= 2 SUPPORT children. When
    # DEVELOP spread thin (24 workers, support scattered), no INITIAL ever
    # crossed 2, the precondition never passed, and the validator sat out
    # entire runs (10-run delta_sweep_small batch: VALIDATE share 0.0,
    # avg_verification_score exactly 0.0). Verification eligibility must not
    # depend on support accumulation — a bare INITIAL is verifiable, and
    # early verification is what steers development toward grounded claims.
    VALIDATE: lambda f: f.n_initials >= 1,
    # REFINE previously gated on n_surviving_unverified, which FieldState.
    # from_store never populates — that path was always 0, so REFINE never
    # fired. Pivot to: there's a cluster worth polishing (i.e., one with
    # >= 2 SUPPORT children), AND we have at least one INITIAL to refine
    # on top of. The action's prompt steers the LLM toward sharpening a
    # specific testable element of the claim.
    REFINE:   lambda f: f.n_clusters_with_support_div_ge_2 >= 1,
}


# ---------------------------------------------------------------------------
# Shared parsing helpers
# ---------------------------------------------------------------------------

_TYPE_LINE_RE = _re.compile(r"^\s*TYPE\s*[:=]\s*(\w+)",
                            _re.IGNORECASE | _re.MULTILINE)
_PARENT_LINE_RE = _re.compile(r"^\s*PARENT\s*[:=]\s*([A-Za-z0-9_\-]+)",
                              _re.IGNORECASE | _re.MULTILINE)


def extract_json_object(text: str) -> Optional[dict]:
    """First balanced {...} block in `text` that parses as a JSON object.

    Replaces the old flat `re.search(r"\\{[^{}]*\\}")`, which could not match
    nested objects: a brace inside the reasoning string, or a reasoning-model
    preamble containing {...}, made it fail or capture garbage — producing
    the silent score=0.5 defaults behind validator_raw.log. This scanner is
    string-aware (quotes / escapes), walks brace depth, and returns the first
    balanced candidate that json.loads to a dict. Returns None when no
    parseable object exists (callers then use their text fallbacks).
    """
    s = text or ""
    i = s.find("{")
    while i != -1:
        depth = 0
        in_str = False
        esc = False
        for j in range(i, len(s)):
            ch = s[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(s[i:j + 1])
                        if isinstance(obj, dict):
                            return obj
                    except ValueError:
                        pass
                    break  # malformed or non-dict candidate; try next '{'
        i = s.find("{", i + 1)
    return None
_SCORE_RE = _re.compile(r"score\s*[:=]\s*([+-]?(?:\d+\.?\d*|\.\d+))",
                        _re.IGNORECASE)
# Drop these lines wholesale — they're internal protocol or pure metadata.
_DROP_LINE_RE = _re.compile(
    r"^\s*(?:TYPE|PARENT|SCORE|CONFIDENCE)\s*[:=].*$",
    _re.IGNORECASE | _re.MULTILINE,
)

# Strip only the LABEL prefix for these — the content that follows on the
# same line is the actual substance ("CRITIQUE: claim is well-supported"
# becomes "claim is well-supported"). Without this, the substance gets
# eaten by either the full-line drop or the junk filter's template-leak
# pattern at filters.py.
_LABEL_PREFIX_RE = _re.compile(
    r"^\s*(?:CRITIQUE|OBJECTION|DEVELOPMENT|REFINEMENT|CLAIM|OBSERVATION|"
    r"ANSWER|RESPONSE|REASONING|NOTE)\s*[:=]\s*",
    _re.IGNORECASE | _re.MULTILINE,
)

_DYNAMIC_TYPES = {INITIAL, SUPPORT, CRITIQUE_POSITIVE, CRITIQUE_NEGATIVE,
                  OBJECTION}


def _strip_meta_lines(text: str) -> str:
    """Remove protocol lines AND strip prompt-scaffold label prefixes.

    Two distinct operations:
      - TYPE / PARENT / SCORE / CONFIDENCE: full line dropped (metadata only).
      - CRITIQUE / OBJECTION / DEVELOPMENT / ...: LABEL stripped, substance
        on the same line is preserved.

    Why this matters: each action prompt instructs the LLM to format its
    response as "CRITIQUE: <text>\\nSCORE: <num>". The model complies, then
    the junk filter's _TEMPLATE_LEAK_PATTERNS matches the echoed label as
    template leakage and rejects the deposit — that was CRITIQUE share = 0%
    across every continuous-pool run.
    """
    if not text:
        return text
    cleaned = _DROP_LINE_RE.sub("", text)
    cleaned = _LABEL_PREFIX_RE.sub("", cleaned)
    cleaned = _re.sub(r"\n{2,}", "\n\n", cleaned)
    return cleaned.strip()


def _parse_type(text: str) -> Optional[str]:
    m = _TYPE_LINE_RE.search(text or "")
    if not m:
        return None
    cand = m.group(1).upper()
    return cand if cand in _DYNAMIC_TYPES else None


def _parse_parent(text: str, store: SignalStore) -> Optional[str]:
    m = _PARENT_LINE_RE.search(text or "")
    if not m:
        return None
    cand = m.group(1)
    if cand.lower() == "none":
        return "__NONE__"
    if store.get(cand) is not None:
        return cand
    return None


def _type_parent_instruction() -> str:
    # Position-neutral wording: this block now sits in the STATIC prefix of
    # every action prompt (before the dynamic signal content) so vLLM prefix
    # caching can reuse it — don't reference "above"/"below".
    return (
        "Before your response, on two separate lines, emit:\n"
        "  TYPE: <SUPPORT|INITIAL|OBJECTION|CRITIQUE_NEGATIVE|CRITIQUE_POSITIVE>\n"
        "  PARENT: <the bracketed signal_id shown in this prompt|none>\n"
        "Use 'none' as PARENT only if seeding a fresh INITIAL.\n"
    )


# ---------------------------------------------------------------------------
# SCOUT
# ---------------------------------------------------------------------------

def scout_prompt(task_prompt: str, retrieved_chunks: list,
                 prior_own_content: Optional[str] = None) -> str:
    from core.config import SCOUT_CLAIMS_PER_CALL
    k = SCOUT_CLAIMS_PER_CALL
    if retrieved_chunks:
        blocks = []
        for c in retrieved_chunks[:5]:
            tag = (getattr(c, "source_tag", "") or "")[:160]
            body = (getattr(c, "text", "") or "")[:800]
            blocks.append(f"[{tag}]\n{body}")
        evidence_text = "\n\n".join(blocks)
        evidence_intro = ("You issued an agentic search and received the "
                          "following evidence.")
    else:
        evidence_text = "(no evidence retrieved — fall back to first principles)"
        evidence_intro = ("No search results available. Reason from "
                          "general knowledge of the task domain.")
    reseed = ""
    if prior_own_content:
        excerpt = prior_own_content.replace("\n", " ").strip()
        reseed = (f'\nYou previously contributed: "{excerpt}"\n'
                  f"Every claim below must be genuinely different from that.\n")
    if k <= 1:
        ask = (
            f"Produce ONE concise initial claim or observation grounded in "
            f"the evidence shown in this prompt. One or two sentences only."
        )
        cue = "CLAIM:"
    else:
        ask = (
            f"Produce {k} DISTINCT initial claims grounded in the evidence "
            f"shown in this prompt, numbered 1. to {k}. Each claim: one or "
            f"two sentences. Each must "
            f"take a genuinely different angle — e.g. a specific mechanism, a "
            f"counterexample or limitation, a cost or trade-off, an affected "
            f"group, a quantitative detail, a second-order consequence. Do "
            f"NOT write {k} paraphrases of the same obvious point; the most "
            f"obvious claim may appear at most once."
        )
        cue = "CLAIMS:\n1."
    # Static-first ordering (prefix-cache): TASK + instruction are identical
    # across every SCOUT call of a run, so vLLM reuses their KV prefix; the
    # per-iteration evidence/reseed follow; the tiny generation cue is last.
    return (
        f"TASK: {task_prompt}\n\n"
        f"{ask}\n\n"
        f"{evidence_intro}\n\n"
        f"---EVIDENCE---\n{evidence_text}\n---END EVIDENCE---\n"
        f"{reseed}\n"
        f"{cue}"
    )


def scout_parse(raw: str) -> ParsedDeposit:
    content = _strip_meta_lines(raw or "")
    return ParsedDeposit(
        signal_type=INITIAL, content=content, strength=0.55,
    )


# Numbered-claim splitter for multi-claim scout responses. Accepts "1.", "1)",
# "2 ." etc. at line start. The prompt pre-seeds "1." so the first claim
# usually arrives without its number — handled by treating leading text
# before the first marker as claim 1.
_CLAIM_MARKER_RE = _re.compile(r"^\s*(\d{1,2})[.)]\s+", _re.MULTILINE)


def split_scout_claims(content: str) -> list[str]:
    """Split a multi-claim scout response into individual claim candidates.

    Returns [content] unchanged when no numbered structure is found, so
    single-claim responses (and SCOUT_CLAIMS_PER_CALL=1 configs) flow
    through identically to the old path. Junk candidates are filtered.
    """
    from core.filters import is_junk_output
    text = (content or "").strip()
    if not text:
        return []
    pieces: list[str] = []
    matches = list(_CLAIM_MARKER_RE.finditer(text))
    if not matches:
        return [text]
    # Text before the first marker is claim 1 (prompt pre-seeds "1.").
    head = text[: matches[0].start()].strip()
    if head:
        pieces.append(head)
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end(): end].strip()
        if body:
            pieces.append(body)
    # Filter junk and trivially-short fragments; keep order (model's own
    # confidence ordering breaks novelty-score ties at the caller).
    cleaned = [
        p.replace("\n", " ").strip()
        for p in pieces
        if len(p.split()) >= 4 and not is_junk_output(p)
    ]
    return cleaned if cleaned else [text]


# Paraphrase-support gate. A SUPPORT that merely restates its parent inflates
# support_diversity without adding information — the root cause of "claims,
# not evidence" fields. Reject when string similarity to the parent is high;
# annotate whether the support carries a new concrete particular (number or
# proper-noun-ish token absent from the parent) for telemetry and strength.
_SUPPORT_PARAPHRASE_RATIO = 0.72
_PARTICULAR_TOKEN_RE = _re.compile(r"\b(?:\d[\d,.%]*|[A-Z][a-z]{2,})\b")


def _new_particulars(content: str, parent_content: str) -> list[str]:
    """Concrete particulars (numbers, capitalized tokens) in `content` that
    are absent from the parent. Sentence-leading capitals are noisy but
    acceptable — absence from the parent is the real filter."""
    parent_l = parent_content.lower()
    out = []
    for tok in _PARTICULAR_TOKEN_RE.findall(content):
        if tok.lower() not in parent_l:
            out.append(tok)
    return out


def support_adds_information(content: str, parent_content: str) -> bool:
    """True when a SUPPORT/CHAIN deposit adds information over its parent.

    Rejects paraphrases: high string similarity to the parent AND no new
    concrete particular. A support with genuinely new tokens (a number, a
    named place/program/study) passes even at moderate similarity; a pure
    reword fails. Used by the worker pool to reject DEVELOP/CHAIN deposits
    that would inflate support_diversity without adding evidence.
    """
    from difflib import SequenceMatcher
    if not content or not parent_content:
        return True
    ratio = SequenceMatcher(
        None, content.lower(), parent_content.lower(),
    ).ratio()
    if ratio < _SUPPORT_PARAPHRASE_RATIO:
        return True
    return bool(_new_particulars(content, parent_content))


# Number-grounding gate (STORM-style source attachment, enforced at deposit).
# The particulars push made workers produce specific figures; real runs showed
# plausible-but-fabricated statistics ('$54M Oslo', '90K tons Copenhagen')
# that no retrieved chunk contained. A figure is GROUNDED only if its digits
# appear in evidence the worker was actually shown (retrieved chunks, the
# parent signal, or the task prompt). Numbers with >= 2 digits are checked;
# single digits ('step 2', 'version 3') are structural noise, not facts.
_NUMBER_TOKEN_RE = _re.compile(r"\d[\d,\.]*")


def _normalize_number(tok: str) -> str:
    return tok.replace(",", "").rstrip(".")


def ungrounded_numbers(content: str, sources: list[str]) -> list[str]:
    """Return numeric tokens in `content` (>= 2 digits) whose normalized
    form does not appear in any source text. Empty list = all figures
    grounded (or no figures present)."""
    src_tokens = {
        _normalize_number(t)
        for s in sources if s
        for t in _NUMBER_TOKEN_RE.findall(s)
    }
    out = []
    for tok in _NUMBER_TOKEN_RE.findall(content or ""):
        norm = _normalize_number(tok)
        if sum(ch.isdigit() for ch in norm) < 2:
            continue
        if norm not in src_tokens:
            out.append(tok)
    return out


def select_novel_claim(candidates: list[str], store: SignalStore) -> str:
    """Pick the candidate claim LEAST similar to the recent INITIAL field.

    Similarity is store-side (embedding cosine, string-ratio fallback) and
    returns scalars only — no signal content reaches the scout's prompt, so
    the no-leak rule and the partition-diversity mechanic are intact. Ties
    (including the empty-store case where every score is 0.0) resolve to the
    earliest candidate, i.e. the model's own highest-confidence claim.
    """
    from core.config import SCOUT_NOVELTY_RECENT_N
    if not candidates:
        return ""
    if len(candidates) == 1:
        return candidates[0]
    scored = [
        (store.max_similarity_to_recent(c, INITIAL, SCOUT_NOVELTY_RECENT_N),
         i, c)
        for i, c in enumerate(candidates)
    ]
    scored.sort(key=lambda t: (t[0], t[1]))
    best_sim, _, best = scored[0]
    if best_sim > 0.0:
        print(f"[scout multiclaim] {len(candidates)} candidates; "
              f"selected least-similar (max_sim={best_sim:.2f})")
    return best


# ---------------------------------------------------------------------------
# Genome-aware atom context block (used by DEVELOP, CRITIQUE, OBJECT, REFINE)
# ---------------------------------------------------------------------------

def _atom_for_develop(genome) -> Optional[object]:
    """Return the atom most in need of development.

    Primary key: lowest verification_score (most under-supported).
    Tie-break: highest weight (load-bearing atoms benefit more from support).
    This ensures that when all atoms are equally unverified (e.g., early in
    the run before validators have run), we target the most important one.
    """
    if genome is None or not genome.atoms:
        return None
    return min(genome.atoms, key=lambda a: (a.verification_score, -a.weight))


def _atom_for_critique(genome) -> Optional[object]:
    """Return the atom with the highest weight (load-bearing claim to evaluate)."""
    if genome is None or not genome.atoms:
        return None
    return max(genome.atoms, key=lambda a: a.weight)


def _atom_for_object(genome) -> Optional[object]:
    """Return the atom most vulnerable to attack: load-bearing (high weight) but weakly verified (low score).

    Metric: verification_score / max(weight, 0.01).
    A low ratio means the atom carries structural importance but lacks verification —
    the ideal target for an adversarial objection.
    min(score/weight) finds load-bearing atoms whose verification is most disproportionately low.
    """
    if genome is None or not genome.atoms:
        return None
    return min(genome.atoms, key=lambda a: a.verification_score / max(a.weight, 0.01))


def _format_atom_block(atom) -> str:
    """Render one AtomFact as a no-leak prompt fragment (text + scalars only, no IDs in prose)."""
    lines = [f"  ATOM [{atom.atom_id}]: {atom.text}"]
    lines.append(f"    verification_score={atom.verification_score:.2f}  "
                 f"weight={atom.weight:.2f}  source={atom.source_tag}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# DEVELOP — produce SUPPORT for an INITIAL
# ---------------------------------------------------------------------------

def develop_prompt(task_prompt: str, target: Signal,
                   dissent: Optional[Signal] = None,
                   retrieval: Optional[list] = None,
                   retrieval_query: str = "",
                   genome=None) -> str:
    dissent_block = ""
    if dissent is not None:
        dissent_block = (
            f"\nA challenge has been raised against this signal "
            f"(strength={dissent.strength:.2f}). Your SUPPORT should "
            f"anticipate and address it:\n"
            f"  [{dissent.id}]: {dissent.content}\n"
        )
    retrieval_block = ""
    if retrieval:
        blocks = []
        for c in retrieval[:3]:
            tag = (getattr(c, "source_tag", "") or "")[:120]
            body = (getattr(c, "text", "") or "")[:400]
            blocks.append(f"[{tag}]\n{body}")
        retrieval_block = (
            f"\nCluster has thin support — you queried {retrieval_query!r}:\n\n"
            + "\n\n".join(blocks) + "\n\n"
        )
    # Genome-aware atom targeting: show the weakest atom and ask to develop it.
    # Falls back to standard single-signal prompt when no genome is available.
    atom = _atom_for_develop(genome)
    if atom is not None:
        atom_block = (
            f"\nThe cluster this signal belongs to has been decomposed into "
            f"atomic facts. The LEAST verified atom is:\n"
            f"{_format_atom_block(atom)}\n"
            f"Produce ONE SUPPORT that provides evidence or elaboration "
            f"specifically for THAT atom. Your deposit metadata will record "
            f"targets_atom={atom.atom_id!r}.\n"
        )
        develop_instruction = (
            f"Target the weakest atom shown in this prompt. One or two "
            f"sentences of "
            f"concrete evidence that directly supports it. Your sentence "
            f"MUST contain at least one concrete particular — a number, a "
            f"named city/program/study, or a date — that is NOT already in "
            f"the signal. Restating the signal in different words is "
            f"worthless and will be rejected."
        )
    else:
        atom_block = ""
        develop_instruction = (
            f"Produce ONE concise development of the signal shown in this "
            f"prompt: supporting "
            f"evidence, an extension, or a refinement that anticipates the "
            f"challenge (if one is shown). One or two sentences. Your "
            f"development MUST add information the signal does not already "
            f"contain — ideally a concrete particular (a number, a named "
            f"city/program/study, a date, a mechanism). Restating the "
            f"signal in different words is worthless and will be rejected."
        )
    # Static-first ordering (prefix-cache): instruction block before the
    # per-iteration signal/dissent/retrieval content.
    return (
        f"TASK: {task_prompt}\n\n"
        f"You observe one signal in the shared store. You do not know who "
        f"deposited it.\n\n"
        f"{develop_instruction}\n\n"
        f"{_type_parent_instruction()}\n"
        f"---SIGNAL [{target.id}]---\n{target.content}\n---END SIGNAL---\n"
        f"{dissent_block}{retrieval_block}{atom_block}\n"
        f"DEVELOPMENT:"
    )


def develop_parse(raw: str, genome=None) -> ParsedDeposit:
    content = _strip_meta_lines(raw or "")
    meta: dict = {}
    if genome is not None:
        atom = _atom_for_develop(genome)
        if atom is not None:
            meta["targets_atom"] = atom.atom_id
    return ParsedDeposit(
        signal_type=SUPPORT, content=content, strength=0.6,
        metadata=meta if meta else None,
    )


# ---------------------------------------------------------------------------
# CHAIN — produce SUPPORT chained off another SUPPORT (deepens lineage)
# ---------------------------------------------------------------------------

def chain_prompt(task_prompt: str, target_support: Signal) -> str:
    # Static-first ordering (prefix-cache): instruction before the signal.
    return (
        f"TASK: {task_prompt}\n\n"
        f"You see a SUPPORT signal already developing a claim. Build the "
        f"next step in the chain: a sub-claim, a downstream consequence, "
        f"or a finer-grained piece of evidence that extends THIS support "
        f"specifically — not the original INITIAL.\n\n"
        f"One or two sentences. Do not restate the original — your step "
        f"must add information it does not contain, ideally a concrete "
        f"particular (a number, a named city/program/study, a date, a "
        f"mechanism). Paraphrase will be rejected.\n\n"
        f"{_type_parent_instruction()}\n"
        f"---SUPPORT [{target_support.id}]---\n{target_support.content}\n"
        f"---END SUPPORT---\n\n"
        f"CHAINED DEVELOPMENT:"
    )


def chain_parse(raw: str) -> ParsedDeposit:
    content = _strip_meta_lines(raw or "")
    return ParsedDeposit(
        signal_type=SUPPORT, content=content, strength=0.6,
    )


# ---------------------------------------------------------------------------
# CRITIQUE — evaluate an INITIAL, valence determines POS vs NEG
# ---------------------------------------------------------------------------

def critique_prompt(task_prompt: str, target: Signal,
                    genome=None) -> str:
    atom = _atom_for_critique(genome)
    if atom is not None:
        atom_block = (
            f"\nThe cluster's most load-bearing atom (highest weight) is:\n"
            f"{_format_atom_block(atom)}\n\n"
            f"Focus your evaluation on WHETHER THAT ATOM STANDS. Does the "
            f"artifact provide adequate grounding for it?\n"
        )
    else:
        atom_block = ""
    # Static-first ordering (prefix-cache): instruction + format before the
    # per-iteration artifact/atom content.
    return (
        f"TASK: {task_prompt}\n\n"
        f"Evaluate the deposited artifact shown in this prompt on its "
        f"merits.\n\n"
        f"Write a brief critique (one or two sentences) and assign a "
        f"quality score in [0, 1] reflecting how well the artifact stands "
        f"as a claim. Score >= 0.5 means well-formed; < 0.5 means it has "
        f"significant weaknesses. Format:\n\n"
        f"CRITIQUE: <your critique>\n"
        f"SCORE: <number between 0 and 1>\n\n"
        f"{_type_parent_instruction()}\n"
        f"---ARTIFACT [{target.id}]---\n{target.content}\n---END ARTIFACT---\n"
        f"{atom_block}\n"
        f"Write the CRITIQUE and SCORE lines now."
    )


def critique_parse(raw: str) -> ParsedDeposit:
    text = (raw or "").strip()
    score = 0.5
    m = _SCORE_RE.search(text)
    if m:
        try:
            score = max(0.0, min(1.0, float(m.group(1))))
        except ValueError:
            pass
    sig_type = CRITIQUE_POSITIVE if score >= 0.5 else CRITIQUE_NEGATIVE
    content = _strip_meta_lines(text)
    return ParsedDeposit(
        signal_type=sig_type, content=content, strength=score,
        metadata={"score": round(score, 4)},
    )


# ---------------------------------------------------------------------------
# OBJECT — adversarial challenge to a cluster (Hater equivalent)
# ---------------------------------------------------------------------------

def object_prompt(task_prompt: str, representatives: list[Signal],
                  task_type: Optional[str] = None,
                  genome=None) -> str:
    rep_lines = "\n".join(
        f"  - [{s.id}] strength={s.strength:.2f}: {s.content}"
        for s in representatives
    )
    if task_type == "creative":
        challenge_instruction = (
            "Find a craft-level weakness that applies to the CLUSTER AS A "
            "WHOLE — not any individual signal. Is the cluster anchoring on "
            "an obvious literary reference or cliché without earning it "
            "through the actual language? Do any of the representative lines "
            "have grammatical errors or broken rhythm? Is this consensus "
            "forming because the theme is genuinely resonant, or because it "
            "appeared first and went unchallenged? "
            "One or two sentences naming the specific weakness."
        )
    else:
        challenge_instruction = (
            "Find a structural weakness that applies to the CLUSTER AS A "
            "WHOLE — not to any individual signal. What shared assumption do "
            "these signals make that might be wrong? What kind of evidence "
            "would they all collectively miss? One or two sentences."
        )
    atom = _atom_for_object(genome)
    if atom is not None:
        atom_block = (
            f"\nThe cluster's most VULNERABLE atom (low verification × high weight) is:\n"
            f"{_format_atom_block(atom)}\n\n"
            f"Your objection should attack WHETHER THAT ATOM IS ACTUALLY SUPPORTED — "
            f"what would disprove it, or what assumption does it silently rely on?\n"
        )
    else:
        atom_block = ""
    # Static-first ordering (prefix-cache): framing + challenge instruction
    # before the per-iteration representatives/atom content.
    return (
        f"TASK: {task_prompt}\n\n"
        f"You see a consensus cluster forming in the shared signal store, "
        f"represented by the signal(s) shown in this prompt.\n\n"
        f"{challenge_instruction}\n\n"
        f"{_type_parent_instruction()}\n"
        f"---CLUSTER REPRESENTATIVES---\n"
        f"{rep_lines}\n"
        f"---END REPRESENTATIVES---\n"
        f"{atom_block}\n"
        f"OBJECTION:"
    )


def object_parse(raw: str) -> ParsedDeposit:
    content = _strip_meta_lines(raw or "")
    return ParsedDeposit(
        signal_type=OBJECTION, content=content, strength=0.6,
    )


# ---------------------------------------------------------------------------
# VALIDATE — external grounding via search_tool
# ---------------------------------------------------------------------------

# Task profiles that should use the non-factual ("engages/quality") prompt.
# These match SURVIVAL_TASK_PROFILES.requires_verification=False in
# core/config.py; kept duplicated here so this module stays import-light.
_NON_FACTUAL_TASKS = {"debate", "analysis", "problem_solving", "creative"}


def validate_prompt(task_prompt: str, target: Signal,
                    external_snippet: str,
                    task_type: Optional[str] = None) -> str:
    """Build the validator prompt.

    Factual tasks (default, coding) ask "does the source support the claim?".
    Non-factual tasks (debate / analysis / problem_solving / creative) ask
    "does the source engage substantively with the topic?" because no web
    source will *confirm* an interpretive claim — sources present arguments,
    not corroboration, and asking for corroboration produces
    `supports: false` with high confidence and a verification score of 0.
    """
    # Static-first ordering (prefix-cache): instruction + JSON schema before
    # the per-iteration claim/snippet; one-line reply cue last.
    if task_type in _NON_FACTUAL_TASKS:
        return (
            f"TASK: {task_prompt}\n\n"
            f"Assess whether the external snippet substantively engages with "
            f"the topic of the claim shown in this prompt. You are NOT asked "
            f"to confirm or deny "
            f"the claim — only to judge whether the snippet brings relevant "
            f"evidence, argument, or scholarly context that bears on it.\n\n"
            f"Reply with EXACTLY this JSON object (no other text):\n"
            f'{{"engages": true|false, "quality": <number in [0,1]>, '
            f'"reasoning": "<one short sentence>"}}\n\n'
            f"`engages` is true if the snippet meaningfully addresses the "
            f"claim's topic (even from a different angle or with different "
            f"conclusions). `quality` rates the snippet's substantive depth "
            f"and source credibility (0 = thin/irrelevant, 1 = rigorous "
            f"and on-topic).\n\n"
            f"---CLAIM [{target.id}]---\n{target.content}\n---END CLAIM---\n\n"
            f"---EXTERNAL SNIPPET---\n{external_snippet}\n---END SNIPPET---\n\n"
            f"Reply now with ONLY the JSON object."
        )
    return (
        f"TASK: {task_prompt}\n\n"
        f"Verify the claim shown in this prompt against the external "
        f"snippet.\n\n"
        f"Reply with EXACTLY this JSON object (no other text):\n"
        f'{{"supports": true|false, "confidence": <number in [0,1]>, '
        f'"reasoning": "<one short sentence>"}}\n\n'
        f"`supports` is true if the snippet plausibly backs the claim, "
        f"false if it contradicts or fails to address it.\n\n"
        f"---CLAIM [{target.id}]---\n{target.content}\n---END CLAIM---\n\n"
        f"---EXTERNAL SNIPPET---\n{external_snippet}\n---END SNIPPET---\n\n"
        f"Reply now with ONLY the JSON object."
    )


def validate_parse(raw: str, task_type: Optional[str] = None) -> ParsedDeposit:
    """Parse the validator's JSON. Schema is task-aware.

    Factual: {supports: bool, confidence: 0..1}
      → score = confidence if supports else max(0, 1 - confidence)
    Non-factual: {engages: bool, quality: 0..1}
      → score = quality if engages else 0.2
      A relevant-but-thin snippet still scores 0.2 (a real signal — the
      topic was at least addressed), so non-factual clusters can
      accumulate verification > 0 instead of getting pinned to 0.0.
    """
    text = (raw or "").strip()
    score = 0.5
    note = text
    parsed = extract_json_object(text)
    if parsed is not None:
        try:
            if task_type in _NON_FACTUAL_TASKS:
                # Non-factual: prefer engages/quality if the model emitted
                # them. Otherwise ADAPT a supports/confidence response
                # (which the model may default to from its training
                # distribution) instead of dropping into the factual branch
                # where supports=false + confidence=0.95 produces score=0.05.
                if "engages" in parsed or "quality" in parsed:
                    engages = bool(parsed.get("engages", False))
                    qual = float(parsed.get("quality", 0.5))
                    qual = max(0.0, min(1.0, qual))
                    # Engaged → quality; not-engaged → 0.2 floor.
                    score = qual if engages else 0.2
                elif "supports" in parsed or "confidence" in parsed:
                    # Schema-mismatch fallback: treat `supports` as `engages`
                    # proxy, treat `confidence` as `quality`. supports=False
                    # in this context means "doesn't confirm" not
                    # "irrelevant" — so the floor is 0.3 (a confident "the
                    # source doesn't support, but the topic is addressed").
                    supports = bool(parsed.get("supports", True))
                    conf = float(parsed.get("confidence", 0.5))
                    conf = max(0.0, min(1.0, conf))
                    score = conf if supports else max(0.3, conf * 0.6)
                else:
                    # Unknown JSON shape — fall back to mid-range and let
                    # downstream dynamics decide.
                    score = 0.5
                note = str(parsed.get("reasoning",
                                      parsed.get("note", ""))).strip() or text
            else:
                # Factual branch (coding): the old supports/confidence
                # semantics — a confident "doesn't support" SHOULD score
                # near zero because that's literal negative evidence.
                supports = bool(parsed.get("supports", False))
                conf = float(parsed.get("confidence", 0.5))
                conf = max(0.0, min(1.0, conf))
                score = conf if supports else max(0.0, 1.0 - conf)
                note = str(parsed.get("reasoning",
                                      parsed.get("note", ""))).strip() or text
        except (ValueError, TypeError):
            pass
    else:
        m = _SCORE_RE.search(text)
        if m:
            try:
                score = max(0.0, min(1.0, float(m.group(1))))
            except ValueError:
                pass
        # On non-factual tasks, an unparseable validator response is more
        # likely "the model went off-format" than "the model rejected the
        # claim". Floor the score so we don't pin verification at 0.
        if task_type in _NON_FACTUAL_TASKS:
            score = max(score, 0.3)
    return ParsedDeposit(
        signal_type=VERIFICATION, content=note, strength=score,
        metadata={"score": round(score, 4)},
    )


# ---------------------------------------------------------------------------
# REFINE — patch a surviving-but-unverified cluster
# ---------------------------------------------------------------------------

def refine_prompt(task_prompt: str, target: Signal,
                  dissent: Optional[Signal] = None,
                  genome=None) -> str:
    """Build a REFINE prompt.

    Two modes:
      - With dissent: rebuttal mode. The renderer must read the supplied
        objection and produce a SUPPORT that DIRECTLY addresses the named
        weakness. Closes the loop that previously made objections cosmetic
        (visits=0) — REFINE is now the deliberative step that consumes
        OBJECTION/CRITIQUE_NEGATIVE signals.
      - Without dissent: legacy polishing mode (sharpen the claim toward
        verifiability).
    """
    # Genome-aware: identify the atom under attack (if dissent carries targets_atom).
    # Falls back to standard prompt when no genome or no targeted atom.
    targeted_atom = None
    if genome is not None and dissent is not None:
        targets_atom_id = (dissent.metadata or {}).get("targets_atom")
        if targets_atom_id:
            targeted_atom = next(
                (a for a in genome.atoms if a.atom_id == targets_atom_id), None
            )
    if targeted_atom is None and genome is not None and dissent is not None:
        # No explicit target: default to the weakest atom (highest priority to strengthen)
        targeted_atom = _atom_for_develop(genome)

    if dissent is not None:
        atom_block = ""
        if targeted_atom is not None:
            atom_block = (
                f"\nThe challenge targets this atom:\n"
                f"{_format_atom_block(targeted_atom)}\n"
                f"Rebut by strengthening THAT ATOM SPECIFICALLY — new evidence, "
                f"a scope qualification, or a counter-example.\n"
            )
        # Static-first ordering (prefix-cache): instruction before the
        # per-iteration claim/challenge/atom content.
        return (
            f"TASK: {task_prompt}\n\n"
            f"The claim shown in this prompt has been challenged by another "
            f"agent. Produce ONE "
            f"SUPPORT that directly rebuts the named weakness — acknowledging "
            f"the challenge, then giving evidence, narrowing the scope, or "
            f"qualifying the original claim so the challenge no longer "
            f"applies. Do NOT ignore the challenge or restate the original "
            f"claim verbatim.\n\n"
            f"One or two sentences. Your SUPPORT should be readable as a "
            f"reply to the challenge: it must engage what the challenge "
            f"actually says.\n\n"
            f"{_type_parent_instruction()}\n"
            f"---CLAIM [{target.id}]---\n{target.content}\n---END CLAIM---\n\n"
            f"---CHALLENGE [{dissent.id}] (strength={dissent.strength:.2f})---\n"
            f"{dissent.content}\n---END CHALLENGE---\n"
            f"{atom_block}\n"
            f"REBUTTAL:"
        )
    return (
        f"TASK: {task_prompt}\n\n"
        f"The claim shown in this prompt survived structural filtering but "
        f"lacks external "
        f"verification. Produce ONE supporting refinement that makes the "
        f"claim more concrete, specific, or testable — narrowing where "
        f"verification could target. One or two sentences.\n\n"
        f"{_type_parent_instruction()}\n"
        f"---CLAIM [{target.id}]---\n{target.content}\n---END CLAIM---\n\n"
        f"REFINEMENT:"
    )


def refine_parse(raw: str, genome=None) -> ParsedDeposit:
    content = _strip_meta_lines(raw or "")
    meta: dict = {}
    if genome is not None:
        atom = _atom_for_develop(genome)   # weakest atom = most needing rebuttal
        if atom is not None:
            meta["targets_atom"] = atom.atom_id
    return ParsedDeposit(
        signal_type=SUPPORT, content=content, strength=0.55,
        metadata=meta if meta else None,
    )


# ---------------------------------------------------------------------------
# Action registry (template + parse + output type)
# ---------------------------------------------------------------------------

@dataclass
class ActionSpec:
    name: str
    output_type: str
    prompt_builder: Callable
    parse: Callable[[str], ParsedDeposit]
    precondition: Callable[[FieldState], bool]
    max_tokens: int
    temperature: float


from core.config import MAX_TOKENS_SCOUT as _MAX_TOKENS_SCOUT

ACTION_REGISTRY: dict[str, ActionSpec] = {
    SCOUT: ActionSpec(
        # max_tokens from config: the budget covers a K-claim portfolio
        # (multi-claim sampling), not a single claim.
        SCOUT, INITIAL, scout_prompt, scout_parse,
        ACTION_PRECONDITIONS[SCOUT], max_tokens=_MAX_TOKENS_SCOUT,
        temperature=0.9,
    ),
    DEVELOP: ActionSpec(
        DEVELOP, SUPPORT, develop_prompt, develop_parse,
        ACTION_PRECONDITIONS[DEVELOP], max_tokens=300, temperature=0.7,
    ),
    CHAIN: ActionSpec(
        CHAIN, SUPPORT, chain_prompt, chain_parse,
        ACTION_PRECONDITIONS[CHAIN], max_tokens=300, temperature=0.7,
    ),
    CRITIQUE: ActionSpec(
        CRITIQUE, CRITIQUE_NEGATIVE, critique_prompt, critique_parse,
        ACTION_PRECONDITIONS[CRITIQUE], max_tokens=200, temperature=0.55,
    ),
    OBJECT: ActionSpec(
        OBJECT, OBJECTION, object_prompt, object_parse,
        ACTION_PRECONDITIONS[OBJECT], max_tokens=300, temperature=0.85,
    ),
    VALIDATE: ActionSpec(
        VALIDATE, VERIFICATION, validate_prompt, validate_parse,
        ACTION_PRECONDITIONS[VALIDATE], max_tokens=150, temperature=0.4,
    ),
    REFINE: ActionSpec(
        REFINE, SUPPORT, refine_prompt, refine_parse,
        ACTION_PRECONDITIONS[REFINE], max_tokens=200, temperature=0.6,
    ),
}
