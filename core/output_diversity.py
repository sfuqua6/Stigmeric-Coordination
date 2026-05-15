"""Output diversity metrics over agent-produced text (§2 directive).

These metrics measure how different the OUTPUTS are — unlike the Jaccard
partition-overlap metric in core/diversity.py which measures input-set overlap.

Two metrics:
    centroid_cosine_distance(texts) — how far texts are from their centroid
    self_bleu(texts)               — inverse of n-gram self-similarity

Both degrade gracefully when fewer than 2 texts are provided.

# FUTURE-CLAUDE NOTE:
# centroid_cosine_distance uses sentence-transformers if available, otherwise
# falls back to a normalised bag-of-words TF vector. The bag-of-words fallback
# is fast but underestimates semantic diversity — it misses paraphrase collapse.
# If you add sentence-transformers as a hard dependency, remove the fallback
# path and update this note.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Optional


# ---------------------------------------------------------------------------
# Bag-of-words vector (fallback when no embedding model)
# ---------------------------------------------------------------------------

_STOP = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "in",
    "and", "or", "but", "for", "on", "at", "by", "with", "as", "that",
    "this", "it", "not", "be", "been", "can", "will", "we", "i", "you",
})


def _bow_vector(text: str) -> dict[str, float]:
    """Return a normalised TF bag-of-words vector."""
    tokens = re.findall(r"[a-z]+", text.lower())
    tokens = [t for t in tokens if t not in _STOP and len(t) > 2]
    if not tokens:
        return {}
    counts = Counter(tokens)
    norm = math.sqrt(sum(v * v for v in counts.values()))
    return {k: v / norm for k, v in counts.items()} if norm > 0 else {}


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    dot = sum(a.get(k, 0.0) * v for k, v in b.items())
    return max(-1.0, min(1.0, dot))


def _embed_bow(texts: list[str]) -> list[dict[str, float]]:
    return [_bow_vector(t) for t in texts]


def _embed_sbert(texts: list[str]) -> Optional[list[list[float]]]:
    """Return sentence-transformers embeddings, or None if unavailable."""
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return model.encode(texts, normalize_embeddings=True).tolist()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# centroid_cosine_distance
# ---------------------------------------------------------------------------

def centroid_cosine_distance(texts: list[str]) -> float:
    """Average cosine distance from each text embedding to the group centroid.

    Returns a value in [0, 1]:
      0 = all texts are identical (no output diversity)
      1 = maximally spread around the centroid

    Falls back to bag-of-words TF vectors if sentence-transformers unavailable.
    """
    texts = [t.strip() for t in texts if t.strip()]
    if len(texts) < 2:
        return 0.0

    sbert = _embed_sbert(texts)
    if sbert is not None:
        n = len(sbert[0])
        centroid = [sum(e[i] for e in sbert) / len(sbert) for i in range(n)]
        c_norm = math.sqrt(sum(v * v for v in centroid))
        if c_norm < 1e-9:
            return 0.0
        centroid = [v / c_norm for v in centroid]
        sims = [
            max(-1.0, min(1.0, sum(e[i] * centroid[i] for i in range(n))))
            for e in sbert
        ]
        avg_sim = sum(sims) / len(sims)
        return round(1.0 - avg_sim, 4)

    # Bag-of-words fallback
    vecs = _embed_bow(texts)
    non_empty = [v for v in vecs if v]
    if len(non_empty) < 2:
        return 0.0
    all_keys = set()
    for v in non_empty:
        all_keys.update(v.keys())
    centroid = {k: sum(v.get(k, 0.0) for v in non_empty) / len(non_empty) for k in all_keys}
    c_norm = math.sqrt(sum(v * v for v in centroid.values()))
    if c_norm < 1e-9:
        return 0.0
    centroid = {k: v / c_norm for k, v in centroid.items()}
    sims = [_cosine(v, centroid) for v in non_empty]
    avg_sim = sum(sims) / len(sims)
    return round(1.0 - avg_sim, 4)


# ---------------------------------------------------------------------------
# Self-BLEU
# ---------------------------------------------------------------------------

def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def _bleu_score(hypothesis: list[str], references: list[list[str]], max_n: int = 3) -> float:
    """Compute a simplified sentence-BLEU of hypothesis against references."""
    if not hypothesis:
        return 0.0
    log_score = 0.0
    weights = [1.0 / max_n] * max_n
    for n, w in enumerate(weights, 1):
        hyp_ngrams = _ngrams(hypothesis, n)
        if not hyp_ngrams:
            continue
        ref_counts: Counter = Counter()
        for ref in references:
            ref_counts |= _ngrams(ref, n)
        clipped = sum(min(c, ref_counts[ng]) for ng, c in hyp_ngrams.items())
        denom = sum(hyp_ngrams.values())
        precision = clipped / denom if denom > 0 else 0.0
        if precision > 0:
            log_score += w * math.log(precision)
    # Brevity penalty
    ref_len = sum(len(r) for r in references) / max(1, len(references))
    hyp_len = len(hypothesis)
    bp = 1.0 if hyp_len >= ref_len else math.exp(1.0 - ref_len / max(1, hyp_len))
    return bp * math.exp(log_score)


def self_bleu(texts: list[str]) -> float:
    """Self-BLEU: average BLEU of each text against all other texts.

    High Self-BLEU means texts are very similar → low output diversity.
    Returns a value in [0, 1]; lower is better (more diverse).
    """
    texts = [t.strip() for t in texts if t.strip()]
    if len(texts) < 2:
        return 0.0
    tokenised = [re.findall(r"[a-z]+", t.lower()) for t in texts]
    scores = []
    for i, hyp in enumerate(tokenised):
        refs = [tokenised[j] for j in range(len(tokenised)) if j != i]
        scores.append(_bleu_score(hyp, refs))
    return round(sum(scores) / len(scores), 4)


# ---------------------------------------------------------------------------
# Combined report
# ---------------------------------------------------------------------------

def format_output_diversity_report(texts: list[str], round_num: int) -> str:
    """Return a one-block summary of output diversity for a round."""
    texts = [t.strip() for t in texts if t.strip()]
    if not texts:
        return f"[round {round_num}] output diversity: no deposits to measure"
    ccd = centroid_cosine_distance(texts)
    sb = self_bleu(texts)
    n = len(texts)
    lines = [
        f"Output diversity (round {round_num}, n={n} deposits):",
        f"  centroid cosine dist = {ccd:.4f}  (0=identical, 1=maximally spread)",
        f"  self-BLEU            = {sb:.4f}  (0=diverse, 1=identical)",
    ]
    return "\n".join(lines)
