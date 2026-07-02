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
from itertools import combinations
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
    """Unit-normalized embeddings via the shared SignalStore loader, or None.

    Previously instantiated a THIRD SentenceTransformer copy of the same
    all-MiniLM-L6-v2 weights (store + search_tool + here). The store's
    loader is now the one canonical embedder and provides the
    transformers-AutoModel fallback path.
    """
    try:
        from core.signal_store import _try_load_embedder
        model = _try_load_embedder()
        if model is None:
            return None
        import numpy as np
        out: list[list[float]] = []
        for t in texts:
            v = np.asarray(model.encode(t), dtype="float32")
            n = float((v ** 2).sum() ** 0.5)
            out.append((v / n).tolist() if n > 0 else v.tolist())
        return out
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

# ---------------------------------------------------------------------------
# Cross-model decomposition (§2.5 — heterogeneous routing ablation)
# ---------------------------------------------------------------------------

def _normalize_vec(v):
    """Return a unit-normalized list of floats; [] if v is empty/None."""
    if not v:
        return []
    s = math.sqrt(sum(x * x for x in v))
    if s < 1e-9:
        return []
    return [x / s for x in v]


def _agent_centroid(texts: list[str], embedder) -> Optional[list[float]]:
    """Average unit-normalized embeddings of texts into one centroid vector.

    Returns None if no texts produced a usable embedding. Falls back to
    BoW when the embedder is unavailable.
    """
    texts = [t for t in (t.strip() for t in texts) if t]
    if not texts:
        return None

    if embedder is not None:
        vecs = []
        for t in texts:
            try:
                raw = embedder.encode(t)
            except Exception:
                continue
            # Convert to native Python list[float]; sentence-transformers returns
            # numpy.ndarray and downstream json.dumps() can't serialize numpy
            # types. .tolist() recursively converts numpy scalars to Python.
            if hasattr(raw, "tolist"):
                vec = raw.tolist()
            elif isinstance(raw, list):
                vec = [float(x) for x in raw]
            else:
                vec = [float(x) for x in list(raw)]
            unit = _normalize_vec(vec)
            if unit:
                vecs.append(unit)
        if vecs:
            n = len(vecs[0])
            centroid = [sum(v[i] for v in vecs) / len(vecs) for i in range(n)]
            return _normalize_vec(centroid)

    # BoW fallback (also used when embedder is None or every encode failed)
    bows = [_bow_vector(t) for t in texts]
    bows = [b for b in bows if b]
    if not bows:
        return None
    keys = set()
    for b in bows:
        keys.update(b.keys())
    keys = sorted(keys)
    centroid = [sum(b.get(k, 0.0) for b in bows) / len(bows) for k in keys]
    unit = _normalize_vec(centroid)
    if not unit:
        return None
    # Encode keys in the vector itself so different agents' BoWs can be
    # compared: prefix-tag each component with its key index by returning
    # a (keys, vec) — but the cross-agent compare path needs the SAME basis.
    # Simplest correct path: rebuild a shared key space when comparing.
    # We return a list keyed by the position in `keys`, and the comparison
    # helper recomputes a shared basis if needed. To keep things simple,
    # carry the key vocabulary as part of the centroid by tagging.
    return {"__bow__": dict(zip(keys, unit))}  # type: ignore[return-value]


def _cosine_centroids(a, b) -> Optional[float]:
    """Cosine similarity between two centroids returned by _agent_centroid.

    Handles both list-of-float (embedder) and BoW-dict (fallback) forms.
    Returns None on incompatible shapes.
    """
    if a is None or b is None:
        return None
    a_is_bow = isinstance(a, dict) and "__bow__" in a
    b_is_bow = isinstance(b, dict) and "__bow__" in b
    if a_is_bow != b_is_bow:
        return None
    if a_is_bow:
        return float(_cosine(a["__bow__"], b["__bow__"]))
    if len(a) != len(b):
        return None
    return float(max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b)))))


def output_diversity_by_model(records, embedder, model_assignment: dict) -> dict:
    """Decompose output diversity into within-model and between-model components.

    records: list[AgentContextRecord] — each record's .deposit_contents holds
             the actual text the agent deposited during this round.
    embedder: SignalStore._embedder or compatible (.encode(text) -> list[float]).
    model_assignment: {role: model_name} from the router's manifest, or
             {'all': model_name} for homogeneous runs.

    Returns:
      {
        'within_model': {model_name: avg_pairwise_cosine_distance},
        'between_model': avg_pairwise_cosine_distance_across_diff_model_agents,
        'delta': between_model - mean(within_model values),
      }

    delta > 0 is direct evidence model heterogeneity decorrelates outputs.
    delta <= 0 means heterogeneity isn't doing what we think.
    """
    # Per-agent (role, centroid), skipping any agent with no usable deposits
    homogeneous_mark = (len(model_assignment) == 1 and "all" in model_assignment)
    only_model = next(iter(model_assignment.values())) if homogeneous_mark else None

    def model_of(role: str) -> str:
        if homogeneous_mark:
            return only_model  # type: ignore[return-value]
        return model_assignment.get(role, "_unassigned_")

    agents = []
    for rec in records:
        texts = [t for t in (rec.deposit_contents or []) if t and t.strip()]
        if not texts:
            continue
        c = _agent_centroid(texts, embedder)
        if c is None:
            continue
        agents.append((model_of(rec.role), c))

    if len(agents) < 2:
        return {"within_model": {}, "between_model": None, "delta": None}

    # Buckets of pairwise cosine distances
    within_pairs: dict = {}
    between_dists: list[float] = []
    for (m_a, c_a), (m_b, c_b) in combinations(agents, 2):
        sim = _cosine_centroids(c_a, c_b)
        if sim is None:
            continue
        dist = 1.0 - sim
        if m_a == m_b:
            within_pairs.setdefault(m_a, []).append(dist)
        else:
            between_dists.append(dist)

    # float() wraps guarantee native Python floats for JSON serialization,
    # in case any upstream value is a numpy scalar that slipped through.
    within_model = {
        m: float(round(sum(ds) / len(ds), 4))
        for m, ds in within_pairs.items() if ds
    }

    # If every agent shares one model, there is no between-model signal.
    distinct_models = {m for m, _ in agents}
    if len(distinct_models) < 2:
        return {"within_model": within_model, "between_model": None, "delta": None}

    if not between_dists:
        return {"within_model": within_model, "between_model": None, "delta": None}

    between = float(sum(between_dists) / len(between_dists))
    mean_within = (
        float(sum(within_model.values()) / len(within_model))
        if within_model else 0.0
    )
    return {
        "within_model": within_model,
        "between_model": float(round(between, 4)),
        "delta": float(round(between - mean_within, 4)),
    }


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
