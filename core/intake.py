"""Corpus partitioning at intake.

The diversity engine. A retrieved corpus C is split into K chunks; each
chunk is assigned to exactly one Scout. Scouts never share their chunks.
This is the only place agents are conditioned on raw evidence; from this
point forward, all conditioning happens via signals in the store.

This module deliberately knows nothing about LLMs or signals — it only
knows how to slice text into chunks and assign them to scout indexes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .config import CHUNK_WORDS, CHUNK_OVERLAP, CHUNKS_PER_SCOUT_MAX


@dataclass
class CorpusChunk:
    """A contiguous slice of source corpus assigned to one scout."""
    chunk_id: str
    text: str
    source_tag: str
    # Set by partition_for_scouts() after chunk assignment. Empty ("") until
    # stamped. Web-search chunks receive "web_<facet_slug>" from FacetPlanner.
    partition_id: str = ""


@dataclass
class ScoutPartition:
    """The set of chunks given to a single scout for one round."""
    scout_index: int
    chunks: list
    # Fix S: web partitions override the default "partition_N" tag so that
    # web-search chunks carry "web_<facet_slug>" partition IDs through the pipeline.
    # Leave empty ("") for corpus partitions — the default property returns the
    # standard tag. Never set this in normal partition_for_scouts() calls.
    custom_partition_id: str = ""

    @property
    def partition_id(self) -> str:
        """Canonical partition tag for this scout slot."""
        if self.custom_partition_id:
            return self.custom_partition_id
        return f"partition_{self.scout_index}"

    @property
    def chunk_ids(self):
        return [c.chunk_id for c in self.chunks]

    def render(self, max_chars: int = 4000, offset: int = 0) -> str:
        """Render chunks as text, optionally rotated by offset (for sub-window variation).

        offset=k starts at chunks[k % len] and wraps around so every iteration
        of a scout sees the same partition contents but in a different order,
        raising output diversity without changing the partition itself.
        """
        if not self.chunks:
            return ""
        n = len(self.chunks)
        rotated = self.chunks[offset % n:] + self.chunks[:offset % n]
        parts = []
        used = 0
        for c in rotated:
            tag = "[" + c.source_tag + "#" + c.chunk_id + "]"
            piece = tag + "\n" + c.text + "\n"
            if used + len(piece) > max_chars:
                room = max(0, max_chars - used)
                parts.append(piece[:room])
                break
            parts.append(piece)
            used += len(piece)
        return "".join(parts).strip()


def chunk_corpus(text: str, source_tag: str = "corpus"):
    """Split a corpus string into overlapping word-chunks."""
    words = _normalize(text).split()
    if not words:
        return []
    chunks = []
    step = max(1, CHUNK_WORDS - CHUNK_OVERLAP)
    i = 0
    n = 0
    while i < len(words):
        slice_words = words[i : i + CHUNK_WORDS]
        chunk_text = " ".join(slice_words)
        chunks.append(CorpusChunk(
            chunk_id="chunk_" + format(n, "04d"),
            text=chunk_text,
            source_tag=source_tag,
        ))
        n += 1
        i += step
    return chunks


# Provenance of the most recent partition_for_scouts call ("semantic" |
# "contiguous" | ""). Surfaced so runs can record which partitioner actually
# fired — the semantic path degrades silently otherwise.
LAST_PARTITIONER: str = ""


def partition_for_scouts(chunks, num_scouts: int):
    """Assign chunks to scouts in non-overlapping SEMANTIC groups.

    The corpus arrives relevance-ranked (MMR output of the agentic search
    stack), so the old contiguous-block split gave scout 0 the best chunks
    and later scouts the tail — partitions were POSITIONAL, not topical, and
    "input diversity" was partly a quality gradient. Semantic partitioning
    clusters chunks around farthest-point seeds in embedding space, so each
    scout explores a topically distinct region of the evidence. Falls back
    to contiguous blocks when the embedder is unavailable or the corpus is
    tiny. Same invariants either way: partition_id stamped on every
    assigned chunk, every scout gets a (possibly empty) partition, at most
    CHUNKS_PER_SCOUT_MAX chunks per scout.
    """
    if num_scouts <= 0:
        return []
    if not chunks:
        return [ScoutPartition(scout_index=i, chunks=[]) for i in range(num_scouts)]

    per_scout = max(1, min(CHUNKS_PER_SCOUT_MAX, (len(chunks) + num_scouts - 1) // num_scouts))

    global LAST_PARTITIONER
    if len(chunks) >= num_scouts * 2:
        semantic = _partition_semantic(chunks, num_scouts, per_scout)
        if semantic is not None:
            LAST_PARTITIONER = "semantic"
            return semantic
        print(f"[intake] contiguous fallback: embedder unavailable "
              f"({len(chunks)} chunks, {num_scouts} scouts)")
    else:
        print(f"[intake] contiguous fallback: corpus too small for semantic "
              f"split ({len(chunks)} chunks < {num_scouts * 2})")

    LAST_PARTITIONER = "contiguous"
    return _partition_contiguous(chunks, num_scouts, per_scout)


def _partition_contiguous(chunks, num_scouts: int, per_scout: int):
    partitions = []
    cursor = 0
    for s in range(num_scouts):
        block = chunks[cursor : cursor + per_scout]
        pid = f"partition_{s}"
        for ch in block:
            ch.partition_id = pid
        partitions.append(ScoutPartition(scout_index=s, chunks=block))
        cursor += per_scout
        if cursor >= len(chunks):
            for s2 in range(s + 1, num_scouts):
                partitions.append(ScoutPartition(scout_index=s2, chunks=[]))
            return partitions
    return partitions


def _partition_semantic(chunks, num_scouts: int, per_scout: int):
    """Farthest-point seeds + capacity-bounded nearest-seed assignment.

    Returns None when embeddings are unavailable (caller falls back).
    """
    try:
        from .signal_store import _try_load_embedder
        model = _try_load_embedder()
        if model is None:
            return None
        import numpy as np
        vecs = []
        for c in chunks:
            v = np.asarray(model.encode(c.text[:600]), dtype="float32")
            n = float((v ** 2).sum() ** 0.5)
            if n <= 0:
                return None
            vecs.append(v / n)
    except Exception:
        return None

    import numpy as np
    V = np.stack(vecs)                       # (N, d), unit rows
    # Farthest-point seeding: start from chunk 0 (highest-ranked), then
    # repeatedly take the chunk least similar to any existing seed.
    seeds = [0]
    while len(seeds) < min(num_scouts, len(chunks)):
        sims_to_seeds = V @ V[seeds].T       # (N, n_seeds)
        max_sim = sims_to_seeds.max(axis=1)
        max_sim[seeds] = np.inf              # never re-pick a seed
        seeds.append(int(max_sim.argmin()))

    # Capacity-bounded assignment: highest-affinity pairs first, so each
    # chunk lands with its most similar seed that still has room.
    sims = V @ V[seeds].T                    # (N, S)
    order = sorted(
        ((float(sims[i, s]), i, s) for i in range(len(chunks))
         for s in range(len(seeds))),
        reverse=True,
    )
    assigned: dict[int, int] = {}
    counts = [0] * num_scouts
    for _sim, i, s in order:
        if i in assigned or counts[s] >= per_scout:
            continue
        assigned[i] = s
        counts[s] += 1

    partitions = [ScoutPartition(scout_index=s, chunks=[]) for s in range(num_scouts)]
    for i, s in sorted(assigned.items()):
        chunks[i].partition_id = f"partition_{s}"
        partitions[s].chunks.append(chunks[i])
    n_used = len(assigned)
    print(f"[intake] semantic partitioning: {n_used}/{len(chunks)} chunks -> "
          f"{sum(1 for p in partitions if p.chunks)} topical partitions "
          f"(<= {per_scout} each; farthest-point seeds)")
    return partitions


# Four section templates, each ~140 words. Loaded from a Python list of
# (tag, body) tuples — kept short so file write succeeds reliably.
_SECTION_TEMPLATES = [
    ("empirical", "Evidence comes from observational data and measurement. Counter-positions dispute the data, the sampling, the choice of summary statistics, or the inference from association to causation. Studies operationalize the thesis by selecting outcome variables, controlling for confounds, and reporting effect sizes with confidence intervals. Critics note that the chosen outcomes are not the only relevant ones, that controls are incomplete, and that effect sizes that appear small in absolute terms may matter at scale. Empirical analyses are legible to outside reviewers because the data and the model are inspectable. They are also vulnerable to charges of measurement validity: are we measuring what we claim to measure, consistently across populations of interest, and are cases outside the measurement boundary substantively different from cases inside it."),
    ("historical", "The historical framing asks how this question has been answered before, in what contexts, and with what success. Past attempts illuminate what kinds of solutions tend to work, what kinds tend to fail, and which framings persist versus which are products of a particular era. Counter-positions dispute the relevance of analogies, argue that present conditions are unique enough to invalidate comparisons, or claim that historical successes have been selectively remembered while failures have been forgotten. The historical framing is useful when contemporary debate has hardened into recognizable camps, because past episodes often cut across present alignments and reveal which features of the disagreement are durable. The hazard is overgeneralization: no two episodes are identical, and treating analogies as identities produces conclusions the analogy cannot bear."),
    ("economic", "The economic framing asks about incentives, costs, and distributional effects. Who pays, who benefits, on what timescale, through what mechanisms. Counter-positions dispute assumed price elasticities, discount rates applied to future consequences, the boundaries of the relevant market, or the assumption that rational actors will respond to nominal incentives in the predicted way. Economic analyses force a reckoning with opportunity costs: what else could the resources committed to this position be used for. The economic framing tends to be uncomfortable for advocates because it requires explicit accounting that other framings can leave implicit. It is also uncomfortable for opponents, because the same accounting applies to counter-proposals. Economic analyses that take both sides seriously usually conclude with a smaller intervention than either side would have proposed alone."),
    ("ethical", "The ethical framing asks about obligations, rights, and the distribution of moral weight. Whose interests count, by how much, and what duties follow. Counter-positions dispute the moral framework being applied, the choice of which agents have standing, and the priority orderings among competing values. Consequentialist arguments emphasize aggregate outcomes; deontological arguments emphasize duties and constraints; virtue-ethics arguments emphasize the character the position cultivates. Each framework yields different conclusions on the same thesis, and the disagreement is often about which framework to apply rather than about facts in the world. The ethical framing is essential because few significant theses are purely empirical: even an ostensibly factual claim usually carries an implicit policy conclusion, and policy conclusions are always ethical."),
]


def trivial_corpus_from_thesis(thesis: str) -> str:
    """Multi-framing placeholder corpus for when no retriever is wired in.

    Each framing is long enough to span at least one chunk, so partitioning
    across NUM_SCOUTS produces genuinely disjoint scout views.
    """
    pieces = []
    for tag, body in _SECTION_TEMPLATES:
        header = "SECTION (" + tag + " framing). Consider the thesis: " + thesis + "."
        # repeat the body twice so each section is robustly over CHUNK_WORDS/2
        pieces.append(header + " " + body + " " + body)
    return "\n\n".join(pieces)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
