"""Knowledge base persistence across swarm runs.

Surviving and rejected signal clusters are saved to disk so that future
runs can:
  - Boost support_diversity for clusters matching prior survivors.
  - Pre-penalize clusters matching prior rejections (dissent_pressure += 0.5).

Design constraints:
  - prior_consensus does NOT re-deposit as scout INITIALs. Scouts must
    remain fresh hypothesis samplers; injecting cached consensus into
    their input space defeats the divergence layer.
  - prior_rejections down-weight matching new clusters only in the
    survival filter, not in scout conditioning.
  - Embeddings are stored alongside cluster entries so similarity checks
    work offline without re-encoding.

Phase C improvements (C3/C4/C7/C8/C9):
  - Decay: on every load, all entries have decay_strength *= 0.95.
    Entries with decay_strength < 0.3 are moved to an archived/ list and
    excluded from projection influence. They remain on disk for inspection.
  - Bounded prior_consensus boost: handled in projection.py (cap at +2 via
    run_count). This file stores run_count and increments it on dedup match.
  - Task-aware retrieval: each entry carries a task_context block. When
    returning prior_consensus/prior_rejections, entries are filtered to
    those whose topic_hash matches the current run, with legacy entries
    (no task_context) included as soft fallback.
  - Contradiction tracking: when a new surviving cluster matches a prior
    rejected_by_field entry (sim >= 0.75) or vice versa, both entries get
    `contradicts` updated with each other's hash. Contradicted surviving
    clusters are demoted to contested in the projection (handled here by
    tagging the entry; projection.py checks the tag).
  - Per-run KB diff: kb_diff.json is written to the run's output directory
    showing new_entries, matched_entries, decayed_to_archive, and
    contradictions_recorded.
  - Content-sanity check before save: entries whose representative content
    fails the junk-output filter are refused (defence-in-depth).
  - Dedup on save: new entries matching existing at similarity >= 0.85
    update in-place rather than appending.
  - Provenance: originating_run records (timestamp, task_type, user_prompt).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Optional

from .projection import SynthesisProjection, ClusterProjection
from .signal_store import SignalStore
from .filters import is_junk_output

_DEFAULT_KB_DIR = "knowledge_base"
_SURVIVING_FILE = "surviving_clusters.json"
_REJECTED_FILE = "rejected_clusters.json"
_CONTESTED_FILE = "contested_clusters.json"
_SCHEMA_VERSION = 2

# Cosine similarity above which a new cluster is considered a duplicate.
_KB_DEDUP_THRESHOLD = 0.85
# Similarity above which a surviving↔rejected pair is a contradiction.
_KB_CONTRADICTION_THRESHOLD = 0.75
# decay_strength below which an entry is moved to the archived list.
_KB_ARCHIVE_THRESHOLD = 0.3
# decay multiplier applied per load cycle.
_KB_DECAY_PER_LOAD = 0.95


def _cosine_sim(a: list[float], b: list[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b)))


def _cluster_hash(representative_content: str) -> str:
    """Stable 16-char SHA-1 prefix of the representative content."""
    return hashlib.sha1(representative_content.encode("utf-8")).hexdigest()[:16]


def _normalized_keyphrase(text: str, max_words: int = 5) -> str:
    """Lowercase, strip stopwords, take first `max_words` content words."""
    stop = {
        "the", "a", "an", "is", "are", "was", "were", "to", "of", "in",
        "and", "or", "but", "for", "on", "at", "by", "with", "as",
        "that", "this", "these", "those", "it", "its", "i", "my", "we",
    }
    words = [w.strip(".,;:?!\"'()[]").lower() for w in text.split()]
    keep = [w for w in words if w and w not in stop]
    return " ".join(keep[:max_words])


def _topic_hash(user_prompt: str) -> str:
    keyphrase = _normalized_keyphrase(user_prompt)
    return hashlib.sha1(keyphrase.encode("utf-8")).hexdigest()[:16]


class KnowledgeBase:
    def __init__(self, kb_dir: Optional[Path] = None):
        if kb_dir is None:
            kb_dir = Path(os.environ.get("SWARM_KB_DIR", _DEFAULT_KB_DIR))
        self.kb_dir = Path(kb_dir)
        self._surviving: list[dict] = []
        self._contested: list[dict] = []
        self._rejected: list[dict] = []
        self._archived: list[dict] = []  # decayed entries, not used in projection

    # ---- load / save -------------------------------------------------------

    def load(self) -> None:
        """Load prior run clusters from disk (no-op if files don't exist).

        Applies decay (decay_strength *= 0.95) on every load.
        Entries with decay_strength < 0.3 are moved to self._archived.
        Warns if any file has a schema_version newer than _SCHEMA_VERSION.
        """
        for attr, path in (
            ("_surviving", self.kb_dir / _SURVIVING_FILE),
            ("_contested", self.kb_dir / _CONTESTED_FILE),
            ("_rejected",  self.kb_dir / _REJECTED_FILE),
        ):
            if not path.exists():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                for entry in data:
                    v = entry.get("schema_version", 1)
                    if v > _SCHEMA_VERSION:
                        print(
                            f"[kb] warning: {path.name} contains schema_version={v} "
                            f"(current={_SCHEMA_VERSION}); some fields may be ignored"
                        )
                        break
                setattr(self, attr, data)
            except Exception as exc:
                print(f"[kb] warning: could not load {path}: {exc}")

        # C3: Apply decay on load and move expired entries to _archived.
        active_s, active_c, active_r = [], [], []
        decayed = 0
        for attr, active_list in (
            ("_surviving", active_s),
            ("_contested", active_c),
            ("_rejected",  active_r),
        ):
            for entry in getattr(self, attr):
                entry["decay_strength"] = entry.get("decay_strength", 1.0) * _KB_DECAY_PER_LOAD
                if entry["decay_strength"] < _KB_ARCHIVE_THRESHOLD:
                    self._archived.append(entry)
                    decayed += 1
                else:
                    active_list.append(entry)
        self._surviving = active_s
        self._contested = active_c
        self._rejected  = active_r
        if decayed:
            print(f"[kb] decayed {decayed} entries to archive (decay_strength < {_KB_ARCHIVE_THRESHOLD})")

    def save(
        self,
        projection: SynthesisProjection,
        store: SignalStore,
        run_meta: dict,
        output_dir: Optional[Path] = None,
    ) -> dict:
        """Append surviving, contested, and rejected_by_field clusters to disk.

        Returns a kb_diff dict; writes kb_diff.json to output_dir when provided.
        """
        self.kb_dir.mkdir(parents=True, exist_ok=True)
        timestamp = run_meta.get("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S"))
        user_prompt = run_meta.get("user_prompt", "")
        task_type = run_meta.get("task_type", "")
        current_topic_hash = _topic_hash(user_prompt)

        task_context = {
            "task_type": task_type,
            "topic_hash": current_topic_hash,
            "user_prompt_excerpt": user_prompt[:200],
        }
        originating_run = {
            "timestamp": timestamp,
            "task_type": task_type,
            "user_prompt": user_prompt,
        }

        new_surviving = self._build_entries(
            projection.surviving, store, timestamp, "surviving",
            originating_run, task_context
        )
        new_contested = self._build_entries(
            projection.contested, store, timestamp, "contested",
            originating_run, task_context
        )
        new_rejected = self._build_entries(
            projection.rejected_by_field, store, timestamp, "rejected_by_field",
            originating_run, task_context
        )

        # C7: Contradiction tracking before merge.
        contradiction_pairs = self._detect_contradictions(
            new_surviving, self._rejected,
            new_rejected, self._surviving,
        )

        saved_s, new_s_hashes, matched_s = self._merge_entries(self._surviving, new_surviving, timestamp)
        saved_c, new_c_hashes, matched_c = self._merge_entries(self._contested, new_contested, timestamp)
        saved_r, new_r_hashes, matched_r = self._merge_entries(self._rejected,  new_rejected,  timestamp)

        (self.kb_dir / _SURVIVING_FILE).write_text(
            json.dumps(self._surviving, indent=2), encoding="utf-8"
        )
        (self.kb_dir / _CONTESTED_FILE).write_text(
            json.dumps(self._contested, indent=2), encoding="utf-8"
        )
        (self.kb_dir / _REJECTED_FILE).write_text(
            json.dumps(self._rejected, indent=2), encoding="utf-8"
        )
        print(
            f"[kb] saved {saved_s} surviving, "
            f"{saved_c} contested, "
            f"{saved_r} rejected -> {self.kb_dir} "
            f"(dedup: {len(new_surviving)-saved_s + len(new_contested)-saved_c + len(new_rejected)-saved_r} merged, "
            f"contradictions: {len(contradiction_pairs)})"
        )

        # C9: Per-run KB diff.
        kb_diff = {
            "new_entries": new_s_hashes + new_c_hashes + new_r_hashes,
            "matched_entries": matched_s + matched_c + matched_r,
            "decayed_to_archive": [
                e.get("cluster_hash", "") for e in self._archived
            ],
            "contradictions_recorded": contradiction_pairs,
        }
        if output_dir is not None:
            try:
                (output_dir / "kb_diff.json").write_text(
                    json.dumps(kb_diff, indent=2), encoding="utf-8"
                )
            except Exception as exc:
                print(f"[kb] could not write kb_diff.json: {exc}")

        return kb_diff

    # ---- accessors ---------------------------------------------------------

    def prior_consensus(self, topic_hash: Optional[str] = None) -> list[dict]:
        """Return surviving entries filtered to the current topic, with legacy fallback."""
        return _filter_by_topic(self._surviving, topic_hash)

    def prior_contested(self, topic_hash: Optional[str] = None) -> list[dict]:
        return _filter_by_topic(self._contested, topic_hash)

    def prior_rejections(self, topic_hash: Optional[str] = None) -> list[dict]:
        return _filter_by_topic(self._rejected, topic_hash)

    def is_empty(self) -> bool:
        return not self._surviving and not self._contested and not self._rejected

    # ---- internal helpers --------------------------------------------------

    def _build_entries(
        self,
        clusters: list[ClusterProjection],
        store: SignalStore,
        timestamp: str,
        status: str,
        originating_run: dict,
        task_context: dict,
    ) -> list[dict]:
        entries: list[dict] = []
        for cp in clusters:
            rep = store.get(cp.representative_id)
            if rep is None:
                continue
            if is_junk_output(rep.content):
                print(
                    f"[kb] junk filter rejected cluster {cp.representative_id} "
                    f"from KB (content looks like agent scratchpad)"
                )
                continue
            entries.append(
                _cluster_to_entry(cp, store, timestamp, status, originating_run, task_context)
            )
        return entries

    def _merge_entries(
        self,
        existing: list[dict],
        new_entries: list[dict],
        timestamp: str,
    ) -> tuple[int, list[str], list[dict]]:
        """Merge new_entries into existing in-place with dedup.

        Returns (n_new_appended, list_of_new_hashes, list_of_match_dicts).
        """
        appended = 0
        new_hashes: list[str] = []
        match_records: list[dict] = []

        for entry in new_entries:
            new_emb = entry.get("representative_embedding")
            new_hash = entry.get("cluster_hash", "")
            new_content = entry.get("representative_content", "")
            matched = False

            for ex in existing:
                ex_emb = ex.get("representative_embedding")
                # Prefer embedding cosine similarity; fall back to string ratio.
                if new_emb is not None and ex_emb is not None:
                    sim = _cosine_sim(new_emb, ex_emb)
                else:
                    from difflib import SequenceMatcher
                    sim = SequenceMatcher(
                        None,
                        new_content.lower(),
                        ex.get("representative_content", "").lower(),
                    ).ratio()

                if sim >= _KB_DEDUP_THRESHOLD:
                    prev_rc = ex.get("run_count", 1)
                    ex["last_seen"] = timestamp
                    ex["run_count"] = prev_rc + 1
                    # Cap support_diversity boost at +1 per match
                    prior_sd = ex.get("support_diversity", 0)
                    new_sd = entry.get("support_diversity", 0)
                    ex["support_diversity"] = prior_sd + min(1, max(0, new_sd - prior_sd))
                    ex["decay_strength"] = 1.0  # refresh on match
                    match_records.append({
                        "hash": ex.get("cluster_hash", ""),
                        "prev_run_count": prev_rc,
                        "new_run_count": ex["run_count"],
                    })
                    matched = True
                    break

            if not matched:
                existing.append(entry)
                appended += 1
                new_hashes.append(new_hash)

        return appended, new_hashes, match_records

    def _detect_contradictions(
        self,
        new_surviving: list[dict],
        prior_rejected: list[dict],
        new_rejected: list[dict],
        prior_surviving: list[dict],
    ) -> list[list[str]]:
        """Flag cross-status pairs as contradictions and annotate both entries.

        Returns list of [hash_a, hash_b] pairs.
        """
        pairs: list[list[str]] = []

        def _check_and_flag(group_a: list[dict], group_b: list[dict]) -> None:
            for ea in group_a:
                emb_a = ea.get("representative_embedding")
                if emb_a is None:
                    continue
                for eb in group_b:
                    emb_b = eb.get("representative_embedding")
                    if emb_b is None:
                        continue
                    sim = _cosine_sim(emb_a, emb_b)
                    if sim >= _KB_CONTRADICTION_THRESHOLD:
                        ha = ea.get("cluster_hash", "")
                        hb = eb.get("cluster_hash", "")
                        if [ha, hb] not in pairs and [hb, ha] not in pairs:
                            # Annotate both entries
                            ea.setdefault("contradicts", [])
                            if hb not in ea["contradicts"]:
                                ea["contradicts"].append(hb)
                            ea["first_contradicted_at"] = ea.get(
                                "first_contradicted_at",
                                time.strftime("%Y-%m-%dT%H:%M:%S"),
                            )
                            eb.setdefault("contradicts", [])
                            if ha not in eb["contradicts"]:
                                eb["contradicts"].append(ha)
                            eb["first_contradicted_at"] = eb.get(
                                "first_contradicted_at",
                                time.strftime("%Y-%m-%dT%H:%M:%S"),
                            )
                            pairs.append([ha, hb])

        _check_and_flag(new_surviving, prior_rejected)
        _check_and_flag(new_rejected, prior_surviving)
        return pairs


# ---------------------------------------------------------------------------
# Task-aware filtering
# ---------------------------------------------------------------------------

def _filter_by_topic(entries: list[dict], topic_hash: Optional[str]) -> list[dict]:
    """Return entries matching topic_hash, plus legacy entries (no task_context)."""
    if topic_hash is None:
        return list(entries)
    matched = []
    legacy = []
    for e in entries:
        tc = e.get("task_context")
        if tc is None:
            legacy.append(e)
        elif tc.get("topic_hash") == topic_hash:
            matched.append(e)
    # Topic-matched entries take priority; include legacy as soft fallback.
    return matched if matched else legacy


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _cluster_to_entry(
    cp: ClusterProjection,
    store: SignalStore,
    timestamp: str,
    status: str,
    originating_run: Optional[dict] = None,
    task_context: Optional[dict] = None,
) -> dict:
    rep = store.get(cp.representative_id)
    rep_emb = store.get_embedding(cp.representative_id)
    rep_content = rep.content if rep else ""

    lineage_ids = (
        cp.member_ids
        + cp.support_set
        + cp.dissent_set
        + cp.verification_set
    )

    return {
        "schema_version": _SCHEMA_VERSION,
        "cluster_hash": _cluster_hash(rep_content),
        "representative_content": rep_content,
        "representative_embedding": rep_emb,
        "member_signal_ids": cp.member_ids,
        "lineage_signal_ids": lineage_ids,
        "partition_origins": cp.partition_origins,
        "support_diversity": cp.support_diversity,
        "dissent_pressure": round(cp.dissent_pressure, 4),
        "verification_score": round(cp.verification_score, 4),
        "run_timestamp": timestamp,
        "status": status,
        # C3 decay / provenance fields
        "first_seen": timestamp,
        "last_seen": timestamp,
        "decay_strength": 1.0,
        "run_count": 1,
        "originating_run": originating_run,
        # C8 task-aware retrieval
        "task_context": task_context,
        # C7 contradiction tracking (populated at save time if contradiction found)
        "contradicts": [],
    }
