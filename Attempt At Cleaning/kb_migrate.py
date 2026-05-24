"""Migration script: old flat-JSON KB format → schema_version=2 format.

Reads surviving_clusters.json, contested_clusters.json, rejected_clusters.json
from the old format (or from quarantine directories), applies the junk filter,
computes cluster_hash, adds missing fields, and writes them back.

Also creates schema_version.json as the KB schema marker.

Usage:
    python kb_migrate.py                          # migrate knowledge_base/
    python kb_migrate.py --kb-dir path/to/kb      # migrate a specific directory
    python kb_migrate.py --from-quarantine        # also check .quarantine_* dirs

Output:
    Prints summary: N migrated, M junk-rejected, K deduplicated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

# Make project root importable
sys.path.insert(0, str(Path(__file__).parent))

from core.filters import is_junk_output

_SCHEMA_VERSION = 2
_FILES = ("surviving_clusters.json", "contested_clusters.json", "rejected_clusters.json")


def _cluster_hash(content: str) -> str:
    return hashlib.sha1(content.encode("utf-8")).hexdigest()[:16]


def _migrate_file(path: Path) -> tuple[list[dict], int, int, int]:
    """Read a KB JSON file, apply migration, return (entries, migrated, junk, deduped)."""
    if not path.exists():
        return [], 0, 0, 0

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"  [skip] could not read {path}: {exc}")
        return [], 0, 0, 0

    migrated = 0
    junk_rejected = 0
    seen_hashes: dict[str, int] = {}  # hash -> index in out
    out: list[dict] = []

    for entry in raw:
        content = entry.get("representative_content", "")

        # Junk filter: reject contaminated entries
        if is_junk_output(content):
            print(f"  [junk] rejected: {content[:60]!r}")
            junk_rejected += 1
            continue

        # Compute stable hash
        chash = _cluster_hash(content)

        # Dedup: if we've seen this hash, update the existing entry
        if chash in seen_hashes:
            idx = seen_hashes[chash]
            out[idx]["run_count"] = out[idx].get("run_count", 1) + 1
            out[idx]["last_seen"] = entry.get("run_timestamp", out[idx].get("last_seen", ""))
            continue

        # Add missing schema_version=2 fields if absent
        ts = entry.get("run_timestamp", time.strftime("%Y-%m-%dT%H:%M:%S"))
        entry.setdefault("schema_version", _SCHEMA_VERSION)
        entry["cluster_hash"] = chash
        entry.setdefault("first_seen", ts)
        entry.setdefault("last_seen", ts)
        entry.setdefault("decay_strength", 1.0)
        entry.setdefault("run_count", 1)
        entry.setdefault("task_context", None)
        entry.setdefault("contradicts", [])

        seen_hashes[chash] = len(out)
        out.append(entry)
        migrated += 1

    deduped = len(raw) - junk_rejected - migrated
    return out, migrated, junk_rejected, deduped


def _write_schema_version(kb_dir: Path) -> None:
    sv_path = kb_dir / "schema_version.json"
    sv_path.write_text(
        json.dumps({"schema_version": _SCHEMA_VERSION, "migrated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}),
        encoding="utf-8",
    )
    print(f"  [ok] wrote schema_version.json (schema_version={_SCHEMA_VERSION})")


def migrate_dir(kb_dir: Path) -> tuple[int, int, int]:
    """Migrate one KB directory. Returns (total_migrated, total_junk, total_deduped)."""
    if not kb_dir.exists():
        print(f"[migrate] directory {kb_dir} does not exist — skipping")
        return 0, 0, 0

    print(f"[migrate] processing {kb_dir}")
    total_m = total_j = total_d = 0

    for fname in _FILES:
        path = kb_dir / fname
        entries, m, j, d = _migrate_file(path)
        total_m += m
        total_j += j
        total_d += d
        if entries or path.exists():
            path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
            print(f"  [{fname}] {m} migrated, {j} junk-rejected, {d} deduplicated")

    _write_schema_version(kb_dir)
    return total_m, total_j, total_d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kb-dir", default="knowledge_base",
                        help="Path to the KB directory (default: knowledge_base/)")
    parser.add_argument("--from-quarantine", action="store_true",
                        help="Also migrate entries from .quarantine_* directories")
    args = parser.parse_args()

    kb_dir = Path(args.kb_dir)

    grand_m = grand_j = grand_d = 0
    m, j, d = migrate_dir(kb_dir)
    grand_m += m; grand_j += j; grand_d += d

    if args.from_quarantine:
        for q in sorted(kb_dir.glob(".quarantine_*")):
            if q.is_dir():
                m, j, d = migrate_dir(q)
                grand_m += m; grand_j += j; grand_d += d

    print(f"\n[migrate] SUMMARY: {grand_m} migrated, {grand_j} junk-rejected, {grand_d} deduplicated")


if __name__ == "__main__":
    main()
