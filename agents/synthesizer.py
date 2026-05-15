"""Synthesizer — two-layer read-out from the surviving signal DAG.

Layer 1 (core/projection.py): pure-Python DAG projection. No LLM.
  Classifies clusters as surviving / contested / weakly_supported / rejected_by_field.
  surviving clusters carry an `unverified` flag when no validator reached them.
  Computes support_diversity, dissent_pressure, verification_score.

Layer 2 (this file): structured multi-call LLM renderer.
  One LLM call per surviving cluster (Section 1) and per contested/dissented
  cluster (Section 2). Deterministic assembly. No single pooled call that lets
  the model hallucinate cross-cluster content.

Output structure
----------------
  Section 1 — Position synthesis      (1 paragraph per surviving cluster)
  Section 2 — Open questions/dissent  (1 paragraph per contested or challenged)
  Section 3 — Considered and filtered (brief, deterministic — no LLM)
  Section 4 — Citations               (deterministic stamp, no LLM)

Why per-cluster calls?
  - Faithfulness: each call only sees one cluster's evidence, so cross-cluster
    content pooling is structurally impossible.
  - Failure isolation: one bad cluster render doesn't corrupt the others.
  - Parallelism: when LLM_CONCURRENCY rises, cluster calls will parallelize.

External grounding (optional)
------------------------------
  At synthesis time, a Wikipedia lookup is performed for each surviving cluster's
  representative claim. The snippet is injected as "[External context]" in the
  renderer prompt and flagged in the output so it's distinguishable from
  agent-deposited content. Never raises; degrades gracefully offline.

Faithfulness audit
------------------
  After rendering, each paragraph is checked: for every cited cluster ID [INITIAL_XXXXX],
  does the surrounding text have at least a 4-gram overlap with that cluster's
  representative content? Failures are collected and written to renderer_audit.json
  when output_dir is provided.

Returns
-------
  answer_text:   str — prose with inline citation tags
  citations:     dict — maps each signal ID to provenance metadata
  lineage_dot:   str — Graphviz DOT of the surviving DAG
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Optional

from core.signal_store import SignalStore
from core.signal_types import INITIAL, SUPPORT, CRITIQUE, OBJECTION, VERIFICATION
from core.config import MAX_TOKENS_SYNTHESIZER
from core.projection import (
    SynthesisProjection,
    ClusterProjection,
    build_projection,
)
from agents.base import strip_reasoning

_RENDERER_TEMPERATURE = 0.3
# Content budgets for renderer prompts (chars, not tokens)
_REPRESENTATIVE_CHARS = 800
_SUPPORT_CHARS = 400
_DISSENT_CHARS = 400
_SUMMARY_CHARS = 150   # for Section 3 "filtered" entries
_EXTERNAL_CHARS = 400  # Wikipedia snippet injected into Section 1 calls


class Synthesizer:
    ROLE = "synthesizer"

    def __init__(self, llm, task_prompt: str):
        self.llm = llm
        self.task_prompt = task_prompt

    async def synthesize(
        self,
        store: SignalStore,
        has_validators: bool = True,
        prior_rejections: Optional[list] = None,
        prior_consensus: Optional[list] = None,
        output_dir: Optional[Path] = None,
    ) -> tuple[str, dict, str]:
        """Run Layer 1 then Layer 2.

        output_dir: when provided, renderer_audit.json is written there.
        Returns (answer_text, citations_dict, lineage_dot_str).
        """
        # Layer 1: pure-Python DAG projection
        projection = build_projection(
            store,
            has_validators=has_validators,
            prior_rejections=prior_rejections,
            prior_consensus=prior_consensus,
        )

        # Layer 2: structured multi-call renderer
        answer_text = await self._render(projection, store, output_dir=output_dir)

        # Build citation and lineage artefacts (pure Python, no LLM)
        citations = _build_citations(projection, store)
        lineage_dot = _build_lineage_dot(projection, store)

        return answer_text, citations, lineage_dot

    # -----------------------------------------------------------------------
    # Layer 2: renderer
    # -----------------------------------------------------------------------

    async def _render(
        self,
        projection: SynthesisProjection,
        store: SignalStore,
        output_dir: Optional[Path] = None,
    ) -> str:
        if projection.no_consensus:
            surv = len(projection.surviving)
            cont = len(projection.contested)
            weak = len(projection.weakly_supported)
            rej  = len(projection.rejected_by_field)
            return (
                f"The swarm did not converge on a stable answer for this task.\n\n"
                f"Projection summary: {surv} surviving, {cont} contested, "
                f"{weak} weakly_supported, {rej} rejected_by_field.\n\n"
                f"Consult signals.json and lineage.dot for the underlying signal "
                f"field. You can re-synthesize with different filter thresholds "
                f"via `python synthesize.py <run_dir>`."
            )

        sections: list[str] = []

        # ------------------------------------------------------------------
        # Section 1: Position synthesis — one LLM call per surviving cluster
        # ------------------------------------------------------------------
        if projection.surviving:
            fragments: list[str] = []
            for cp in projection.surviving:
                fragment = await self._render_cluster_position(cp, store)
                if fragment:
                    fragments.append(fragment)
            if fragments:
                sections.append(
                    "## 1. POSITION SYNTHESIS\n\n" + "\n\n".join(fragments)
                )

        # ------------------------------------------------------------------
        # Section 2: Open questions and dissent — per contested cluster AND
        # per surviving cluster that attracted any dissent
        # ------------------------------------------------------------------
        dissent_candidates: list[ClusterProjection] = list(projection.contested)
        dissent_candidates += [
            cp for cp in projection.surviving if cp.dissent_set
        ]
        if dissent_candidates:
            fragments = []
            for cp in dissent_candidates:
                fragment = await self._render_cluster_dissent(cp, store)
                if fragment:
                    fragments.append(fragment)
            if fragments:
                sections.append(
                    "## 2. OPEN QUESTIONS AND DISSENT\n\n" + "\n\n".join(fragments)
                )

        # ------------------------------------------------------------------
        # Section 3: Considered and filtered — deterministic, no LLM call
        # Hard cap at 5 entries. Sort: rejected_by_field first (most diagnostic),
        # then weakly_supported by descending verification_score.
        # ------------------------------------------------------------------
        rej_sorted = sorted(
            projection.rejected_by_field,
            key=lambda c: c.verification_score,
            reverse=True,
        )
        weak_sorted = sorted(
            projection.weakly_supported,
            key=lambda c: c.verification_score,
            reverse=True,
        )
        filtered = (rej_sorted + weak_sorted)[:5]
        if filtered:
            lines: list[str] = []
            for cp in filtered:
                rep = store.get(cp.representative_id)
                content = _truncate(rep.content, _SUMMARY_CHARS) if rep else cp.representative_id
                if cp.status == "rejected_by_field":
                    reason = f"rejected: dissent_pressure={cp.dissent_pressure:.2f} > 1.5"
                else:
                    reason = f"filtered: support_diversity={cp.support_diversity} < 2"
                lines.append(f"- [{cp.representative_id}] {content}  ({reason})")
            sections.append(
                "## 3. CONSIDERED AND FILTERED\n\n"
                + "\n".join(lines)
            )

        # ------------------------------------------------------------------
        # Assemble sections
        # ------------------------------------------------------------------
        answer = "\n\n".join(sections)

        # Section 4 — Citations: deterministic stamp appended to the answer
        answer = _stamp_citations(answer, projection, store)

        # ------------------------------------------------------------------
        # Post-hoc faithfulness audit
        # ------------------------------------------------------------------
        audit_flags = _build_faithfulness_audit(answer, projection, store)
        if output_dir is not None:
            _write_faithfulness_audit(audit_flags, output_dir)
        elif audit_flags:
            print(
                f"[synthesizer] faithfulness audit: {len(audit_flags)} flag(s) "
                f"(pass output_dir to write renderer_audit.json)"
            )

        return answer

    # -----------------------------------------------------------------------
    # Per-cluster render helpers
    # -----------------------------------------------------------------------

    async def _render_cluster_position(
        self, cp: ClusterProjection, store: SignalStore
    ) -> str:
        """Render a one-paragraph position statement for one surviving cluster."""
        rep = store.get(cp.representative_id)
        if rep is None:
            return ""

        content = _truncate(rep.content, _REPRESENTATIVE_CHARS)
        unver_note = " (not externally verified)" if cp.unverified else ""

        # Gather support excerpts (up to 3)
        support_lines: list[str] = []
        for sid in cp.support_set[:3]:
            s = store.get(sid)
            if s:
                support_lines.append(
                    f"  - [{sid}] {_truncate(s.content, _SUPPORT_CHARS)}"
                )

        # External grounding (optional — Wikipedia at synthesis time)
        ext_ctx = _get_external_context(rep.content)
        ext_block = (
            f"\n[External context, not agent-derived]: {ext_ctx}\n"
            if ext_ctx else ""
        )

        partition_note = (
            f"Partition origins: {', '.join(cp.partition_origins)}"
            if cp.partition_origins else ""
        )

        support_block = (
            "\nSupporting evidence:\n" + "\n".join(support_lines)
            if support_lines else ""
        )

        prompt = (
            f"TASK: {self.task_prompt}\n\n"
            f"Write one clear paragraph synthesizing the following surviving claim "
            f"and its supporting evidence. Cite [{cp.representative_id}] inline. "
            f"Do not introduce claims not present in the input. "
            f"If the claim is marked 'not externally verified', keep that phrase.\n\n"
            f"Claim [{cp.representative_id}]: {content}{unver_note}\n"
            f"{support_block}"
            f"{ext_block}"
            f"{partition_note}\n\n"
            f"PARAGRAPH:"
        )

        raw = await self.llm.generate(
            prompt, role=self.ROLE,
            max_tokens=MAX_TOKENS_SYNTHESIZER,
            temperature=_RENDERER_TEMPERATURE,
        )
        return strip_reasoning(raw.strip())

    async def _render_cluster_dissent(
        self, cp: ClusterProjection, store: SignalStore
    ) -> str:
        """Render a one-paragraph dissent summary for one contested or challenged cluster."""
        rep = store.get(cp.representative_id)
        if rep is None:
            return ""

        content = _truncate(rep.content, _REPRESENTATIVE_CHARS)
        strongest_dissent = _strongest_signal_content(cp.dissent_set, store)
        counter = (
            _truncate(strongest_dissent, _DISSENT_CHARS)
            if strongest_dissent else "(no explicit counter-position deposited)"
        )

        prompt = (
            f"TASK: {self.task_prompt}\n\n"
            f"Write one paragraph describing the following {'contested' if cp.status == 'contested' else 'challenged'} "
            f"claim, the strongest counter-position, and what evidence would resolve the dispute. "
            f"Cite [{cp.representative_id}] inline. Do not take sides.\n\n"
            f"Claim [{cp.representative_id}] (status={cp.status}, "
            f"dissent_pressure={cp.dissent_pressure:.2f}): {content}\n\n"
            f"Strongest counter-position: {counter}\n\n"
            f"PARAGRAPH:"
        )

        raw = await self.llm.generate(
            prompt, role=self.ROLE,
            max_tokens=MAX_TOKENS_SYNTHESIZER,
            temperature=_RENDERER_TEMPERATURE,
        )
        return strip_reasoning(raw.strip())


# ---------------------------------------------------------------------------
# Post-generation citation stamping (Section 4 — deterministic)
# ---------------------------------------------------------------------------

def _stamp_citations(
    answer: str,
    projection: SynthesisProjection,
    store: SignalStore,
) -> str:
    """Append a deterministic CITATIONS block to the rendered answer.

    Each surviving/contested cluster gets one entry:
        CLAIM [INITIAL_00023]:
          Supports:      SUPPORT_00041, SUPPORT_00058
          Challenges:    CRITIQUE_00047
          Verifications: VERIFICATION_00062
          Partition:     partition_2
          Coverage:      2 partition(s) contributed to this cluster
    """
    active = projection.surviving + projection.contested
    if not active:
        return answer

    lines = ["", "", "## 4. CITATIONS", "=" * 60]
    for cp in active:
        rep = store.get(cp.representative_id)
        excerpt = _truncate(rep.content, 80) if rep else cp.representative_id
        status_tag = f"[{cp.status.upper()}]" if cp.status != "surviving" else ""
        lines.append(f"CLAIM {status_tag} [{cp.representative_id}]: {excerpt}")
        if cp.support_set:
            lines.append(f"  Supports:      {', '.join(cp.support_set[:8])}")
        if cp.dissent_set:
            lines.append(f"  Challenges:    {', '.join(cp.dissent_set[:5])}")
        if cp.verification_set:
            lines.append(f"  Verifications: {', '.join(cp.verification_set[:5])}")
        if cp.partition_origins:
            lines.append(f"  Partition:     {', '.join(cp.partition_origins)}")
        lines.append(
            f"  support_diversity={cp.support_diversity}  "
            f"dissent_pressure={cp.dissent_pressure:.2f}  "
            f"verification_score={cp.verification_score:.2f}"
        )
        lines.append("")

    return answer.rstrip() + "\n".join(lines)


# ---------------------------------------------------------------------------
# Faithfulness audit
# ---------------------------------------------------------------------------

_CITATION_RE = re.compile(r"\[([A-Z]+_\d+)\]")


def _build_faithfulness_audit(
    answer: str,
    projection: SynthesisProjection,
    store: SignalStore,
) -> list[dict]:
    """Check each cited cluster ID in the prose for faithfulness.

    Two checks per cited ID:
      1. Existence: the ID must correspond to a real signal in the store.
         Fabricated IDs (e.g. [OPP_00145] that never existed) are flagged.
      2. 4-gram overlap: for surviving/contested cluster reps, the paragraph
         must share at least one 4-word sequence with that cluster's content.
         Catches wrong-cluster citation and hallucinated prose.

    Also raises post-hoc flags for decoder pathology (these don't reject —
    they make the audit file useful for cross-run comparison):
      3. orphan_think_tag: paragraph contains a literal </think> or <think>.
      4. scratchpad_in_prose: paragraph matches the scratchpad marker regex.
      5. truncated_mid_sentence: paragraph ends without terminal punctuation
         and is followed by a section break (double newline).
    """
    from agents.base import _SCRATCHPAD_RE

    cluster_content: dict[str, str] = {}
    for cp in projection.surviving + projection.contested:
        rep = store.get(cp.representative_id)
        if rep:
            cluster_content[cp.representative_id] = rep.content.lower()

    # Build the complete set of real signal IDs from the store (for existence check).
    valid_signal_ids: set[str] = {s.id for s in store.all()}

    flags: list[dict] = []
    paragraphs = [p.strip() for p in answer.split("\n\n") if p.strip()]

    for i, para in enumerate(paragraphs):
        cited_ids = _CITATION_RE.findall(para)

        for cid in cited_ids:
            # Check 1: ID must exist in the signal store (fabrication check).
            if cid not in valid_signal_ids:
                # Grab up to 200 chars of surrounding context.
                excerpt_start = max(0, answer.find(para) - 50)
                surrounding = answer[excerpt_start: excerpt_start + 200]
                flags.append({
                    "issue": "fabricated citation",
                    "cited_id": cid,
                    "paragraph_excerpt": surrounding,
                })
                continue

            # Check 2: 4-gram overlap with cluster representative content.
            if cid not in cluster_content:
                continue  # not a cluster rep — skip overlap check
            cluster_words = cluster_content[cid].split()
            if len(cluster_words) < 4:
                continue
            para_lower = para.lower()
            found_overlap = any(
                " ".join(cluster_words[j: j + 4]) in para_lower
                for j in range(len(cluster_words) - 3)
            )
            if not found_overlap:
                flags.append({
                    "issue": "no 4-gram overlap between paragraph and cited cluster content",
                    "cited_id": cid,
                    "paragraph_excerpt": para[:300],
                    "cluster_content_excerpt": cluster_content[cid][:300],
                })

        # Check 3: orphan think tags in prose (post-hoc, no rejection).
        if "</think>" in para or "<think>" in para.lower():
            flags.append({
                "issue": "orphan_think_tag",
                "paragraph_excerpt": para[:300],
            })

        # Check 4: scratchpad marker survived into prose.
        if _SCRATCHPAD_RE.search(para):
            flags.append({
                "issue": "scratchpad_in_prose",
                "paragraph_excerpt": para[:300],
            })

        # Check 5: truncated mid-sentence (no terminal punctuation before section break).
        if i < len(paragraphs) - 1:
            last_char = para.rstrip()[-1] if para.rstrip() else ""
            if last_char not in {".", "!", "?", '"', ")", "'"}:
                flags.append({
                    "issue": "truncated_mid_sentence",
                    "paragraph_excerpt": para[-200:],
                })

    return flags


def _write_faithfulness_audit(flags: list[dict], output_dir: Path) -> None:
    audit = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_flags": len(flags),
        "flags": flags,
    }
    try:
        (output_dir / "renderer_audit.json").write_text(
            json.dumps(audit, indent=2), encoding="utf-8"
        )
        if flags:
            print(
                f"[synthesizer] faithfulness audit: {len(flags)} flag(s) "
                f"written to renderer_audit.json"
            )
        else:
            print("[synthesizer] faithfulness audit: 0 flags (all citations faithful)")
    except Exception as exc:
        print(f"[synthesizer] could not write renderer_audit.json: {exc}")


# ---------------------------------------------------------------------------
# External grounding at synthesis time
# ---------------------------------------------------------------------------

def _get_external_context(content: str) -> Optional[str]:
    """Best-effort Wikipedia snippet for a cluster's representative claim.

    Injected into the Section 1 renderer prompt as "[External context]" so it's
    distinguishable from agent-deposited content. Returns None on any failure.
    """
    try:
        keyphrase = _extract_keyphrase(content, max_words=4)
        snippet = _wiki_lookup(keyphrase)
        if not snippet or "(no Wikipedia" in snippet or "(wikipedia package" in snippet:
            return None
        return snippet[:_EXTERNAL_CHARS]
    except Exception:
        return None


def _extract_keyphrase(text: str, max_words: int = 5) -> str:
    stop = {
        "the", "a", "an", "is", "are", "was", "were", "to", "of", "in",
        "and", "or", "but", "for", "on", "at", "by", "with", "as",
        "that", "this", "these", "those", "it", "its",
    }
    words = [w.strip(".,;:?!\"'()[]") for w in text.split()]
    keep = [w for w in words if w and w.lower() not in stop]
    return " ".join(keep[:max_words]) or text[:50]


def _wiki_lookup(query: str) -> str:
    try:
        import wikipedia  # type: ignore
        try:
            return wikipedia.summary(query, sentences=2, auto_suggest=True, redirect=True)[:600]
        except Exception:
            try:
                hits = wikipedia.search(query, results=1)
                if hits:
                    return wikipedia.summary(hits[0], sentences=2)[:600]
            except Exception:
                pass
        return f"(no Wikipedia article found for query: {query!r})"
    except ImportError:
        return f"(wikipedia package not installed; query was: {query!r})"


# ---------------------------------------------------------------------------
# Citation helpers
# ---------------------------------------------------------------------------

def _citation_tag(cp: ClusterProjection) -> str:
    """Build inline citation tag like [INITIAL_00023 ← SUPPORT_00041 | CRITIQUE_00047]."""
    support_part = ", ".join(cp.support_set[:5])
    dissent_part = ", ".join(cp.dissent_set[:3])
    ver_part = ", ".join(cp.verification_set[:3])

    parts = []
    if support_part:
        parts.append(support_part)
    tag_inner = cp.representative_id
    if parts:
        tag_inner += " ← " + " | ".join(parts)
    if dissent_part:
        tag_inner += f" | {dissent_part}"
    if ver_part:
        tag_inner += f" | {ver_part}"
    return f"[{tag_inner}]"


def _build_citations(
    projection: SynthesisProjection,
    store: SignalStore,
) -> dict:
    """Build citations.json content mapping signal IDs to provenance metadata."""
    result: dict = {}

    all_clusters = (
        projection.surviving
        + projection.contested
        + projection.weakly_supported
        + projection.rejected_by_field
    )

    for cp in all_clusters:
        all_ids = (
            [cp.representative_id]
            + cp.member_ids
            + cp.support_set
            + cp.dissent_set
            + cp.verification_set
        )
        for sid in all_ids:
            if sid in result:
                continue
            sig = store.get(sid)
            if sig is None:
                continue
            result[sid] = {
                "depositor_role": sig.depositor,
                "depositor_agent_id": sig.metadata.get("depositor_agent_id", ""),
                "scout_agent_id": sig.metadata.get("scout_agent_id", ""),
                "partition_origin": _parse_partition_tag_from_sig(sig),
                "deposit_timestamp": sig.timestamp,
                "content_excerpt": sig.content[:200],
                "chunk_ids": sig.metadata.get("chunk_ids", []),
                "parent_id": sig.parent_id,
                "signal_type": sig.type,
                "strength": round(sig.strength, 4),
            }

    return result


def _parse_partition_tag_from_sig(sig) -> str:
    scout_id = sig.metadata.get("scout_agent_id", "")
    if not scout_id:
        return ""
    parts = scout_id.split("_")
    if len(parts) >= 3:
        return f"partition_{parts[2]}"
    return ""


# ---------------------------------------------------------------------------
# Lineage DOT graph
# ---------------------------------------------------------------------------

def _build_lineage_dot(
    projection: SynthesisProjection,
    store: SignalStore,
) -> str:
    """Build a Graphviz DOT string for the surviving + contested DAG."""
    lines = ["digraph lineage {", "  rankdir=TB;", "  node [fontsize=10];"]

    active_clusters = projection.surviving + projection.contested

    node_attrs = {
        INITIAL:      'shape=box style=filled fillcolor="#d0e8ff"',
        SUPPORT:      'shape=ellipse style=filled fillcolor="#d0f0d0"',
        CRITIQUE:     'shape=diamond style=filled fillcolor="#ffd0d0"',
        OBJECTION:    'shape=diamond style=filled fillcolor="#ffb0b0"',
        VERIFICATION: 'shape=hexagon style=filled fillcolor="#ffffd0"',
    }

    seen_nodes: set[str] = set()
    seen_edges: set[tuple] = set()

    def add_node(sig_id: str) -> None:
        if sig_id in seen_nodes:
            return
        seen_nodes.add(sig_id)
        sig = store.get(sig_id)
        if sig is None:
            return
        attr = node_attrs.get(sig.type, "")
        label = f"{sig_id}\\nstr={sig.strength:.2f}"
        lines.append(f'  "{sig_id}" [label="{label}" {attr}];')

    def add_edge(parent_id: str, child_id: str) -> None:
        if (parent_id, child_id) in seen_edges:
            return
        seen_edges.add((parent_id, child_id))
        lines.append(f'  "{child_id}" -> "{parent_id}";')

    for cp in active_clusters:
        all_ids = (
            cp.member_ids
            + cp.support_set
            + cp.dissent_set
            + cp.verification_set
        )
        for sid in all_ids:
            add_node(sid)
            sig = store.get(sid)
            if sig and sig.parent_id:
                add_node(sig.parent_id)
                add_edge(sig.parent_id, sid)

    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _truncate(text: str, max_chars: int = _REPRESENTATIVE_CHARS) -> str:
    text = text.replace("\n", " ").strip()
    if len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text


def _strongest_signal_content(signal_ids: list, store: SignalStore) -> str:
    best = None
    best_strength = -1.0
    for sid in signal_ids:
        s = store.get(sid)
        if s and s.strength > best_strength:
            best_strength = s.strength
            best = s.content
    return best or ""
