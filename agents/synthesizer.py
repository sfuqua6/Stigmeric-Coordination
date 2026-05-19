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
from core.signal_types import (
    INITIAL, SUPPORT, CRITIQUE, CRITIQUE_POSITIVE, CRITIQUE_NEGATIVE,
    OBJECTION, VERIFICATION,
)
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

# Fallback cap when the LLM planner returns nothing usable. The planner
# normally chooses the render set itself based on a structural digest that
# carries no signal content — see _plan_synthesis below. The cap only fires
# when the planner call fails or yields zero valid cluster IDs.
_SECTION_1_FALLBACK_CAP = 5
# Hard cap on Section 3 entries (after merging tail-of-surviving, unverified,
# rejected_by_field, and weakly_supported). Was 5; raised because we now
# route more buckets through this section.
_SECTION_3_RENDER_CAP = 10
# Threshold above which the run-end summary prints a faithfulness warning.
_AUDIT_WARNING_THRESHOLD = 20

# Planner is a single LLM call ahead of per-cluster rendering. Its prompt
# carries STRUCTURAL metadata only (cluster IDs, support / dissent / ver
# counts, scores) — never Signal.content — so the synthesizer can decide
# what to surface without ingesting the full DAG. Per-cluster renderers
# then read the specific signals the plan selected. This satisfies the
# no-leak rule: the planner sees structure, the renderer sees content,
# neither sees other agents' reasoning chains.
_PLAN_MAX_TOKENS = 1500
_PLAN_TEMPERATURE = 0.2

# Prompt-interpreter call. Reads the user's task prompt and extracts the
# structural contract (form, regime, constraints) BEFORE any rendering, so
# downstream stages know whether they're producing a haiku, a function,
# or an argument. Decouples task category (coarse) from output shape (fine).
_INTERPRET_MAX_TOKENS = 600
_INTERPRET_TEMPERATURE = 0.1

# Final integration call for cohesive output strategies. Sees the surviving
# cluster representatives + supports as raw materials and the prompt
# contract as the shape directive. Higher temp than per-cluster (which
# rendered prose paragraphs from a fixed template) because the integration
# call is producing the final user-facing artifact and benefits from a
# little freedom — but capped so coding outputs stay deterministic enough.
_INTEGRATE_MAX_TOKENS = 2500
_INTEGRATE_TEMPERATURE_EXPLORATION = 0.7
_INTEGRATE_TEMPERATURE_OPTIMIZATION = 0.3

# Map task_type to output strategy. Tasks not listed fall back to
# "sectioned" (the original multi-section format). Per the design memo:
# debate/analysis -> sectioned (positions presented separately is the
# whole point); creative/problem_solving -> cohesive_exploration (one
# unified user-facing artifact); coding -> cohesive_optimization (one
# working implementation).
_OUTPUT_STRATEGY_BY_TASK = {
    "debate":          "sectioned",
    "analysis":        "sectioned",
    "problem_solving": "cohesive_exploration",
    "creative":        "cohesive_exploration",
    "coding":          "cohesive_optimization",
}

# Composite cluster priority score used for rendering order.
# Higher support_diversity and verification_score = more credible.
# Higher dissent_pressure = more contested = lower priority for Section 1.
def _cluster_priority(cp) -> float:
    return (cp.support_diversity * max(0.01, cp.verification_score)
            / max(0.01, cp.dissent_pressure))


def _rank_clusters(clusters: list) -> list:
    """Sort clusters by composite priority score descending."""
    return sorted(clusters, key=_cluster_priority, reverse=True)


# Minimum embedding-cosine distance between two clusters picked into the
# Section-1 render set. Pure top-N by priority routinely picked 5
# near-duplicates from the same conceptual cluster family — the post-mortem
# showed all 5 paragraphs restating "the debate over free will...". MMR-style
# diversity-aware selection: each new pick must be at least DIVERSITY_MIN_DIST
# from every already-picked cluster. 0.35 ≈ "topically distinct".
_SECTION_1_DIVERSITY_MIN_DIST = 0.35


def _select_diverse_clusters(
    clusters: list, store, k: int,
    min_dist: float = _SECTION_1_DIVERSITY_MIN_DIST,
) -> tuple[list, list]:
    """Pick up to `k` clusters ranked by priority but spread by embedding distance.

    Greedy MMR: take the highest-priority cluster; for each subsequent slot
    pick the next-highest-priority cluster whose representative embedding is
    >= `min_dist` from EVERY already-picked cluster's representative.
    Clusters skipped for distance reasons are returned in the second list
    (Section-3 tail).

    Falls back to a pure top-N when embeddings aren't available — same
    behavior as before, but the tail list still gets populated so Section 3
    is correct.
    """
    ranked = _rank_clusters(clusters)
    if not ranked:
        return [], []
    picked: list = []
    picked_embs: list = []
    tail: list = []
    for cp in ranked:
        if len(picked) >= k:
            tail.append(cp)
            continue
        emb = store.get_embedding(cp.representative_id)
        if emb is None:
            # No embedding — take it unconditionally if there's a slot,
            # otherwise drop to tail.
            picked.append(cp)
            picked_embs.append(None)
            continue
        too_close = False
        for prev_emb in picked_embs:
            if prev_emb is None:
                continue
            sim = sum(a * b for a, b in zip(emb, prev_emb))
            if (1.0 - sim) < min_dist:
                too_close = True
                break
        if too_close:
            tail.append(cp)
            continue
        picked.append(cp)
        picked_embs.append(emb)
    return picked, tail


def _detect_inter_cluster_contradictions(
    clusters: list, store: SignalStore
) -> list[tuple]:
    """Find pairs of surviving clusters that appear contradictory.

    A pair is flagged as contradictory when:
      - Both have dissent_pressure > 0.3 (both have field opposition)
      - Their representative content shares at least 3 content words
        (same topic) but the dissent signals overlap (same challenger claims)

    Returns list of (cp_a, cp_b) pairs. Small-n heuristic — fast enough.
    """
    contradictions = []
    import re as _re
    _stop = frozenset({"the","a","an","is","are","to","of","in","and","or","it","be"})

    def content_words(text: str) -> set[str]:
        return {w.lower() for w in _re.findall(r"[a-z]+", text.lower())
                if w.lower() not in _stop and len(w) > 3}

    for i, ca in enumerate(clusters):
        if ca.dissent_pressure < 0.3:
            continue
        rep_a = store.get(ca.representative_id)
        if rep_a is None:
            continue
        words_a = content_words(rep_a.content)
        for cb in clusters[i+1:]:
            if cb.dissent_pressure < 0.3:
                continue
            rep_b = store.get(cb.representative_id)
            if rep_b is None:
                continue
            words_b = content_words(rep_b.content)
            shared_topic = words_a & words_b
            shared_dissent = set(ca.dissent_set) & set(cb.dissent_set)
            if len(shared_topic) >= 3 and shared_dissent:
                contradictions.append((ca, cb))
    return contradictions


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
        task_type: Optional[str] = None,
    ) -> tuple[str, dict, str]:
        """Run Layer 1 then Layer 2.

        output_dir: when provided, renderer_audit.json is written there.
        task_type: drives the survival profile + position-taking variants
        in the per-cluster renderer prompts.
        Returns (answer_text, citations_dict, lineage_dot_str).
        """
        self._task_type = task_type
        # Layer 1: pure-Python DAG projection
        projection = build_projection(
            store,
            has_validators=has_validators,
            prior_rejections=prior_rejections,
            prior_consensus=prior_consensus,
            task_type=task_type,
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
            # Bug 3: write a minimal renderer_audit.json on the no-consensus
            # short-circuit so summary.json reads audit_flags=0 (ran clean,
            # zero flags) rather than the -1 sentinel that meant "file
            # missing". Skip the actual faithfulness audit — there's no
            # rendered answer to audit.
            if output_dir is not None:
                _write_no_consensus_audit(output_dir)
            return (
                f"The swarm did not converge on a stable answer for this task.\n\n"
                f"Projection summary: {surv} surviving, {cont} contested, "
                f"{weak} weakly_supported, {rej} rejected_by_field.\n\n"
                f"Consult signals.json and lineage.dot for the underlying signal "
                f"field. You can re-synthesize with different filter thresholds "
                f"via `python synthesize.py <run_dir>`."
            )

        # ------------------------------------------------------------------
        # Stage 1: interpret the user's prompt into a structural contract.
        # Decouples task category (coarse) from output shape (fine). The
        # contract drives Stage 3's cohesive integration call.
        # ------------------------------------------------------------------
        contract = await self._interpret_prompt(self.task_prompt)
        self._prompt_contract = contract  # stashed for downstream + summary.json
        strategy = _OUTPUT_STRATEGY_BY_TASK.get(
            getattr(self, "_task_type", None), "sectioned",
        )
        print(f"[synthesizer] contract: form={contract.get('form')!r} "
              f"regime={contract.get('regime')!r} strategy={strategy!r}")

        sections: list[str] = []

        # ------------------------------------------------------------------
        # Synthesis plan FIRST — the executive summary then references it.
        # ------------------------------------------------------------------
        plan = await self._plan_synthesis(projection, store)
        plan_render_ids: set = set(plan.get("render_full", []))
        plan_section3_ids: set = set(plan.get("section3_only", []))
        merge_groups: list = list(plan.get("merge_groups", []))
        plan_notes: str = plan.get("notes", "")

        # ------------------------------------------------------------------
        # COHESIVE strategies short-circuit the multi-section render path.
        # The integration call produces the actual user-facing artifact;
        # process notes (exec summary, plan, citations) are appended
        # deterministically below.
        # ------------------------------------------------------------------
        if strategy in ("cohesive_exploration", "cohesive_optimization"):
            render_id_list = list(plan_render_ids) or [
                cp.representative_id for cp in
                (_rank_clusters(projection.surviving) or projection.contested)[:5]
            ]
            try:
                if strategy == "cohesive_exploration":
                    artifact = await self._render_cohesive_exploration(
                        contract, projection, store, render_id_list,
                    )
                else:
                    artifact = await self._render_cohesive_optimization(
                        contract, projection, store, render_id_list,
                    )
            except Exception as exc:
                print(f"[synthesizer] cohesive render failed: "
                      f"{type(exc).__name__}: {exc}; "
                      f"falling back to sectioned output")
                strategy = "sectioned"
                artifact = None

            if strategy != "sectioned":
                # Assemble: artifact at the top, then deterministic process
                # notes (exec summary, plan), then citations stamp.
                exec_notes = await self._render_executive_summary(
                    projection, store, plan_notes=plan_notes,
                )
                parts = [artifact]
                parts.append("\n\n---\n\n## PROCESS NOTES\n")
                parts.append(exec_notes)
                answer = "\n\n".join(p for p in parts if p)
                answer = _stamp_citations(answer, projection, store,
                                           merge_groups=merge_groups)
                # Soft audit for cohesive outputs (looser than 4-gram).
                try:
                    audit_flags = _build_cohesive_audit(
                        artifact, contract, projection, store,
                    )
                    if output_dir is not None:
                        _write_faithfulness_audit(audit_flags, output_dir)
                except Exception as exc:
                    print(f"[synthesizer] cohesive audit crashed: "
                          f"{type(exc).__name__}: {exc}")
                    if output_dir is not None:
                        _write_crashed_audit(output_dir, exc)
                return answer

        # ------------------------------------------------------------------
        # SECTIONED strategy (debate/analysis fallback): keep the existing
        # multi-section render path. Executive summary appears first.
        # ------------------------------------------------------------------
        exec_summary = await self._render_executive_summary(
            projection, store, plan_notes=plan_notes,
        )
        if exec_summary:
            sections.append(exec_summary)

        # ------------------------------------------------------------------
        # Section 1: per-cluster rendering, driven by the synthesis plan
        # already produced above (no hard cap on Section-1 size).
        # ------------------------------------------------------------------
        if projection.surviving:
            rendered_surviving = [
                cp for cp in projection.surviving
                if cp.representative_id in plan_render_ids
            ]
            section1_tail = [
                cp for cp in projection.surviving
                if cp.representative_id not in plan_render_ids
            ]
            # Preserve priority order within the planned set.
            rendered_surviving = _rank_clusters(rendered_surviving)
        else:
            rendered_surviving = []
            section1_tail = []
        if rendered_surviving:
            fragments: list[str] = []
            for cp in rendered_surviving:
                fragment = await self._render_cluster_position(cp, store)
                if fragment:
                    # Ensure paragraph ends with terminal punctuation.
                    if fragment and fragment[-1] not in ".!?\")'":
                        fragment += "."
                    fragments.append(fragment)
            if fragments:
                sections.append(
                    "## 1. POSITION SYNTHESIS\n\n" + "\n\n".join(fragments)
                )

        # ------------------------------------------------------------------
        # Section 2: Open questions and dissent — per contested cluster AND
        # per surviving cluster that attracted any dissent.
        # Also flags inter-cluster contradictions detected above.
        # ------------------------------------------------------------------
        contradictions = _detect_inter_cluster_contradictions(
            projection.surviving, store
        )
        dissent_candidates: list[ClusterProjection] = list(projection.contested)
        dissent_candidates += [
            cp for cp in projection.surviving if cp.dissent_set
        ]
        if dissent_candidates or contradictions:
            fragments = []
            for cp in dissent_candidates:
                fragment = await self._render_cluster_dissent(cp, store)
                if fragment:
                    if fragment[-1] not in ".!?\")'":
                        fragment += "."
                    fragments.append(fragment)
            for ca, cb in contradictions:
                fragments.append(
                    f"[INTER-CLUSTER CONTRADICTION] Clusters "
                    f"[{ca.representative_id}] and [{cb.representative_id}] "
                    f"share topic content and share dissent signals "
                    f"({', '.join(set(ca.dissent_set) & set(cb.dissent_set))[:3]}). "
                    f"These may be two framings of the same contested claim."
                )
            if fragments:
                sections.append(
                    "## 2. OPEN QUESTIONS AND DISSENT\n\n" + "\n\n".join(fragments)
                )

        # ------------------------------------------------------------------
        # Section 3: Considered and filtered — deterministic, no LLM call.
        # Now aggregates four buckets in priority order:
        #   1. rejected_by_field      (most diagnostic — explicit field rejection)
        #   2. unverified             (would have survived but lacked credibility)
        #   3. tail-of-surviving      (clusters the planner demoted to section 3)
        #   4. weakly_supported       (insufficient distinct supporters)
        # Each sorted by descending verification_score within its bucket.
        # ------------------------------------------------------------------
        rej_sorted = sorted(
            projection.rejected_by_field,
            key=lambda c: c.verification_score, reverse=True,
        )
        unver_sorted = sorted(
            projection.unverified,
            key=lambda c: c.verification_score, reverse=True,
        )
        weak_sorted = sorted(
            projection.weakly_supported,
            key=lambda c: c.verification_score, reverse=True,
        )
        # section1_tail is already ranked by priority.
        filtered = (rej_sorted + unver_sorted + section1_tail + weak_sorted)[
            :_SECTION_3_RENDER_CAP
        ]
        if filtered:
            lines: list[str] = []
            for cp in filtered:
                rep = store.get(cp.representative_id)
                content = _truncate(rep.content, _SUMMARY_CHARS) if rep else cp.representative_id
                if cp.status == "rejected_by_field":
                    reason = f"rejected: dissent_pressure={cp.dissent_pressure:.2f} > 1.5"
                elif cp.status == "unverified":
                    reason = (f"held: no verification, no dissent, "
                              f"support_diversity={cp.support_diversity} < 4")
                elif cp.status == "surviving":
                    reason = (f"tail (planner demoted): "
                              f"support_diversity={cp.support_diversity}, "
                              f"verification_score={cp.verification_score:.2f}")
                else:
                    reason = f"filtered: support_diversity={cp.support_diversity} < 3"
                lines.append(f"- [{cp.representative_id}] {content}  ({reason})")
            sections.append(
                "## 3. CONSIDERED AND FILTERED\n\n"
                + "\n".join(lines)
            )

        # ------------------------------------------------------------------
        # Assemble sections
        # ------------------------------------------------------------------
        answer = "\n\n".join(sections)

        # Section 4 — Citations: deterministic stamp appended to the answer.
        # Pass merge_groups so the reader can see when the planner treated
        # two clusters as a single position.
        answer = _stamp_citations(answer, projection, store,
                                   merge_groups=merge_groups)

        # ------------------------------------------------------------------
        # Post-hoc faithfulness audit
        # ------------------------------------------------------------------
        # Bug 3: wrap so an audit crash writes a -2 sentinel + error message
        # rather than leaving renderer_audit.json absent (which made summary
        # report audit_flags=-1 ambiguously). Sentinel semantics now:
        #     0  = audit ran clean
        #     N>=1 = audit ran, found N flags
        #     -2 = audit crashed (audit_error field carries the reason)
        try:
            audit_flags = _build_faithfulness_audit(answer, projection, store)
            if output_dir is not None:
                _write_faithfulness_audit(audit_flags, output_dir)
            elif audit_flags:
                print(
                    f"[synthesizer] faithfulness audit: {len(audit_flags)} flag(s) "
                    f"(pass output_dir to write renderer_audit.json)"
                )
            # Loud end-of-run warning when the audit is heavily flagged. 20+
            # is the empirical threshold past which a renderer pass is
            # producing more noise than signal and the synthesis prose
            # should not be trusted without manual review.
            if len(audit_flags) >= _AUDIT_WARNING_THRESHOLD:
                audit_path = (
                    str(output_dir / "renderer_audit.json")
                    if output_dir is not None else "(audit not written — no output_dir)"
                )
                print(
                    f"[synthesizer] WARNING: {len(audit_flags)} faithfulness "
                    f"flags (>= {_AUDIT_WARNING_THRESHOLD}). Synthesis prose "
                    f"may diverge from cited signals; review "
                    f"{audit_path} before citing this run."
                )
        except Exception as exc:
            print(f"[synthesizer] faithfulness audit crashed: "
                  f"{type(exc).__name__}: {exc}")
            if output_dir is not None:
                _write_crashed_audit(output_dir, exc)

        return answer

    # -----------------------------------------------------------------------
    # Stage 1: Prompt interpreter — extracts the structural contract
    # -----------------------------------------------------------------------

    async def _interpret_prompt(self, user_prompt: str) -> dict:
        """Extract a structured contract from the user's task prompt.

        Decouples *task category* (coarse: creative/coding/debate/...) from
        *output form* (fine: haiku vs short story vs slogan; binary-search
        function vs JSON parser; etc.). Same swarm behavior produces the
        surviving ideas in all cases; this contract is what tells the
        synthesizer what *shape* to render them in.

        Returns a dict with at least:
            regime:     "exploration" | "optimization"
            form:       short string naming the target form
            structural: list of hard constraints (line count, signature,
                        complexity bound, format markers)
            soft:       list of style/voice/theme cues
            length_hint: "short" | "medium" | "long"

        Falls back to a heuristic contract on parse failure so downstream
        rendering can still proceed.
        """
        prompt = (
            f"Read the following user prompt and extract its structural "
            f"contract as JSON. Two regimes exist:\n"
            f"  - exploration: no oracle answer; reader wants novel/"
            f"interesting ideas (haiku, story, essay, slogan, argument, "
            f"action plan, analysis).\n"
            f"  - optimization: a correct answer or class of correct "
            f"answers exists (code, math, formal proof).\n\n"
            f"---USER PROMPT---\n{user_prompt}\n---END USER PROMPT---\n\n"
            f"Reply with EXACTLY this JSON object (no other text):\n"
            f'{{"regime": "exploration" | "optimization",\n'
            f'  "form": "<short label, e.g. haiku | short_story | slogan | '
            f'argument | function | action_plan | analysis>",\n'
            f'  "structural": ["<hard constraint>", ...],\n'
            f'  "soft":       ["<style/theme cue>", ...],\n'
            f'  "length_hint": "short" | "medium" | "long",\n'
            f'  "audience":   "<one short phrase>"}}\n\n'
            f"Hard constraints are things you can VERIFY post-hoc (line "
            f"count, syllable structure, function signature, presence of "
            f"a section, language requirement). Soft cues are stylistic "
            f"directions (tone, voice, themes) that the renderer must "
            f"interpret. If the prompt is vague, infer the most likely "
            f"form rather than listing nothing.\n\n"
            f"CONTRACT:"
        )
        try:
            raw = await self.llm.generate(
                prompt, role=self.ROLE,
                max_tokens=_INTERPRET_MAX_TOKENS,
                temperature=_INTERPRET_TEMPERATURE,
            )
        except Exception as exc:
            print(f"[synthesizer] prompt-interpret call failed: "
                  f"{type(exc).__name__}: {exc}")
            return self._fallback_contract(user_prompt)
        block = _extract_json_block(raw or "")
        if block is None:
            return self._fallback_contract(user_prompt)
        try:
            contract = json.loads(_repair_json(block))
        except Exception:
            return self._fallback_contract(user_prompt)
        # Coerce / sanitize fields. Anything missing gets a default.
        out = {
            "regime":      str(contract.get("regime", "exploration")).strip().lower(),
            "form":        str(contract.get("form", "free_response")).strip(),
            "structural":  [str(x) for x in contract.get("structural", []) if x],
            "soft":        [str(x) for x in contract.get("soft", []) if x],
            "length_hint": str(contract.get("length_hint", "medium")).strip().lower(),
            "audience":    str(contract.get("audience", "general")).strip(),
        }
        if out["regime"] not in ("exploration", "optimization"):
            out["regime"] = "exploration"
        if out["length_hint"] not in ("short", "medium", "long"):
            out["length_hint"] = "medium"
        return out

    def _fallback_contract(self, user_prompt: str) -> dict:
        """Heuristic contract used when the interpreter call fails.

        Inferred entirely from the task_type recorded on self._task_type.
        Less precise than the LLM extraction but always available.
        """
        tt = getattr(self, "_task_type", None) or "analysis"
        regime = "optimization" if tt == "coding" else "exploration"
        form_by_task = {
            "creative":        "free_creative",
            "coding":          "function",
            "problem_solving": "action_plan",
            "debate":          "argument",
            "analysis":        "analysis",
        }
        return {
            "regime":      regime,
            "form":        form_by_task.get(tt, "free_response"),
            "structural":  [],
            "soft":        [],
            "length_hint": "medium",
            "audience":    "general",
        }

    # -----------------------------------------------------------------------
    # Planner: structural digest -> render plan (no content ingestion)
    # -----------------------------------------------------------------------

    async def _plan_synthesis(
        self, projection: SynthesisProjection, store: SignalStore,
    ) -> dict:
        """Ask the LLM to plan the synthesis from STRUCTURE alone.

        The planner is shown a digest of the surviving + contested clusters:
        cluster IDs, type, support / dissent / verification *counts* and
        *scores*, support_depth, and a short preview (first 80 chars of the
        representative claim — just enough to let the planner topic-cluster
        without consuming the full DAG). It is NOT shown the support /
        dissent / verification signal contents.

        It returns JSON:
            {"render_full":   [<cluster_id>, ...],
             "section3_only": [<cluster_id>, ...],
             "merge_groups":  [[<cluster_id>, <cluster_id>], ...],
             "notes":         "<one-sentence overview>"}

        `render_full` clusters get a full Section-1 paragraph via the
        per-cluster renderer. `section3_only` clusters are demoted. Merge
        groups are recorded in the citation block so the reader can see
        the planner identified them as one position. Falls back to the
        legacy diversity-aware top-N selection if the call or parse fails.

        No leak rule: the digest carries only IDs and aggregate counts,
        not other agents' reasoning. The per-cluster renderer is the only
        layer that reads Signal.content.
        """
        candidates = list(projection.surviving) + list(projection.contested)
        if not candidates:
            return {"render_full": [], "section3_only": [], "merge_groups": [], "notes": ""}

        # Build the digest. ID + short preview + structural metrics.
        lines: list[str] = []
        for cp in candidates:
            rep = store.get(cp.representative_id)
            preview = _truncate(rep.content, 80) if rep else "(no rep)"
            lines.append(
                f"- {cp.representative_id}  status={cp.status}  "
                f"n_supports={len(cp.support_set)}  "
                f"n_dissent={len(cp.dissent_set)}  "
                f"n_verifications={len(cp.verification_set)}  "
                f"support_diversity={cp.support_diversity}  "
                f"support_depth={cp.support_depth}  "
                f"verification_score={cp.verification_score:.2f}  "
                f"dissent_pressure={cp.dissent_pressure:.2f}  "
                f"preview=\"{preview}\""
            )
        digest = "\n".join(lines)

        prompt = (
            f"TASK: {self.task_prompt}\n\n"
            f"You are planning the structure of a synthesis. You see ONLY a "
            f"structural digest of claim clusters: their IDs, counts of "
            f"supporting / dissenting / verifying signals, scores, and an "
            f"80-character preview. You do NOT see the underlying signals' "
            f"content — that gets rendered in a separate pass per cluster.\n\n"
            f"Your job: decide which clusters deserve a full paragraph in "
            f"Section 1 (POSITION SYNTHESIS) and which can be demoted to "
            f"Section 3 (CONSIDERED AND FILTERED). Pick clusters that are:\n"
            f"  1. Topically distinct from each other (avoid 5 paragraphs "
            f"     restating the same position).\n"
            f"  2. Well-supported (high support_diversity, support_depth >= 2).\n"
            f"  3. Verified where possible (verification_score > 0.3).\n"
            f"  4. Contested clusters are valuable — surface them.\n\n"
            f"If two clusters look like the same position by their previews, "
            f"name them in a merge_group rather than rendering both.\n\n"
            f"You can choose up to {len(candidates)} clusters for "
            f"render_full — there is NO fixed cap. The downstream renderer "
            f"will render exactly what you list. Prefer a slightly smaller, "
            f"sharper set over a larger noisy one.\n\n"
            f"---DIGEST---\n{digest}\n---END DIGEST---\n\n"
            f"Reply with EXACTLY this JSON object (no other text):\n"
            f'{{"render_full":   ["<cluster_id>", ...],\n'
            f'  "section3_only": ["<cluster_id>", ...],\n'
            f'  "merge_groups":  [["<cluster_id>", "<cluster_id>"], ...],\n'
            f'  "notes": "<one sentence overview of the plan>"}}\n\n'
            f"PLAN:"
        )

        try:
            raw = await self.llm.generate(
                prompt, role=self.ROLE,
                max_tokens=_PLAN_MAX_TOKENS,
                temperature=_PLAN_TEMPERATURE,
            )
        except Exception as exc:
            print(f"[synthesizer] plan call failed: {type(exc).__name__}: {exc}")
            return self._fallback_plan(candidates, store)
        return self._parse_plan(raw, candidates, store)

    def _parse_plan(
        self, raw: str, candidates: list, store: SignalStore,
    ) -> dict:
        """Parse the planner JSON. Falls back to legacy selection on failure."""
        block = _extract_json_block(raw or "")
        if block is None:
            print(f"[synthesizer] plan parse: no JSON block; using fallback")
            return self._fallback_plan(candidates, store)
        try:
            plan = json.loads(_repair_json(block))
        except Exception as exc:
            print(f"[synthesizer] plan JSON parse error: {exc}; using fallback")
            return self._fallback_plan(candidates, store)

        valid_ids = {cp.representative_id for cp in candidates}
        render_full = [cid for cid in plan.get("render_full", []) if cid in valid_ids]
        section3_only = [cid for cid in plan.get("section3_only", []) if cid in valid_ids]
        merge_groups = []
        for grp in plan.get("merge_groups", []):
            if isinstance(grp, list):
                cleaned = [cid for cid in grp if cid in valid_ids]
                if len(cleaned) >= 2:
                    merge_groups.append(cleaned)
        notes = str(plan.get("notes", "")).strip()

        # The plan must contain *something* renderable; if not, fall back.
        if not render_full:
            print(f"[synthesizer] plan empty after validation; using fallback")
            return self._fallback_plan(candidates, store)

        # Implicit demotion: any candidate not in render_full and not in a
        # merge group's secondary slot drops to section3.
        used = set(render_full)
        for grp in merge_groups:
            used.update(grp)
        section3_set = set(section3_only)
        for cp in candidates:
            if cp.representative_id not in used and cp.representative_id not in section3_set:
                section3_set.add(cp.representative_id)

        return {
            "render_full": render_full,
            "section3_only": sorted(section3_set),
            "merge_groups": merge_groups,
            "notes": notes,
        }

    def _fallback_plan(self, candidates: list, store: SignalStore) -> dict:
        """Legacy diversity-aware top-N selection. Used when the LLM plan fails."""
        surviving = [c for c in candidates if c.status == "surviving"]
        picked, tail = _select_diverse_clusters(
            surviving, store, _SECTION_1_FALLBACK_CAP,
        )
        return {
            "render_full":   [cp.representative_id for cp in picked],
            "section3_only": [cp.representative_id for cp in tail
                              + [c for c in candidates if c.status != "surviving"]],
            "merge_groups":  [],
            "notes":         "(fallback: diversity-aware top-N — planner unavailable)",
        }

    # -----------------------------------------------------------------------
    # Executive summary
    # -----------------------------------------------------------------------

    async def _render_executive_summary(
        self, projection: SynthesisProjection, store: SignalStore,
        plan_notes: str = "",
    ) -> str:
        """Deterministic executive summary built directly from the projection.

        Previously this was an LLM call that produced "Fourteen distinct
        clusters" when there were 41, and routinely inverted the polarity
        of contested clusters in prose. Counting and polarity belong in
        Python: the projection has the numbers, so we render them straight.

        Kept async to preserve the existing await at the call site, but no
        LLM call is made — runtime is microseconds.
        """
        n_surv = len(projection.surviving)
        n_cont = len(projection.contested)
        n_unver = len(projection.unverified)
        n_rej = len(projection.rejected_by_field)
        n_weak = len(projection.weakly_supported)
        total = n_surv + n_cont + n_unver + n_rej + n_weak

        breakdown_parts: list[str] = []
        breakdown_parts.append(f"{n_surv} survived field pressure")
        breakdown_parts.append(f"{n_cont} contested")
        if n_unver:
            breakdown_parts.append(f"{n_unver} held as unverified")
        if n_rej:
            breakdown_parts.append(f"{n_rej} rejected")
        if n_weak:
            breakdown_parts.append(f"{n_weak} weakly supported")
        breakdown = ", ".join(breakdown_parts)

        lines = [
            "## EXECUTIVE SUMMARY",
            "",
            f"Of {total} claim cluster(s) projected from the signal field: {breakdown}.",
        ]

        if projection.surviving:
            ranked = _rank_clusters(projection.surviving)
            top = ranked[0]
            rep = store.get(top.representative_id)
            if rep:
                excerpt = _truncate(rep.content, 160)
                lines.append(
                    f"Strongest surviving cluster [{top.representative_id}] "
                    f"(support_diversity={top.support_diversity}, "
                    f"verification_score={top.verification_score:.2f}, "
                    f"dissent_pressure={top.dissent_pressure:.2f}): {excerpt}"
                )

        if projection.contested:
            top_cont = max(
                projection.contested,
                key=lambda c: c.dissent_pressure,
            )
            rep_c = store.get(top_cont.representative_id)
            if rep_c:
                excerpt = _truncate(rep_c.content, 160)
                lines.append(
                    f"Most contested cluster [{top_cont.representative_id}] "
                    f"(dissent_pressure={top_cont.dissent_pressure:.2f}): {excerpt}"
                )

        # Planner overview — surface the structural reasoning the planner
        # used, so the reader can see *why* this many paragraphs (vs. some
        # other set of clusters). Empty when the planner fell back.
        if plan_notes:
            lines.append("")
            lines.append(f"Synthesis plan: {plan_notes}")

        return "\n".join(lines)

    # -----------------------------------------------------------------------
    # Stage 3: cohesive output — produces the user-facing artifact
    # -----------------------------------------------------------------------

    def _gather_raw_materials(
        self, projection: SynthesisProjection, store: SignalStore,
        cluster_ids: list[str],
    ) -> list[dict]:
        """Pack cluster reps + their best supports as integration-call input.

        Returns a list of {id, rep_content, support_excerpts: [...]} dicts.
        Used by both cohesive_exploration and cohesive_optimization. Reads
        only Signal.content — no reasoning chains, no metadata that could
        leak deposit ordering.
        """
        materials: list[dict] = []
        for cid in cluster_ids:
            cp = next(
                (c for c in projection.surviving + projection.contested
                 if c.representative_id == cid),
                None,
            )
            if cp is None:
                continue
            rep = store.get(cp.representative_id)
            if rep is None:
                continue
            supports: list[str] = []
            for sid in cp.support_set[:4]:
                s = store.get(sid)
                if s and s.content:
                    supports.append(_truncate(s.content, _SUPPORT_CHARS))
                    store.mark_read(sid)
            store.mark_read(cp.representative_id)
            materials.append({
                "id": cp.representative_id,
                "rep_content": _truncate(rep.content, _REPRESENTATIVE_CHARS),
                "support_excerpts": supports,
                "verification_score": cp.verification_score,
                "support_diversity": cp.support_diversity,
            })
        return materials

    def _format_materials_block(self, materials: list[dict]) -> str:
        """Render raw materials into a labeled block the integration call reads."""
        if not materials:
            return "(no raw materials surfaced — synthesizer must fall back to the task prompt alone)"
        lines: list[str] = []
        for m in materials:
            lines.append(f"--- THREAD [{m['id']}] (support_diversity="
                          f"{m['support_diversity']}, "
                          f"verification_score={m['verification_score']:.2f}) ---")
            lines.append(m["rep_content"])
            for ex in m["support_excerpts"]:
                lines.append(f"  · {ex}")
            lines.append("")
        return "\n".join(lines).rstrip()

    async def _render_cohesive_exploration(
        self,
        contract: dict,
        projection: SynthesisProjection,
        store: SignalStore,
        render_ids: list[str],
    ) -> str:
        """Produce ONE unified artifact (haiku/story/plan/etc) from cluster threads.

        The integration call sees:
          - the prompt contract (form, structural constraints, soft cues)
          - the raw materials block (surviving cluster reps + supports)
          - a sharp instruction to produce THE THING, not a description of it.

        Single LLM call. Output is the final user-facing artifact, with
        process notes appended deterministically below. Faithfulness for
        cohesive outputs is graded by the soft audit (see
        _build_cohesive_audit) — content-word overlap and structural
        plausibility, not 4-gram exact match.
        """
        materials = self._gather_raw_materials(projection, store, render_ids)
        materials_block = self._format_materials_block(materials)

        form = contract.get("form", "free_response")
        structural = contract.get("structural", []) or ["(none explicit)"]
        soft = contract.get("soft", []) or ["(none explicit)"]
        length_hint = contract.get("length_hint", "medium")

        struct_block = "\n".join(f"  · {x}" for x in structural)
        soft_block = "\n".join(f"  · {x}" for x in soft)

        prompt = (
            f"USER PROMPT: {self.task_prompt}\n\n"
            f"You are producing the FINAL user-facing artifact. Do not "
            f"describe what the artifact would contain — write the artifact "
            f"itself. The swarm has already explored the conceptual space "
            f"and surfaced the threads below; your job is to integrate them "
            f"into a single coherent work that matches the prompt's form.\n\n"
            f"FORM: {form}\n"
            f"LENGTH HINT: {length_hint}\n\n"
            f"HARD STRUCTURAL CONSTRAINTS (must satisfy):\n{struct_block}\n\n"
            f"SOFT CUES (style/voice/themes):\n{soft_block}\n\n"
            f"RAW MATERIALS — surviving threads from the swarm's exploration. "
            f"These are NOT meant to be quoted verbatim; they are the kernels "
            f"of ideas you should integrate, transform, or build on. The "
            f"final artifact should USE these ideas, not list them.\n\n"
            f"{materials_block}\n\n"
            f"RULES:\n"
            f"  1. Produce the artifact itself — no preface, no 'Here is...', "
            f"no meta-commentary, no markdown headers unless the form demands them.\n"
            f"  2. Honor every hard structural constraint exactly.\n"
            f"  3. Integrate at least two distinct threads — don't just paraphrase "
            f"the strongest one.\n"
            f"  4. Do NOT invent named entities, citations, or quotes that aren't "
            f"in the raw materials.\n"
            f"  5. No bracketed cluster IDs (e.g. [INITIAL_00023]) in the final "
            f"artifact — those are process metadata.\n\n"
            f"ARTIFACT:"
        )
        raw = await self.llm.generate(
            prompt, role=self.ROLE,
            max_tokens=_INTEGRATE_MAX_TOKENS,
            temperature=_INTEGRATE_TEMPERATURE_EXPLORATION,
        )
        return strip_reasoning(raw.strip())

    async def _render_cohesive_optimization(
        self,
        contract: dict,
        projection: SynthesisProjection,
        store: SignalStore,
        render_ids: list[str],
    ) -> str:
        """Produce ONE working implementation from candidate approaches.

        Optimization regime: a correct answer exists. The swarm's surviving
        clusters are candidate approaches; the integration call selects the
        best one (or merges them) and emits a working solution that satisfies
        the spec. Same single-call shape as the exploration path, but lower
        temperature and a spec-shaped instruction.
        """
        materials = self._gather_raw_materials(projection, store, render_ids)
        materials_block = self._format_materials_block(materials)

        form = contract.get("form", "function")
        structural = contract.get("structural", []) or ["(none explicit)"]
        soft = contract.get("soft", []) or ["(none explicit)"]

        struct_block = "\n".join(f"  · {x}" for x in structural)
        soft_block = "\n".join(f"  · {x}" for x in soft)

        prompt = (
            f"USER PROMPT: {self.task_prompt}\n\n"
            f"You are producing the FINAL implementation. The swarm has "
            f"surfaced candidate approaches below — your job is to select "
            f"the best one (or merge compatible elements) and write a "
            f"working, correct solution. The output is the implementation "
            f"itself, not commentary about it.\n\n"
            f"FORM: {form}\n\n"
            f"HARD SPEC (signature, complexity bounds, language, behavior):\n"
            f"{struct_block}\n\n"
            f"SOFT GUIDANCE (idioms, style):\n{soft_block}\n\n"
            f"CANDIDATE APPROACHES — surviving threads from the swarm's "
            f"exploration. Pick the strongest, or merge compatible ideas. "
            f"Do NOT include every approach — produce ONE solution.\n\n"
            f"{materials_block}\n\n"
            f"RULES:\n"
            f"  1. Output the code/solution itself — no preface, no 'Here is...', "
            f"no English commentary outside docstrings/comments.\n"
            f"  2. Match the hard spec exactly (signature, types, complexity).\n"
            f"  3. Include a brief docstring stating the algorithm + complexity.\n"
            f"  4. Handle edge cases the swarm surfaced.\n"
            f"  5. No bracketed cluster IDs in the final code.\n\n"
            f"IMPLEMENTATION:"
        )
        raw = await self.llm.generate(
            prompt, role=self.ROLE,
            max_tokens=_INTEGRATE_MAX_TOKENS,
            temperature=_INTEGRATE_TEMPERATURE_OPTIMIZATION,
        )
        return strip_reasoning(raw.strip())

    # -----------------------------------------------------------------------
    # Per-cluster render helpers
    # -----------------------------------------------------------------------

    async def _render_cluster_position(
        self, cp: ClusterProjection, store: SignalStore
    ) -> str:
        """Render a one-paragraph position statement for one surviving cluster.

        Prompt is anti-parametric: it forbids framing introductions (the
        "the debate over X..." pathology where every paragraph restates the
        same setup) and forces the paragraph to LEAD with the cluster's
        specific claim. On non-factual tasks (debate / analysis /
        problem_solving) it also explicitly asks the renderer to characterize
        which way the supplied evidence actually points rather than hedging
        with "remains contentious."
        """
        rep = store.get(cp.representative_id)
        if rep is None:
            return ""

        content = _truncate(rep.content, _REPRESENTATIVE_CHARS)
        # Only annotate "not externally verified" on factual tasks. For debate
        # / analysis / problem_solving the projection no longer flags unverified
        # for absent web confirmation (sources don't corroborate interpretive
        # claims) — appending the phrase would just be reflexive hedging.
        task_type = getattr(self, "_task_type", None)
        is_non_factual = task_type in {"debate", "analysis", "problem_solving", "creative"}
        unver_note = (
            " (not externally verified)"
            if (cp.unverified and not is_non_factual) else ""
        )

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

        # Dissent injection: if surviving cluster has meaningful field pressure,
        # include the strongest dissent signal and ask for an acknowledgement.
        dissent_block = ""
        if cp.dissent_pressure > 0.5 and cp.dissent_set:
            strongest = _strongest_signal_content(cp.dissent_set, store)
            if strongest:
                dissent_block = (
                    f"\nCounter-position (dissent_pressure={cp.dissent_pressure:.2f}): "
                    f"{_truncate(strongest, _DISSENT_CHARS)}\n"
                    f"Briefly acknowledge this counter-position in one sentence "
                    f"at the end of the paragraph.\n"
                )

        # Position-taking instruction varies by task type. Non-factual tasks
        # need an actual stance on the supplied evidence; factual tasks need
        # to stick to what's verifiable. Both forbid the "the debate over X"
        # framing intro that produced 5 near-duplicate paragraphs.
        if is_non_factual:
            stance_instruction = (
                f"Open the paragraph with the specific position this cluster "
                f"advances — do NOT begin with framing prose like 'The debate "
                f"over...', 'The question of...', 'X is a contested issue...'. "
                f"State what the supplied evidence ACTUALLY argues, then trace "
                f"why the supporting signals back it up. If the evidence leans "
                f"toward one side, say so; do NOT default to 'remains contentious' "
                f"unless the supplied signals themselves are split."
            )
        else:
            stance_instruction = (
                f"Open the paragraph with the specific claim — do NOT begin "
                f"with framing prose ('The debate over...', 'X has long been...'). "
                f"State the claim directly and trace why the supporting "
                f"signals back it up."
            )

        prompt = (
            f"TASK: {self.task_prompt}\n\n"
            f"Synthesize the following surviving claim into one focused "
            f"paragraph. Cite [{cp.representative_id}] inline.\n\n"
            f"HARD RULES (failure to follow = the paragraph is unusable):\n"
            f"  1. Use ONLY the supplied claim, supporting evidence, and "
            f"counter-position below. Do NOT introduce concepts, examples, "
            f"theories, or arguments not present in the input (no parametric "
            f"knowledge bleed — no quantum-physics gestures, no thinkers not "
            f"already cited).\n"
            f"  2. {stance_instruction}\n"
            f"  3. Do not repeat the task framing across paragraphs.\n\n"
            f"Claim [{cp.representative_id}]: {content}{unver_note}\n"
            f"support_diversity={cp.support_diversity}  "
            f"verification_score={cp.verification_score:.2f}\n"
            f"{support_block}"
            f"{dissent_block}"
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
    merge_groups: Optional[list] = None,
) -> str:
    """Append a deterministic CITATIONS block to the rendered answer.

    Each surviving/contested cluster gets one entry:
        CLAIM [INITIAL_00023]:
          Supports:      SUPPORT_00041, SUPPORT_00058
          Challenges:    CRITIQUE_00047
          Verifications: VERIFICATION_00062
          Partition:     partition_2
          Coverage:      2 partition(s) contributed to this cluster

    When the planner identified merge groups, those appear as a trailing
    block so the reader sees which clusters were treated as one position.
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

    # Merge groups: planner-identified equivalent clusters.
    if merge_groups:
        lines.append("MERGED POSITIONS (planner identified as same claim):")
        for grp in merge_groups:
            lines.append(f"  - {' <=> '.join(grp)}")
        lines.append("")

    return answer.rstrip() + "\n".join(lines)


# ---------------------------------------------------------------------------
# Faithfulness audit
# ---------------------------------------------------------------------------

_CITATION_RE = re.compile(r"\[([A-Z]+_\d+)\]")


def _build_cohesive_audit(
    artifact: str,
    contract: dict,
    projection: SynthesisProjection,
    store: SignalStore,
) -> list[dict]:
    """Soft audit for cohesive outputs (creative / coding / problem_solving).

    The hard 4-gram overlap audit doesn't fit a haiku — by design the
    artifact transforms cluster content rather than quoting it. Instead
    we check:

      1. content_word_overlap_low: fewer than K distinct content words from
         surviving cluster reps appear in the artifact. Flags total
         hallucination — the integration call ignored its source material.
      2. structural_violation: hard constraints from contract.structural
         that can be checked deterministically (line count, presence of
         a specific token, expected length range).
      3. fabricated_id_in_artifact: a bracketed [INITIAL_XXXXX]-style tag
         leaked into the artifact (process metadata in user-facing output).
      4. length_implausible: artifact is way shorter / longer than the
         length_hint suggests (e.g. a "haiku" that's 500 chars).

    Returns the same flag-list shape as _build_faithfulness_audit so the
    audit writer / summary.json reader doesn't need to branch.
    """
    flags: list[dict] = []
    _STOP = frozenset({
        "the","a","an","is","are","to","of","in","and","or","it","be",
        "that","this","with","for","on","as","by","at","from","but","not",
    })
    _CITATION_TAG = re.compile(r"\[[A-Z]+_\d+\]")

    rep_word_pool: set[str] = set()
    for cp in projection.surviving + projection.contested:
        rep = store.get(cp.representative_id)
        if not rep:
            continue
        for w in re.findall(r"[a-z]{4,}", rep.content.lower()):
            if w not in _STOP:
                rep_word_pool.add(w)

    artifact_words: set[str] = set()
    for w in re.findall(r"[a-z]{4,}", (artifact or "").lower()):
        if w not in _STOP:
            artifact_words.add(w)
    overlap = rep_word_pool & artifact_words
    # Threshold: at least 4 distinct content words from surviving threads
    # should appear in a non-trivial artifact. Haikus often dip below this;
    # we only flag if the artifact is longer than ~100 chars (i.e. not a
    # form where lexical brevity is the whole point).
    if len(artifact or "") > 100 and len(overlap) < 4:
        flags.append({
            "issue": "content_word_overlap_low",
            "overlap_count": len(overlap),
            "rep_pool_size": len(rep_word_pool),
            "artifact_length": len(artifact or ""),
        })

    tag_hits = _CITATION_TAG.findall(artifact or "")
    if tag_hits:
        flags.append({
            "issue": "fabricated_id_in_artifact",
            "tags": tag_hits[:10],
        })

    length_hint = contract.get("length_hint", "medium")
    artifact_len = len(artifact or "")
    # Loose per-hint bounds. Calibrated against form-typical lengths;
    # crosses-the-line cases (a "short" creative artifact 2000+ chars,
    # or a "long" one under 200) are the ones worth surfacing.
    if length_hint == "short" and artifact_len > 1500:
        flags.append({
            "issue": "length_implausible",
            "detail": f"length={artifact_len} exceeds 'short' bound (1500)",
        })
    elif length_hint == "long" and artifact_len < 300:
        flags.append({
            "issue": "length_implausible",
            "detail": f"length={artifact_len} below 'long' floor (300)",
        })

    # Form-specific structural checks. Conservative — only fire on signals
    # we can verify cheaply and unambiguously.
    form = contract.get("form", "").lower()
    if form == "haiku":
        # A haiku is three lines. We don't enforce syllable count (that
        # requires phonetic analysis), but line count we can check.
        non_empty_lines = [ln for ln in (artifact or "").splitlines() if ln.strip()]
        if non_empty_lines and len(non_empty_lines) != 3:
            flags.append({
                "issue": "structural_violation",
                "form": "haiku",
                "detail": f"expected 3 non-empty lines, got {len(non_empty_lines)}",
            })
    elif form in ("function", "code"):
        # For code: try ast.parse if it looks like Python.
        if any(tok in (artifact or "") for tok in ("def ", "class ", "import ")):
            try:
                import ast as _ast
                _ast.parse(artifact or "")
            except SyntaxError as exc:
                flags.append({
                    "issue": "structural_violation",
                    "form": form,
                    "detail": f"python syntax error: {exc}",
                })
    return flags


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

    # Track which section each paragraph belongs to so prose-only checks
    # (truncation, short-paragraph) don't fire on Section 3 list items or
    # Section 4 citation blocks — both are deterministic structured output
    # that lacks terminal punctuation by design and produced ~22 false
    # positives per run before this gate.
    paragraph_sections: list[str] = []
    current_section = "preamble"
    for para in paragraphs:
        # Detect section headers. Synthesizer emits them as a Markdown H2
        # with a digit prefix ("## 1. POSITION SYNTHESIS") or the literal
        # "## EXECUTIVE SUMMARY".
        stripped = para.lstrip()
        if stripped.startswith("## EXECUTIVE SUMMARY") or stripped.startswith("# EXECUTIVE SUMMARY"):
            current_section = "exec_summary"
        elif stripped.startswith("## 1.") or "POSITION SYNTHESIS" in stripped.split("\n", 1)[0]:
            current_section = "section_1"
        elif stripped.startswith("## 2.") or "OPEN QUESTIONS" in stripped.split("\n", 1)[0]:
            current_section = "section_2"
        elif stripped.startswith("## 3.") or "CONSIDERED AND FILTERED" in stripped.split("\n", 1)[0]:
            current_section = "section_3"
        elif stripped.startswith("## 4.") or "CITATIONS" in stripped.split("\n", 1)[0]:
            current_section = "section_4"
        paragraph_sections.append(current_section)
    # Prose-only sections: truncation + short-paragraph checks run only here.
    _PROSE_SECTIONS = {"preamble", "exec_summary", "section_1", "section_2"}

    for i, para in enumerate(paragraphs):
        cited_ids = _CITATION_RE.findall(para)
        section_name = paragraph_sections[i]
        is_prose = section_name in _PROSE_SECTIONS

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
        # Prose sections only. Section 3 (list) and Section 4 (citation block)
        # are deterministic structured output that lacks terminal punctuation
        # by design — running this check there produced ~22 FPs/run all
        # pointing at the trailing `support_diversity=...` line of each
        # citation entry.
        if is_prose and i < len(paragraphs) - 1:
            last_char = para.rstrip()[-1] if para.rstrip() else ""
            if last_char not in {".", "!", "?", '"', ")", "'"}:
                flags.append({
                    "issue": "truncated_mid_sentence",
                    "paragraph_excerpt": para[-200:],
                    "section": section_name,
                })

        # Check 6: suspiciously short paragraph (likely incomplete render).
        # Prose only — Section 3 list entries are legitimately short.
        if is_prose and cited_ids and len(para) < 50:
            flags.append({
                "issue": "suspiciously_short_paragraph",
                "paragraph_excerpt": para,
                "length": len(para),
                "section": section_name,
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


def _write_no_consensus_audit(output_dir: Path) -> None:
    """No-consensus short-circuit: there's no rendered answer to audit, so
    write total_flags=0 (audit ran clean) rather than leaving the file
    absent — which downstream reads as the -1 'unknown' sentinel."""
    audit = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_flags": 0,
        "flags": [],
        "reason": "no_consensus",
    }
    try:
        (output_dir / "renderer_audit.json").write_text(
            json.dumps(audit, indent=2), encoding="utf-8"
        )
    except Exception as exc:
        print(f"[synthesizer] could not write no-consensus renderer_audit.json: {exc}")


def _write_crashed_audit(output_dir: Path, exc: Exception) -> None:
    """Audit crashed mid-call. Write total_flags=-2 with an audit_error
    field so downstream summary.audit_flags is unambiguous (-2 = crashed,
    not -1 = file missing for unknown reason)."""
    audit = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_flags": -2,
        "flags": [],
        "audit_error": f"{type(exc).__name__}: {exc}",
    }
    try:
        (output_dir / "renderer_audit.json").write_text(
            json.dumps(audit, indent=2), encoding="utf-8"
        )
    except Exception as exc2:
        print(f"[synthesizer] could not write crashed-audit renderer_audit.json: {exc2}")


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
        INITIAL:           'shape=box style=filled fillcolor="#d0e8ff"',
        SUPPORT:           'shape=ellipse style=filled fillcolor="#d0f0d0"',
        CRITIQUE_POSITIVE: 'shape=diamond style=filled fillcolor="#d0ffd0"',
        CRITIQUE_NEGATIVE: 'shape=diamond style=filled fillcolor="#ffd0d0"',
        CRITIQUE:          'shape=diamond style=filled fillcolor="#ffd0d0"',  # legacy
        OBJECTION:         'shape=diamond style=filled fillcolor="#ffb0b0"',
        VERIFICATION:      'shape=hexagon style=filled fillcolor="#ffffd0"',
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

def _extract_json_block(text: str) -> Optional[str]:
    """Return the first complete {...} block from text using brace counting.

    The greedy regex r"\{[\s\S]*\}" fails when the model appends a comment,
    a trailing explanation, or a second JSON object after the real one —
    it grabs from the first '{' to the LAST '}', producing an invalid span.
    This function stops at the matching closing brace of the FIRST '{'.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(text[start:], start):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _repair_json(s: str) -> str:
    """Fix the most common LLM JSON formatting mistakes before parsing."""
    # Trailing commas before } or ] — the most frequent failure mode.
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return s


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
