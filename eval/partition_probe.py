"""Partition -> output-diversity micro-probe.

Tests the load-bearing assumption of the whole architecture (CLAUDE.md
principle #2) in isolation, with no swarm machinery:

    "Information partitioning as the diversity engine. Diversity comes from
     *what each agent has been shown*, not from prompt or temperature tweaks."

The probe asks: for a FIXED model, prompt template, and temperature, does
giving N independent calls DISJOINT evidence partitions separate their
outputs more than giving the same N calls IDENTICAL evidence?

Conditions (same prompt template, same model, same temperature, same
evidence-block size per call — the ONLY manipulated variable is whether the
evidence differs across calls):

  P  partitioned   N calls; call i sees partition i of the corpus
                   (semantic farthest-point partitioning, same code path as
                   the legacy scout intake: core.intake.partition_for_scouts).
  U  uniform       N calls; every call sees the SAME evidence block (a
                   stride-sample of the corpus, sized like a median
                   partition). Output spread here is pure sampling noise.
  H  hot-uniform   (optional, --include-hot) same as U but at --hot-temp
                   (default 1.0). Directly tests the "not from temperature
                   tweaks" half of the claim: if H separates outputs as much
                   as P, a temperature knob buys whatever partitioning buys.

Metrics per condition (from core.output_diversity, the same metrics the
pipeline logs):
  self_bleu                 lexical self-similarity; LOWER = more diverse
  centroid_cosine_distance  embedding dispersion;    HIGHER = more diverse

Decision rule (pre-registered here, before results):
  Partitioning "moves diversity" iff, across prompts, P beats U on BOTH
  metrics with a two-sided sign-test p < 0.05 AND the mean self_bleu gap
  is > 0.03 (below that, the effect is too small to plausibly survive the
  downstream dedup/clustering that already merges near-duplicates).
  If P ~= U, no downstream stigmergic machinery can recover input diversity
  that never reached the outputs — the honest pivot is model-family
  diversity + retrieval + synthesis prompt, not the signal store.
  If the partitioner was not 'semantic', the P condition tests contiguous
  rank-slicing and the verdict must be withheld.

Usage:
    # plumbing check only (MOCK outputs are SHA1-seeded noise — no signal):
    MOCK_LLM=1 python -m eval.partition_probe --mini 4 --corpus placeholder

    # real, offline corpus (tests partitioning of the engineered corpus):
    python -m eval.partition_probe --mini 8 --corpus placeholder

    # real, retrieved corpus (the honest version; needs network):
    GROQ_API_KEY=... python -m eval.partition_probe --mini 20 \
        --model llama-3.1-8b-instant --corpus retrieve --include-hot

Writes eval/results/partition_probe_<ts>/{rows.jsonl,report.md,meta.json}.

Honesty notes:
  - MOCK runs prove plumbing only (P0.1). The report is stamped MOCK.
  - --corpus placeholder partitions the engineered 4-framing corpus; a P>U
    result there shows the mechanism CAN work when partitions are
    topically engineered, not that real retrieval yields such partitions.
    Only --corpus retrieve tests the deployed configuration.
  - Temperature must be > 0 or condition U produces near-identical outputs
    by construction and the comparison is vacuous.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import statistics
import sys
import time
import traceback
from pathlib import Path

# Allow `python eval/partition_probe.py` as well as `python -m eval.partition_probe`.
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import intake as intake_mod  # noqa: E402 -- for LAST_PARTITIONER
from core.intake import CorpusChunk, ScoutPartition, partition_for_scouts, \
    trivial_corpus_from_thesis  # noqa: E402
from core.output_diversity import centroid_cosine_distance, self_bleu  # noqa: E402
from eval import prompts as promptset  # noqa: E402
from eval.ab_harness import Cost, DirectModel  # noqa: E402


def _probe_embedder() -> str:
    """Cheap one-shot probe: which embedding path will output_diversity /
    intake partitioning actually use this run — 'sbert' or 'bow_fallback'.

    Neither core.output_diversity nor core.intake exposes which path fired;
    they just silently fall back. Load the shared embedder once up front so
    every row can carry accurate provenance instead of guessing.
    """
    try:
        from core.signal_store import _try_load_embedder
        model = _try_load_embedder()
        return "sbert" if model is not None else "bow_fallback"
    except Exception as exc:
        print(f"[probe] embedder probe failed ({type(exc).__name__}: {exc}); "
              f"assuming bow_fallback")
        return "bow_fallback"


# ---------------------------------------------------------------------------
# Corpus building
# ---------------------------------------------------------------------------

# Probe-local chunking: smaller chunks than the pipeline's CHUNK_WORDS so the
# placeholder corpus (~1200 words) still yields >= n_agents non-empty
# partitions. No overlap — partitions must be strictly disjoint for the
# manipulation to mean anything.
_PROBE_CHUNK_WORDS = 160

_EVIDENCE_MAX_CHARS = 3200


def _chunk(text: str, source_tag: str, words: int = _PROBE_CHUNK_WORDS) -> list:
    toks = text.split()
    out = []
    for n, i in enumerate(range(0, len(toks), words)):
        piece = " ".join(toks[i:i + words])
        if piece.strip():
            out.append(CorpusChunk(chunk_id=f"chunk_{n:04d}", text=piece,
                                   source_tag=source_tag))
    return out


def _build_corpus(prompt_text: str, mode: str) -> tuple[list, str]:
    """Return (chunks, actual_mode). Falls back to placeholder loudly."""
    if mode == "retrieve":
        try:
            from core.retrieval import CompositeRetriever
            raw = CompositeRetriever().retrieve(prompt_text, target_chars=12000)
            # Re-chunk retrieved text at probe granularity, preserving source tags.
            chunks: list = []
            for rc in raw:
                chunks.extend(_chunk(rc.text, rc.source_tag))
            for n, c in enumerate(chunks):
                c.chunk_id = f"chunk_{n:04d}"
            if len(chunks) >= 4:
                return chunks, "retrieve"
            print(f"[probe] retrieval returned only {len(chunks)} probe-chunks; "
                  f"falling back to placeholder for this prompt")
        except Exception as exc:
            print(f"[probe] retrieval failed ({type(exc).__name__}: {exc}); "
                  f"falling back to placeholder")
    return _chunk(trivial_corpus_from_thesis(prompt_text), "placeholder"), "placeholder"


def _uniform_evidence(chunks: list, k: int) -> str:
    """The U-condition evidence: a stride-sample of k chunks spanning the
    whole corpus, identical for every call. Same render format as P."""
    n = len(chunks)
    k = max(1, min(k, n))
    if k <= 1 or n <= 1:
        # Degenerate stride (k=1, or a 1-chunk corpus): nothing to space out,
        # just take the top chunk. Guards a ZeroDivisionError at k-1==0 that
        # killed a real run (median partition size 1 chunk).
        idxs = [0]
    elif k == n:
        idxs = list(range(n))
    else:
        idxs = sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})
    sample = [chunks[i] for i in idxs]
    part = ScoutPartition(scout_index=0, chunks=sample,
                          custom_partition_id="uniform")
    return part.render(max_chars=_EVIDENCE_MAX_CHARS)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

# One template for every condition — the ONLY thing that varies is {evidence}.
_TEMPLATE = (
    "You are given evidence excerpts about a question. Read them, then state "
    "the single strongest, most specific claim you can defend about the "
    "question, grounded in the evidence you were shown. 2-4 sentences, "
    "concrete and self-contained. Do not restate the question; do not "
    "mention the excerpts or your reasoning process.\n\n"
    "QUESTION: {question}\n\n"
    "EVIDENCE:\n{evidence}\n\n"
    "CLAIM:"
)


async def _gen_batch(dm: DirectModel, question: str, evidences: list[str],
                     max_tokens: int, temperature: float, cost: Cost) -> list[str]:
    async def one(ev: str) -> str:
        prompt = _TEMPLATE.format(question=question, evidence=ev)
        try:
            return await dm.generate(prompt, max_tokens=max_tokens,
                                     temperature=temperature, cost=cost)
        except Exception as exc:
            print(f"[probe] generate failed: {type(exc).__name__}: {exc}")
            return ""
    return list(await asyncio.gather(*[one(ev) for ev in evidences]))


# ---------------------------------------------------------------------------
# Condition V: verbalized sampling (literature-strongest cheap diversity
# lever) as a comparison arm. Same identical evidence as U, same
# temperature, but the prompt asks for 5 different candidate claims with
# probabilities; we take the first listed candidate as the "output" so it
# is scored on the same footing as P/U/H (one claim per call).
# ---------------------------------------------------------------------------

_V_TEMPLATE = (
    "You are given evidence excerpts about a question. Read them, then generate "
    "5 DIFFERENT candidate claims that could answer the question, each grounded "
    "in the evidence you were shown. For each, estimate the probability (0-1) "
    "that it is the single strongest, most defensible claim. Number them 1-5. "
    "Do not explain your reasoning; do not restate the question.\n\n"
    "QUESTION: {question}\n\n"
    "EVIDENCE:\n{evidence}\n\n"
    "Format exactly as:\n"
    "1. [probability] <claim>\n"
    "2. [probability] <claim>\n"
    "3. [probability] <claim>\n"
    "4. [probability] <claim>\n"
    "5. [probability] <claim>\n\n"
    "CANDIDATES:"
)

_FIRST_CANDIDATE_RE = re.compile(r"1\.\s*(.+?)(?=\n\s*2\.|\Z)", re.DOTALL)
_LEADING_PROB_RE = re.compile(r"^\[?\s*\d*\.?\d+\s*\]?\s*[:\-]?\s*")


def _extract_first_candidate(text: str) -> str:
    """Pull candidate #1's claim text out of a verbalized-sampling response."""
    if not text or not text.strip():
        return ""
    m = _FIRST_CANDIDATE_RE.search(text)
    claim = m.group(1).strip() if m else text.strip()
    claim = _LEADING_PROB_RE.sub("", claim).strip()
    return claim


async def _gen_batch_v(dm: DirectModel, question: str, evidences: list[str],
                       max_tokens: int, temperature: float, cost: Cost) -> list[str]:
    async def one(ev: str) -> str:
        prompt = _V_TEMPLATE.format(question=question, evidence=ev)
        try:
            raw = await dm.generate(prompt, max_tokens=max_tokens * 3,
                                    temperature=temperature, cost=cost)
        except Exception as exc:
            print(f"[probe] generate (V) failed: {type(exc).__name__}: {exc}")
            return ""
        return _extract_first_candidate(raw)
    return list(await asyncio.gather(*[one(ev) for ev in evidences]))


# ---------------------------------------------------------------------------
# Degenerate-output filtering
# ---------------------------------------------------------------------------

_MIN_ALPHA_TOKENS = 30


def _filter_degenerate(texts: list[str]) -> tuple[list[str], int]:
    """Drop outputs with fewer than _MIN_ALPHA_TOKENS alphabetic tokens.

    A degenerate output (e.g. a run of underscores, or near-empty text) can
    score as maximally "diverse" against everything else by both metrics —
    it poisons self_bleu (no shared n-grams) and centroid_cosine_distance
    (embeds far from the centroid of real text) alike. Returns
    (surviving_texts, n_dropped).
    """
    kept: list[str] = []
    n_dropped = 0
    for t in texts:
        n_alpha = len(re.findall(r"[a-zA-Z]+", t))
        if n_alpha < _MIN_ALPHA_TOKENS:
            n_dropped += 1
        else:
            kept.append(t)
    return kept, n_dropped


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def _sign_test_p(wins: int, losses: int) -> float:
    """Exact two-sided binomial sign test, ties dropped."""
    n = wins + losses
    if n == 0:
        return 1.0
    def cdf_le(k: int) -> float:
        return sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    p_low = cdf_le(min(wins, losses))
    return min(1.0, 2.0 * p_low)


def _fmt(x) -> str:
    return "-" if x is None else f"{x:.3f}"


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def _write_meta(out_dir: Path, ts: str, is_mock: bool, dm: DirectModel, args,
                rows: list[dict], cost: Cost, embedder_used: str,
                status: str) -> None:
    (out_dir / "meta.json").write_text(json.dumps({
        "timestamp": ts, "mock": is_mock, "model": dm.label,
        "backend": args.backend, "corpus": args.corpus,
        "n_agents": args.n_agents, "temperature": args.temperature,
        "hot_temp": args.hot_temp if args.include_hot else None,
        "include_hot": args.include_hot, "v_condition": args.v_condition,
        "max_tokens": args.max_tokens, "n_prompts_run": len(rows),
        "prompt_ids": [r["pid"] for r in rows],
        "llm_calls": cost.llm_calls, "latency_s": round(cost.latency_s, 1),
        "embedder": embedder_used, "status": status,
    }, indent=2), encoding="utf-8")


async def run_probe(args) -> Path:
    is_mock = os.environ.get("MOCK_LLM", "").strip() not in ("", "0", "false", "False")
    if args.temperature <= 0:
        raise SystemExit("[probe] --temperature must be > 0: at temp 0 the "
                         "uniform condition is deterministic and the "
                         "comparison is vacuous.")

    plist = promptset.mini(args.mini) if args.mini else promptset.DEFAULT_SET
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(__file__).parent / "results" / f"partition_probe_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    embedder_used = _probe_embedder()
    print(f"[probe] embedder path: {embedder_used}")

    dm = DirectModel(args.model, force_local=args.backend == "local")
    cost = Cost()
    conditions = (["P", "U"] + (["H"] if args.include_hot else [])
                 + (["V"] if args.v_condition else []))

    rows: list[dict] = []
    rows_path = out_dir / "rows.jsonl"

    # Write meta.json immediately so a crash on prompt 1 still leaves valid
    # artifacts on disk (the run that motivated this: died on prompt 5,
    # report.md/meta.json never touched, everything lost).
    _write_meta(out_dir, ts, is_mock, dm, args, rows, cost, embedder_used,
               status="running")

    completed = False
    crash_info: str | None = None
    try:
        for p in plist:
            chunks, corpus_mode = _build_corpus(p.text, args.corpus)
            partitions = partition_for_scouts(chunks, args.n_agents)
            partitioner = intake_mod.LAST_PARTITIONER
            if partitioner != "semantic":
                print(f"[probe] *** WARNING: partitioner={partitioner!r} for "
                      f"{p.pid} — semantic gate needs len(chunks) >= "
                      f"n_agents*2 ({len(chunks)} chunks, {args.n_agents} "
                      f"agents); P condition is testing CONTIGUOUS "
                      f"rank-slicing, not topical partitioning ***")
            nonempty = [pt for pt in partitions if pt.chunks]
            n_eff = len(nonempty)
            if n_eff < 3:
                print(f"[probe] {p.pid}: only {n_eff} non-empty partitions "
                      f"({len(chunks)} chunks) — skipped")
                continue

            p_evidence = [pt.render(max_chars=_EVIDENCE_MAX_CHARS) for pt in nonempty]
            med_size = int(statistics.median(len(pt.chunks) for pt in nonempty))
            u_evidence = [_uniform_evidence(chunks, med_size)] * n_eff

            print(f"[probe] {p.pid}: {len(chunks)} chunks -> {n_eff} partitions "
                  f"(median {med_size} chunks) corpus={corpus_mode} "
                  f"partitioner={partitioner}")

            outs: dict[str, list[str]] = {}
            outs["P"] = await _gen_batch(dm, p.text, p_evidence,
                                         args.max_tokens, args.temperature, cost)
            outs["U"] = await _gen_batch(dm, p.text, u_evidence,
                                         args.max_tokens, args.temperature, cost)
            if args.include_hot:
                outs["H"] = await _gen_batch(dm, p.text, u_evidence,
                                             args.max_tokens, args.hot_temp, cost)
            if args.v_condition:
                outs["V"] = await _gen_batch_v(dm, p.text, u_evidence,
                                               args.max_tokens, args.temperature, cost)

            row: dict = {"pid": p.pid, "task": p.task, "prompt": p.text,
                         "n_eff": n_eff, "n_chunks": len(chunks),
                         "corpus_mode": corpus_mode, "partitioner": partitioner,
                         "embedder": embedder_used, "outputs": {},
                         "output_lens": {}}
            for c in conditions:
                raw_texts = [t for t in outs[c] if t.strip()]
                row["output_lens"][c] = [len(t) for t in raw_texts]
                texts, n_degen = _filter_degenerate(raw_texts)
                row[f"n_degenerate_{c}"] = n_degen
                if len(texts) < 2:
                    print(f"[probe] {p.pid}/{c}: only {len(texts)} "
                          f"non-degenerate outputs (of {len(raw_texts)} "
                          f"non-empty, {n_degen} degenerate) — metrics set "
                          f"to null for this condition")
                    row["outputs"][c] = texts
                    row[f"self_bleu_{c}"] = None
                    row[f"centroid_dist_{c}"] = None
                    continue
                row["outputs"][c] = texts
                row[f"self_bleu_{c}"] = round(self_bleu(texts), 4)
                row[f"centroid_dist_{c}"] = round(centroid_cosine_distance(texts), 4)

            rows.append(row)
            with rows_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

            # Rewrite report.md/meta.json after EVERY prompt so a crash never
            # loses completed work.
            report = _render_report(rows, conditions, args, cost, is_mock,
                                    partial=True)
            (out_dir / "report.md").write_text(report, encoding="utf-8")
            _write_meta(out_dir, ts, is_mock, dm, args, rows, cost,
                       embedder_used, status="running")

        completed = True
    except BaseException as exc:
        crash_info = f"{type(exc).__name__}: {exc}"
        print(f"[probe] *** run interrupted: {crash_info} ***")
        traceback.print_exc()
        raise
    finally:
        report = _render_report(rows, conditions, args, cost, is_mock,
                                partial=not completed, crash_info=crash_info)
        (out_dir / "report.md").write_text(report, encoding="utf-8")
        _write_meta(out_dir, ts, is_mock, dm, args, rows, cost, embedder_used,
                   status="completed" if completed else "PARTIAL")
        print("\n" + report)
        print(f"[probe] artifacts: {out_dir}")
    return out_dir


def _paired_rows(rows: list[dict], a: str, b: str) -> list[dict]:
    """Rows where BOTH condition a and condition b have non-null metrics."""
    return [r for r in rows
           if r.get(f"self_bleu_{a}") is not None
           and r.get(f"self_bleu_{b}") is not None
           and r.get(f"centroid_dist_{a}") is not None
           and r.get(f"centroid_dist_{b}") is not None]


def _render_report(rows: list[dict], conditions: list[str], args,
                   cost: Cost, is_mock: bool, partial: bool = False,
                   crash_info: str | None = None) -> str:
    lines: list[str] = []
    title = "# Partition -> output-diversity probe"
    if is_mock:
        title += "  (MOCK — plumbing check only; numbers carry NO behavioral signal)"
    lines.append(title)
    if partial:
        lines.append("")
        note = "**PARTIAL — run did not complete.**"
        if crash_info:
            note += f" Interrupted by: `{crash_info}`."
        note += (" Rows below reflect only the prompts that finished before "
                "the run stopped; artifacts were written incrementally so "
                "no completed work is lost.")
        lines.append(note)
    lines.append("")
    cond_desc = "P=partitioned evidence, U=identical evidence"
    if "H" in conditions:
        cond_desc += ", H=identical evidence @ hot temperature"
    if "V" in conditions:
        cond_desc += ", V=identical evidence + verbalized-sampling prompt (first of 5 candidates)"
    lines.append(f"Conditions: {cond_desc}. n_agents={args.n_agents}, "
                f"temp={args.temperature}, corpus={args.corpus}, "
                f"model={args.model or 'local'}, llm_calls={cost.llm_calls}, "
                f"embedder={rows[0]['embedder'] if rows else '?'}.")
    lines.append("")
    if not rows:
        lines.append("No prompts completed — nothing to report.")
        return "\n".join(lines)

    partitioners = {r.get("partitioner", "") for r in rows}
    non_semantic_rows = [r for r in rows if r.get("partitioner") != "semantic"]
    if non_semantic_rows:
        lines.append(f"**WARNING: {len(non_semantic_rows)}/{len(rows)} prompt(s) "
                     f"used a non-semantic partitioner ({sorted(partitioners)}). "
                     f"The semantic gate requires len(chunks) >= n_agents*2; "
                     f"for these rows, condition P tested CONTIGUOUS "
                     f"rank-slicing, not topical partitioning.**")
        lines.append("")

    hdr = "| pid | n | corpus | partitioner |"
    sep = "|---|---|---|---|"
    for c in conditions:
        hdr += f" bleu_{c} |"
        sep += "---|"
    for c in conditions:
        hdr += f" cdist_{c} |"
        sep += "---|"
    for c in conditions:
        hdr += f" degen_{c} |"
        sep += "---|"
    lines += [hdr, sep]
    for r in rows:
        cells = [r["pid"], str(r["n_eff"]), r["corpus_mode"],
                 r.get("partitioner", "?")]
        cells += [_fmt(r.get(f"self_bleu_{c}")) for c in conditions]
        cells += [_fmt(r.get(f"centroid_dist_{c}")) for c in conditions]
        cells += [str(r.get(f"n_degenerate_{c}", 0)) for c in conditions]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # Paired comparison P vs U (and P vs H / P vs V when present). Only rows
    # where BOTH conditions have non-null metrics (>=2 non-degenerate
    # outputs survived) enter each comparison.
    for other in [c for c in conditions if c != "P"]:
        paired = _paired_rows(rows, "P", other)
        n_skipped = len(rows) - len(paired)
        lines.append(f"## P vs {other}  (n={len(paired)} prompts"
                     + (f", {n_skipped} skipped for null metrics" if n_skipped else "")
                     + ")")
        lines.append("")
        if len(paired) == 0:
            lines.append("No comparable rows (every row had a null metric "
                         "for one side).")
            lines.append("")
            continue
        db = [r[f"self_bleu_P"] - r[f"self_bleu_{other}"] for r in paired]
        dc = [r[f"centroid_dist_P"] - r[f"centroid_dist_{other}"] for r in paired]
        # P more diverse: lower self_bleu, higher centroid distance.
        bleu_w = sum(1 for d in db if d < 0); bleu_l = sum(1 for d in db if d > 0)
        cd_w = sum(1 for d in dc if d > 0); cd_l = sum(1 for d in dc if d < 0)
        lines.append(f"- self_bleu:     mean delta {statistics.mean(db):+.4f} "
                     f"(negative = P more diverse); P wins {bleu_w}/{bleu_w+bleu_l}, "
                     f"sign-test p={_sign_test_p(bleu_w, bleu_l):.3f}")
        lines.append(f"- centroid_dist: mean delta {statistics.mean(dc):+.4f} "
                     f"(positive = P more diverse); P wins {cd_w}/{cd_w+cd_l}, "
                     f"sign-test p={_sign_test_p(cd_w, cd_l):.3f}")
        lines.append("")

    # Pre-registered verdict (P vs U only).
    lines.append("## Verdict (pre-registered rule)")
    lines.append("")
    paired_pu = _paired_rows(rows, "P", "U")
    if is_mock:
        lines.append("MOCK run — verdict withheld (plumbing only).")
    elif partial:
        lines.append("Run did not complete — verdict withheld pending a full run.")
    elif non_semantic_rows:
        lines.append("Partitioner was not 'semantic' for one or more prompts "
                     "(see WARNING above) — per the pre-registered rule "
                     "extension, the verdict is WITHHELD. Condition P tested "
                     "contiguous rank-slicing for those rows, not the "
                     "semantic partitioning the pipeline actually uses.")
    elif len(paired_pu) < 8:
        lines.append(f"n={len(paired_pu)} < 8 comparable prompts — "
                     f"underpowered; verdict withheld. (Directional numbers "
                     f"above only.)")
    else:
        db = [r["self_bleu_P"] - r["self_bleu_U"] for r in paired_pu]
        dc = [r["centroid_dist_P"] - r["centroid_dist_U"] for r in paired_pu]
        bleu_p = _sign_test_p(sum(1 for d in db if d < 0), sum(1 for d in db if d > 0))
        cd_p = _sign_test_p(sum(1 for d in dc if d > 0), sum(1 for d in dc if d < 0))
        passed = (bleu_p < 0.05 and cd_p < 0.05
                  and statistics.mean(db) < -0.03
                  and statistics.mean(dc) > 0)
        if passed:
            lines.append("PARTITIONING MOVES OUTPUT DIVERSITY on this corpus mode. "
                         "This licenses fixing the default pipeline to actually use "
                         "partitions — it does not by itself show a quality win.")
        else:
            lines.append("Partitioning did NOT separate outputs beyond sampling "
                         "noise (per the pre-registered rule). Input partitioning "
                         "cannot be the diversity engine for this model/corpus; "
                         "downstream stigmergic machinery cannot recover diversity "
                         "that never reached the outputs.")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Partition -> output-diversity micro-probe (no swarm machinery)")
    ap.add_argument("--mini", type=int, default=0,
                    help="use only the first N pre-registered prompts (0 = full set)")
    ap.add_argument("--n-agents", type=int, default=6,
                    help="independent calls / partitions per condition (default 6)")
    ap.add_argument("--model", default=None,
                    help="model name (Groq model id when GROQ_API_KEY is set; "
                         "otherwise the local engine is used and this is ignored)")
    ap.add_argument("--backend", choices=["auto", "local"], default="auto",
                    help="'local' forces the local engine even with GROQ_API_KEY set")
    ap.add_argument("--corpus", choices=["placeholder", "retrieve"],
                    default="retrieve",
                    help="evidence source; 'retrieve' is the deployed config, "
                         "'placeholder' is offline (engineered 4-framing corpus)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=220)
    ap.add_argument("--include-hot", action="store_true",
                    help="add condition H: identical evidence at --hot-temp, to "
                         "compare partitioning against a plain temperature knob")
    ap.add_argument("--hot-temp", type=float, default=1.0)
    ap.add_argument("--v-condition", action="store_true",
                    help="add condition V: same identical evidence as U, same "
                         "temperature, but a verbalized-sampling prompt (ask "
                         "for 5 different candidate claims with probabilities, "
                         "take the first as the output) — the literature's "
                         "strongest cheap diversity lever, as a comparison arm")
    return ap


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    asyncio.run(run_probe(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
