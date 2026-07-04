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
import statistics
import sys
import time
from pathlib import Path

# Allow `python eval/partition_probe.py` as well as `python -m eval.partition_probe`.
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.intake import CorpusChunk, ScoutPartition, partition_for_scouts, \
    trivial_corpus_from_thesis  # noqa: E402
from core.output_diversity import centroid_cosine_distance, self_bleu  # noqa: E402
from eval import prompts as promptset  # noqa: E402
from eval.ab_harness import Cost, DirectModel  # noqa: E402


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
    if k == n:
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

    dm = DirectModel(args.model, force_local=args.backend == "local")
    cost = Cost()
    conditions = ["P", "U"] + (["H"] if args.include_hot else [])

    rows: list[dict] = []
    rows_path = out_dir / "rows.jsonl"

    for p in plist:
        chunks, corpus_mode = _build_corpus(p.text, args.corpus)
        partitions = partition_for_scouts(chunks, args.n_agents)
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
              f"(median {med_size} chunks) corpus={corpus_mode}")

        outs: dict[str, list[str]] = {}
        outs["P"] = await _gen_batch(dm, p.text, p_evidence,
                                     args.max_tokens, args.temperature, cost)
        outs["U"] = await _gen_batch(dm, p.text, u_evidence,
                                     args.max_tokens, args.temperature, cost)
        if args.include_hot:
            outs["H"] = await _gen_batch(dm, p.text, u_evidence,
                                         args.max_tokens, args.hot_temp, cost)

        row: dict = {"pid": p.pid, "task": p.task, "prompt": p.text,
                     "n_eff": n_eff, "n_chunks": len(chunks),
                     "corpus_mode": corpus_mode, "outputs": {}}
        ok = True
        for c in conditions:
            texts = [t for t in outs[c] if t.strip()]
            if len(texts) < 3:
                print(f"[probe] {p.pid}/{c}: only {len(texts)} non-empty "
                      f"outputs — prompt skipped")
                ok = False
                break
            row["outputs"][c] = texts
            row[f"self_bleu_{c}"] = round(self_bleu(texts), 4)
            row[f"centroid_dist_{c}"] = round(centroid_cosine_distance(texts), 4)
        if not ok:
            continue
        rows.append(row)
        with rows_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ---- aggregate ---------------------------------------------------------
    report = _render_report(rows, conditions, args, cost, is_mock)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    (out_dir / "meta.json").write_text(json.dumps({
        "timestamp": ts, "mock": is_mock, "model": dm.label,
        "backend": args.backend, "corpus": args.corpus,
        "n_agents": args.n_agents, "temperature": args.temperature,
        "hot_temp": args.hot_temp if args.include_hot else None,
        "max_tokens": args.max_tokens, "n_prompts_run": len(rows),
        "prompt_ids": [r["pid"] for r in rows],
        "llm_calls": cost.llm_calls, "latency_s": round(cost.latency_s, 1),
    }, indent=2), encoding="utf-8")
    print("\n" + report)
    print(f"[probe] artifacts: {out_dir}")
    return out_dir


def _render_report(rows: list[dict], conditions: list[str], args,
                   cost: Cost, is_mock: bool) -> str:
    lines: list[str] = []
    title = "# Partition -> output-diversity probe"
    if is_mock:
        title += "  (MOCK — plumbing check only; numbers carry NO behavioral signal)"
    lines.append(title)
    lines.append("")
    lines.append(f"Conditions: P=partitioned evidence, U=identical evidence"
                 + (", H=identical evidence @ hot temperature" if "H" in conditions else "")
                 + f". n_agents={args.n_agents}, temp={args.temperature}, "
                   f"corpus={args.corpus}, model={args.model or 'local'}, "
                   f"llm_calls={cost.llm_calls}.")
    lines.append("")
    if not rows:
        lines.append("No prompts completed — nothing to report.")
        return "\n".join(lines)

    hdr = "| pid | n | corpus |"
    sep = "|---|---|---|"
    for c in conditions:
        hdr += f" bleu_{c} |"
        sep += "---|"
    for c in conditions:
        hdr += f" cdist_{c} |"
        sep += "---|"
    lines += [hdr, sep]
    for r in rows:
        cells = [r["pid"], str(r["n_eff"]), r["corpus_mode"]]
        cells += [_fmt(r.get(f"self_bleu_{c}")) for c in conditions]
        cells += [_fmt(r.get(f"centroid_dist_{c}")) for c in conditions]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # Paired comparison P vs U (and P vs H when present).
    for other in [c for c in conditions if c != "P"]:
        db = [r[f"self_bleu_P"] - r[f"self_bleu_{other}"] for r in rows]
        dc = [r[f"centroid_dist_P"] - r[f"centroid_dist_{other}"] for r in rows]
        # P more diverse: lower self_bleu, higher centroid distance.
        bleu_w = sum(1 for d in db if d < 0); bleu_l = sum(1 for d in db if d > 0)
        cd_w = sum(1 for d in dc if d > 0); cd_l = sum(1 for d in dc if d < 0)
        lines.append(f"## P vs {other}  (n={len(rows)} prompts)")
        lines.append("")
        lines.append(f"- self_bleu:     mean delta {statistics.mean(db):+.4f} "
                     f"(negative = P more diverse); P wins {bleu_w}/{bleu_w+bleu_l}, "
                     f"sign-test p={_sign_test_p(bleu_w, bleu_l):.3f}")
        lines.append(f"- centroid_dist: mean delta {statistics.mean(dc):+.4f} "
                     f"(positive = P more diverse); P wins {cd_w}/{cd_w+cd_l}, "
                     f"sign-test p={_sign_test_p(cd_w, cd_l):.3f}")
        lines.append("")

    # Pre-registered verdict (P vs U only).
    db = [r["self_bleu_P"] - r["self_bleu_U"] for r in rows]
    dc = [r["centroid_dist_P"] - r["centroid_dist_U"] for r in rows]
    bleu_p = _sign_test_p(sum(1 for d in db if d < 0), sum(1 for d in db if d > 0))
    cd_p = _sign_test_p(sum(1 for d in dc if d > 0), sum(1 for d in dc if d < 0))
    passed = (bleu_p < 0.05 and cd_p < 0.05
              and statistics.mean(db) < -0.03
              and statistics.mean(dc) > 0)
    lines.append("## Verdict (pre-registered rule)")
    lines.append("")
    if is_mock:
        lines.append("MOCK run — verdict withheld (plumbing only).")
    elif len(rows) < 8:
        lines.append(f"n={len(rows)} < 8 prompts — underpowered; verdict withheld. "
                     f"(Directional numbers above only.)")
    elif passed:
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
    return ap


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    asyncio.run(run_probe(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
