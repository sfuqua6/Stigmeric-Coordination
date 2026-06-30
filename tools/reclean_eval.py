"""Re-clean the swarm (condition A) answers in a conditions.jsonl and report the
word-count collapse. Writes a sibling <stem>_cleaned.jsonl with condition A's
answer replaced by the reader-facing text (Sections 1[+2] only). Re-point
eval/judge.py at the cleaned file to re-judge A vs B/C/D with no GPU / no swarm
re-run required."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.clean_answer import split_answer


def main(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    out = []
    for r in rows:
        if r.get("condition") == "A":
            before = len(r["answer"].split())
            reader, _diag = split_answer(r["answer"])
            r["answer"] = reader
            after = len(reader.split())
            print(f"  {r.get('pid', '?'):12s} A: {before} -> {after} words")
        out.append(r)
    dst = Path(path).with_name(Path(path).stem + "_cleaned.jsonl")
    dst.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out),
                   encoding="utf-8")
    print(f"wrote {dst}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python tools/reclean_eval.py <conditions.jsonl>")
        raise SystemExit(2)
    main(sys.argv[1])
