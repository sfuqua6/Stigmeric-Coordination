"""Generator for notebooks/colab_swarm.ipynb (robust JSON > hand-writing).

Run:  python notebooks/_build_colab.py
Re-run to regenerate after editing cell text below.

Design (2026-07-02 rewrite) — the notebook is organized around producing JUDGED
EVIDENCE and not losing or polluting it. Three failures from real sessions are
engineered out:
  1. LOST RESULTS  — the Jun-30 session's eval/results (conditions.jsonl,
     scores.json) were never synced to Drive; only run dirs survived. Every
     phase now ends with sync_drive().
  2. BROKEN SWEEPS — delta_sweep_small burned ~3.5 h across 10 runs with the
     validator never firing and clustering inert. health_check() now gates
     every run before more GPU time is spent.
  3. DEAD EMBEDDER — huggingface-hub 1.x / torchcodec silently kill
     sentence-transformers -> singleton clusters. Cell 5 preflights the
     embedder BEFORE any model download or run.
"""
import json
import os

def md(src): return {"cell_type": "markdown", "metadata": {}, "source": src}
def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src}

cells = []

# ---------------------------------------------------------------- header
cells.append(md(
"""# Stigmergic Swarm — Colab evidence runner

One notebook, one mission: **produce a judged verdict on the swarm and get it
safely into Drive.** Phases:

| Phase | Cells | What |
|---|---|---|
| Setup | 1–6 | Drive, repo, deps, GPU tier, backend/keys |
| Preflight | 7–8 | embedder + mock smoke + the health/sync helpers — **do not skip** |
| Single run | 9–10 | one real swarm run, health-gated, inspected |
| **Eval (the deliverable)** | 11–13 | conditions A/B/E (+D), blind packet → Claude judges → report |
| Sweep | 14 | Δ_amp across model sizes (optional) |

### Rules of evidence (from CLAUDE.md — read before celebrating)
- A verdict with `agreement: false` is position bias, not signal.
- **Condition E** (direct call + the swarm's own synthesis prompt) is the
  attribution control. Never report A-vs-B without A-vs-E.
- A run that fails `health_check()` is not evidence for anything. Fix, rerun.
- Mock output proves plumbing, never behavior.

### Backend cheat-sheet (Cell 6 is the only config cell)
| Backend | `SWARM_BACKEND` | GPU? | Best for |
|---|---|---|---|
| **hybrid** (recommended) | `hybrid` + Groq key | yes | local model for volume + Groq 70B hater |
| groq-only | *(unset)* + Groq key | no | zero GPU units; watch RPM/TPD |
| local-only | `local`, no key | yes | no rate limits |

Colab GPUs: T4 16 GB → 3B local, L4 24 GB → 14B (sweet spot), A100 40 GB → 32B-AWQ.
Free Groq ≈ 30 req/min/model + a daily token cap; only the hater rides Groq by
default (the synthesizer's end-of-run burst blows the free TPD — it runs local).

> **Push your local commits first** — Cell 3 clones the GitHub repo.
> On reconnect: re-run Cells 1–6 and Cell 8, then continue where you were."""))

# ---------------------------------------------------------------- 1 drive
cells.append(md("## Cell 1 — Mount Drive"))
cells.append(code(
"""from google.colab import drive
drive.mount('/content/drive')"""))

# ---------------------------------------------------------------- 2 paths
cells.append(md(
"""## Cell 2 — Paths & environment

Run dirs land on Drive directly (`SWARM_OUTPUTS_BASE_DIR`). Eval results do
NOT — they are written under the repo and copied by `sync_drive()` (Cell 8)."""))
cells.append(code(
"""import os
DRIVE_BASE = '/content/drive/MyDrive/swarm'
os.environ['SWARM_OUTPUTS_BASE_DIR']    = f'{DRIVE_BASE}/runs'
os.environ['SWARM_KB_DIR']              = f'{DRIVE_BASE}/knowledge_base'
os.environ['SWARM_RETRIEVAL_CACHE_DIR'] = f'{DRIVE_BASE}/retrieval_cache'
for d in ('runs', 'knowledge_base', 'retrieval_cache', 'eval_results'):
    os.makedirs(f'{DRIVE_BASE}/{d}', exist_ok=True)
os.environ['HF_HOME'] = '/content/hf_cache'; os.makedirs('/content/hf_cache', exist_ok=True)
os.environ['COLAB'] = '1'; os.environ['SWARM_QUIET_LIBS'] = '1'
print('persistent dirs under', DRIVE_BASE)"""))

# ---------------------------------------------------------------- 3 clone
cells.append(md("## Cell 3 — Clone / pull the repo"))
cells.append(code(
"""import os, subprocess
REPO_URL='https://github.com/sfuqua6/Stigmeric-Coordination.git'
REPO_BRANCH='cleanup/restructure'   # branch to run (set 'main' after merge)
REPO_ROOT='/content/swarm_repo'
if not os.path.exists(REPO_ROOT):
    subprocess.run(['git','clone','--branch',REPO_BRANCH,REPO_URL,REPO_ROOT],
                   check=True)
else:
    subprocess.run(['git','-C',REPO_ROOT,'fetch','origin'], check=True)
    subprocess.run(['git','-C',REPO_ROOT,'checkout',REPO_BRANCH], check=True)
    subprocess.run(['git','-C',REPO_ROOT,'pull','origin',REPO_BRANCH], check=True)
os.chdir(REPO_ROOT)
print('cwd:', os.getcwd())
print(subprocess.run(['git','-C',REPO_ROOT,'log','--oneline','-3'],
                     capture_output=True, text=True).stdout)"""))

# ---------------------------------------------------------------- 4 deps
cells.append(md(
"""## Cell 4 — Install dependencies

Installs core deps + `openai` (Groq client) always; vLLM **only if a GPU is
present**, CUDA-version-matched. Pins that exist for a reason:
- `huggingface_hub<1.0` — hub 1.x silently kills sentence-transformers
  (→ no embedder → singleton clusters).
- `torchcodec` is force-removed — its broken native lib poisons the
  sentence-transformers import chain on the Colab image.
- `vllm<0.20` on CUDA 12 — 0.20+ links libcudart.so.13."""))
cells.append(code(
"""import subprocess, sys, re, shutil
subprocess.run([sys.executable,'-m','pip','install','-q',
    'openai','sentence-transformers','cohere','datasets','faiss-cpu',
    'wikipedia','ddgs','requests','beautifulsoup4','tqdm',
    'huggingface_hub>=0.26.0,<1.0'], check=True)
subprocess.run([sys.executable,'-m','pip','uninstall','-y','-q','torchcodec'],
               check=False)
print('core deps installed; torchcodec removed.')

has_gpu = shutil.which('nvidia-smi') is not None
if not has_gpu:
    print('No GPU detected — skipping vLLM (Groq-only / CPU runtime).')
else:
    _nvcc = subprocess.run(['nvcc','--version'], capture_output=True, text=True)
    _maj = 12
    for ln in _nvcc.stdout.splitlines():
        m = re.search(r'release (\\d+)\\.', ln)
        if m: _maj = int(m.group(1)); break
    spec = 'vllm' if _maj >= 13 else 'vllm<0.20.0'
    print(f'CUDA {_maj}: installing {spec} ...')
    subprocess.run([sys.executable,'-m','pip','install','-q',spec,'bitsandbytes>=0.46.1'], check=True)
    try:
        import importlib; importlib.import_module('vllm.engine.async_llm_engine')
        import vllm; print(f'vLLM {vllm.__version__} OK')
    except Exception as e:
        print(f'WARNING vllm import failed: {e}\\n  Try Runtime>Restart, re-run this cell.')"""))

# ---------------------------------------------------------------- 5 gpu + embedder preflight
cells.append(md(
"""## Cell 5 — GPU tier + embedder preflight

The embedder check is the point of this cell: if sentence-transformers can't
load, clustering and dedup are silently OFF and every downstream number is
garbage (this is exactly how the 10-run `delta_sweep_small` batch died).
**Hard stop here beats a wasted sweep.**"""))
cells.append(code(
"""import shutil, sys, subprocess
sys.path.insert(0, '.')
if shutil.which('nvidia-smi'):
    print(subprocess.run(['nvidia-smi','--query-gpu=name,memory.total',
        '--format=csv,noheader'], capture_output=True, text=True).stdout.strip())
else:
    print('No GPU (CPU runtime) — use groq-only backend.')
from core import config
print(f'tier={config._TIER!r}  local_model={config.MODEL_NAME!r}  dtype={config.VLLM_DTYPE!r}')

# --- embedder preflight (do not proceed on failure) ---
try:
    from sentence_transformers import SentenceTransformer
    _m = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    v = _m.encode(['preflight'])
    assert v is not None and len(v) == 1
    print('embedder preflight: OK (all-MiniLM-L6-v2 loads and encodes)')
except Exception as e:
    raise RuntimeError(
        'EMBEDDER PREFLIGHT FAILED — clustering/dedup would silently die.\\n'
        'Fix before running anything: pip install \"huggingface-hub>=0.26.0,<1.0\", '
        'pip uninstall torchcodec, then Runtime>Restart.\\n'
        f'Underlying error: {e}')"""))

# ---------------------------------------------------------------- 6 backend & keys
cells.append(md(
"""## Cell 6 — Backend & keys  ⭐ the one cell to configure

- **hybrid** (recommended): Groq key + `SWARM_BACKEND='hybrid'`. Needs a GPU.
- **groq-only**: Groq key, `SWARM_BACKEND` unset. No GPU needed.
- **local-only**: no key, `SWARM_BACKEND='local'`. Needs a GPU.

Keys: free Groq at console.groq.com/keys · optional Tavily (better retrieval
than DDG, 1000 free/mo) at tavily.com."""))
cells.append(code(
"""import os
GROQ_API_KEY   = ''        # <-- paste Groq key (hybrid / groq-only); blank = local-only
TAVILY_API_KEY = ''        # <-- optional; blank = DDG (rate-limited, noisier)
SWARM_BACKEND  = 'hybrid'  # 'hybrid' | 'local' | ''(=groq-only when key set)

if GROQ_API_KEY:   os.environ['GROQ_API_KEY'] = GROQ_API_KEY
if TAVILY_API_KEY: os.environ['TAVILY_API_KEY'] = TAVILY_API_KEY
if SWARM_BACKEND:  os.environ['SWARM_BACKEND'] = SWARM_BACKEND
else:              os.environ.pop('SWARM_BACKEND', None)

os.environ['SWARM_MAX_TIME_S'] = '1800'
# Roles riding Groq in hybrid. Default hater-only: the synthesizer's token
# burst exhausts the free TPD mid-render. 'synthesizer,hater' needs paid quota.
os.environ['SWARM_HYBRID_GROQ_ROLES'] = 'hater'
# Optional: pin the local model (else tier auto-selects).
# os.environ['SWARM_MODEL'] = 'Qwen/Qwen2.5-14B-Instruct'

_k = 'set' if os.environ.get('GROQ_API_KEY') else 'BLANK'
_t = 'set' if os.environ.get('TAVILY_API_KEY') else 'BLANK (DDG primary)'
print(f'GROQ_API_KEY: {_k} | TAVILY_API_KEY: {_t} | '
      f\"SWARM_BACKEND: {os.environ.get('SWARM_BACKEND','(groq-only or local)')}\")"""))

# ---------------------------------------------------------------- 7 mock smoke
cells.append(md("## Cell 7 — Mock smoke test (no GPU / key / network; plumbing only)"))
cells.append(code(
"""!MOCK_LLM=1 SWARM_MIN_TIME_S=0 SWARM_MIN_ITERATIONS=5 SWARM_MAX_ITERATIONS=20 \\
    python run_swarm.py debate "Cities should ban private cars" --corpus=placeholder"""))

# ---------------------------------------------------------------- 8 helpers
cells.append(md(
"""## Cell 8 — `health_check()` + `sync_drive()` — the session's two safety rails

- `health_check(run_dir)` fails loudly on the structural breakages seen in real
  sessions: dead embedder, inert clustering, validator that never ran, engines
  unrecorded. **A run that fails this is not evidence.**
- `sync_drive()` copies `outputs/` and `eval/results/` into Drive. The Jun-30
  session lost its `conditions.jsonl`/`scores.json` on disconnect because only
  run dirs were persisted. Every phase below calls it; call it manually anytime."""))
cells.append(code(
"""import json, shutil, os
from pathlib import Path

def health_check(run_dir, gate=True):
    \"\"\"Structural sanity for one swarm run dir. Returns True if sound.\"\"\"
    run_dir = Path(run_dir)
    sp = run_dir / 'summary.json'
    if not sp.exists():
        print(f'  FAIL {run_dir.name}: no summary.json (crashed run)')
        if gate: raise RuntimeError(f'{run_dir.name}: no summary.json')
        return False
    s = json.loads(sp.read_text())
    probs, warns = [], []
    emb = (s.get('clustering') or {}).get('embedder') or s.get('embedder')
    if not emb or 'UNAVAILABLE' in str(emb).upper():
        probs.append(f'embedder dead ({emb!r}) -> singleton clusters, dedup off')
    cl = s.get('clustering') or {}
    if cl.get('n_initial_signals', 0) >= 8 and cl.get('multi_member_clusters', 0) == 0:
        probs.append('clustering inert: every INITIAL is a singleton')
    if not s.get('engines'):
        probs.append('engines not recorded — router never engaged')
    shares = s.get('action_shares') or {}
    if s.get('task_type') in ('debate', 'analysis') and not shares.get('VALIDATE'):
        probs.append('validator never ran (VALIDATE share = 0) -> verification numbers are fake')
    av = s.get('avg_verification_score')
    if av is not None and av <= 0:
        warns.append(f'avg_verification_score={av} (<= 0)')
    if not s.get('quality_met'):
        warns.append(f\"quality_met=false (halt: {s.get('convergence_reason')})\")
    for p in probs: print(f'  FAIL {run_dir.name}: {p}')
    for w in warns: print(f'  warn {run_dir.name}: {w}')
    if not probs and not warns: print(f'  ok   {run_dir.name}')
    if probs and gate:
        raise RuntimeError(f'{run_dir.name}: broken run — fix before spending more GPU time')
    return not probs

def latest_run():
    base = Path(os.environ.get('SWARM_OUTPUTS_BASE_DIR', 'outputs'))
    base = base if base.exists() else Path('outputs')
    runs = [p for p in base.glob('*') if p.is_dir()]
    return max(runs, key=lambda p: p.stat().st_mtime) if runs else None

def sync_drive():
    \"\"\"Persist everything judge-relevant to Drive. Cheap; run often.\"\"\"
    for src, dst in (('outputs', f'{DRIVE_BASE}/runs_repo'),
                     ('eval/results', f'{DRIVE_BASE}/eval_results')):
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
            print(f'synced {src} -> {dst}')

print('health_check / latest_run / sync_drive defined')"""))

# ---------------------------------------------------------------- 9 single run
cells.append(md(
"""## Cell 9 — Single real run (health-gated)

Same command for every backend — Cell 6 decides routing. First GPU run
downloads the local model (~3–30 min). Task types: debate | analysis |
creative | problem_solving | coding."""))
cells.append(code(
"""TASK   = 'debate'
PROMPT = 'Cities should ban private cars to fight climate change.'
# --workers: keep modest on free Groq so you don't slam the RPM cap.
!python run_swarm.py "{TASK}" "{PROMPT}" --workers=8

r = latest_run()
if r: health_check(r)
sync_drive()"""))

# ---------------------------------------------------------------- 10 inspect
cells.append(md("## Cell 10 — Inspect the latest run"))
cells.append(code(
"""import json
r = latest_run()
if not r:
    print('No runs yet — run Cell 9.')
else:
    print('run:', r, '\\n')
    if (r/'answer.txt').exists():
        print('--- answer.txt (reader answer) ---')
        print((r/'answer.txt').read_text()[:4000])
    if (r/'diagnostics.md').exists():
        print(f\"\\n(diagnostics.md present: {len((r/'diagnostics.md').read_text())} chars of field telemetry)\")
    s = json.loads((r/'summary.json').read_text())
    print('\\n--- summary ---')
    for k in ('task_type','bundle','engines','convergence_reason','quality_met',
              'wall_clock_s','total_iterations','n_clusters','clustering',
              'avg_verification_score','max_verification_score','output_diversity'):
        if k in s: print(f'  {k}: {s[k]}')"""))

# ---------------------------------------------------------------- 11 eval generate
cells.append(md(
"""## ⭐ Cell 11 — Eval: generate conditions A / B / E (+D optional)

The experiment that decides the project. All conditions on the SAME
pre-registered prompt set (`eval/prompts.py`):

| | Condition | What it isolates |
|---|---|---|
| **A** | `swarm(M)` | the orchestration |
| **B** | `direct(M)` | the model alone — the amplification baseline |
| **E** | `synth-prompt(M)` | one call given the swarm's own synthesis instruction — **the attribution control**. If E ties A, the value was the prompt, not the swarm |
| D | `direct(M+)` | a stronger model — the "small+swarm beats big-direct" claim (add `--strong-model`) |

Deltas: **Δ_amp = A−B**, **Δ_vs_prompt = A−E**, **Δ_vs_strong = A−D**.
Every condition-A run is health-gated before anything is judged; results sync
to Drive immediately (nothing depends on the session surviving)."""))
cells.append(code(
"""import time
EXP     = f\"abe_{time.strftime('%Y%m%d_%H%M')}\"   # experiment name (one per eval)
M_MODEL = 'llama-3.1-8b-instant'                    # model M for A/B/E; blank -> local
N_PROMPTS = 8                                        # bump toward 20+ for a real verdict

!python -m eval.ab_harness --name "{EXP}" --mini {N_PROMPTS} --conditions ABE --model "{M_MODEL}"
# For condition D add:  --conditions ABDE --strong-model llama-3.3-70b-versatile

# Health-gate every condition-A (swarm) run generated by this experiment.
from pathlib import Path
import os
base = Path(os.environ.get('SWARM_OUTPUTS_BASE_DIR', 'outputs'))
a_runs = sorted(base.glob(f'delta_{EXP}_*'))
print(f'\\nhealth-gating {len(a_runs)} condition-A run(s):')
bad = [r for r in a_runs if not health_check(r, gate=False)]
sync_drive()
if bad:
    raise RuntimeError(f'{len(bad)} broken swarm run(s) — do NOT judge this batch; '
                       'fix the failure and regenerate.')"""))

# ---------------------------------------------------------------- 12 pack
cells.append(md(
"""## Cell 12 — Build the blind judging packet → hand to Claude

Colab's judging job ends here: it emits raw, blind, format-normalized answer
pairs (both orders, shuffled, swarm tells stripped). **Claude is the judge** —
a stronger and different-family model than anything in the conditions, so no
self-preference. The packet is split into bounded parts (`--max-chars`) so
nothing truncates mid-item; feed Claude one part per turn."""))
cells.append(code(
"""!python -m eval.judge eval/results/"{EXP}" --pack   # add --max-chars 40000 for smaller parts
from google.colab import files
import glob
parts = sorted(glob.glob(f'eval/results/{EXP}/judging_packet*.md'))
print(f'{len(parts)} part(s) — give each to Claude in its own turn:')
for p in parts: print('  ', p)
sync_drive()
files.download(f'eval/results/{EXP}/judging_packets.zip')"""))

# ---------------------------------------------------------------- 13 verdicts
cells.append(md(
"""## Cell 13 — Verdicts back → report

Claude returns `1` / `2` / `tie` per item id. Upload `verdicts.json` (or paste
it inline), then this cell tallies Wilson CIs, deltas, and cost multiples —
pure arithmetic, no LLM.

Reading the report: a **real win** = Wilson lower bound clearly above 50%
**with order-agreement** — and check Δ_vs_prompt (A−E) before crediting the
orchestration."""))
cells.append(code(
"""# Option A: upload the verdicts file Claude produced
# from google.colab import files; files.upload()   # -> verdicts.json
# Option B: paste inline:
# import json; json.dump({...}, open(f'eval/results/{EXP}/verdicts.json','w'), indent=2)

!python -m eval.judge eval/results/"{EXP}" --score-verdicts verdicts.json
import pathlib
rep = pathlib.Path(f'eval/results/{EXP}/report.md')
print(rep.read_text() if rep.exists() else 'fill verdicts.json first')
sync_drive()"""))

# ---------------------------------------------------------------- 14 sweep
cells.append(md(
"""## Cell 14 — Size sweep: Δ_amp across model strengths (optional)

A vs B at several sizes of M → plot Δ_amp against size. Flat/rising = the
thesis scales; falling to zero = small-model crutch. Each arm is health-gated
and synced before its packet is built — a broken arm is skipped, not judged."""))
cells.append(code(
"""import subprocess, os
from pathlib import Path
SWEEP = {
    'small': 'llama-3.1-8b-instant',
    'mid':   'llama-3.3-70b-versatile',
    # 'frontier': '...',   # add a 3rd point
}
base = Path(os.environ.get('SWARM_OUTPUTS_BASE_DIR', 'outputs'))
for size, model in SWEEP.items():
    name = f'sweep_{size}'
    subprocess.run(['python','-m','eval.ab_harness','--name',name,
                    '--conditions','AB','--model',model], check=False)
    runs = sorted(base.glob(f'delta_{name}_*'))
    ok = all(health_check(r, gate=False) for r in runs) if runs else False
    sync_drive()
    if not ok:
        print(f'!! arm {name!r} has broken runs — NOT packing it for judging'); continue
    subprocess.run(['python','-m','eval.judge',f'eval/results/{name}','--pack'],
                   check=False)
print('Hand each healthy arm\\'s judging_packets.zip to Claude, return verdicts, then:')
for size in SWEEP:
    print(f'  !python -m eval.judge eval/results/sweep_{size} --score-verdicts verdicts.json')"""))

# ---------------------------------------------------------------- troubleshooting
cells.append(md(
"""## Troubleshooting

**`health_check` FAIL: validator never ran** — the eval/task config suppressed
VALIDATE or the validator crashed at startup; check the run banner and
`validator_raw.log`. Do not judge the batch.

**`health_check` FAIL: embedder dead / singleton clusters** — huggingface-hub
1.x or torchcodec poisoning (Cell 4 handles both; re-run it, then
Runtime>Restart). If the embedder is fine but clusters are still singletons,
lower `SWARM_CLUSTER_JOIN_THRESHOLD` (e.g. 0.6).

**Groq `429` / TPD exhausted** — free tier. Lower `--workers`, keep
`SWARM_HYBRID_GROQ_ROLES='hater'`, or wait for the daily reset. The token
bucket in `core/llm_groq.py` backs off automatically.

**`ImportError: libcudart.so.13`** — vLLM 0.20+ wants CUDA 13; Colab is CUDA 12.
Cell 4 pins `vllm<0.20.0` — re-run it; never bare `pip install vllm`.

**OOM loading the local model** — pin smaller in Cell 6
(`SWARM_MODEL='Qwen/Qwen2.5-3B-Instruct'` on T4) or pass `--small`.

**`Engine core initialization failed`** — pipeline called in-kernel. Always use
the `!python run_swarm.py ...` shell form (vLLM needs a subprocess).

**Session died mid-eval** — everything already written is on Drive
(`sync_drive()` runs after each phase). Re-run Cells 1–6 + 8, set `EXP` to the
old name by hand, and continue from where the last sync left off."""))

nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3",
      "display_name": "Python 3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}

out_path = os.path.join(os.path.dirname(__file__), "colab_swarm.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(f"wrote {out_path} ({len(cells)} cells)")
