"""Generator for notebooks/colab_swarm.ipynb (robust JSON > hand-writing).

Run:  python notebooks/_build_colab.py
Produces a Colab notebook with THREE backends as a first-class choice:
  hybrid (local GPU + Groq, recommended), groq-only (no GPU), local-only (vLLM).
Re-run to regenerate after editing cell text below.
"""
import json
import os

def md(src): return {"cell_type": "markdown", "metadata": {}, "source": src}
def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src}

cells = []

cells.append(md(
"""# Stigmergic Swarm — Colab Runner (hybrid local-GPU + Groq)

Runs the pipeline at the repository root. Pick a **backend** — that's the
one real decision:

| Backend | `SWARM_BACKEND` | GPU? | Groq quota | Best for |
|---|---|---|---|---|
| **Hybrid (recommended)** | `hybrid` | yes | tiny (~10-15 calls/run) | best models per role; free-Groq-safe |
| Groq-only | *(unset)* + key | no (CPU ok) | higher | zero GPU units; full model-family diversity |
| Local-only | `local` *(unset key)* | yes | none | one strong local model, no rate limits |

### Colab GPUs (what to expect)
| Tier | GPU | Local model (auto by tier) | Note |
|---|---|---|---|
| Free | **T4 16 GB** | Qwen2.5-3B | hybrid works (3B local + Groq synth/hater) |
| Pro | **L4 24 GB** | Qwen2.5-14B | **sweet spot** for hybrid |
| Pro+ | **A100 40 GB** | Qwen2.5-32B (AWQ) | strongest local |

### Free Groq — limits & model options (verify at console.groq.com/settings/limits)
Free tier ≈ **30 req/min per model** + a **daily token cap (TPD)**; the 70B has the
tightest TPD. By default ONLY the **hater** (~1-2/round) hits Groq — the token-heavy
**synthesizer runs LOCAL** because its end-of-run burst exhausts the free TPD
mid-render (saw this in groqrun1.txt: 429s, revision skipped, half-rendered).

| Role on Groq | Model | Why |
|---|---|---|
| hater | `llama-3.3-70b-versatile` | large model, distinct from local Qwen → real adversarial diversity |
| *(alt)* | `llama-3.1-8b-instant` | fast/cheap fallback |
| *(synthesizer, only w/ paid quota)* | `llama-3.3-70b-versatile` | depth; re-add via the role split below |

Tune the split: `SWARM_HYBRID_GROQ_ROLES="hater"` (default) or `"synthesizer,hater"`
if you have paid Groq quota; `GROQ_ROLE_HATER=...` to swap the hater model.

> **Push your local commits first** — Cell 3 clones `sfuqua6/Stigmeric-Coordination`.
> Run cells top-to-bottom; on reconnect re-run 1-5."""))

cells.append(md("## Cell 1 — Mount Drive (persist runs/KB across sessions)"))
cells.append(code(
"""from google.colab import drive
drive.mount('/content/drive')"""))

cells.append(md("## Cell 2 — Paths & environment"))
cells.append(code(
"""import os
DRIVE_BASE = '/content/drive/MyDrive/swarm'
os.environ['SWARM_OUTPUTS_BASE_DIR']    = f'{DRIVE_BASE}/runs'
os.environ['SWARM_KB_DIR']              = f'{DRIVE_BASE}/knowledge_base'
os.environ['SWARM_RETRIEVAL_CACHE_DIR'] = f'{DRIVE_BASE}/retrieval_cache'
for d in ('runs', 'knowledge_base', 'retrieval_cache', 'corpora'):
    os.makedirs(f'{DRIVE_BASE}/{d}', exist_ok=True)
os.environ['HF_HOME'] = '/content/hf_cache'; os.makedirs('/content/hf_cache', exist_ok=True)
os.environ['COLAB'] = '1'; os.environ['SWARM_QUIET_LIBS'] = '1'
print('persistent dirs under', DRIVE_BASE)"""))

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

cells.append(md(
"""## Cell 4 — Install dependencies

Always installs `openai` (Groq backend) + core deps. Installs **vLLM only if a GPU
is present** (CUDA-version-matched) — so a Groq-only CPU runtime skips the heavy,
fragile vLLM install entirely. First GPU run: ~5-10 min."""))
cells.append(code(
"""import subprocess, sys, re, shutil
# Always: Groq client + core deps (light, work on CPU runtimes).
subprocess.run([sys.executable,'-m','pip','install','-q',
    'openai','sentence-transformers','cohere','datasets','faiss-cpu',
    # huggingface_hub<1.0: hub 1.x breaks sentence-transformers' model load
    # (silently -> no embedder -> singleton clusters; see groqrun1.txt).
    'wikipedia','ddgs','requests','beautifulsoup4','tqdm','huggingface_hub>=0.26.0,<1.0'], check=True)
# torchcodec's native lib fails against the Colab image's FFmpeg and poisons
# the sentence-transformers import chain (RuntimeError: Could not load
# libtorchcodec -> embedder UNAVAILABLE -> 63/63 singleton clusters; see
# multisamplegroq.txt). The swarm decodes no audio — remove it so optional
# imports degrade cleanly. The pipeline also has a transformers AutoModel
# embedder fallback if anything similar recurs.
subprocess.run([sys.executable,'-m','pip','uninstall','-y','-q','torchcodec'],
               check=False)
print('core + openai (Groq) deps installed; torchcodec removed.')

has_gpu = shutil.which('nvidia-smi') is not None
if not has_gpu:
    print('No GPU detected — skipping vLLM (Groq-only / CPU runtime). Set SWARM_BACKEND unset + GROQ_API_KEY.')
else:
    _nvcc = subprocess.run(['nvcc','--version'], capture_output=True, text=True)
    _maj = 12
    for ln in _nvcc.stdout.splitlines():
        m = re.search(r'release (\\d+)\\.', ln)
        if m: _maj = int(m.group(1)); break
    spec = 'vllm' if _maj >= 13 else 'vllm<0.20.0'   # 0.20+ links libcudart.so.13 (CUDA13)
    print(f'CUDA {_maj}: installing {spec} ...')
    subprocess.run([sys.executable,'-m','pip','install','-q',spec,'bitsandbytes>=0.46.1'], check=True)
    try:
        import importlib; importlib.import_module('vllm.engine.async_llm_engine')
        import vllm; print(f'vLLM {vllm.__version__} OK')
    except Exception as e:
        print(f'WARNING vllm import failed: {e}\\n  Try Runtime>Restart, re-run this cell.')"""))

cells.append(md("## Cell 5 — GPU & tier detection"))
cells.append(code(
"""import shutil, sys
sys.path.insert(0, '.')
if shutil.which('nvidia-smi'):
    import subprocess; print(subprocess.run(['nvidia-smi','--query-gpu=name,memory.total',
        '--format=csv,noheader'], capture_output=True, text=True).stdout.strip())
else:
    print('No GPU (CPU runtime) — use Groq-only backend.')
from core import config
print(f'tier={config._TIER!r}  local_model={config.MODEL_NAME!r}  dtype={config.VLLM_DTYPE!r}')"""))

cells.append(md(
"""## Cell 6 — Backend & keys  ⭐ the one cell to configure

- **Hybrid** (recommended): paste a free `GROQ_API_KEY` AND set `SWARM_BACKEND='hybrid'`. Needs a GPU.
- **Groq-only**: paste `GROQ_API_KEY`, leave `SWARM_BACKEND` unset. No GPU needed.
- **Local-only**: leave `GROQ_API_KEY` blank, set `SWARM_BACKEND='local'`. Needs a GPU.

Free Groq key: https://console.groq.com/keys"""))
cells.append(code(
"""import os
GROQ_API_KEY = ''        # <-- paste free Groq key (hybrid or groq-only); blank = local-only
SWARM_BACKEND = 'hybrid' # 'hybrid' | 'local' | ''(=groq-only when key set)

if GROQ_API_KEY: os.environ['GROQ_API_KEY'] = GROQ_API_KEY
if SWARM_BACKEND: os.environ['SWARM_BACKEND'] = SWARM_BACKEND
else: os.environ.pop('SWARM_BACKEND', None)

# Groq is ~2.5s/iter and free-tier rate-limited — give the run room to converge.
os.environ['SWARM_MAX_TIME_S'] = '1800'
# Which roles ride Groq in hybrid (rest run local). Default 'hater' only — the
# synthesizer runs LOCAL (its token burst blows the free-tier daily limit). Use
# 'synthesizer,hater' only if you have paid Groq quota.
os.environ['SWARM_HYBRID_GROQ_ROLES'] = 'hater'
# Optional: pin the local model (else tier auto-selects). e.g. T4: 3B, L4: 14B.
# os.environ['SWARM_MODEL'] = 'Qwen/Qwen2.5-14B-Instruct'

# --- Clustering quality knobs (the blob+dust fix; defaults are good) ---
# A/B the fix: set both to 0 to reproduce the old single-mega-cluster behaviour.
# os.environ['SWARM_CLUSTER_JOIN_SIZE_PENALTY'] = '0.03'  # 0 disables size penalty
# os.environ['SWARM_CLUSTER_RECLUSTER_EVERY']   = '25'    # 0 disables periodic recluster

_key = 'set' if os.environ.get('GROQ_API_KEY') else 'BLANK'
print(f"GROQ_API_KEY: {_key} | SWARM_BACKEND: {os.environ.get('SWARM_BACKEND','(groq-only or local)')}")"""))

cells.append(md("## Cell 7 — MOCK sanity check (no GPU / no key / no network)"))
cells.append(code(
"""!MOCK_LLM=1 SWARM_MIN_TIME_S=0 SWARM_MIN_ITERATIONS=5 SWARM_MAX_ITERATIONS=20 \\
    python run_swarm.py debate "Cities should ban private cars" --corpus=placeholder"""))

cells.append(md(
"""## Cell 8 — Real run

Uses whatever you set in Cell 6. Same command for all backends — `SWARM_BACKEND` +
`GROQ_API_KEY` decide routing. Task types: debate | analysis | creative |
problem_solving | coding. First GPU run downloads the local model (~3-30 min)."""))
cells.append(code(
"""TASK   = 'debate'
PROMPT = 'Cities should ban private cars to fight climate change.'
# --workers: keep modest on free Groq so you don't slam the RPM cap.
!python run_swarm.py "{TASK}" "{PROMPT}" --workers=8"""))

cells.append(md(
"""### Backend cheat-sheet
```bash
# hybrid  (Cell 6: key + SWARM_BACKEND='hybrid'):     local GPU model + Groq synth/hater
# groq    (Cell 6: key + SWARM_BACKEND=''):           all roles on Groq, no GPU
# local   (Cell 6: no key + SWARM_BACKEND='local'):   vLLM only; add --small on T4
!python run_swarm.py debate "..." --small --workers=8        # T4 local-only
!python run_swarm.py debate "..." --mode=baseline --workers=8 # A/B: independent agents, no stigmergy
```"""))

cells.append(md("## Cell 9 — Inspect the latest run"))
cells.append(code(
"""import json, os
from pathlib import Path
base = Path(os.environ.get('SWARM_OUTPUTS_BASE_DIR','outputs'))
base = base if base.exists() else Path('outputs')
runs = sorted(base.glob('*'), key=lambda p: p.stat().st_mtime) if base.exists() else []
if not runs:
    print('No runs yet — run Cell 8.')
else:
    latest = runs[-1]; print('latest:', latest, '\\n')
    if (latest/'answer.txt').exists(): print((latest/'answer.txt').read_text()[:4000])
    if (latest/'summary.json').exists():
        s = json.loads((latest/'summary.json').read_text())
        for k in ('task_type','bundle','n_clusters','clustering','embedder',
                  'degraded','silent_roles','convergence_reason','wall_clock_s'):
            if k in s: print(f'  {k}: {s[k]}')"""))

cells.append(md(
"""## ⭐ Amplification-delta experiment (`eval/`) — the headline deliverable

This is the experiment that decides the project: does wrapping a model **M** in
the swarm beat just **calling M directly**? Measured blind, across model sizes.

Five conditions on ONE pre-registered prompt set (`eval/prompts.py`):

| | Condition | What |
|---|---|---|
| **A** | `swarm(M)` | your `run_swarm` output |
| **B** | `direct(M)` | one *strong* call to M — the real baseline |
| **C** | `cheap-scaffold(M)` | verify-then-revise (or best-of-N) — the "worth it vs 5 min of effort?" control |
| **D** | `direct(M+)` | one call to a *stronger* model — the "small+swarm beats big-direct" claim |
| **E** | `synth-prompt(M)` | one call to M given the swarm's own synthesis instruction (map positions → context-dependence → second-order effects → conditions conclusion) — the **attribution control**: if E ties A, the value was the prompt, not the orchestration |

Deltas: **Δ_amp = A−B**, **Δ_vs_simple = A−C**, **Δ_vs_strong = A−D**,
**Δ_vs_prompt = A−E**.

**Colab generates the answers; Claude is the judge.** Colab strips the swarm's
formatting tells and emits a **blind packet** (`judging_packet.md`) — answers
shown as anonymized Response 1/2, both orders, shuffled. You hand that to Claude
(a far stronger judge than any free Groq model you'd otherwise settle for); drop
the verdicts back and Colab tallies a **win-rate with a Wilson 95% CI** and a
**cost multiple** — no weaker LLM judge anywhere in the loop.

> **Guards against fooling yourself:** strong baseline B (not a strawman),
> hard format-normalization, **Claude as a neutral judge** (a different family
> from the Groq/local condition models — no self-preference), randomized
> both-orders (ties on disagreement), and a **pre-registered** prompt set (no
> peeking). A real win = Wilson lower bound clearly above 50%."""))

cells.append(md("### Cell 10 — Minimal first run (A vs B vs E, 8 prompts)"))
cells.append(code(
"""# Generate conditions A (swarm) + B (direct-M) + E (synthesis-prompt control).
# E costs one extra call per prompt and answers the attribution question.
# Mock plumbing: prefix MOCK_LLM=1 and add SWARM_MIN_TIME_S=0 SWARM_MIN_ITERATIONS=5.
M_MODEL = 'llama-3.1-8b-instant'   # model M (A/B/E). Blank -> local/mock.
!python -m eval.ab_harness --name delta_min --mini 8 --conditions ABE --model "{M_MODEL}"
"""))

cells.append(md(
"""### Cell 11 — Build the blind judging packet — **this is the Colab deliverable**

Colab's job ends here: it emits the **raw, blind, normalized answers** for
**Claude** to judge. No weaker LLM judge runs — the judge is Claude, not some
Groq model you'd otherwise settle for.

Every pair is shown twice (both orders) as anonymized *Response 1 / Response 2*,
swarm formatting stripped, items shuffled — so the judge can't tell which is the
swarm.

**The packet is split into bounded PARTS** (`--max-chars`, default 60 KB each)
so a big run can't truncate mid-item when you paste it to Claude — a dropped
item would be a silently dropped judgment. Feed Claude **one part per turn**;
verdicts merge across all parts by item id. Everything is bundled in
`judging_packets.zip` for one download."""))
cells.append(code(
"""!python -m eval.judge eval/results/delta_min --pack   # add --max-chars 40000 for smaller parts
from google.colab import files
import glob
parts = sorted(glob.glob('eval/results/delta_min/judging_packet*.md'))
print(f'{len(parts)} part(s) — give each to Claude in its own turn:')
for p in parts: print('  ', p)
files.download('eval/results/delta_min/judging_packets.zip')   # all parts, one download
"""))

cells.append(md(
"""### Cell 11b — Paste Claude's verdicts back, assemble the report

Claude returns a verdict (`1` / `2` / `tie`) per item id. Put them in
`verdicts.json` (or have Claude emit the whole JSON and upload it), then assemble
the report — Wilson CIs, deltas, cost multiples. **No LLM runs here either**;
this is pure tallying of Claude's judgments."""))
cells.append(code(
"""# Option A: upload a verdicts.json Claude produced
# from google.colab import files; files.upload()   # -> verdicts.json
# Option B: paste the dict inline:
# import json; json.dump({...}, open('eval/results/delta_min/verdicts.json','w'), indent=2)

!python -m eval.judge eval/results/delta_min --score-verdicts verdicts.json
import pathlib
rep = pathlib.Path('eval/results/delta_min/report.md')
print(rep.read_text() if rep.exists() else 'fill verdicts.json first')
"""))

cells.append(md(
"""### Cell 12 — The size sweep (the actual deliverable)

Run **A vs B** at three model strengths so you can plot Δ_amp against size.
Flat/rising = the thesis holds and is huge; falling to zero/negative = it's a
small-model crutch. Each arm sets a different M; point `--model` at small / mid /
frontier. This generates a **packet per arm for Claude to judge** — same blind
protocol, no weaker LLM judge."""))
cells.append(code(
"""SWEEP = {
    'small':    'llama-3.1-8b-instant',
    'mid':      'llama-3.3-70b-versatile',
    # 'frontier': 'deepseek-r1-distill-llama-70b',   # add a 3rd point
}
import subprocess, pathlib
for size, model in SWEEP.items():
    name = f'sweep_{size}'
    subprocess.run(['python','-m','eval.ab_harness','--name',name,
                    '--conditions','AB','--model',model], check=False)
    subprocess.run(['python','-m','eval.judge',f'eval/results/{name}','--pack'],
                   check=False)
print('Hand each arm\\'s judging_packets.zip (its parts) to Claude, drop the verdicts back, then:')
for size in SWEEP:
    print(f"  !python -m eval.judge eval/results/sweep_{size} --score-verdicts verdicts.json")
"""))

cells.append(md("### Cell 12b — Δ_amp(M) size curve (after all arms are judged)"))
cells.append(code(
"""import json, pathlib
print('Δ_amp(M) size curve:')
print(f"{'size':10} {'model':32} {'win-rate':>9} {'Wilson95':>16} real?")
SWEEP = {'small':'llama-3.1-8b-instant','mid':'llama-3.3-70b-versatile'}
for size, model in SWEEP.items():
    sc = pathlib.Path(f'eval/results/sweep_{size}/scores.json')
    if not sc.exists():
        print(f'{size:10} (not judged yet)'); continue
    r = json.loads(sc.read_text())['pairs'].get('A_vs_B', {})
    wl = r.get('wilson95', [0,0])
    print(f"{size:10} {model:32} {r.get('win_rate',0):>8.0%} "
          f"[{wl[0]:.2f},{wl[1]:.2f}]   {r.get('real_win')}")
"""))

cells.append(md(
"""## Troubleshooting

**Groq `429` / rate-limited or quota exhausted** — free tier. The token-bucket in
`core/llm_groq.py` backs off automatically; if it persists, lower `--workers`, keep
`SWARM_HYBRID_GROQ_ROLES` small (just `hater`), or wait for the daily TPD reset.
Switch to local-only (Cell 6: blank key, `SWARM_BACKEND='local'`) to run with no Groq.

**All clusters are singletons / `summary.json` shows `embedder: UNAVAILABLE`** — the
sentence-transformers model failed to load (usually `huggingface-hub` 1.x vs the
required `<1.0`), so semantic clustering + dedup are OFF. Cell 4 now pins a
compatible hub; if it still fails, run `pip install "huggingface-hub>=0.26.0,<1.0"`
and restart. If `embedder` is active but clusters are still all singletons, lower
`SWARM_CLUSTER_JOIN_THRESHOLD` (e.g. 0.6).

**`ImportError: libcudart.so.13`** (local/hybrid) — vLLM 0.20+ wants CUDA 13; Colab is
CUDA 12. Cell 4 installs `vllm<0.20.0` automatically — re-run it; do NOT `pip install vllm`.

**OOM loading the local model** (hybrid/local) — the local model is too big for the GPU.
Pin a smaller one in Cell 6: `os.environ['SWARM_MODEL']='Qwen/Qwen2.5-3B-Instruct'` (T4)
or `7B` (L4), or use `--small`.

**`Engine core initialization failed`** — pipeline called in-kernel. Always use the
`!python run_swarm.py ...` shell form (vLLM needs a subprocess).

**Hybrid: synthesizer/hater not on Groq** — check Cell 6 set BOTH the key and
`SWARM_BACKEND='hybrid'`; the run banner prints `hybrid backend (local + Groq): {...}`."""))

nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3",
      "display_name": "Python 3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}

out_path = os.path.join(os.path.dirname(__file__), "colab_swarm.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(f"wrote {out_path} ({len(cells)} cells)")
