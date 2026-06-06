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

Runs the cleaned pipeline (`Attempt At Cleaning/`). Pick a **backend** — that's the
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
tightest TPD. The hybrid design exists precisely for this: only **synthesizer**
(~1 call/cluster) and **hater** (~1-2/round) hit Groq, so the free quota lasts.

| Role on Groq | Model | Why |
|---|---|---|
| synthesizer | `llama-3.3-70b-versatile` | depth; few calls/run |
| hater | `meta-llama/llama-4-maverick-17b-128e-instruct` | different family = real adversarial diversity |
| *(alt)* | `llama-3.1-8b-instant` | fast/cheap fallback |
| *(alt reasoning)* | `qwen-qwq-32b`, `deepseek-r1-distill-llama-70b` | r1-distill is sometimes decommissioned mid-run |

Tune the split: `SWARM_HYBRID_GROQ_ROLES="synthesizer,hater"`, `GROQ_ROLE_HATER=...`.

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
REPO_ROOT='/content/swarm_repo'
if not os.path.exists(REPO_ROOT):
    subprocess.run(['git','clone',REPO_URL,REPO_ROOT], check=True)
else:
    subprocess.run(['git','-C',REPO_ROOT,'pull'], check=True)
os.chdir(f'{REPO_ROOT}/Attempt At Cleaning')
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
    'wikipedia','ddgs','requests','beautifulsoup4','tqdm','huggingface_hub'], check=True)
print('core + openai (Groq) deps installed.')

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
# Which roles ride Groq in hybrid (rest run local). Keep small for free quota.
os.environ['SWARM_HYBRID_GROQ_ROLES'] = 'synthesizer,hater'
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
        for k in ('task_type','bundle','n_clusters','convergence_reason','wall_clock_s'):
            if k in s: print(f'  {k}: {s[k]}')"""))

cells.append(md(
"""## Troubleshooting

**Groq `429` / rate-limited or quota exhausted** — free tier. The token-bucket in
`core/llm_groq.py` backs off automatically; if it persists, lower `--workers`, keep
`SWARM_HYBRID_GROQ_ROLES` small (synthesizer,hater), or wait for the daily TPD reset.
Switch to local-only (Cell 6: blank key, `SWARM_BACKEND='local'`) to run with no Groq.

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
