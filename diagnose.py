"""Diagnostic for the swarm LLM stack.

Runs each test in its own subprocess so an OS kill (exit code -1, 137, 9,
or any negative number) doesn't take down the whole diagnostic. Reports a
summary at the end that identifies the failure point.

Usage:
    python diagnose.py

You can also test a different model:
    $env:SWARM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
    python diagnose.py
"""

import os
import subprocess
import sys
import textwrap

MODEL = os.environ.get("SWARM_MODEL") or "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
SMALL_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"   # tiny, ~1 GB, almost always loads


def run_step(label, code, timeout=600):
    """Run one test in a subprocess. Returns (ok, exit_code, stdout, stderr)."""
    print()
    print("=" * 70)
    print(f"  STEP: {label}")
    print("=" * 70)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=timeout,
        )
        if proc.stdout:
            print(proc.stdout.rstrip())
        if proc.stderr:
            print("--- stderr ---")
            print(proc.stderr.rstrip())
        print(f"--- exit code: {proc.returncode} ---")
        ok = (proc.returncode == 0)
        # negative or > 128 typically means OS-level kill
        if proc.returncode < 0 or proc.returncode in (137, 139, 143):
            print(f"!!! Process was KILLED by the OS (likely RAM/commit pressure or SEGV).")
        return ok, proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        print(f"!!! Timed out after {timeout}s.")
        return False, -999, "", ""
    except Exception as e:
        print(f"!!! Subprocess launch failed: {e}")
        return False, -998, "", str(e)


def main():
    print()
    print("#" * 70)
    print(f"  Swarm LLM diagnostic")
    print(f"  Target model: {MODEL}")
    print(f"  Probe model:  {SMALL_MODEL}")
    print("#" * 70)

    results = {}

    # 1. environment
    code = textwrap.dedent("""
        import platform, sys, os
        print('python:', sys.version.split()[0])
        print('platform:', platform.platform())
        try:
            import psutil
            vm = psutil.virtual_memory()
            sm = psutil.swap_memory()
            print(f'ram total:      {vm.total/1e9:.1f} GB')
            print(f'ram available:  {vm.available/1e9:.1f} GB')
            print(f'swap total:     {sm.total/1e9:.1f} GB')
            print(f'swap free:      {sm.free/1e9:.1f} GB')
        except ImportError:
            print('psutil not installed (pip install psutil for RAM info)')
        for pkg in ('torch', 'transformers', 'accelerate', 'bitsandbytes', 'safetensors'):
            try:
                m = __import__(pkg)
                print(f'{pkg:<14} {getattr(m, "__version__", "unknown")}')
            except ImportError as e:
                print(f'{pkg:<14} NOT INSTALLED ({e})')
    """)
    results["env"] = run_step("environment (versions, RAM, swap)", code, timeout=60)

    # 2. torch + CUDA
    code = textwrap.dedent("""
        import torch
        print('torch.cuda.is_available:', torch.cuda.is_available())
        if torch.cuda.is_available():
            print('cuda version:', torch.version.cuda)
            print('device:', torch.cuda.get_device_name(0))
            free, total = torch.cuda.mem_get_info()
            print(f'gpu mem: {free/1e9:.2f} GB free / {total/1e9:.2f} GB total')
    """)
    results["torch"] = run_step("torch + CUDA", code, timeout=60)

    # 3. bitsandbytes — does it import cleanly?
    code = textwrap.dedent("""
        import bitsandbytes as bnb
        print('bitsandbytes version:', bnb.__version__)
        # bnb 0.42+ removed cuda_setup module; test the kernel directly instead
        import torch
        if torch.cuda.is_available():
            from bitsandbytes.nn import Linear4bit
            layer = Linear4bit(32, 16, bias=False).cuda()
            x = torch.randn(2, 32, device='cuda', dtype=torch.float16)
            y = layer(x)
            print('4-bit kernel forward pass succeeded:', y.shape)
        else:
            print('skipping kernel test, no CUDA')
    """)
    results["bnb"] = run_step("bitsandbytes 4-bit kernel sanity", code, timeout=60)

    # 4. load a tiny model end-to-end (catches everything except RAM pressure)
    code = textwrap.dedent(f"""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
        tok = AutoTokenizer.from_pretrained({SMALL_MODEL!r})
        print('tokenizer loaded')
        model = AutoModelForCausalLM.from_pretrained(
            {SMALL_MODEL!r},
            quantization_config=bnb,
            device_map='auto',
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
        print('small model loaded ok')
        inputs = tok('Hello', return_tensors='pt').to(next(model.parameters()).device)
        out = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        decoded = tok.decode(out[0], skip_special_tokens=True)
        print('decode:', decoded.encode('ascii', 'replace').decode('ascii'))
    """)
    results["small"] = run_step(f"load {SMALL_MODEL} (small model, full pipeline)", code, timeout=300)

    # 5. the actual target model
    code = textwrap.dedent(f"""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
        tok = AutoTokenizer.from_pretrained({MODEL!r})
        print('tokenizer loaded')
        # report RAM right before the heavy load
        try:
            import psutil
            vm = psutil.virtual_memory()
            print(f'ram available before load: {{vm.available/1e9:.1f}} GB')
        except ImportError:
            pass
        model = AutoModelForCausalLM.from_pretrained(
            {MODEL!r},
            quantization_config=bnb,
            device_map='auto',
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            max_memory={{0: '5000MiB', 'cpu': '30GiB'}},
        )
        print('TARGET model loaded ok')
        device_map = getattr(model, 'hf_device_map', {{}})
        on_gpu = sum(1 for d in device_map.values() if str(d).startswith('cuda') or d == 0)
        on_cpu = sum(1 for d in device_map.values() if str(d) == 'cpu')
        print(f'dispatch: {{on_gpu}} on GPU, {{on_cpu}} on CPU, total {{len(device_map)}}')
    """)
    results["target"] = run_step(f"load {MODEL} (target, the failing one)", code, timeout=900)

    # summary
    print()
    print("#" * 70)
    print("  SUMMARY")
    print("#" * 70)
    for name, (ok, code, _, _) in results.items():
        status = "OK" if ok else f"FAIL (exit {code})"
        print(f"  {name:<10} {status}")

    failed = [k for k, (ok, _, _, _) in results.items() if not ok]
    if not failed:
        print("\nAll steps passed. The wrapper should work.")
        return
    print()
    print("Diagnosis:")
    if "env" in failed:
        print("  - Basic environment broken; check python install.")
        return
    if "torch" in failed:
        print("  - PyTorch/CUDA not working. Reinstall torch with matching CUDA.")
        return
    if "bnb" in failed:
        print("  - bitsandbytes broken. On Windows try:")
        print("      pip install --upgrade bitsandbytes")
        print("      pip install bitsandbytes --index-url https://jllllll.github.io/bitsandbytes-windows-webui  # for older bnb on Windows")
        return
    if "small" in failed and "target" in failed:
        print("  - Even the small model failed but bnb kernels work.")
        print("  - This is unusual. Check the stderr of the 'small' step above.")
        return
    if "small" not in failed and "target" in failed:
        print("  - Small model loads fine; target model fails.")
        print("  - This is almost certainly system RAM / commit limit pressure")
        print("    during the fp16->4bit quantization transient (~14 GB peak).")
        print()
        print("  Try in order:")
        print("    1. Raise pagefile: System Properties -> Advanced -> Performance")
        print("       Settings -> Advanced -> Virtual Memory -> set custom size")
        print("       to 32768 MB initial / 65536 MB max. Reboot.")
        print("    2. Close other apps before launching (browsers, IDEs, Docker).")
        print("    3. Use a 3B model as a stepping stone:")
        print("         $env:SWARM_MODEL = 'Qwen/Qwen2.5-3B-Instruct'")
        print("    4. Switch to a pre-quantized GGUF model via llama-cpp-python")
        print("       (loads directly at 4-bit, no fp16 transient peak):")
        print("         pip install llama-cpp-python")
        print("         (and wire it into core/llm.py as an alternative backend)")


if __name__ == "__main__":
    main()
