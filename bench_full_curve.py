#!/usr/bin/env python3
"""
Full KV cache curve: prefill vs CPU-DRAM vs SSD (nodirect, gs=1)
Uses container's built-in lmcache (orig scripts, no local code override).
SSD cache is cleared before each SSD run.
15+ context lengths (8K-448K), dense in 64K-256K.

Modes:
  1. CPU/DRAM  (local_cpu=true,  gs=1)
  2. SSD nodirect (local_disk=nvme, gs=1) — SSD cache cleared first
"""
import requests, subprocess, time

BASE        = "http://localhost:8000/v1/completions"
MODEL       = "qwen14b1m"
N_RET       = 3
WRITE_SLEEP = 20

TARGETS  = [8, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 192, 224, 256, 320, 384, 448]

SENTENCE     = "The quick brown fox jumped over the lazy dog near the river. "
TOKS_PER_REP = 13
LENGTHS      = [(f"{t}K", SENTENCE * (t * 1000 // TOKS_PER_REP)) for t in TARGETS]

CPU_LAUNCH = "/root/qwen14b1m_tp8_cpu_orig.sh"
SSD_LAUNCH = "/root/qwen14b1m_tp8_nodirect_orig.sh"

SSD_DIRS = [
    "/mnt/nvme0/lmcache/qwen14b1m",
    "/mnt/nvme2/lmcache/qwen14b1m",
    "/mnt/nvme3/lmcache/qwen14b1m",
    "/mnt/nvme4/lmcache/qwen14b1m",
    "/mnt/nvme5/lmcache/qwen14b1m",
    "/mnt/nvme6/lmcache/qwen14b1m",
    "/mnt/nvme8/lmcache/qwen14b1m",
    "/mnt/nvme9/lmcache/qwen14b1m",
]


def shell(cmd):
    return subprocess.run(cmd, shell=True, check=False, capture_output=True, text=True)

def wait_ready(timeout=900):
    for i in range(timeout // 3):
        if shell("curl -fsS http://127.0.0.1:8000/v1/models").returncode == 0:
            print(f"    server ready ({i*3}s)", flush=True)
            return
        if i % 20 == 0 and i > 0:
            print(f"    still waiting... {i*3}s", flush=True)
        time.sleep(3)
    raise TimeoutError("server not ready after 900s")

def clear_ssd():
    for d in SSD_DIRS:
        shell(f"rm -rf {d}/*")
    print("    SSD cache cleared", flush=True)

def launch(script, gs=1):
    print(f"\n*** launch: {script} gs={gs} ***", flush=True)
    shell("docker rm -f qwen14b1m-tp8 2>/dev/null")
    shell(f"bash {script} {gs}")
    wait_ready()
    do_req("Hello world.", "warmup")

def do_req(prompt, label, max_tokens=1):
    t0 = time.perf_counter()
    r = requests.post(BASE, json={
        "model": MODEL, "prompt": prompt,
        "max_tokens": max_tokens, "temperature": 0, "stream": False,
    }, timeout=900)
    elapsed = time.perf_counter() - t0
    r.raise_for_status()
    ptok = r.json().get("usage", {}).get("prompt_tokens", "?")
    print(f"    [{label}] {elapsed:.3f}s  tokens={ptok}", flush=True)
    return elapsed, ptok

def measure(prompt, tag):
    prefill_t, actual_toks = do_req(prompt, "prefill")
    print(f"    [sleep {WRITE_SLEEP}s for KV write]", flush=True)
    time.sleep(WRITE_SLEEP)
    ret_times = []
    for i in range(N_RET):
        t, _ = do_req(prompt, f"ret-{i+1}")
        ret_times.append(t)
    avg = sum(ret_times) / len(ret_times)
    warm = sum(ret_times[1:]) / max(len(ret_times) - 1, 1)
    print(f"  [{tag}] prefill={prefill_t:.3f}s avg_ret={avg:.3f}s avg_warm={warm:.3f}s",
          flush=True)
    return prefill_t, avg, warm, actual_toks


cpu_res  = {}
ssd1_res = {}

# ── Mode 1: CPU/DRAM ──────────────────────────────────────────────────────────
launch(CPU_LAUNCH, gs=1)
print("\n" + "="*60)
print("  MODE 1: CPU/DRAM (local_cpu=true, gs=1, orig lmcache)")
print("="*60)
for tag, prompt in LENGTHS:
    print(f"\n[{tag}]", flush=True)
    pf, avg, warm, tok = measure(prompt, f"cpu-{tag}")
    cpu_res[tag] = (pf, avg, warm, tok)

# ── Mode 2: SSD nodirect gs=1 ─────────────────────────────────────────────────
clear_ssd()
launch(SSD_LAUNCH, gs=1)
print("\n" + "="*60)
print("  MODE 2: SSD nodirect gs=1 (orig lmcache)")
print("="*60)
for tag, prompt in LENGTHS:
    print(f"\n[{tag}]", flush=True)
    pf, avg, warm, tok = measure(prompt, f"ssd1-{tag}")
    ssd1_res[tag] = (pf, avg, warm, tok)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n\n" + "="*88)
print(f"  {'Length':<8} {'Tokens':>8} {'Prefill':>10} {'CPU_ret':>10} "
      f"{'SSD1_ret':>10} {'vsCPU':>8} {'vsSSD1':>8}")
print("  " + "-"*82)
for tag, _ in LENGTHS:
    pf, cpu_r, _, tok = cpu_res.get(tag,  (float("nan"),)*4)
    _,  ssd_r, _, _   = ssd1_res.get(tag, (float("nan"),)*4)
    r_cpu  = pf / cpu_r if cpu_r > 0 else float("nan")
    r_ssd  = pf / ssd_r if ssd_r > 0 else float("nan")
    print(f"  {tag:<8} {str(tok):>8} {pf:>10.3f} {cpu_r:>10.3f} "
          f"{ssd_r:>10.3f} {r_cpu:>7.1f}x {r_ssd:>7.1f}x")
print("  vsCPU/vsSSD1 = prefill / retrieve  (>1 = cache faster than recompute)")
print("="*88)
