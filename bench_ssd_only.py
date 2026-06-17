#!/usr/bin/env python3
"""
SSD nodirect bench (orig lmcache, use_layerwise=false, max_local_cpu_size=20).
Records both prefill (recompute) and SSD retrieve times.
SSD cache is cleared before run.
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

# Reference SSD retrieve times from lmcache_speedup_multidisk_20260507.csv
REF_SSD = {
    "16K":  0.129, "64K":  0.390, "128K": 0.794,
    "256K": 1.629, "320K": 2.056, "384K": 2.300,
}


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
        shell(f"find {d} -type f -delete 2>/dev/null || true")
    print("    SSD cache cleared", flush=True)

def launch(script):
    print(f"\n*** launch: {script} ***", flush=True)
    shell("docker rm -f qwen14b1m-tp8 2>/dev/null")
    shell(f"bash {script}")
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


ssd_res = {}

clear_ssd()
launch(SSD_LAUNCH)
print("\n" + "="*60)
print("  SSD nodirect (orig lmcache, use_layerwise=false)")
print("="*60)
for tag, prompt in LENGTHS:
    print(f"\n[{tag}]", flush=True)
    pf, avg, warm, tok = measure(prompt, f"ssd-{tag}")
    ssd_res[tag] = (pf, avg, warm, tok)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n\n" + "="*88)
print(f"  {'Length':<8} {'Tokens':>8} {'SSD_pf':>10} {'SSD_ret':>10} "
      f"{'Ref_ret':>10} {'vs_ref':>8} {'pf/ret':>8}")
print("  " + "-"*82)
for tag, _ in LENGTHS:
    pf, ret, warm, tok = ssd_res.get(tag, (float("nan"),)*4)
    ref = REF_SSD.get(tag, float("nan"))
    vs_ref = ret / ref if ref == ref and ref > 0 else float("nan")
    ratio  = pf / ret if ret > 0 else float("nan")
    print(f"  {tag:<8} {str(tok):>8} {pf:>10.3f} {ret:>10.3f} "
          f"{ref:>10.3f} {vs_ref:>7.2f}x {ratio:>7.1f}x")
print("  vs_ref: <1 = faster than May-7 reference, pf/ret = speedup vs recompute")
print("="*88)
