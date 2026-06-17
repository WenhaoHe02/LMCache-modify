#!/usr/bin/env python3
"""
SSD retrieve benchmark: nodirect vs O_DIRECT, gs=1
A few key lengths to compare against the multidisk reference data.
Each mode: clear SSD cache → launch → prefill+sleep → retrieve×3
"""
import requests, subprocess, time

BASE        = "http://localhost:8000/v1/completions"
MODEL       = "qwen14b1m"
N_RET       = 3
WRITE_SLEEP = 20

NODIRECT_SH = "/root/qwen14b1m_tp8_nodirect.sh"
ODIRECT_SH  = "/root/qwen14b1m_tp8_odirect.sh"

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

SENTENCE     = "The quick brown fox jumped over the lazy dog near the river. "
TOKS_PER_REP = 13
TARGETS      = [64, 128, 256, 320]
LENGTHS      = [(f"{t}K", SENTENCE * (t * 1000 // TOKS_PER_REP)) for t in TARGETS]


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
    print(f"  [{tag}] prefill={prefill_t:.3f}s avg_ret={avg:.3f}s", flush=True)
    return prefill_t, avg, actual_toks


nodirect_res = {}
odirect_res  = {}

# ── Mode 1: nodirect gs=1 ─────────────────────────────────────────────────────
clear_ssd()
launch(NODIRECT_SH, gs=1)
print("\n" + "="*60)
print("  MODE 1: SSD nodirect gs=1")
print("="*60)
for tag, prompt in LENGTHS:
    print(f"\n[{tag}]", flush=True)
    pf, avg, tok = measure(prompt, f"nodirect-{tag}")
    nodirect_res[tag] = (pf, avg, tok)

# ── Mode 2: O_DIRECT gs=1 ─────────────────────────────────────────────────────
clear_ssd()
launch(ODIRECT_SH, gs=1)
print("\n" + "="*60)
print("  MODE 2: SSD O_DIRECT gs=1")
print("="*60)
for tag, prompt in LENGTHS:
    print(f"\n[{tag}]", flush=True)
    pf, avg, tok = measure(prompt, f"odirect-{tag}")
    odirect_res[tag] = (pf, avg, tok)

# ── Summary ───────────────────────────────────────────────────────────────────
# Reference from lmcache_speedup_multidisk_20260507.csv (closest token counts)
REF_SSD = {"64K": 0.390, "128K": 0.794, "256K": 1.629, "320K": 2.056}

print("\n\n" + "="*80)
print(f"  {'Length':<8} {'Ref_SSD':>10} {'Nodirect':>10} {'O_DIRECT':>10} {'nd/ref':>8} {'od/ref':>8}")
print("  " + "-"*74)
for tag, _ in LENGTHS:
    ref  = REF_SSD.get(tag, float("nan"))
    nd   = nodirect_res.get(tag, (0, float("nan"), 0))[1]
    od   = odirect_res.get(tag,  (0, float("nan"), 0))[1]
    print(f"  {tag:<8} {ref:>10.3f} {nd:>10.3f} {od:>10.3f} "
          f"{nd/ref:>7.2f}x {od/ref:>7.2f}x")
print("  nd/ref and od/ref: >1 = slower than reference, <1 = faster")
print("="*80)
