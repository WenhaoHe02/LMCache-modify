#!/usr/bin/env python3
"""
CPU/DRAM KV cache curve: prefill time vs retrieve time, 16 context lengths.
prefill = 1st request (recompute KV from scratch, no cache hit)
retrieve = subsequent requests (load KV from DRAM)
"""
import requests, subprocess, time

BASE   = "http://localhost:8000/v1/completions"
MODEL  = "qwen14b1m"
N_RET  = 3

SENTENCE = "The quick brown fox jumped over the lazy dog near the river. "
TOKS_PER_REP = 13  # approximate; actual count reported by server

# 16 target lengths
TARGETS = [8, 16, 24, 32, 48, 64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512]

LENGTHS = [(f"{t}K", SENTENCE * (t * 1000 // TOKS_PER_REP)) for t in TARGETS]


def do_request(prompt, label, max_tokens=1):
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

def flush_gpu():
    do_request("Hello.", "flush")

def wait_ready(timeout=900):
    for i in range(timeout // 3):
        if subprocess.run("curl -fsS http://127.0.0.1:8000/v1/models",
                          shell=True, capture_output=True).returncode == 0:
            print(f"    server ready ({i*3}s)", flush=True)
            return
        if i % 20 == 0 and i > 0:
            print(f"    still waiting... {i*3}s", flush=True)
        time.sleep(3)
    raise TimeoutError("server not ready")


# ── Launch CPU container ──────────────────────────────────────────────────────
subprocess.run("docker rm -f qwen14b1m-tp8 2>/dev/null || true", shell=True)
subprocess.run("bash /root/qwen14b1m_tp8_cpu.sh 1", shell=True, check=False)
wait_ready()
do_request("Hello world.", "server-warmup")

print("\n" + "="*64)
print("  CPU/DRAM backend — prefill vs retrieve")
print("="*64)

results = {}

for tag, prompt in LENGTHS:
    print(f"\n[{tag}]", flush=True)

    # prefill: first request hits the model with no cached KV
    prefill_t, actual_toks = do_request(prompt, "prefill")
    flush_gpu()

    # retrieve: KV now in DRAM, reload N_RET times
    ret_times = []
    for i in range(N_RET):
        t, _ = do_request(prompt, f"ret-{i+1}")
        ret_times.append(t)
        if i < N_RET - 1:
            flush_gpu()

    avg_ret  = sum(ret_times) / len(ret_times)
    avg_warm = sum(ret_times[1:]) / max(len(ret_times) - 1, 1)
    print(f"  [{tag}] prefill={prefill_t:.3f}s  avg_ret={avg_ret:.3f}s  "
          f"avg_warm={avg_warm:.3f}s  tokens={actual_toks}", flush=True)
    results[tag] = (prefill_t, avg_ret, avg_warm, actual_toks)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n\n" + "="*80)
print(f"  {'Length':<8} {'Tokens':>8} {'Prefill':>10} {'CPU_ret':>10} "
      f"{'CPU_warm':>10} {'ratio':>8}")
print("  " + "-"*72)
for tag in [t for t, _ in LENGTHS]:
    pf, cr, cw, tok = results[tag]
    ratio = pf / cr if cr > 0 else float("nan")
    print(f"  {tag:<8} {str(tok):>8} {pf:>10.3f} {cr:>10.3f} "
          f"{cw:>10.3f} {ratio:>7.1f}x")
print("  ratio = prefill / cpu_ret  (how much faster than recompute)")
print("="*80)
