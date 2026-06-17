#!/usr/bin/env python3
"""
Benchmark: LMCACHE_IO_OVERLAP=0 vs LMCACHE_IO_OVERLAP=1, both gs=4, O_DIRECT=True.

思路4: swap order of next(get_generator) and mem_obj_consumer.send()
so NVMe DMA (next group SSD read) and PCIe DMA (current group H2D copy)
run concurrently on independent controllers.

Toggled by LMCACHE_IO_OVERLAP env var (default=1, set 0 to disable).

Run A: LMCACHE_IO_OVERLAP=0  — sequential (scatter first, then submit next IO)
Run B: LMCACHE_IO_OVERLAP=1  — overlap (submit next IO before scatter)

gs=1 baselines from prior tp=8 O_DIRECT run (hardcoded for reference):
  32K: 2.134s, 64K: 5.629s, 128K: 10.287s, 256K: 19.947s
"""
import requests, subprocess, time

BASE       = "http://localhost:8000/v1/completions"
MODEL      = "qwen14b1m"
LAUNCH_SH  = "/root/qwen14b1m_tp8_odirect.sh"
N_RETRIEVE = 3

GS1_BASELINE = {"32K": 2.134, "64K": 5.629, "128K": 10.287, "256K": 19.947}

SENTENCES = [
    ("32K",   "The quick brown fox jumped over the lazy dog near the river. ",   2500),
    ("64K",   "She sells seashells by the seashore where the waves crash now. ",  5000),
    ("128K",  "How much wood could a woodchuck chuck if it could chuck wood. ",   10000),
    ("256K",  "Peter Piper picked a peck of pickled peppers from the garden. ",   20000),
]


def shell(cmd, check=True):
    return subprocess.run(cmd, shell=True, check=check,
                          capture_output=True, text=True)

def wait_ready(timeout=900):
    for i in range(timeout // 3):
        if shell("curl -fsS http://127.0.0.1:8000/v1/models", check=False).returncode == 0:
            print(f"    server ready ({i*3}s)", flush=True)
            return
        if i % 20 == 0 and i > 0:
            print(f"    still waiting... {i*3}s", flush=True)
        time.sleep(3)
    raise TimeoutError("server not ready")

def launch_container(gs: int, overlap: int):
    """Start container with given gs and LMCACHE_IO_OVERLAP value."""
    print(f"\n*** launch gs={gs}, LMCACHE_IO_OVERLAP={overlap} ***", flush=True)
    # Temporarily inject -e LMCACHE_IO_OVERLAP=<value> into the launch script
    shell(f"cp {LAUNCH_SH} /tmp/launch_tp8_backup.sh")
    shell(f"sed -i 's/--entrypoint/-e LMCACHE_IO_OVERLAP={overlap} --entrypoint/' {LAUNCH_SH}")
    try:
        shell(f"bash {LAUNCH_SH} {gs}")
    finally:
        shell(f"cp /tmp/launch_tp8_backup.sh {LAUNCH_SH}")
    wait_ready()

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

def measure_retrieve(prompt, tag, n=N_RETRIEVE):
    times = []
    for i in range(n):
        t, _ = do_request(prompt, f"retrieve-{i+1}")
        times.append(t)
        if i < n - 1:
            flush_gpu()
    avg = sum(times) / len(times)
    print(f"    [{tag} avg={avg:.3f}s]", flush=True)
    return times, avg


prompts = {tag: sentence * reps for tag, sentence, reps in SENTENCES}

# ── Run A: overlap OFF ────────────────────────────────────────────────────────
launch_container(gs=4, overlap=0)
do_request("Hello world.", "server-warmup")
do_request(prompts[SENTENCES[0][0]], "warmup")
flush_gpu()

print("\n" + "="*64, flush=True)
print("  gs=4, LMCACHE_IO_OVERLAP=0  (sequential: no NVMe/PCIe overlap)", flush=True)
print("="*64, flush=True)

off_avgs = {}
for tag, _, _ in SENTENCES:
    print(f"\n[{tag}] gs=4 overlap=OFF ...", flush=True)
    _, off_avgs[tag] = measure_retrieve(prompts[tag], "gs4-OFF")
    flush_gpu()

# ── Run B: overlap ON ─────────────────────────────────────────────────────────
launch_container(gs=4, overlap=1)
do_request("Hello world.", "server-warmup")
do_request(prompts[SENTENCES[0][0]], "warmup")
flush_gpu()

print("\n" + "="*64, flush=True)
print("  gs=4, LMCACHE_IO_OVERLAP=1  (思路4: NVMe/PCIe overlap enabled)", flush=True)
print("="*64, flush=True)

on_avgs = {}
for tag, _, _ in SENTENCES:
    print(f"\n[{tag}] gs=4 overlap=ON ...", flush=True)
    _, on_avgs[tag] = measure_retrieve(prompts[tag], "gs4-ON")
    flush_gpu()

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n\n" + "="*80, flush=True)
print(f"  {'Length':<8} {'gs=1 base':>10} {'gs4 OFF':>10} {'gs4 ON':>10} {'delta%':>9} {'speedup':>9}", flush=True)
print("  " + "-"*74, flush=True)
for tag, _, _ in SENTENCES:
    base = GS1_BASELINE.get(tag, float("nan"))
    off  = off_avgs[tag]
    on   = on_avgs[tag]
    # positive delta = ON faster; negative = ON slower
    delta = (off - on) / off * 100
    spd   = off / on
    print(f"  {tag:<8} {base:>10.3f} {off:>10.3f} {on:>10.3f} {delta:>+9.1f}% {spd:>9.2f}x", flush=True)
print("="*80, flush=True)
