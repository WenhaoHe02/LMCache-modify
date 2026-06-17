#!/usr/bin/env python3
"""
Group-size prefetch benchmark with O_DIRECT=True, tp=4 (GPU 4-7).

Phase 1 (gs=1): populate all lengths + retrieve gs=1
Phase 2 (gs=4, ONE restart): retrieve gs=4 on same SSD data

Output: Length | Tokens | Populate(重算) | Retrieve gs=1 | Retrieve gs=4 | Speedup
"""
import requests, subprocess, time

BASE        = "http://localhost:8000/v1/completions"
MODEL       = "qwen14b1m"
LAUNCH_SH   = "/root/qwen14b1m_tp4_odirect.sh"
CONTAINER   = "qwen14b1m-tp4"
N_RETRIEVE  = 3
WAIT_SSD_S  = 40

# tp=4 with O_DIRECT — keeping lengths that complete within vLLM timeout
# ~13 tokens/rep empirical for these sentences
SENTENCES = [
    ("32K",   "The quick brown fox jumped over the lazy dog near the river. ",   2500),
    ("64K",   "She sells seashells by the seashore where the waves crash now. ",  5000),
    ("128K",  "How much wood could a woodchuck chuck if it could chuck wood. ",   10000),
    ("256K",  "Peter Piper picked a peck of pickled peppers from the garden. ",   20000),
]


def shell(cmd, check=True):
    return subprocess.run(cmd, shell=True, check=check,
                          capture_output=True, text=True)

def wait_ready(timeout=400):
    for i in range(timeout // 3):
        if shell("curl -fsS http://127.0.0.1:8000/v1/models", check=False).returncode == 0:
            print(f"    server ready ({i*3}s)", flush=True)
            return
        if i % 10 == 0 and i > 0:
            print(f"    waiting... {i*3}s", flush=True)
        time.sleep(3)
    raise TimeoutError("server not ready")

def switch_gs(gs: int):
    print(f"\n  *** switching to group_size={gs} ***", flush=True)
    shell(f"bash {LAUNCH_SH} {gs}")
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

def warmup(prompt):
    print("    [warmup — discarded]", flush=True)
    do_request(prompt, "warmup")
    flush_gpu()

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


# ── startup ───────────────────────────────────────────────────────────────────
print("Clearing SSD caches ...", flush=True)
for d in ["/mnt/nvme0", "/mnt/nvme2", "/mnt/nvme3", "/mnt/nvme4",
          "/mnt/nvme5", "/mnt/nvme6", "/mnt/nvme8", "/mnt/nvme9"]:
    shell(f"find {d}/lmcache/qwen14b1m -mindepth 1 -delete 2>/dev/null || true", check=False)

switch_gs(1)

# ── Phase 1: gs=1 ─────────────────────────────────────────────────────────────
print("\n" + "="*64, flush=True)
print("  PHASE 1: populate + retrieve gs=1  (O_DIRECT=True, tp=4)", flush=True)
print("="*64, flush=True)

pop_times = {}
gs1_avgs  = {}
prompts   = {}

do_request("Hello world.", "server-warmup")

for tag, sentence, reps in SENTENCES:
    prompt = sentence * reps
    prompts[tag] = prompt

    print(f"\n  [{tag}] populate ...", flush=True)
    t_pop, actual_toks = do_request(prompt, "populate")
    pop_times[tag] = (t_pop, actual_toks)

    print(f"  [{tag}] waiting {WAIT_SSD_S}s for SSD write ...", flush=True)
    time.sleep(WAIT_SSD_S)
    flush_gpu()

    print(f"  [{tag}] retrieve gs=1 ...", flush=True)
    _, gs1_avgs[tag] = measure_retrieve(prompt, "gs=1")
    flush_gpu()

# ── Phase 2: gs=4 ─────────────────────────────────────────────────────────────
switch_gs(4)

print("\n" + "="*64, flush=True)
print("  PHASE 2: retrieve gs=4  (O_DIRECT=True, same SSD data)", flush=True)
print("="*64, flush=True)

gs4_avgs = {}
warmup(prompts[SENTENCES[0][0]])

for tag, sentence, reps in SENTENCES:
    print(f"\n  [{tag}] retrieve gs=4 ...", flush=True)
    _, gs4_avgs[tag] = measure_retrieve(prompts[tag], "gs=4")
    flush_gpu()

# ── summary ───────────────────────────────────────────────────────────────────
print("\n\n" + "="*72, flush=True)
print(f"  {'Length':<8} {'Tokens':>8} {'Populate':>10} {'gs=1 ret':>10} {'gs=4 ret':>10} {'Speedup':>9}", flush=True)
print("  " + "-"*68, flush=True)
for tag, sentence, reps in SENTENCES:
    t_pop, tok = pop_times[tag]
    g1  = gs1_avgs[tag]
    g4  = gs4_avgs[tag]
    spd = g1 / g4
    print(f"  {tag:<8} {tok:>8} {t_pop:>10.3f} {g1:>10.3f} {g4:>10.3f} {spd:>9.2f}x", flush=True)
print("="*72, flush=True)
