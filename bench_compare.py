#!/usr/bin/env python3
"""
Group-size prefetch benchmark: gs=1 vs gs=4 retrieve latency.

Phase 1 (gs=1 container):
  For each length: populate SSD, then measure retrieve(gs=1)
Phase 2 (gs=4 container, ONE restart):
  For each length: reuse same SSD data, measure retrieve(gs=4)

Output table: Length | Tokens | Populate(重算) | Retrieve gs=1 | Retrieve gs=4 | Speedup

Correctness:
  - Each length uses a DISTINCT sentence (no prompt is a prefix of another)
  - SSD cleared once at start
  - drop_caches before every retrieve (no page-cache shortcuts)
  - flush_gpu after populate and between retrieve runs
  - Warmup retrieve after each container start (discarded from averages)
"""
import requests, subprocess, time

BASE        = "http://localhost:8000/v1/completions"
MODEL       = "qwen14b1m"
CONTAINER   = "qwen14b1m-tp8"
CONFIG_FILE = "/mnt/nvme0/lmcache_ssd_nodirect.yaml"
N_RETRIEVE  = 3
WAIT_SSD_S  = 25

# 5 distinct sentences — none is a prefix of another
# ~13 tokens / rep (empirical: 3200 reps → 41601 tokens)
SENTENCES = [
    ("32K",   "The quick brown fox jumped over the lazy dog near the river. ",   2500),
    ("64K",   "She sells seashells by the seashore where the waves crash now. ",  5000),
    ("128K",  "How much wood could a woodchuck chuck if it could chuck wood. ",   10000),
    ("256K",  "Peter Piper picked a peck of pickled peppers from the garden. ",   20000),
    ("512K",  "Betty Botter bought some butter but she said the butter bitter. ", 38000),
]
# → ~32500 / ~65000 / ~130000 / ~260000 / ~494000 tokens


def shell(cmd, check=True):
    return subprocess.run(cmd, shell=True, check=check,
                          capture_output=True, text=True)

def drop_page_cache():
    shell("sync && echo 3 > /proc/sys/vm/drop_caches", check=False)

def wait_ready(timeout=300):
    for i in range(timeout // 3):
        if shell("curl -fsS http://127.0.0.1:8000/v1/models", check=False).returncode == 0:
            print(f"    server ready ({i*3}s)", flush=True)
            return
        if i % 10 == 0 and i > 0:
            print(f"    waiting... {i*3}s", flush=True)
        time.sleep(3)
    raise TimeoutError("server not ready after timeout")

def switch_gs(gs: int):
    print(f"\n  *** switching to group_size={gs} (one-time restart) ***", flush=True)
    shell(f"sed -i 's/^layer_group_size:.*/layer_group_size: {gs}/' {CONFIG_FILE}")
    shell(f"docker restart {CONTAINER}")
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
    """One warmup retrieve after container restart — not counted in averages."""
    print("    [warmup retrieve — discarded]", flush=True)
    drop_page_cache()
    do_request(prompt, "warmup")
    flush_gpu()

def measure_retrieve(prompt, tag, n=N_RETRIEVE):
    times = []
    for i in range(n):
        drop_page_cache()
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
print("  PHASE 1: populate + retrieve with group_size=1", flush=True)
print("="*64, flush=True)

pop_times  = {}
gs1_avgs   = {}
prompts    = {}

# warmup server with a tiny request first
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
print("  PHASE 2: retrieve with group_size=4 (same SSD data)", flush=True)
print("="*64, flush=True)

gs4_avgs = {}

# warmup after restart
warmup(prompts[SENTENCES[0][0]])

for tag, sentence, reps in SENTENCES:
    prompt = prompts[tag]
    print(f"\n  [{tag}] retrieve gs=4 ...", flush=True)
    _, gs4_avgs[tag] = measure_retrieve(prompt, "gs=4")
    flush_gpu()

# ── summary ───────────────────────────────────────────────────────────────────
print("\n\n" + "="*72, flush=True)
print(f"  {'Length':<8} {'Tokens':>8} {'Populate':>10} {'gs=1 ret':>10} {'gs=4 ret':>10} {'Speedup':>9}", flush=True)
print("  " + "-"*68, flush=True)
for tag, sentence, reps in SENTENCES:
    t_pop, tok = pop_times[tag]
    g1 = gs1_avgs[tag]
    g4 = gs4_avgs[tag]
    spd = g1 / g4
    print(f"  {tag:<8} {tok:>8} {t_pop:>10.3f} {g1:>10.3f} {g4:>10.3f} {spd:>9.2f}x", flush=True)
print("="*72, flush=True)
