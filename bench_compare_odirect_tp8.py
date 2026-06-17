# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
"""
Group-size prefetch benchmark with O_DIRECT=True, tp=8 (all GPUs).

Phase 1 (gs=1): clear SSD once, populate all lengths, then retrieve gs=1.
Phase 2 (gs=4, one restart): retrieve gs=4 from the same SSD data.

The script deliberately does not delete SSD files or drop page cache after the
server starts. Each benchmark length uses a unique prefix so the stored keys do
not overlap across lengths.
"""

# Standard
import subprocess
import time
from typing import Any

# Third Party
import requests

BASE = "http://localhost:8000/v1/completions"
MODEL = "qwen14b1m"
LAUNCH_SH = "/root/qwen14b1m_tp8_odirect.sh"
N_RETRIEVE = 3
WAIT_SSD_S = 50

SSD_DIRS = [
    "/mnt/nvme0",
    "/mnt/nvme2",
    "/mnt/nvme3",
    "/mnt/nvme4",
    "/mnt/nvme5",
    "/mnt/nvme6",
    "/mnt/nvme8",
    "/mnt/nvme9",
]

SENTENCES = [
    ("32K", "The quick brown fox jumped over the lazy dog near the river. ", 2500),
    ("64K", "She sells seashells by the seashore where the waves crash now. ", 5000),
    ("128K", "How much wood could a woodchuck chuck if it could chuck wood. ", 10000),
    ("256K", "Peter Piper picked a peck of pickled peppers from the garden. ", 20000),
    ("512K", "Betty Botter bought some butter but she said the butter bitter. ", 38000),
]


def shell(cmd: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        shell=True,
        check=check,
        capture_output=True,
        text=True,
    )


def clear_ssd_cache_once() -> None:
    print("Clearing SSD cache once before benchmark startup ...", flush=True)
    for directory in SSD_DIRS:
        shell(
            f"find {directory}/lmcache/qwen14b1m -mindepth 1 -delete "
            "2>/dev/null || true",
            check=False,
        )


def make_prompt(tag: str, sentence: str, reps: int) -> str:
    prefix = f"[BENCH-ODIRECT-TP8-{tag}-UNIQUE-PREFIX]\n"
    return prefix + sentence * reps


def wait_ready(timeout: int = 900) -> None:
    for i in range(timeout // 3):
        result = shell("curl -fsS http://127.0.0.1:8000/v1/models", check=False)
        if result.returncode == 0:
            print(f"    server ready ({i * 3}s)", flush=True)
            return
        if i % 10 == 0 and i > 0:
            print(f"    waiting... {i * 3}s", flush=True)
        time.sleep(3)
    raise TimeoutError("server not ready")


def switch_gs(gs: int) -> None:
    print(f"\n  *** switching to group_size={gs} ***", flush=True)
    shell(f"bash {LAUNCH_SH} {gs}")
    wait_ready()


def do_request(prompt: str, label: str, max_tokens: int = 1) -> tuple[float, Any]:
    t0 = time.perf_counter()
    response = requests.post(
        BASE,
        json={
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": False,
        },
        timeout=900,
    )
    elapsed = time.perf_counter() - t0
    response.raise_for_status()
    prompt_tokens = response.json().get("usage", {}).get("prompt_tokens", "?")
    print(f"    [{label}] {elapsed:.3f}s  tokens={prompt_tokens}", flush=True)
    return elapsed, prompt_tokens


def flush_gpu() -> None:
    do_request("Hello.", "flush")


def warmup(prompt: str) -> None:
    print("    [warmup discarded]", flush=True)
    do_request(prompt, "warmup")
    flush_gpu()


def measure_retrieve(
    prompt: str,
    tag: str,
    n: int = N_RETRIEVE,
) -> tuple[list[float], float]:
    times = []
    for i in range(n):
        elapsed, _ = do_request(prompt, f"retrieve-{i + 1}")
        times.append(elapsed)
        if i < n - 1:
            flush_gpu()
    avg = sum(times) / len(times)
    print(f"    [{tag} avg={avg:.3f}s]", flush=True)
    return times, avg


clear_ssd_cache_once()
switch_gs(1)

print("\n" + "=" * 64, flush=True)
print("  PHASE 1: populate + retrieve gs=1  (O_DIRECT=True, tp=8)", flush=True)
print("  SSD cache will not be deleted again until the script exits.", flush=True)
print("=" * 64, flush=True)

pop_times = {}
gs1_avgs = {}
prompts = {}

do_request("Hello world.", "server-warmup")

for tag, sentence, reps in SENTENCES:
    prompt = make_prompt(tag, sentence, reps)
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

switch_gs(4)

print("\n" + "=" * 64, flush=True)
print("  PHASE 2: retrieve gs=4  (O_DIRECT=True, same SSD data)", flush=True)
print("  SSD cache is kept intact after the initial startup cleanup.", flush=True)
print("=" * 64, flush=True)

gs4_avgs = {}
warmup(prompts[SENTENCES[0][0]])

for tag, sentence, reps in SENTENCES:
    print(f"\n  [{tag}] retrieve gs=4 ...", flush=True)
    _, gs4_avgs[tag] = measure_retrieve(prompts[tag], "gs=4")
    flush_gpu()

print("\n\n" + "=" * 72, flush=True)
print(
    f"  {'Length':<8} {'Tokens':>8} {'Populate':>10} "
    f"{'gs=1 ret':>10} {'gs=4 ret':>10} {'Speedup':>9}",
    flush=True,
)
print("  " + "-" * 68, flush=True)
for tag, sentence, reps in SENTENCES:
    t_pop, tok = pop_times[tag]
    g1 = gs1_avgs[tag]
    g4 = gs4_avgs[tag]
    spd = g1 / g4
    print(
        f"  {tag:<8} {tok:>8} {t_pop:>10.3f} "
        f"{g1:>10.3f} {g4:>10.3f} {spd:>9.2f}x",
        flush=True,
    )
print("=" * 72, flush=True)
