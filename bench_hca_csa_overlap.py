#!/usr/bin/env python3
"""
Benchmark: HCA overlapping + CSA decode-prefetch feature isolation.

Run this script *inside* the gpu002 environment (the server must be localhost:8000).

Workflow
--------
Phase 1 – Populate SSD cache:
  Send a large prompt → waits for SSD flush (30s sleep) → cache populated.

Phase 2 – Measure (N_REPS repeats):
  a) TTFT  : send same prompt with max_tokens=1 (cache hit, tests HCA prefetch overlap)
  b) Decode: send same prompt with max_tokens=DECODE_TOKENS (tests CSA decode prefetch)

Report: mean ± std for TTFT and token/s under current env config label.

Usage (run 3 times, edit container env between runs):
  python3 bench_hca_csa_overlap.py --label baseline
  # restart container with LMCACHE_HCA_ENABLE_PREFETCH=1
  python3 bench_hca_csa_overlap.py --label hca_on
  # restart container with LMCACHE_INDEXER_ENABLE_DECODE_PREFETCH=1
  python3 bench_hca_csa_overlap.py --label csa_decode_on
"""
import argparse
import statistics
import subprocess
import time

import requests

BASE = "http://localhost:8000/v1/completions"
MODEL = "/mnt/nvme0/models/DeepSeek-V4-Pro"

# ~13 tokens/rep — use 4000 reps → ~52K tokens
SENTENCE = "The quick brown fox jumped over the lazy dog near the river bank. "
REPS = 4000  # ~52K tokens

N_REPS = 3
DECODE_TOKENS = 50  # tokens to generate in decode test
SSD_FLUSH_WAIT = 45  # seconds to wait for SSD write after populate


def drop_page_cache() -> None:
    subprocess.run(
        "sync && echo 3 > /proc/sys/vm/drop_caches",
        shell=True, check=False, capture_output=True,
    )


def do_request(prompt: str, max_tokens: int, label: str) -> tuple[float, int, float]:
    """Returns (wall_time_s, prompt_tokens, tokens_per_sec_decode)."""
    t0 = time.perf_counter()
    r = requests.post(
        BASE,
        json={
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": False,
        },
        timeout=1800,
    )
    elapsed = time.perf_counter() - t0
    r.raise_for_status()
    usage = r.json().get("usage", {})
    ptok = usage.get("prompt_tokens", 0)
    ctok = usage.get("completion_tokens", 0)
    tps = ctok / elapsed if elapsed > 0 else 0.0
    print(f"  [{label}] {elapsed:.3f}s  prompt={ptok}  gen={ctok}  gen_tps={tps:.1f}", flush=True)
    return elapsed, ptok, tps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="current", help="config label for output")
    parser.add_argument("--reps", type=int, default=REPS, help="sentence repetitions (~13 tok each)")
    parser.add_argument("--n-reps", type=int, default=N_REPS, help="benchmark repeats")
    parser.add_argument("--decode-tokens", type=int, default=DECODE_TOKENS)
    parser.add_argument("--skip-populate", action="store_true", help="skip populate phase")
    args = parser.parse_args()

    prompt = SENTENCE * args.reps
    print(f"\n{'='*60}", flush=True)
    print(f"Config: {args.label}", flush=True)
    env_keys = [
        "LMCACHE_HCA_ENABLE_PREFETCH",
        "LMCACHE_INDEXER_ENABLE_DECODE_PREFETCH",
        "LMCACHE_INDEXER_ENABLE_REUSE_PREFETCH",
        "LMCACHE_INDEXER_ENABLE_PREFETCH",
    ]
    import os
    for k in env_keys:
        print(f"  {k}={os.environ.get(k, '(unset)')}", flush=True)
    print(f"  Prompt reps={args.reps}, decode_tokens={args.decode_tokens}", flush=True)

    # Phase 1: Populate
    if not args.skip_populate:
        print("\n[Phase 1] Populate SSD cache ...", flush=True)
        do_request(prompt, 1, "populate")
        print(f"  Waiting {SSD_FLUSH_WAIT}s for SSD flush ...", flush=True)
        time.sleep(SSD_FLUSH_WAIT)
        drop_page_cache()
        print("  Page cache dropped.", flush=True)

    # Phase 2a: TTFT test (max_tokens=1, cache-hit)
    print(f"\n[Phase 2a] TTFT test (max_tokens=1, n={args.n_reps}) ...", flush=True)
    ttft_times: list[float] = []
    for i in range(args.n_reps):
        drop_page_cache()
        t, ptok, _ = do_request(prompt, 1, f"ttft-{i+1}")
        ttft_times.append(t)
        time.sleep(2)

    # Phase 2b: Decode test (max_tokens=DECODE_TOKENS, cache-hit)
    print(f"\n[Phase 2b] Decode test (max_tokens={args.decode_tokens}, n={args.n_reps}) ...", flush=True)
    decode_tps: list[float] = []
    for i in range(args.n_reps):
        drop_page_cache()
        _, _, tps = do_request(prompt, args.decode_tokens, f"decode-{i+1}")
        decode_tps.append(tps)
        time.sleep(2)

    # Summary
    def fmt(vals: list[float]) -> str:
        if len(vals) == 1:
            return f"{vals[0]:.3f}"
        return f"{statistics.mean(vals):.3f} ± {statistics.stdev(vals):.3f}"

    print(f"\n{'='*60}", flush=True)
    print(f"RESULTS [{args.label}]", flush=True)
    print(f"  TTFT (s):       {fmt(ttft_times)}", flush=True)
    print(f"  Decode (tok/s): {fmt(decode_tps)}", flush=True)
    print(f"  prompt_tokens:  ~{ptok}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
