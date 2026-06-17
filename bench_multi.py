#!/usr/bin/env python3
"""
Multi-length LMCache benchmark.
  - populate : cold recompute (重算) — unique prompt per length, no prior cache
  - retrieve : SSD cache reuse (复用) — page cache dropped, GPU blocks freed

Token calibration (empirical, Qwen2.5-14B tokenizer):
  "The quick brown fox..." x1 unit (~45 chars) ≈ 11 tokens
  We use 4 distinct sentences so prompts are never prefixes of each other.
  Each BASE_* is ~45 chars repeated enough times to reach the target length.

  Target  | sentence repetitions
  --------|---------------------
  32K     | BASE_A * 3200   (~32000 tokens, ~144KB text)
  64K     | BASE_B * 6400
  100K    | BASE_C * 10000
  128K    | BASE_D * 12800
"""
import requests, time, subprocess

BASE  = "http://localhost:8000/v1/completions"
MODEL = "qwen14b1m"

# ~11 tokens per repetition (short English sentences, BPE-friendly)
BASE_A = "The quick brown fox jumps over the lazy dog near the riverbank. "   # 64 chars
BASE_B = "She sells seashells by the seashore where the waves crash loudly. "  # 66 chars
BASE_C = "How much wood would a woodchuck chuck if it could chuck wood today. "  # 68 chars
BASE_D = "Peter Piper picked a peck of pickled peppers from the garden plot. "    # 67 chars

# (label, sentence, repetitions)  — calibrated for ~target token count
# Each sentence ≈ 13 tokens  →  32K/13 ≈ 2462 reps, 64K/13 ≈ 4923 reps, etc.
LENGTHS = [
    ("32K",   BASE_A, 2500),    # ~32500 tokens
    ("64K",   BASE_B, 5000),    # ~65000 tokens
    ("100K",  BASE_C, 7700),    # ~100100 tokens
    ("128K",  BASE_D, 9900),    # ~128700 tokens
]

DROP_CACHES = True
N_RETRIEVE  = 2
FLUSH_PROMPT = "Hello."

def drop_page_cache():
    try:
        subprocess.run("sync && echo 3 > /proc/sys/vm/drop_caches",
                       shell=True, check=True)
    except Exception as e:
        print(f"  [warn] drop_caches failed: {e}", flush=True)

def do_request(prompt, label, max_tokens=1):
    t0 = time.perf_counter()
    r = requests.post(BASE, json={
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }, timeout=900)
    elapsed = time.perf_counter() - t0
    r.raise_for_status()
    ptok = r.json().get("usage", {}).get("prompt_tokens", "?")
    print(f"  [{label}] TTFT={elapsed:.3f}s  tokens={ptok}", flush=True)
    return elapsed, ptok

def flush_gpu_kv():
    do_request(FLUSH_PROMPT, "flush-gpu", max_tokens=1)

results = []

for tag, sentence, reps in LENGTHS:
    prompt = sentence * reps
    print(f"\n{'='*60}", flush=True)
    print(f"  Length ~{tag}  (reps={reps})", flush=True)
    print(f"{'='*60}", flush=True)

    print("  [populate] cold recompute ...", flush=True)
    t_pop, actual_tokens = do_request(prompt, "populate")

    print("  waiting 20s for async SSD write ...", flush=True)
    time.sleep(20)

    flush_gpu_kv()

    retrieves = []
    for i in range(N_RETRIEVE):
        if DROP_CACHES:
            print(f"  dropping page cache before retrieve-{i+1} ...", flush=True)
            drop_page_cache()
        t_ret, _ = do_request(prompt, f"retrieve-{i+1}")
        retrieves.append(t_ret)
        if i < N_RETRIEVE - 1:
            flush_gpu_kv()

    avg_ret = sum(retrieves) / len(retrieves)
    speedup  = t_pop / avg_ret
    results.append((tag, actual_tokens, t_pop, retrieves, avg_ret, speedup))
    print(f"\n  >> populate={t_pop:.3f}s  retrieve_avg={avg_ret:.3f}s  speedup={speedup:.2f}x",
          flush=True)

print("\n" + "="*70, flush=True)
print(f"  {'Length':<8} {'Tokens':>8} {'Populate(s)':>12} {'Retrieve_avg':>14} {'Speedup':>9}", flush=True)
print("  " + "-"*66, flush=True)
for tag, tok, t_pop, rets, avg_ret, spd in results:
    print(f"  {tag:<8} {tok:>8} {t_pop:>12.3f} {avg_ret:>14.3f} {spd:>9.2f}x", flush=True)
print("="*70, flush=True)
