#!/usr/bin/env python3
"""
Bench Tutti raw LBA write/read path on DSV4.
cold2: 20K-token prefill -> KV written to raw LBA
hit1:  same prompt -> KV loaded from raw LBA
hit2:  same prompt again -> should match hit1
Target: hit << cold (goal: hit < 1s, cold ~11-15s)
"""
import requests
import time

BASE        = "http://localhost:8000/v1/completions"
MODEL       = "deepseek-v4-pro"
MAX_TOKENS  = 1
WRITE_SLEEP = 20   # seconds to wait after cold for KV store to flush

SENTENCE     = "The quick brown fox jumped over the lazy dog near the river. "
TOKS_PER_REP = 13
PROMPT_KTOKS = 20  # ~20K input tokens
PROMPT = SENTENCE * (PROMPT_KTOKS * 1000 // TOKS_PER_REP)


def do_req(label: str, prompt: str, max_tokens: int = MAX_TOKENS) -> tuple[float, int]:
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
        timeout=600,
    )
    elapsed = time.perf_counter() - t0
    r.raise_for_status()
    ptok = r.json().get("usage", {}).get("prompt_tokens", "?")
    print(f"  [{label:8s}] {elapsed:.3f}s  prompt_tokens={ptok}", flush=True)
    return elapsed, ptok


def wait_server(timeout: int = 120) -> None:
    for _ in range(timeout):
        try:
            r = requests.get("http://localhost:8000/v1/models", timeout=3)
            if r.status_code == 200:
                print("Server ready.", flush=True)
                return
        except Exception:
            pass
        time.sleep(1)
    raise TimeoutError("server not ready")


def main() -> None:
    print("=== bench_raw_cold2_small  (Tutti raw LBA) ===")
    wait_server()

    print(f"\n[warm-up] short prompt to prime vLLM scheduling")
    do_req("warmup", "Hello world.")

    print(f"\n[cold2]  ~{PROMPT_KTOKS}K token prefill -> write to raw LBA")
    cold_t, ptok = do_req("cold2", PROMPT)

    print(f"\n[sleep {WRITE_SLEEP}s] waiting for KV store flush to NVMe ...")
    time.sleep(WRITE_SLEEP)

    print(f"\n[hit1]   same prompt -> read from raw LBA")
    hit1_t, _ = do_req("hit1", PROMPT)

    print(f"\n[hit2]   repeat -> should match hit1")
    hit2_t, _ = do_req("hit2", PROMPT)

    print("\n=== Results ===")
    print(f"  cold2 = {cold_t:.3f}s  ({ptok} tokens)")
    print(f"  hit1  = {hit1_t:.3f}s  speedup = {cold_t/hit1_t:.1f}x")
    print(f"  hit2  = {hit2_t:.3f}s  speedup = {cold_t/hit2_t:.1f}x")
    if hit1_t < 1.0:
        print("  PASS: hit1 < 1s target")
    else:
        print(f"  MISS: hit1={hit1_t:.3f}s exceeds 1s target")


if __name__ == "__main__":
    main()
