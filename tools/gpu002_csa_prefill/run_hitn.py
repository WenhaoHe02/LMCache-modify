#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run DSv4 prefill cache-hit benchmarks without dropping page cache."""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request


ENDPOINT = os.environ.get("VLLM_ENDPOINT", "http://127.0.0.1:8000")
MODEL = os.environ.get("MODEL_NAME", "deepseek-v4-pro")
BENCH_ID = os.environ.get("BENCH_ID", "csa_attention_kv_prefill_20260630_fixed")
BASE = (
    "This deterministic cache reuse benchmark discusses DeepSeek V4 sparse "
    "attention, LMCache retrieval, Tutti GPU direct storage, and prefill "
    "overlap. "
)
QUESTION = (
    "\n\nAnswer with exactly one word. Complete the phrase: Ancient and"
)


def post_json(path: str, payload: dict[str, object], timeout: float) -> dict[str, object]:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{ENDPOINT}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode())


def count_tokens(prompt: str) -> int:
    obj = post_json(
        "/tokenize",
        {"model": MODEL, "prompt": prompt},
        timeout=120,
    )
    return int(obj["count"])


def build_prompt(target_tokens: int) -> str:
    lo = 1
    hi = 1024
    prefix = f"Benchmark id: {BENCH_ID}.\n"
    while count_tokens(prefix + BASE * hi + QUESTION) < target_tokens:
        hi *= 2
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count_tokens(prefix + BASE * mid + QUESTION) <= target_tokens:
            lo = mid
        else:
            hi = mid - 1
    return prefix + BASE * lo + QUESTION


def call(label: str, prompt: str) -> dict[str, object]:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": 0,
        "stream": False,
    }
    started = time.perf_counter()
    try:
        obj = post_json("/v1/completions", payload, timeout=900)
        elapsed = time.perf_counter() - started
        choice = obj.get("choices", [{}])[0]
        row = {
            "label": label,
            "status": 200,
            "elapsed_s": round(elapsed, 6),
            "usage": obj.get("usage"),
            "text": choice.get("text", ""),
            "finish_reason": choice.get("finish_reason"),
        }
    except urllib.error.HTTPError as exc:
        elapsed = time.perf_counter() - started
        row = {
            "label": label,
            "status": exc.code,
            "elapsed_s": round(elapsed, 6),
            "error": exc.read().decode(errors="replace")[:500],
        }
    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - started
        row = {
            "label": label,
            "status": "exception",
            "elapsed_s": round(elapsed, 6),
            "error": repr(exc),
        }
    print(json.dumps(row, ensure_ascii=False), flush=True)
    return row


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "seed_and_hits"
    target_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
    sleep_s = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
    if mode not in {"seed_and_hits", "hits_only", "seed_and_one_hit", "hit_once"}:
        raise SystemExit(f"unknown mode: {mode}")

    prompt = build_prompt(target_tokens)
    prompt_tokens = count_tokens(prompt)
    print(
        json.dumps(
            {
                "event": "prompt_ready",
                "mode": mode,
                "target_tokens": target_tokens,
                "prompt_tokens": prompt_tokens,
                "chars": len(prompt),
                "sleep_s": sleep_s,
                "bench_id": BENCH_ID,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    if mode == "seed_and_one_hit":
        labels = ["cold_store", "hit-1"]
    elif mode == "hit_once":
        labels = ["hit-1"]
    else:
        labels = []
        if mode == "seed_and_hits":
            labels.append("cold_store")
        labels.extend(["hit-1", "hit-2", "hit-3"])

    for idx, label in enumerate(labels):
        call(label, prompt)
        if idx != len(labels) - 1:
            print(
                json.dumps({"event": "sleep", "seconds": sleep_s}, ensure_ascii=False),
                flush=True,
            )
            time.sleep(sleep_s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
