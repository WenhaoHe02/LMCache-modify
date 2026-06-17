# SPDX-License-Identifier: Apache-2.0
"""Run DSv4 full-hit timing for one already-started vLLM service."""

from __future__ import annotations

# Standard
import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

# Third Party
import requests


BASE_URL = "http://127.0.0.1:8000/v1/completions"
MODEL = "deepseek-v4-pro"
TOKENS_PER_REP = 13
SENTENCES = [
    "The quick brown fox jumped over the lazy dog near the river bank. ",
    "She sells seashells by the seashore while cold waves crash nearby. ",
    "Peter Piper picked a peck of pickled peppers from the garden row. ",
    "Bright lanterns flickered softly above the quiet market square. ",
    "Careful engineers measured every signal before changing the system. ",
]


def drop_page_cache() -> None:
    """Drop Linux page cache best-effort before a full-hit request."""
    subprocess.run(
        "sync && echo 3 > /proc/sys/vm/drop_caches",
        shell=True,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def do_request(prompt: str, max_tokens: int, label: str) -> tuple[float, int, int]:
    """Send one completion request and return wall time plus token counts."""
    started = time.perf_counter()
    response = requests.post(
        BASE_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": False,
        },
        timeout=1800,
    )
    elapsed = time.perf_counter() - started
    response.raise_for_status()
    usage = response.json().get("usage", {})
    prompt_tokens = int(usage.get("prompt_tokens", 0))
    completion_tokens = int(usage.get("completion_tokens", 0))
    print(
        f"    [{label}] {elapsed:.3f}s prompt={prompt_tokens} gen={completion_tokens}",
        flush=True,
    )
    return elapsed, prompt_tokens, completion_tokens


def build_prompt(target_k: int, sentence_index: int) -> str:
    """Build a distinct prompt near ``target_k`` thousand tokens."""
    reps = max(1, target_k * 1000 // TOKENS_PER_REP)
    sentence = SENTENCES[sentence_index % len(SENTENCES)]
    return sentence * reps


def main() -> None:
    """Run the benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--lengths", default="32,64,128,192,248")
    parser.add_argument("--n-ret", type=int, default=2)
    parser.add_argument("--write-sleep", type=float, default=45.0)
    parser.add_argument("--out", default="/tmp/dsv4_full_hit_results.jsonl")
    args = parser.parse_args()

    lengths = [int(part) for part in args.lengths.split(",") if part.strip()]
    out_path = Path(args.out)
    print(f"label={args.label} lengths={lengths} n_ret={args.n_ret}", flush=True)

    do_request("Hello world.", 1, "warmup")
    with out_path.open("a", encoding="utf-8") as out_file:
        for index, target_k in enumerate(lengths):
            prompt = build_prompt(target_k, index)
            print(f"\n[{target_k}K] populate", flush=True)
            populate_s, prompt_tokens, _ = do_request(prompt, 1, "populate")
            print(f"    sleep {args.write_sleep:.1f}s for KV write", flush=True)
            time.sleep(args.write_sleep)

            retrieve_times = []
            for run_id in range(args.n_ret):
                drop_page_cache()
                elapsed, _, _ = do_request(prompt, 1, f"full-hit-{run_id + 1}")
                retrieve_times.append(elapsed)
                time.sleep(2)

            mean_s = statistics.mean(retrieve_times)
            std_s = statistics.stdev(retrieve_times) if len(retrieve_times) > 1 else 0.0
            result = {
                "label": args.label,
                "target_k": target_k,
                "prompt_tokens": prompt_tokens,
                "populate_s": populate_s,
                "retrieve_times_s": retrieve_times,
                "retrieve_mean_s": mean_s,
                "retrieve_std_s": std_s,
            }
            out_file.write(json.dumps(result, sort_keys=True) + "\n")
            out_file.flush()
            print(
                f"  RESULT {args.label} {target_k}K tokens={prompt_tokens} "
                f"populate={populate_s:.3f}s full_hit={mean_s:.3f}+/-{std_s:.3f}s",
                flush=True,
            )


if __name__ == "__main__":
    main()
