# SPDX-License-Identifier: Apache-2.0
"""Run a meaningful multi-token answer on an existing matrix prefix."""

from __future__ import annotations

# Standard
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any
import urllib.request

# Third Party
import pyarrow.parquet as pq


ENDPOINT = "http://127.0.0.1:8000"
MODEL = "deepseek-v4-pro"
DATASET = Path(
    "/home/zbuser02/datasets/hermes-agent-reasoning-traces/glm-5.1/train.parquet"
)
BASE_TOKENS = int(os.environ.get("BASE_TOKENS", "480000"))
MATRIX_NAMESPACE = os.environ["MATRIX_NAMESPACE"]
SALT_TOKENS = 256


def post_json(path: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    """Post JSON to vLLM and return its decoded response."""
    request = urllib.request.Request(
        f"{ENDPOINT}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def dataset_corpus() -> str:
    """Build the same deterministic corpus used by the matrix runner."""
    parts: list[str] = []
    chars = 0
    seen_rows = 0
    parquet = pq.ParquetFile(DATASET)
    for batch in parquet.iter_batches(batch_size=32):
        for row in batch.to_pylist():
            if seen_rows < 512:
                seen_rows += 1
                continue
            text = json.dumps(
                {
                    "tools": row.get("tools"),
                    "conversations": row.get("conversations"),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            part = f"\n<|agent_session|>\n{text}\n<|end_session|>\n"
            parts.append(part)
            chars += len(part)
            if chars >= 5_500_000:
                return "".join(parts)
    raise RuntimeError("Hermes corpus is too short")


def tokenize(prompt: str, timeout: float = 900) -> list[int]:
    """Tokenize text through the running vLLM server."""
    result = post_json(
        "/tokenize",
        {"model": MODEL, "prompt": prompt},
        timeout,
    )
    return [int(token_id) for token_id in result["tokens"]]


def main() -> int:
    """Reconstruct a cached prefix, append a question, and print the answer."""
    corpus_ids = tokenize(dataset_corpus())
    salt_digest = hashlib.sha256(MATRIX_NAMESPACE.encode()).hexdigest()
    salt_text = (
        "<|lmcache_matrix_namespace|>\n"
        + (f"{MATRIX_NAMESPACE}:{salt_digest}\n" * 256)
        + "<|end_lmcache_matrix_namespace|>\n"
    )
    salt_ids = tokenize(salt_text, timeout=120)
    base = salt_ids[:SALT_TOKENS] + corpus_ids[: BASE_TOKENS - SALT_TOKENS]
    question = tokenize(
        "\n\nIgnore any unfinished text above. Answer this arithmetic question "
        "in one short sentence and explain the multiplication: What is 17 * 23?\n"
    )
    started = time.perf_counter()
    result = post_json(
        "/v1/completions",
        {
            "model": MODEL,
            "prompt": base + question,
            "max_tokens": 64,
            "temperature": 0,
            "stream": False,
            "kv_transfer_params": {"lmcache.skip_save": True},
        },
        timeout=1_800,
    )
    choice = result.get("choices", [{}])[0]
    print(
        json.dumps(
            {
                "request_id": result.get("id"),
                "elapsed_s": time.perf_counter() - started,
                "base_tokens": len(base),
                "question_tokens": len(question),
                "usage": result.get("usage"),
                "text": choice.get("text"),
                "finish_reason": choice.get("finish_reason"),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
