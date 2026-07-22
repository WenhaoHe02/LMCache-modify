# SPDX-License-Identifier: Apache-2.0
"""Run a meaningful long-prefix correctness completion against vLLM."""

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


ENDPOINT = os.environ.get("VLLM_ENDPOINT", "http://127.0.0.1:8000")
MODEL = os.environ.get("MODEL_NAME", "deepseek-v4-pro")
DATASET = Path(
    "/home/zbuser02/datasets/hermes-agent-reasoning-traces/glm-5.1/train.parquet"
)
BASE_TOKENS = int(os.environ.get("BASE_TOKENS", "480000"))
RECOMPUTE_TOKENS = int(os.environ.get("RECOMPUTE_TOKENS", "8192"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "160"))
MIN_TOKENS = int(os.environ.get("MIN_TOKENS", "96"))
STORE_WAIT_S = float(os.environ.get("STORE_WAIT_S", "60"))
MODE_LABEL = os.environ.get("MODE_LABEL", "unknown")
SKIP_COLD_STORE = os.environ.get("SKIP_COLD_STORE", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
NUM_HITS = int(os.environ.get("NUM_HITS", "1"))
LOGPROBS = int(os.environ.get("LOGPROBS", "0"))
SKIP_ROWS = 512
QUESTION = """

<|user|>
This is a deterministic KV-cache correctness test. Ignore the preceding
dataset dump and answer this final request only. A storage system reads 6.4
GiB in 0.8 seconds. Explain, in at least five numbered steps and at least 120
English words, how to calculate its effective GiB/s and GB/s bandwidth. Also
calculate 37 * 29. State all three numerical results before the explanation,
then end with exactly this line:
CHECKSUM=1073
<|assistant|>
""".strip()


def post_json(path: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    """POST a JSON payload and return the decoded response."""
    request = urllib.request.Request(
        f"{ENDPOINT}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def emit(event: str, **fields: Any) -> None:
    """Emit one machine-readable progress record."""
    print(json.dumps({"event": event, **fields}, ensure_ascii=False), flush=True)


def corpus_token_ids(minimum_tokens: int) -> list[int]:
    """Build and tokenize enough deterministic Hermes rows."""
    parts: list[str] = []
    chars = 0
    seen_rows = 0
    parquet = pq.ParquetFile(DATASET)
    for batch in parquet.iter_batches(batch_size=32):
        for row in batch.to_pylist():
            if seen_rows < SKIP_ROWS:
                seen_rows += 1
                continue
            parts.append(
                "\n<|agent_session|>\n"
                + json.dumps(
                    {
                        "tools": row.get("tools"),
                        "conversations": row.get("conversations"),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n<|end_session|>\n"
            )
            chars += len(parts[-1])
            if chars < 3_500_000:
                continue
            tokenized = post_json(
                "/tokenize",
                {"model": MODEL, "prompt": "".join(parts)},
                timeout=600,
            )
            ids = [int(token_id) for token_id in tokenized["tokens"]]
            if len(ids) >= minimum_tokens:
                return ids
    raise RuntimeError("Hermes corpus does not contain enough tokens")


def complete(label: str, prompt: list[int], max_tokens: int) -> dict[str, Any]:
    """Run one deterministic streaming completion and record exact TTFT."""
    started = time.perf_counter()
    payload: dict[str, Any] = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if max_tokens > 1:
        payload["min_tokens"] = min(MIN_TOKENS, max_tokens)
    if LOGPROBS > 0:
        payload["logprobs"] = LOGPROBS
    request = urllib.request.Request(
        f"{ENDPOINT}/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    text_parts: list[str] = []
    first_token_at: float | None = None
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    first_logprobs: dict[str, Any] | None = None
    with urllib.request.urlopen(request, timeout=1_800) as response:
        for raw_line in response:
            line = raw_line.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line.removeprefix("data:").strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage") is not None:
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            piece = str(choice.get("text", ""))
            if piece and first_token_at is None:
                first_token_at = time.perf_counter()
                raw_logprobs = choice.get("logprobs")
                if isinstance(raw_logprobs, dict):
                    first_logprobs = raw_logprobs
            text_parts.append(piece)
            if choice.get("finish_reason") is not None:
                finish_reason = str(choice["finish_reason"])
    completed_at = time.perf_counter()
    if first_token_at is None:
        raise RuntimeError(f"{label} returned no output token")
    text = "".join(text_parts)
    output_tokens = post_json(
        "/tokenize",
        {"model": MODEL, "prompt": text},
        timeout=300,
    )["tokens"]
    result = {
        "event": "request_complete",
        "mode": MODE_LABEL,
        "label": label,
        "status": 200,
        "ttft_s": first_token_at - started,
        "elapsed_s": completed_at - started,
        "finish_reason": finish_reason,
        "usage": usage,
        "output_text": text,
        "output_token_ids": output_tokens,
        "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "first_logprobs": first_logprobs,
    }
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main() -> int:
    """Cold-store 480K tokens, then recompute 8192 and decode a real answer."""
    question_ids = [
        int(token_id)
        for token_id in post_json(
            "/tokenize",
            {"model": MODEL, "prompt": QUESTION},
            timeout=300,
        )["tokens"]
    ]
    if len(question_ids) >= RECOMPUTE_TOKENS:
        raise RuntimeError("correctness question is too long")
    required = BASE_TOKENS + RECOMPUTE_TOKENS
    corpus_ids = corpus_token_ids(required)
    base_prompt = corpus_ids[:BASE_TOKENS]
    bridge_len = RECOMPUTE_TOKENS - len(question_ids)
    continuation = corpus_ids[BASE_TOKENS : BASE_TOKENS + bridge_len] + question_ids
    hit_prompt = base_prompt + continuation
    emit(
        "prompt_ready",
        mode=MODE_LABEL,
        base_tokens=len(base_prompt),
        recompute_tokens=len(continuation),
        hit_tokens=len(hit_prompt),
        question_tokens=len(question_ids),
        max_tokens=MAX_TOKENS,
        min_tokens=MIN_TOKENS,
        base_sha256=hashlib.sha256(
            ",".join(str(token_id) for token_id in base_prompt).encode()
        ).hexdigest(),
        hit_sha256=hashlib.sha256(
            ",".join(str(token_id) for token_id in hit_prompt).encode()
        ).hexdigest(),
    )
    if SKIP_COLD_STORE:
        emit("cold_store_skipped", reason="existing_compact_objects")
    else:
        complete("cold_store", base_prompt, 1)
        emit("sleep", seconds=STORE_WAIT_S)
        time.sleep(STORE_WAIT_S)
    for trial in range(NUM_HITS):
        hit = complete(
            f"meaningful_hit_480k8192_trial{trial + 1}",
            hit_prompt,
            MAX_TOKENS,
        )
        completion_tokens = int(
            (hit.get("usage") or {}).get("completion_tokens", 0)
        )
        if completion_tokens < MIN_TOKENS:
            raise RuntimeError(
                f"decoded only {completion_tokens} tokens, "
                f"expected at least {MIN_TOKENS}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
