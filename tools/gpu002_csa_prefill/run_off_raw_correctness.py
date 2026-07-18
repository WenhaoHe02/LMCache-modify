# SPDX-License-Identifier: Apache-2.0
"""Validate long-output correctness across an exact LMCache replay."""

# Standard
import hashlib
import json
import os
import time
import urllib.request
from typing import Any


ENDPOINT = os.environ.get("VLLM_ENDPOINT", "http://127.0.0.1:8000")
MODEL = os.environ.get("MODEL_NAME", "deepseek-v4-pro")
TARGET_PROMPT_TOKENS = int(os.environ.get("TARGET_PROMPT_TOKENS", "1024"))
REPLAY_WAIT_SECONDS = float(os.environ.get("REPLAY_WAIT_SECONDS", "30"))
RUN_REPLAY = os.environ.get("RUN_REPLAY", "1").lower() in {
    "1",
    "true",
    "yes",
    "on",
}

CONTEXT_UNIT = (
    "A distributed payment service uses a replicated ledger. Each transfer "
    "has a unique request identifier, debits one account, credits another, "
    "and may be retried after a client timeout. Replicas can fail between "
    "committing the ledger entry and returning the response. Auditors require "
    "the sum of debits and credits to remain equal. "
)
QUESTION = (
    "\nUsing the scenario above, write 300 to 380 English words explaining why "
    "the transfer operation must be idempotent. Use exactly four numbered "
    "sections titled Invariant, Duplicate-request failure, Idempotency-key "
    "solution, and Audit checks. Include a T1/T2/T3 failure timeline and end "
    "with exactly: Therefore, retries cannot create extra money."
)


def post_json(path: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    """POST JSON to the local vLLM endpoint and return the decoded response."""
    request = urllib.request.Request(
        f"{ENDPOINT}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def count_tokens(prompt: str) -> int:
    """Return the server tokenizer count for one prompt."""
    response = post_json(
        "/tokenize",
        {"model": MODEL, "prompt": prompt},
        timeout=120,
    )
    return int(response["count"])


def build_prompt() -> str:
    """Build a deterministic prompt longer than one LMCache chunk."""
    low = 1
    high = 16
    while count_tokens(CONTEXT_UNIT * high + QUESTION) < TARGET_PROMPT_TOKENS:
        high *= 2
    while low < high:
        middle = (low + high) // 2
        if count_tokens(CONTEXT_UNIT * middle + QUESTION) < TARGET_PROMPT_TOKENS:
            low = middle + 1
        else:
            high = middle
    return CONTEXT_UNIT * low + QUESTION


def run_completion(prompt: str) -> dict[str, Any]:
    """Generate one deterministic long chat completion."""
    started = time.perf_counter()
    response = post_json(
        "/v1/chat/completions",
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 600,
            "temperature": 0,
            "stream": False,
        },
        timeout=900,
    )
    response["client_elapsed_s"] = round(time.perf_counter() - started, 6)
    return response


def summarize(label: str, response: dict[str, Any]) -> dict[str, Any]:
    """Return correctness and content checks for one response."""
    choice = response["choices"][0]
    text = choice["message"]["content"]
    usage = response["usage"]
    return {
        "label": label,
        "elapsed_s": response["client_elapsed_s"],
        "finish_reason": choice["finish_reason"],
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "words": len(text.split()),
        "section_count": sum(
            marker in text
            for marker in (
                "Invariant",
                "Duplicate-request failure",
                "Idempotency-key solution",
                "Audit checks",
            )
        ),
        "has_timeline": all(marker in text for marker in ("T1", "T2", "T3")),
        "required_end": text.rstrip().endswith(
            "Therefore, retries cannot create extra money."
        ),
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "text": text,
    }


def main() -> int:
    """Run a cold generation and exact replay, then report equivalence."""
    prompt = build_prompt()
    prompt_tokens = count_tokens(prompt)
    cold = summarize("cold", run_completion(prompt))
    print(json.dumps(cold, ensure_ascii=False), flush=True)
    if not RUN_REPLAY:
        checks = {
            "event": "correctness",
            "prompt_tokens": prompt_tokens,
            "completion_over_256": cold["completion_tokens"] > 256,
            "structure_valid": (
                cold["finish_reason"] == "stop"
                and cold["section_count"] == 4
                and cold["has_timeline"]
                and cold["required_end"]
            ),
        }
        print(json.dumps(checks), flush=True)
        return 0 if all(checks.values()) else 1
    time.sleep(REPLAY_WAIT_SECONDS)
    replay = summarize("replay", run_completion(prompt))
    print(json.dumps(replay, ensure_ascii=False), flush=True)
    checks = {
        "event": "correctness",
        "prompt_tokens": prompt_tokens,
        "completion_over_256": replay["completion_tokens"] > 256,
        "content_identical": cold["text_sha256"] == replay["text_sha256"],
        "structure_valid": (
            replay["finish_reason"] == "stop"
            and replay["section_count"] == 4
            and replay["has_timeline"]
            and replay["required_end"]
        ),
    }
    print(json.dumps(checks), flush=True)
    return (
        0
        if all(
            (
                checks["completion_over_256"],
                checks["content_identical"],
                checks["structure_valid"],
            )
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
