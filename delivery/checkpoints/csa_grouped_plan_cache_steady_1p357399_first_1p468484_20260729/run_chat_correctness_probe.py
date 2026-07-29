# SPDX-License-Identifier: Apache-2.0
"""Verify cold and hit output semantics through the chat API."""

from __future__ import annotations

# Standard
import json
import os
import time
from typing import Any
import urllib.request


ENDPOINT = "http://127.0.0.1:8000"
MODEL = "deepseek-v4-pro"


def post_json(path: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    """Post JSON to vLLM and return its decoded response."""
    request = urllib.request.Request(
        f"{ENDPOINT}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def request(label: str, content: str, *, skip_save: bool) -> dict[str, Any]:
    """Run one deterministic chat request and return a concise record."""
    payload: dict[str, Any] = {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": int(os.environ.get("CHAT_MAX_TOKENS", "96")),
        "temperature": 0,
        "stream": False,
    }
    if skip_save:
        payload["kv_transfer_params"] = {"lmcache.skip_save": True}
    started = time.perf_counter()
    response = post_json("/v1/chat/completions", payload, timeout=1_800)
    choice = response.get("choices", [{}])[0]
    result = {
        "label": label,
        "request_id": response.get("id"),
        "elapsed_s": time.perf_counter() - started,
        "usage": response.get("usage"),
        "content": choice.get("message", {}).get("content"),
        "finish_reason": choice.get("finish_reason"),
    }
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main() -> int:
    """Store and hit the same meaningful long chat prompt."""
    nonce = os.environ.get(
        "CHAT_NONCE",
        f"chat-correctness-{time.time_ns()}",
    )
    background_repeats = int(os.environ.get("CHAT_BACKGROUND_REPEATS", "4000"))
    background = (
        f"Experiment identifier: {nonce}. "
        "The following repeated sentence is irrelevant background context. "
        * background_repeats
    )
    content = (
        background
        + "\nIgnore the irrelevant background. What is 17 multiplied by 23? "
        "Give the numeric answer and one short multiplication explanation."
    )
    skip_cold = os.environ.get("CHAT_SKIP_COLD", "0") == "1"
    cold = None if skip_cold else request("cold", content, skip_save=False)
    if cold is not None:
        time.sleep(float(os.environ.get("COLD_WAIT_SECONDS", "30")))
    hit_repeats = int(os.environ.get("CHAT_HIT_REPEATS", "1"))
    hits = [
        request(f"hit_{index + 1}", content, skip_save=True)
        for index in range(hit_repeats)
    ]
    cold_text = "391" if cold is None else str(cold.get("content") or "")
    hit_texts = [str(hit.get("content") or "") for hit in hits]
    valid = "391" in cold_text and all("391" in text for text in hit_texts)
    print(json.dumps({"valid": valid}, ensure_ascii=False))
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
