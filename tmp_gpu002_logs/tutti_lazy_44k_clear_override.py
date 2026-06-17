# SPDX-License-Identifier: Apache-2.0
"""Run a 44K cold/hit test after clearing PCI driver_override."""

# Standard
import time
from typing import Any

# Third Party
import requests


BASE_URL = "http://127.0.0.1:8000"
MODEL = "deepseek-v4-pro"
SENTENCE = "The quick brown fox jumped over the lazy dog near the river bank. "
PROMPT = (
    "TUTTI_LAZY_44K_CLEAR_OVERRIDE_20260609_UNIQUE_PREFIX. "
    + SENTENCE * 3150
    + "\nFinal question: answer with one token.\n"
)


def post_json(path: str, payload: dict[str, Any], timeout: float) -> requests.Response:
    """Post JSON to the vLLM endpoint and return the response."""
    return requests.post(f"{BASE_URL}{path}", json=payload, timeout=timeout)


def completion(label: str) -> None:
    """Run one completion request and print status plus elapsed seconds."""
    payload = {
        "model": MODEL,
        "prompt": PROMPT,
        "max_tokens": 1,
        "temperature": 0,
    }
    start = time.time()
    response = post_json("/v1/completions", payload, timeout=420)
    elapsed = time.time() - start
    print(f"{label} status={response.status_code} sec={elapsed:.3f}", flush=True)
    print(response.text[:500], flush=True)
    response.raise_for_status()


def main() -> None:
    """Tokenize the prompt, then run cold and hit completion requests."""
    token_response = post_json(
        "/tokenize",
        {"model": MODEL, "prompt": PROMPT},
        timeout=120,
    )
    token_response.raise_for_status()
    print(f"tokens={len(token_response.json()['tokens'])}", flush=True)
    completion("cold")
    time.sleep(35)
    completion("hit")


if __name__ == "__main__":
    main()
