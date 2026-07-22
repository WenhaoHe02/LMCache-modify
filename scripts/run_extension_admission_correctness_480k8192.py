# SPDX-License-Identifier: Apache-2.0
"""Validate that a deferred hit suffix is usable by a later extension hit."""

from __future__ import annotations

# Standard
import hashlib
import json
import os
import time

# Local
from run_meaningful_correctness_480k8192 import (
    BASE_TOKENS,
    MODEL,
    STORE_WAIT_S,
    complete,
    corpus_token_ids,
    emit,
    post_json,
)


FIRST_SUFFIX_TOKENS = int(os.environ.get("FIRST_SUFFIX_TOKENS", "8192"))
EXTENSION_TOKENS = int(os.environ.get("EXTENSION_TOKENS", "1024"))
ADMISSION_WAIT_S = float(os.environ.get("ADMISSION_WAIT_S", "15"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "96"))
QUESTION = """

<|user|>
Ignore all earlier text. Calculate 37 * 29 and 6.4 / 0.8. State both
results, briefly explain the arithmetic, and end with exactly this line:
CHECKSUM=1073
<|assistant|>
""".strip()


def main() -> int:
    """Cold-store, extend once, then consume that deferred extension."""
    question_ids = [
        int(token_id)
        for token_id in post_json(
            "/tokenize",
            {"model": MODEL, "prompt": QUESTION},
            timeout=300,
        )["tokens"]
    ]
    if len(question_ids) >= EXTENSION_TOKENS:
        raise RuntimeError("extension correctness question is too long")

    required = BASE_TOKENS + FIRST_SUFFIX_TOKENS + EXTENSION_TOKENS
    corpus_ids = corpus_token_ids(required)
    base_prompt = corpus_ids[:BASE_TOKENS]
    first_hit_prompt = corpus_ids[: BASE_TOKENS + FIRST_SUFFIX_TOKENS]
    extension_bridge = corpus_ids[
        BASE_TOKENS
        + FIRST_SUFFIX_TOKENS : required
        - len(question_ids)
    ]
    extension_prompt = first_hit_prompt + extension_bridge + question_ids
    emit(
        "extension_prompt_ready",
        base_tokens=len(base_prompt),
        first_hit_tokens=len(first_hit_prompt),
        extension_tokens=len(extension_prompt),
        question_tokens=len(question_ids),
        first_hit_sha256=hashlib.sha256(
            ",".join(str(token_id) for token_id in first_hit_prompt).encode()
        ).hexdigest(),
    )

    complete("cold_store", base_prompt, 1)
    emit("sleep", seconds=STORE_WAIT_S, reason="cold_admission")
    time.sleep(STORE_WAIT_S)
    complete("first_hit_store_suffix", first_hit_prompt, 1)
    emit("sleep", seconds=ADMISSION_WAIT_S, reason="deferred_suffix_admission")
    time.sleep(ADMISSION_WAIT_S)
    result = complete("extension_hit", extension_prompt, MAX_TOKENS)
    text = str(result["output_text"])
    checks = {
        "has_checksum": "CHECKSUM=1073" in text,
        "has_product": "1073" in text,
        "has_division": "8" in text,
    }
    print(json.dumps({"event": "semantic_checks", **checks}), flush=True)
    if not all(checks.values()):
        raise RuntimeError(f"extension answer failed semantic checks: {checks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
