# SPDX-License-Identifier: Apache-2.0
"""Run a parameterized Trial-2 Hermes cache-hit prefill."""

from __future__ import annotations

# Standard
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any
import urllib.error
import urllib.request

# Third Party
import pyarrow.parquet as pq


ENDPOINT = os.environ.get("VLLM_ENDPOINT", "http://127.0.0.1:8000")
MP_PROFILE_ENDPOINT = os.environ.get("LMCACHE_MP_PROFILE_ENDPOINT", "").rstrip("/")
PROFILE_STOP = os.environ.get("PROFILE_STOP", "1") == "1"
MODEL = os.environ.get("MODEL_NAME", "deepseek-v4-pro")
DATASET = Path(
    "/home/zbuser02/datasets/hermes-agent-reasoning-traces/glm-5.1/train.parquet"
)
BASE_TOKENS = int(os.environ.get("BASE_TOKENS", "480000"))
RECOMPUTE_TOKENS = int(os.environ.get("RECOMPUTE_TOKENS", "200"))
HIT_TOKENS = BASE_TOKENS + RECOMPUTE_TOKENS
SKIP_ROWS = int(os.environ.get("SKIP_ROWS", "512"))


def post_json(
    path: str,
    payload: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    """POST a JSON object and return the decoded response."""
    request = urllib.request.Request(
        f"{ENDPOINT}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
    return json.loads(body.decode()) if body else {}


def set_mp_cuda_profiler(enabled: bool) -> None:
    """Start or stop CUDA profiling inside an LMCache MP server.

    This optional hook lets an Nsight Systems capture include CUDA work issued
    by the server-driven transfer process. It is inactive unless
    ``LMCACHE_MP_PROFILE_ENDPOINT`` is set.

    Args:
        enabled: Start the CUDA profiler when true; stop it when false.

    Raises:
        RuntimeError: If the MP server rejects the profiling script.
    """
    if not MP_PROFILE_ENDPOINT:
        return
    action = "start" if enabled else "stop"
    script = (
        "import torch\n"
        f"torch.cuda.profiler.{action}()\n"
        f"result = 'cuda_profiler_{action}ed'\n"
    ).encode()
    boundary = "lmcache-cuda-profiler-boundary"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="script"; '
        f'filename="cuda_profiler_{action}.py"\r\n'
        "Content-Type: text/x-python\r\n\r\n"
    ).encode() + script + f"\r\n--{boundary}--\r\n".encode()
    request = urllib.request.Request(
        f"{MP_PROFILE_ENDPOINT}/run_script",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        result = response.read().decode(errors="replace")
    expected = f"cuda_profiler_{action}ed"
    if expected not in result:
        raise RuntimeError(
            f"LMCache MP CUDA profiler {action} failed: {result[:500]}"
        )
    emit("mp_cuda_profiler", action=action, response=result)


def count_tokens(prompt: str) -> int:
    """Return the server tokenizer length for a prompt."""
    response = post_json(
        "/tokenize",
        {"model": MODEL, "prompt": prompt},
        timeout=300,
    )
    return int(response["count"])


def dataset_corpus(minimum_chars: int = 3_500_000) -> str:
    """Build the exact deterministic Trial-2 corpus used by coverage tests."""
    parts: list[str] = []
    chars = 0
    seen_rows = 0
    parquet = pq.ParquetFile(DATASET)
    for batch in parquet.iter_batches(batch_size=32):
        for row in batch.to_pylist():
            if seen_rows < SKIP_ROWS:
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
            if chars >= minimum_chars:
                return "".join(parts)
    raise RuntimeError("Hermes Trial-2 corpus is too short")


def trim_to_token_target(corpus: str, target_tokens: int) -> tuple[str, int]:
    """Return the longest character prefix not exceeding a token target."""
    if count_tokens(corpus) < target_tokens:
        raise RuntimeError(f"corpus is shorter than {target_tokens} tokens")
    low = 1
    high = len(corpus)
    best = corpus[:1]
    best_count = count_tokens(best)
    while low <= high:
        middle = (low + high) // 2
        candidate = corpus[:middle]
        count = count_tokens(candidate)
        if count <= target_tokens:
            best = candidate
            best_count = count
            low = middle + 1
        else:
            high = middle - 1
    return best, best_count


def emit(event: str, **fields: Any) -> None:
    """Emit one structured progress record."""
    print(json.dumps({"event": event, **fields}, ensure_ascii=False), flush=True)


def completion(label: str, prompt: str | list[int]) -> dict[str, Any]:
    """Run one completion and emit its status and elapsed time."""
    started = time.perf_counter()
    try:
        response = post_json(
            "/v1/completions",
            {
                "model": MODEL,
                "prompt": prompt,
                "max_tokens": 1,
                "temperature": 0,
                "stream": False,
            },
            timeout=1_800,
        )
    except urllib.error.HTTPError as exc:
        emit(
            "request_complete",
            label=label,
            status=exc.code,
            elapsed_s=time.perf_counter() - started,
            error=exc.read().decode(errors="replace")[:1_000],
        )
        raise
    result = {
        "label": label,
        "status": 200,
        "elapsed_s": time.perf_counter() - started,
        "usage": response.get("usage"),
    }
    emit("request_complete", **result)
    return result


def run_distinct_prompt_hits(
    corpus: str,
    store_wait_s: float,
    num_warmup_hits: int,
    num_hits: int,
    warmup_wait_s: float,
    hit_wait_s: float,
    profile: bool,
) -> int:
    """Run hits with one shared prefix and disjoint continuation token IDs."""
    if num_warmup_hits != 1:
        raise ValueError("distinct-prompt mode requires exactly one warmup")
    tokenized = post_json(
        "/tokenize",
        {"model": MODEL, "prompt": corpus},
        timeout=600,
    )
    corpus_ids = [int(token_id) for token_id in tokenized["tokens"]]
    required_tokens = BASE_TOKENS + (num_hits + 1) * RECOMPUTE_TOKENS
    if len(corpus_ids) < required_tokens:
        raise RuntimeError(
            f"corpus has {len(corpus_ids)} tokens, need {required_tokens}"
        )
    base_prompt = corpus_ids[:BASE_TOKENS]
    prompts: list[list[int]] = []
    continuation_hashes: list[str] = []
    for variant_index in range(num_hits + 1):
        tail_start = BASE_TOKENS + variant_index * RECOMPUTE_TOKENS
        tail_end = tail_start + RECOMPUTE_TOKENS
        continuation = corpus_ids[tail_start:tail_end]
        prompts.append(base_prompt + continuation)
        continuation_hashes.append(
            hashlib.sha256(
                ",".join(str(token_id) for token_id in continuation).encode()
            ).hexdigest()
        )
    emit(
        "prompt_ready",
        base_tokens=len(base_prompt),
        hit_tokens=len(prompts[0]),
        recompute_tokens=RECOMPUTE_TOKENS,
        distinct_hit_prompts=True,
        continuation_hashes=continuation_hashes,
        profile=profile,
    )

    completion("cold_store", base_prompt)
    emit("sleep", seconds=store_wait_s)
    time.sleep(store_wait_s)
    completion(f"warmup_trial2_{RECOMPUTE_TOKENS}_1", prompts[0])
    if warmup_wait_s > 0:
        emit("sleep", seconds=warmup_wait_s, reason="after_warmup")
        time.sleep(warmup_wait_s)
    if profile:
        set_mp_cuda_profiler(True)
        try:
            post_json("/start_profile", {}, timeout=120)
        except BaseException:
            set_mp_cuda_profiler(False)
            raise
        emit("profile_started")
    try:
        for hit_index, prompt in enumerate(prompts[1:], start=1):
            completion(f"hit_trial2_{RECOMPUTE_TOKENS}_{hit_index}", prompt)
            if hit_index < num_hits and hit_wait_s > 0:
                emit("sleep", seconds=hit_wait_s)
                time.sleep(hit_wait_s)
    finally:
        if profile and PROFILE_STOP:
            try:
                post_json("/stop_profile", {}, timeout=600)
                emit("profile_stopped")
            finally:
                set_mp_cuda_profiler(False)
        elif profile:
            emit("profile_left_running_for_process_exit")
    return 0


def main() -> int:
    """Build the trace, cold-store the prefix, then run cache-hit prefills."""
    store_wait_s = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    num_hits = int(os.environ.get("NUM_HITS", "4"))
    num_warmup_hits = int(os.environ.get("NUM_WARMUP_HITS", "1"))
    hit_wait_s = float(os.environ.get("HIT_WAIT_S", "5"))
    warmup_wait_s = float(os.environ.get("WARMUP_WAIT_S", str(hit_wait_s)))
    shape_sequence_text = os.environ.get("RECOMPUTE_SEQUENCE", "").strip()
    shape_sequence = (
        [int(value) for value in shape_sequence_text.split(",")]
        if shape_sequence_text
        else []
    )
    warmup_recompute = int(
        os.environ.get("WARMUP_RECOMPUTE_TOKENS", str(RECOMPUTE_TOKENS))
    )
    if num_hits <= 0:
        raise ValueError("NUM_HITS must be positive")
    if num_warmup_hits <= 0:
        raise ValueError("NUM_WARMUP_HITS must be positive")
    if BASE_TOKENS <= 0 or RECOMPUTE_TOKENS <= 0:
        raise ValueError("BASE_TOKENS and RECOMPUTE_TOKENS must be positive")
    if warmup_recompute <= 0 or any(value <= 0 for value in shape_sequence):
        raise ValueError("warmup and sequence recompute lengths must be positive")
    if shape_sequence and num_hits != len(shape_sequence):
        raise ValueError("NUM_HITS must match RECOMPUTE_SEQUENCE length")
    profile = os.environ.get("ENABLE_TORCH_PROFILE", "0") == "1"
    emit("building_corpus", dataset=str(DATASET), skip_rows=SKIP_ROWS)
    corpus = dataset_corpus()
    if os.environ.get("DISTINCT_HIT_PROMPTS", "0") == "1":
        if shape_sequence:
            raise ValueError(
                "distinct-prompt mode does not support shape sequences"
            )
        return run_distinct_prompt_hits(
            corpus,
            store_wait_s,
            num_warmup_hits,
            num_hits,
            warmup_wait_s,
            hit_wait_s,
            profile,
        )
    measured_recomputes = shape_sequence or [RECOMPUTE_TOKENS]
    all_recomputes = set(measured_recomputes)
    all_recomputes.add(RECOMPUTE_TOKENS)
    all_recomputes.add(warmup_recompute)
    maximum_tokens = BASE_TOKENS + max(all_recomputes)
    maximum_prompt, maximum_count = trim_to_token_target(corpus, maximum_tokens)
    base_prompt, base_count = trim_to_token_target(maximum_prompt, BASE_TOKENS)
    prompts: dict[int, str] = {}
    prompt_counts: dict[int, int] = {}
    for recompute in sorted(all_recomputes):
        prompt, count = trim_to_token_target(
            maximum_prompt,
            BASE_TOKENS + recompute,
        )
        prompts[recompute] = prompt
        prompt_counts[recompute] = count
    if base_count != BASE_TOKENS or maximum_count != maximum_tokens:
        raise RuntimeError(
            "exact token targets unavailable: "
            f"base={base_count}, maximum={maximum_count}"
        )
    for recompute, count in prompt_counts.items():
        if count != BASE_TOKENS + recompute:
            raise RuntimeError(
                f"exact token target unavailable for recompute={recompute}: {count}"
            )
        if not prompts[recompute].startswith(base_prompt):
            raise RuntimeError("cold prompt is not a character prefix of hit prompt")
    emit(
        "prompt_ready",
        base_tokens=base_count,
        hit_tokens=BASE_TOKENS + RECOMPUTE_TOKENS,
        recompute_tokens=RECOMPUTE_TOKENS,
        warmup_recompute_tokens=warmup_recompute,
        recompute_sequence=shape_sequence,
        base_chars=len(base_prompt),
        hit_chars=len(prompts[RECOMPUTE_TOKENS]),
        base_sha256=hashlib.sha256(base_prompt.encode()).hexdigest(),
        hit_sha256=hashlib.sha256(
            prompts[RECOMPUTE_TOKENS].encode()
        ).hexdigest(),
        profile=profile,
    )

    completion("cold_store", base_prompt)
    emit("sleep", seconds=store_wait_s)
    time.sleep(store_wait_s)

    for warmup_index in range(1, num_warmup_hits + 1):
        completion(
            f"warmup_trial2_{warmup_recompute}_{warmup_index}",
            prompts[warmup_recompute],
        )
    if warmup_wait_s > 0:
        emit("sleep", seconds=warmup_wait_s, reason="after_warmup")
        time.sleep(warmup_wait_s)

    if profile:
        set_mp_cuda_profiler(True)
        try:
            post_json("/start_profile", {}, timeout=120)
        except BaseException:
            set_mp_cuda_profiler(False)
            raise
        emit("profile_started")
    hit_results: list[dict[str, Any]] = []
    try:
        recomputes = shape_sequence or [RECOMPUTE_TOKENS] * num_hits
        for hit_index, recompute in enumerate(recomputes, start=1):
            label = (
                f"hit_shape_{recompute}_{hit_index}"
                if shape_sequence
                else f"hit_trial2_{recompute}_{hit_index}"
            )
            hit_results.append(
                completion(label, prompts[recompute])
            )
            if hit_index < num_hits and hit_wait_s > 0:
                emit("sleep", seconds=hit_wait_s)
                time.sleep(hit_wait_s)
    finally:
        if profile and PROFILE_STOP:
            try:
                post_json("/stop_profile", {}, timeout=600)
                emit("profile_stopped")
            finally:
                set_mp_cuda_profiler(False)
        elif profile:
            emit("profile_left_running_for_process_exit")
    if any(hit["status"] != 200 for hit in hit_results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
