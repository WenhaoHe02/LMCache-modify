# SPDX-License-Identifier: Apache-2.0
"""Run an isolated multi-shape TTFT matrix against one cached prefix."""

from __future__ import annotations

# Standard
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any
import urllib.error
import urllib.request

# Third Party
import pyarrow.parquet as pq


ENDPOINT = os.environ.get("VLLM_ENDPOINT", "http://127.0.0.1:8000")
MODEL = os.environ.get("MODEL_NAME", "deepseek-v4-pro")
DATASET = Path(
    "/home/zbuser02/datasets/hermes-agent-reasoning-traces/glm-5.1/train.parquet"
)
BASE_TOKENS = int(os.environ["BASE_TOKENS"])
RECOMPUTE_SIZES = [int(value) for value in os.environ["RECOMPUTE_SIZES"].split(",")]
NUM_HITS = int(os.environ.get("NUM_HITS", "3"))
SKIP_ROWS = int(os.environ.get("SKIP_ROWS", "512"))
STORE_WAIT_S = float(os.environ.get("STORE_WAIT_S", "20"))
BETWEEN_REQUEST_S = float(os.environ.get("BETWEEN_REQUEST_S", "1"))
MINIMUM_CHARS = int(os.environ.get("MINIMUM_CHARS", "5500000"))
CONTAINER_NAME = os.environ.get("CONTAINER_NAME", "dsv4-csa-cp8-on")
VALIDATION_TIMEOUT_S = float(os.environ.get("VALIDATION_TIMEOUT_S", "15"))
ADMISSION_TIMEOUT_S = float(os.environ.get("ADMISSION_TIMEOUT_S", "120"))
MATRIX_NAMESPACE = os.environ.get(
    "MATRIX_NAMESPACE",
    f"matrix-{BASE_TOKENS}-{SKIP_ROWS}-{time.time_ns()}",
)
SALT_TOKENS = int(os.environ.get("SALT_TOKENS", "256"))
EXPECTED_COLD_HIT_TOKENS = int(os.environ.get("EXPECTED_COLD_HIT_TOKENS", "0"))
PREWARM_KERNEL_SIZES = [
    int(value)
    for value in os.environ.get("PREWARM_KERNEL_SIZES", "").split(",")
    if value
]
REQUIRE_CSA_MANIFEST = os.environ.get("REQUIRE_CSA_MANIFEST", "1") == "1"

ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def emit(event: str, **fields: Any) -> None:
    """Write one structured result record."""
    print(json.dumps({"event": event, **fields}, ensure_ascii=False), flush=True)


def post_json(
    path: str,
    payload: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    """POST JSON to the serving endpoint and return the decoded response."""
    request = urllib.request.Request(
        f"{ENDPOINT}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
    return json.loads(body.decode()) if body else {}


def server_logs_since(started_epoch: float) -> str:
    """Return container logs emitted since just before one request."""
    completed = subprocess.run(
        [
            "sudo",
            "docker",
            "logs",
            "--since",
            str(max(0, int(started_epoch) - 1)),
            CONTAINER_NAME,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return ANSI_ESCAPE.sub("", completed.stdout + completed.stderr)


def server_logs_between(started_epoch: float, finished_epoch: float) -> str:
    """Return only logs emitted during one completed request.

    A short tail allowance includes asynchronous log flushing without moving the
    start boundary into the preceding request.
    """
    completed = subprocess.run(
        [
            "sudo",
            "docker",
            "logs",
            "--since",
            str(started_epoch),
            "--until",
            str(finished_epoch + 0.2),
            CONTAINER_NAME,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return ANSI_ESCAPE.sub("", completed.stdout + completed.stderr)


def wait_for_hit_validation(
    request_id: str,
    expected_hit_tokens: int,
    started_epoch: float,
    finished_epoch: float,
) -> dict[str, Any]:
    """Wait for and validate the server-side cache-hit decision."""
    deadline = time.monotonic() + VALIDATION_TIMEOUT_S
    hit_pattern = re.compile(
        rf"Reqid: {re.escape(request_id)}-[^,]*,.*LMCache hit tokens: (\d+)"
    )
    logs = ""
    hit_tokens: int | None = None
    while time.monotonic() < deadline:
        logs = server_logs_between(started_epoch, max(finished_epoch, time.time()))
        match = hit_pattern.search(logs)
        if match is not None:
            hit_tokens = int(match.group(1))
            break
        time.sleep(0.25)
    indexer_streams = len(
        re.findall(
            rf"IndexerSSDManager: compact native indexer stream started "
            rf"request={re.escape(request_id)}-[^ ]* layers=21",
            logs,
        )
    )
    shard_gathers = logs.count("CSA_SHARD_GATHER")
    fatal_markers = [
        marker
        for marker in (
            "poll_timeout",
            "staging capacity exceeded",
            "Error processing lookup request",
            "Traceback (most recent call last)",
        )
        if marker in logs
    ]
    gather_by_rank = {
        rank: len(
            re.findall(
                rf"Worker_TP{rank}_EP\d+.*CSA_SHARD_GATHER",
                logs,
            )
        )
        for rank in range(8)
    }
    if any(count not in {0, 21} for count in gather_by_rank.values()):
        fatal_markers.append("partial_shard_gather")
    valid = hit_tokens == expected_hit_tokens and not fatal_markers
    result = {
        "request_id": request_id,
        "expected_hit_tokens": expected_hit_tokens,
        "server_hit_tokens": hit_tokens,
        "indexer_streams": indexer_streams,
        "shard_gathers": shard_gathers,
        "shard_gathers_by_rank": gather_by_rank,
        "fatal_markers": fatal_markers,
        "valid": valid,
    }
    emit("request_validation", **result)
    return result


def wait_for_admission(started_epoch: float) -> dict[str, Any]:
    """Wait until all eight ranks publish the cold base generation."""
    deadline = time.monotonic() + ADMISSION_TIMEOUT_S
    ranks: set[int] = set()
    zero_layer_snapshots = 0
    logs = ""
    pattern = re.compile(
        rf"Worker_TP(\d+)_EP\d+.*CSA layout published:.*"
        rf"covered_tokens={BASE_TOKENS}\b"
    )
    while time.monotonic() < deadline:
        logs = server_logs_since(started_epoch)
        ranks = {int(value) for value in pattern.findall(logs)}
        zero_layer_snapshots = len(
            re.findall(
                rf"attention layer-major snapshot.*layers=0.*"
                rf"tokens={BASE_TOKENS}\b",
                logs,
            )
        )
        if len(ranks) == 8 or zero_layer_snapshots > 0:
            break
        time.sleep(0.5)
    result = {
        "base_tokens": BASE_TOKENS,
        "ready_ranks": sorted(ranks),
        "zero_layer_snapshots": zero_layer_snapshots,
        "valid": len(ranks) == 8 and zero_layer_snapshots == 0,
    }
    emit("admission_validation", **result)
    return result


def dataset_corpus() -> str:
    """Build a deterministic token source starting at ``SKIP_ROWS``."""
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
            if chars >= MINIMUM_CHARS:
                return "".join(parts)
    raise RuntimeError("Hermes corpus is shorter than MINIMUM_CHARS")


def completion(
    label: str,
    prompt: list[int],
    phase: str,
    *,
    skip_save: bool = False,
) -> dict[str, Any]:
    """Issue one deterministic one-token request and report TTFT."""
    prompt_sha256 = hashlib.sha256(
        ",".join(str(token_id) for token_id in prompt).encode()
    ).hexdigest()
    started_epoch = time.time()
    started = time.perf_counter()
    try:
        payload: dict[str, Any] = {
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0,
            "stream": False,
        }
        if skip_save:
            payload["kv_transfer_params"] = {"lmcache.skip_save": True}
        response = post_json(
            "/v1/completions",
            payload,
            timeout=1_800,
        )
    except urllib.error.HTTPError as exc:
        emit(
            "request_complete",
            label=label,
            phase=phase,
            status=exc.code,
            started_epoch=started_epoch,
            elapsed_s=time.perf_counter() - started,
            prompt_tokens=len(prompt),
            prompt_sha256=prompt_sha256,
            error=exc.read().decode(errors="replace")[:2_000],
        )
        raise
    finished_epoch = time.time()
    result = {
        "label": label,
        "phase": phase,
        "status": 200,
        "request_id": response.get("id"),
        "started_epoch": started_epoch,
        "elapsed_s": time.perf_counter() - started,
        "finished_epoch": finished_epoch,
        "prompt_tokens": len(prompt),
        "prompt_sha256": prompt_sha256,
        "usage": response.get("usage"),
        "output_token_ids": response.get("choices", [{}])[0].get("token_ids"),
        "skip_save": skip_save,
    }
    emit("request_complete", **result)
    return result


def main() -> int:
    """Cold-store one base, then benchmark disjoint recompute suffixes."""
    if BASE_TOKENS <= 0 or NUM_HITS <= 0:
        raise ValueError("BASE_TOKENS and NUM_HITS must be positive")
    if not RECOMPUTE_SIZES or any(size <= 0 for size in RECOMPUTE_SIZES):
        raise ValueError("RECOMPUTE_SIZES must contain positive values")

    emit(
        "matrix_start",
        base_tokens=BASE_TOKENS,
        recompute_sizes=RECOMPUTE_SIZES,
        num_hits=NUM_HITS,
        skip_rows=SKIP_ROWS,
        store_wait_s=STORE_WAIT_S,
        matrix_namespace=MATRIX_NAMESPACE,
        salt_tokens=SALT_TOKENS,
        expected_cold_hit_tokens=EXPECTED_COLD_HIT_TOKENS,
        prewarm_kernel_sizes=PREWARM_KERNEL_SIZES,
        require_csa_manifest=REQUIRE_CSA_MANIFEST,
    )
    corpus = dataset_corpus()
    corpus_tokenized = post_json(
        "/tokenize",
        {"model": MODEL, "prompt": corpus},
        timeout=900,
    )
    corpus_ids = [int(token_id) for token_id in corpus_tokenized["tokens"]]
    salt_digest = hashlib.sha256(MATRIX_NAMESPACE.encode()).hexdigest()
    salt_text = (
        "<|lmcache_matrix_namespace|>\n"
        + (f"{MATRIX_NAMESPACE}:{salt_digest}\n" * 256)
        + "<|end_lmcache_matrix_namespace|>\n"
    )
    salt_tokenized = post_json(
        "/tokenize",
        {"model": MODEL, "prompt": salt_text},
        timeout=120,
    )
    salt_ids = [int(token_id) for token_id in salt_tokenized["tokens"]]
    if SALT_TOKENS < 256 or SALT_TOKENS % 256:
        raise ValueError("SALT_TOKENS must be a positive multiple of 256")
    if BASE_TOKENS <= SALT_TOKENS or len(salt_ids) < SALT_TOKENS:
        raise RuntimeError("matrix namespace salt cannot fill the requested prefix")
    corpus_base_tokens = BASE_TOKENS - SALT_TOKENS
    needed = corpus_base_tokens + (NUM_HITS + 1) * sum(RECOMPUTE_SIZES)
    if len(corpus_ids) < needed:
        raise RuntimeError(f"corpus has {len(corpus_ids)} tokens, need {needed}")

    base_prompt = salt_ids[:SALT_TOKENS] + corpus_ids[:corpus_base_tokens]

    for kernel_size in PREWARM_KERNEL_SIZES:
        if kernel_size <= 0:
            raise ValueError("PREWARM_KERNEL_SIZES must contain positive values")
        repeats = (kernel_size + len(salt_ids) - 1) // len(salt_ids)
        completion(
            f"kernel_prewarm_{kernel_size}",
            (salt_ids * repeats)[:kernel_size],
            "kernel_prewarm",
            skip_save=True,
        )
    emit(
        "base_ready",
        base_tokens=len(base_prompt),
        source_tokens=len(corpus_ids),
        base_sha256=hashlib.sha256(
            ",".join(str(token_id) for token_id in base_prompt).encode()
        ).hexdigest(),
        first_chunk_sha256=hashlib.sha256(
            ",".join(str(token_id) for token_id in base_prompt[:256]).encode()
        ).hexdigest(),
        salt_sha256=salt_digest,
    )
    cold = completion("cold_store", base_prompt, "cold")
    cold_validation = wait_for_hit_validation(
        cold["request_id"],
        EXPECTED_COLD_HIT_TOKENS,
        cold["started_epoch"],
        cold["finished_epoch"],
    )
    admission = (
        wait_for_admission(cold["started_epoch"])
        if REQUIRE_CSA_MANIFEST
        else {"valid": True, "skipped": True}
    )
    if not REQUIRE_CSA_MANIFEST:
        emit(
            "admission_validation",
            base_tokens=BASE_TOKENS,
            valid=True,
            skipped=True,
        )
    if STORE_WAIT_S > 0:
        emit("sleep", seconds=STORE_WAIT_S, reason="after_manifest_ready")
        time.sleep(STORE_WAIT_S)

    cursor = corpus_base_tokens
    failures = int(not cold_validation["valid"] or not admission["valid"])
    for recompute_tokens in RECOMPUTE_SIZES:
        prompts: list[list[int]] = []
        continuation_hashes: list[str] = []
        for _ in range(NUM_HITS + 1):
            continuation = corpus_ids[cursor : cursor + recompute_tokens]
            cursor += recompute_tokens
            prompts.append(base_prompt + continuation)
            continuation_hashes.append(
                hashlib.sha256(
                    ",".join(str(token_id) for token_id in continuation).encode()
                ).hexdigest()
            )
        emit(
            "shape_start",
            base_tokens=BASE_TOKENS,
            recompute_tokens=recompute_tokens,
            continuation_hashes=continuation_hashes,
        )
        warmup = completion(
            f"warmup_{BASE_TOKENS}_{recompute_tokens}",
            prompts[0],
            "warmup",
            skip_save=True,
        )
        warmup_validation = wait_for_hit_validation(
            warmup["request_id"],
            BASE_TOKENS,
            warmup["started_epoch"],
            warmup["finished_epoch"],
        )
        failures += int(not warmup_validation["valid"])
        time.sleep(BETWEEN_REQUEST_S)
        for hit_index, prompt in enumerate(prompts[1:], start=1):
            result = completion(
                f"hit_{BASE_TOKENS}_{recompute_tokens}_{hit_index}",
                prompt,
                "hit",
                skip_save=True,
            )
            failures += result["status"] != 200
            validation = wait_for_hit_validation(
                result["request_id"],
                BASE_TOKENS,
                result["started_epoch"],
                result["finished_epoch"],
            )
            failures += int(not validation["valid"])
            if hit_index < NUM_HITS:
                time.sleep(BETWEEN_REQUEST_S)
        emit(
            "shape_end",
            base_tokens=BASE_TOKENS,
            recompute_tokens=recompute_tokens,
        )

    emit("matrix_end", failures=failures)
    return int(failures != 0)


if __name__ == "__main__":
    raise SystemExit(main())
