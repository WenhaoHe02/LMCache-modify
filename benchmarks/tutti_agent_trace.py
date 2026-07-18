# SPDX-License-Identifier: Apache-2.0
"""Inspect and replay multi-turn agent traces against an LMCache-backed engine."""

from __future__ import annotations

# Standard
import argparse
import asyncio
from collections.abc import Iterable
import json
import math
from pathlib import Path
import random
import statistics
import time
from typing import Any

# Third Party
import pyarrow.parquet as pq
import requests


ROLE_NAMES = {
    "gpt": "assistant",
    "human": "user",
}
MODEL_INPUT_ROLES = {"human", "user", "tool"}


def iter_rows(path: Path) -> Iterable[dict[str, Any]]:
    """Yield rows from a Parquet agent-trajectory dataset.

    Args:
        path: Input Parquet file.

    Yields:
        Dataset rows converted to Python dictionaries.
    """
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=64):
        yield from batch.to_pylist()


def message_role(message: dict[str, Any]) -> str:
    """Return the normalized role name for one conversation message.

    Args:
        message: A ShareGPT- or OpenAI-style message dictionary.

    Returns:
        Normalized role name.
    """
    role = str(message.get("from", message.get("role", "unknown")))
    return ROLE_NAMES.get(role, role)


def message_text(message: dict[str, Any]) -> str:
    """Return the textual payload for one conversation message.

    Args:
        message: A ShareGPT- or OpenAI-style message dictionary.

    Returns:
        Message content serialized as text.
    """
    value = message.get("value", message.get("content", ""))
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def build_model_input_prefixes(row: dict[str, Any]) -> list[str]:
    """Build the growing prompts at every agent inference boundary.

    An inference boundary occurs after a user or tool message. Earlier
    assistant and tool messages remain byte-identical prefixes, which makes
    these prompts suitable for measuring multi-turn KV reuse.

    Args:
        row: One trajectory row containing ``conversations`` and optionally
            ``tools``.

    Returns:
        Serialized prompts in chronological order.
    """
    parts: list[str] = []
    tools = row.get("tools")
    if tools:
        tools_text = tools if isinstance(tools, str) else json.dumps(
            tools, ensure_ascii=False, sort_keys=True
        )
        parts.append(f"<|tools|>\n{tools_text}\n<|end|>\n")

    prefixes: list[str] = []
    for raw_message in row.get("conversations", []):
        if not isinstance(raw_message, dict):
            continue
        role = message_role(raw_message)
        parts.append(f"<|{role}|>\n{message_text(raw_message)}\n<|end|>\n")
        original_role = str(raw_message.get("from", raw_message.get("role", "")))
        if original_role in MODEL_INPUT_ROLES or role in MODEL_INPUT_ROLES:
            prefixes.append("".join(parts))
    return prefixes


def percentile(values: list[float], fraction: float) -> float:
    """Return a nearest-rank percentile.

    Args:
        values: Numeric samples.
        fraction: Percentile expressed in the inclusive range ``[0, 1]``.

    Returns:
        Requested percentile, or NaN when ``values`` is empty.
    """
    if not values:
        return math.nan
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def summarize(values: list[float]) -> dict[str, float]:
    """Summarize a numeric sample.

    Args:
        values: Numeric samples.

    Returns:
        Count, mean, and selected percentiles.
    """
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else math.nan,
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p99": percentile(values, 0.99),
        "min": min(values) if values else math.nan,
        "max": max(values) if values else math.nan,
    }


def inspect_dataset(args: argparse.Namespace) -> None:
    """Print session, turn, character, and estimated-token distributions.

    Args:
        args: Parsed ``inspect`` arguments.
    """
    session_chars: list[float] = []
    inference_turns: list[float] = []
    prefix_chars: list[float] = []
    exact_session_tokens: list[float] = []
    categories: dict[str, int] = {}
    tokenizer = None
    if args.tokenizer is not None:
        # Third Party
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer,
            trust_remote_code=True,
        )

    for index, row in enumerate(iter_rows(args.dataset)):
        if args.limit and index >= args.limit:
            break
        prefixes = build_model_input_prefixes(row)
        if not prefixes:
            continue
        session_chars.append(float(len(prefixes[-1])))
        if tokenizer is not None:
            exact_session_tokens.append(
                float(len(tokenizer.encode(prefixes[-1], add_special_tokens=False)))
            )
        inference_turns.append(float(len(prefixes)))
        prefix_chars.extend(float(len(prompt)) for prompt in prefixes)
        category = str(row.get("category", "unknown"))
        categories[category] = categories.get(category, 0) + 1

    result = {
        "dataset": str(args.dataset),
        "sessions": len(session_chars),
        "inference_turns": summarize(inference_turns),
        "final_prompt_chars": summarize(session_chars),
        "all_prefix_chars": summarize(prefix_chars),
        "estimated_final_prompt_tokens": summarize(
            [value / args.chars_per_token for value in session_chars]
        ),
        "exact_final_prompt_tokens": (
            summarize(exact_session_tokens) if tokenizer is not None else None
        ),
        "categories": categories,
        "chars_per_token_assumption": args.chars_per_token,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


def select_sessions(
    dataset: Path,
    num_sessions: int,
    min_estimated_tokens: int,
    max_estimated_tokens: int,
    chars_per_token: float,
    seed: int,
    session_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Select deterministic sessions within an estimated token-length band.

    Args:
        dataset: Input Parquet file.
        num_sessions: Number of sessions to select.
        min_estimated_tokens: Minimum final-prefix length.
        max_estimated_tokens: Maximum final-prefix length.
        chars_per_token: Character-to-token estimate used before requests.
        seed: Random selection seed.
        session_ids: Exact dataset session IDs to select, in replay order. When
            provided, deterministic reservoir sampling is bypassed.

    Returns:
        Selected session dictionaries with prepared prompts.

    Raises:
        RuntimeError: If too few eligible sessions exist.
    """
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    eligible_count = 0
    requested_ids = set(session_ids or [])
    for row in iter_rows(dataset):
        session_id = str(row.get("id", f"row-{eligible_count}"))
        if requested_ids and session_id not in requested_ids:
            continue
        prefixes = build_model_input_prefixes(row)
        if len(prefixes) < 2:
            continue
        final_estimate = len(prefixes[-1]) / chars_per_token
        if not min_estimated_tokens <= final_estimate <= max_estimated_tokens:
            continue
        usable = [
            prompt
            for prompt in prefixes
            if len(prompt) / chars_per_token >= min_estimated_tokens / 2
        ]
        if len(usable) < 2:
            continue
        candidate = {
            "session_id": session_id,
            "category": str(row.get("category", "unknown")),
            "prefixes": usable,
            "final_estimated_tokens": round(final_estimate),
        }
        eligible_count += 1
        if requested_ids:
            selected.append(candidate)
        elif len(selected) < num_sessions:
            selected.append(candidate)
        else:
            replacement_index = rng.randrange(eligible_count)
            if replacement_index < num_sessions:
                selected[replacement_index] = candidate

    if requested_ids:
        found = {session["session_id"] for session in selected}
        missing = requested_ids - found
        if missing:
            raise RuntimeError(
                "Requested sessions were missing or outside the length band: "
                + ", ".join(sorted(missing))
            )
        order = {
            session_id: index
            for index, session_id in enumerate(session_ids or [])
        }
        return sorted(selected, key=lambda session: order[session["session_id"]])
    if eligible_count < num_sessions:
        raise RuntimeError(
            f"Only {eligible_count} eligible sessions; requested {num_sessions}"
        )
    return selected


def post_completion(
    url: str,
    model: str,
    prompt: str,
    timeout: float,
) -> dict[str, Any]:
    """Send one non-streaming, one-token completion request.

    Args:
        url: OpenAI-compatible completions endpoint.
        model: Served model name.
        prompt: Serialized agent conversation prefix.
        timeout: HTTP timeout in seconds.

    Returns:
        Request timing, status, token count, and error information.
    """
    started = time.perf_counter()
    try:
        response = requests.post(
            url,
            json={
                "model": model,
                "prompt": prompt,
                "max_tokens": 1,
                "temperature": 0,
                "stream": False,
            },
            timeout=timeout,
        )
        elapsed = time.perf_counter() - started
        response.raise_for_status()
        body = response.json()
        usage = body.get("usage", {})
        return {
            "ok": True,
            "latency_s": elapsed,
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "error": "",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "latency_s": time.perf_counter() - started,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "error": repr(exc),
        }


async def run_rate(
    args: argparse.Namespace,
    sessions: list[dict[str, Any]],
    qps: float,
    output_file: Any,
) -> dict[str, Any]:
    """Run one open-loop Poisson arrival-rate experiment.

    Args:
        args: Parsed ``replay`` arguments.
        sessions: Prepared multi-turn sessions.
        qps: Offered request rate.
        output_file: Open JSONL output stream.

    Returns:
        Summary dictionary for this offered rate.
    """
    rng = random.Random(args.seed + round(qps * 1_000_000))
    events: list[tuple[dict[str, Any], int, str]] = []
    for request_index in range(args.requests_per_rate):
        session = sessions[request_index % len(sessions)]
        prefixes = session["prefixes"]
        turn_index = (request_index // len(sessions)) % len(prefixes)
        events.append((session, turn_index, prefixes[turn_index]))

    scheduled_offsets: list[float] = []
    offset = 0.0
    for _ in events:
        scheduled_offsets.append(offset)
        offset += rng.expovariate(qps)

    run_started = time.perf_counter()

    async def dispatch(
        request_index: int,
        event: tuple[dict[str, Any], int, str],
        scheduled_offset: float,
    ) -> dict[str, Any]:
        delay = run_started + scheduled_offset - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        actual_submit_offset = time.perf_counter() - run_started
        session, turn_index, prompt = event
        result = await asyncio.to_thread(
            post_completion,
            args.url,
            args.model,
            prompt,
            args.timeout,
        )
        result.update(
            {
                "type": "request",
                "qps": qps,
                "request_index": request_index,
                "session_id": session["session_id"],
                "category": session["category"],
                "turn_index": turn_index,
                "scheduled_offset_s": scheduled_offset,
                "actual_submit_offset_s": actual_submit_offset,
                "submit_lag_s": actual_submit_offset - scheduled_offset,
            }
        )
        return result

    tasks = [
        asyncio.create_task(dispatch(index, event, scheduled_offsets[index]))
        for index, event in enumerate(events)
    ]
    results = await asyncio.gather(*tasks)
    finished = time.perf_counter()
    for result in results:
        output_file.write(json.dumps(result, sort_keys=True) + "\n")
    output_file.flush()

    good = [result for result in results if result["ok"]]
    latencies = [float(result["latency_s"]) for result in good]
    prompt_tokens = [float(result["prompt_tokens"]) for result in good]
    elapsed = finished - run_started
    summary = {
        "type": "summary",
        "qps": qps,
        "offered_requests": len(results),
        "successful_requests": len(good),
        "failed_requests": len(results) - len(good),
        "elapsed_s": elapsed,
        "achieved_qps": len(good) / elapsed if elapsed else 0.0,
        "latency_s": summarize(latencies),
        "prompt_tokens": summarize(prompt_tokens),
    }
    output_file.write(json.dumps(summary, sort_keys=True) + "\n")
    output_file.flush()
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


async def replay_dataset(args: argparse.Namespace) -> None:
    """Populate selected sessions and run the arrival-rate sweep.

    Args:
        args: Parsed ``replay`` arguments.
    """
    sessions = select_sessions(
        args.dataset,
        args.num_sessions,
        args.min_estimated_tokens,
        args.max_estimated_tokens,
        args.chars_per_token,
        args.seed,
        args.session_ids,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    selection = [
        {
            "session_id": session["session_id"],
            "category": session["category"],
            "turns": len(session["prefixes"]),
            "final_estimated_tokens": session["final_estimated_tokens"],
        }
        for session in sessions
    ]
    print(json.dumps({"selected_sessions": selection}, indent=2), flush=True)

    with args.output.open("w", encoding="utf-8") as output_file:
        output_file.write(
            json.dumps(
                {
                    "type": "config",
                    "dataset": str(args.dataset),
                    "model": args.model,
                    "qps": args.qps,
                    "requests_per_rate": args.requests_per_rate,
                    "sessions": selection,
                    "arrival_process": "poisson-open-loop",
                    "seed": args.seed,
                },
                sort_keys=True,
            )
            + "\n"
        )

        for session in sessions:
            result = await asyncio.to_thread(
                post_completion,
                args.url,
                args.model,
                session["prefixes"][-1],
                args.timeout,
            )
            result.update(
                {
                    "type": "populate",
                    "session_id": session["session_id"],
                    "category": session["category"],
                }
            )
            output_file.write(json.dumps(result, sort_keys=True) + "\n")
            output_file.flush()
            print(json.dumps(result, sort_keys=True), flush=True)

        print(f"waiting {args.store_wait_s:.1f}s for asynchronous stores", flush=True)
        await asyncio.sleep(args.store_wait_s)
        for qps in args.qps:
            await run_rate(args, sessions, qps, output_file)
            await asyncio.sleep(args.between_rates_s)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("--dataset", type=Path, required=True)
    inspect_parser.add_argument("--chars-per-token", type=float, default=4.0)
    inspect_parser.add_argument("--limit", type=int, default=0)
    inspect_parser.add_argument("--tokenizer", type=Path)

    replay_parser = subparsers.add_parser("replay")
    replay_parser.add_argument("--dataset", type=Path, required=True)
    replay_parser.add_argument(
        "--url", default="http://127.0.0.1:8000/v1/completions"
    )
    replay_parser.add_argument("--model", default="deepseek-v4-pro")
    replay_parser.add_argument("--num-sessions", type=int, default=4)
    replay_parser.add_argument("--min-estimated-tokens", type=int, default=8192)
    replay_parser.add_argument("--max-estimated-tokens", type=int, default=98304)
    replay_parser.add_argument("--chars-per-token", type=float, default=4.0)
    replay_parser.add_argument("--qps", type=float, nargs="+", required=True)
    replay_parser.add_argument("--requests-per-rate", type=int, default=12)
    replay_parser.add_argument("--store-wait-s", type=float, default=30.0)
    replay_parser.add_argument("--between-rates-s", type=float, default=10.0)
    replay_parser.add_argument("--timeout", type=float, default=900.0)
    replay_parser.add_argument("--seed", type=int, default=20260715)
    replay_parser.add_argument(
        "--session-ids",
        nargs="+",
        help="Exact session IDs to replay in the supplied order.",
    )
    replay_parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Run dataset inspection or replay."""
    args = parse_args()
    if args.command == "inspect":
        inspect_dataset(args)
    else:
        asyncio.run(replay_dataset(args))


if __name__ == "__main__":
    main()
