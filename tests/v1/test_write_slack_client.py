# SPDX-License-Identifier: Apache-2.0
"""Tests for TP worker write-slack fan-out."""

# Standard
from unittest.mock import patch
import json
import urllib.error

# Third Party
import pytest

# First Party
from lmcache.v1.write_slack_client import TuttiWriteSlackFanoutClient


class _Response:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def test_tool_call_begin_and_end_fan_out_to_every_worker() -> None:
    client = TuttiWriteSlackFanoutClient.from_worker_ports(worker_count=2)
    calls: list[tuple[str, dict]] = []

    def urlopen(request, timeout):
        payload = json.loads(request.data.decode("utf-8"))
        calls.append((request.full_url, payload))
        if request.full_url.endswith("/begin"):
            port = int(request.full_url.split(":")[2].split("/")[0])
            return _Response({"token": port})
        return _Response({"removed": True})

    with patch("urllib.request.urlopen", side_effect=urlopen):
        handle = client.begin_tool_call(expected_duration_s=3.0)
        client.end(handle)

    assert len(handle.worker_tokens) == 2
    assert sum(url.endswith("/begin") for url, _ in calls) == 2
    assert sum(url.endswith("/end") for url, _ in calls) == 2
    assert all(
        payload.get("expected_duration_s") == 3.0
        for url, payload in calls
        if url.endswith("/begin")
    )


def test_partial_begin_failure_rolls_back_successful_workers() -> None:
    client = TuttiWriteSlackFanoutClient.from_worker_ports(worker_count=2)
    closed_tokens: list[int] = []

    def urlopen(request, timeout):
        payload = json.loads(request.data.decode("utf-8"))
        if request.full_url.endswith("/begin"):
            if ":7001/" in request.full_url:
                raise urllib.error.URLError("worker unavailable")
            return _Response({"token": 11})
        closed_tokens.append(int(payload["token"]))
        return _Response({"removed": True})

    with (
        patch("urllib.request.urlopen", side_effect=urlopen),
        pytest.raises(RuntimeError, match="begin failed"),
    ):
        client.begin_tool_call()

    assert closed_tokens == [11]
