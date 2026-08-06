# SPDX-License-Identifier: Apache-2.0
"""Tests for vLLM frontend tool-call slack tracking."""

# Standard
from types import SimpleNamespace
import asyncio
import json

# First Party
from lmcache.integration.vllm.tool_slack_hook import VllmToolSlackTracker
from lmcache.v1.write_slack_client import WriteSlackHandle


class _Client:
    def __init__(self) -> None:
        self.begun: list[float] = []
        self.ended: list[WriteSlackHandle] = []
        self.handle = WriteSlackHandle("tool_call", ())

    def begin_tool_call(self, expected_duration_s: float) -> WriteSlackHandle:
        self.begun.append(expected_duration_s)
        return self.handle

    def end(self, handle: WriteSlackHandle) -> None:
        self.ended.append(handle)


def _raw_request(**headers: str) -> SimpleNamespace:
    return SimpleNamespace(headers=headers)


def test_stream_tool_call_opens_then_continuation_closes_slack() -> None:
    client = _Client()
    tracker = VllmToolSlackTracker(client, default_duration_s=10.0)  # type: ignore[arg-type]

    async def run() -> list[str]:
        async def stream():
            yield "data: " + json.dumps(
                {
                    "choices": [
                        {
                            "delta": {"tool_calls": [{"id": "call-1"}]},
                            "finish_reason": None,
                        }
                    ]
                }
            )
            yield "data: " + json.dumps(
                {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}
            )

        chunks = [
            chunk
            async for chunk in tracker.wrap_stream(
                stream(),
                _raw_request(**{"x-lmcache-tool-slack-seconds": "4"}),
            )
        ]
        continuation = SimpleNamespace(
            messages=[{"role": "tool", "tool_call_id": "call-1"}]
        )
        await tracker.before_request(continuation, _raw_request())
        return chunks

    chunks = asyncio.run(run())

    assert len(chunks) == 2
    assert client.begun == [4.0]
    assert client.ended == [client.handle]


def test_session_header_supports_xml_tool_protocol_without_call_id() -> None:
    client = _Client()
    tracker = VllmToolSlackTracker(client, default_duration_s=7.0)  # type: ignore[arg-type]
    response = {
        "choices": [
            {
                "message": {"content": "<tool_call>...</tool_call>"},
                "finish_reason": "stop",
            }
        ]
    }
    raw_request = _raw_request(**{"x-lmcache-agent-session-id": "agent-7"})

    async def run() -> None:
        await tracker.observe_full_response(response, raw_request)
        continuation = SimpleNamespace(messages=[])
        await tracker.before_request(continuation, raw_request)

    asyncio.run(run())

    assert client.begun == [7.0]
    assert client.ended == [client.handle]


def test_tools_in_request_without_confirmed_output_does_not_open_slack() -> None:
    client = _Client()
    tracker = VllmToolSlackTracker(client)  # type: ignore[arg-type]
    response = {
        "choices": [
            {
                "message": {"content": "normal answer"},
                "finish_reason": "stop",
            }
        ]
    }

    asyncio.run(tracker.observe_full_response(response, _raw_request()))

    assert client.begun == []
