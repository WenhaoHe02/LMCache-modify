# SPDX-License-Identifier: Apache-2.0
"""vLLM OpenAI frontend hook for real tool-call write slack."""

# Standard
import asyncio
import json
import os
import threading
from collections.abc import AsyncGenerator
from functools import wraps
from typing import Any, Optional

# First Party
from lmcache.logging import init_logger
from lmcache.v1.write_slack_client import (
    TuttiWriteSlackFanoutClient,
    WriteSlackHandle,
)

logger = init_logger(__name__)

_SESSION_HEADER = "x-lmcache-agent-session-id"
_DURATION_HEADER = "x-lmcache-tool-slack-seconds"


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        return dumped if isinstance(dumped, dict) else {}
    return {}


def _request_close_keys(request: Any, raw_request: Any) -> set[str]:
    keys: set[str] = set()
    for message in getattr(request, "messages", None) or []:
        data = _as_dict(message)
        if data.get("role") != "tool":
            continue
        tool_call_id = data.get("tool_call_id")
        if tool_call_id:
            keys.add(f"tool:{tool_call_id}")
    session_id = _header(raw_request, _SESSION_HEADER)
    if session_id:
        keys.add(f"session:{session_id}")
    return keys


def _header(raw_request: Any, name: str) -> Optional[str]:
    headers = getattr(raw_request, "headers", None)
    if headers is None:
        return None
    value = headers.get(name)
    normalized = str(value).strip() if value is not None else ""
    return normalized or None


def _expected_duration(raw_request: Any, default_s: float) -> float:
    raw_value = _header(raw_request, _DURATION_HEADER)
    if raw_value is None:
        return default_s
    try:
        value = float(raw_value)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r", _DURATION_HEADER, raw_value)
        return default_s
    return value if value > 0 else default_s


def _tool_keys_from_choice(choice: dict[str, Any]) -> set[str]:
    message = _as_dict(choice.get("message"))
    delta = _as_dict(choice.get("delta"))
    tool_calls = message.get("tool_calls") or delta.get("tool_calls") or []
    keys = set()
    for tool_call in tool_calls:
        tool_call_id = _as_dict(tool_call).get("id")
        if tool_call_id:
            keys.add(f"tool:{tool_call_id}")
    return keys


def _choice_contains_xml_tool_call(choice: dict[str, Any]) -> bool:
    message = _as_dict(choice.get("message"))
    delta = _as_dict(choice.get("delta"))
    text = "".join(
        str(value or "")
        for value in (
            message.get("content"),
            message.get("reasoning_content"),
            delta.get("content"),
            delta.get("reasoning_content"),
        )
    )
    return "<tool_call>" in text


class VllmToolSlackTracker:
    """Map parsed OpenAI tool calls to worker write-slack handles.

    Args:
        client: TP-worker fan-out client.
        default_duration_s: Failsafe lifetime for a tool window.

    Raises:
        ValueError: If the failsafe duration is not positive.
    """

    def __init__(
        self,
        client: TuttiWriteSlackFanoutClient,
        default_duration_s: float = 30.0,
    ) -> None:
        if default_duration_s <= 0:
            raise ValueError("default_duration_s must be positive")
        self._client = client
        self._default_duration_s = default_duration_s
        self._lock = threading.Lock()
        self._handles_by_key: dict[str, WriteSlackHandle] = {}

    async def before_request(self, request: Any, raw_request: Any) -> None:
        """Close matching tool windows before scheduling a continuation.

        Args:
            request: vLLM chat-completion request.
            raw_request: Starlette request containing optional session headers.
        """
        await self._close_keys(_request_close_keys(request, raw_request))

    async def observe_full_response(self, response: Any, raw_request: Any) -> None:
        """Open slack when a non-streaming response confirms tool calls.

        Args:
            response: vLLM chat-completion response.
            raw_request: Starlette request containing optional session headers.
        """
        data = _as_dict(response)
        keys: set[str] = set()
        confirmed = False
        for choice_value in data.get("choices") or []:
            choice = _as_dict(choice_value)
            if choice.get("finish_reason") == "tool_calls":
                confirmed = True
                keys.update(_tool_keys_from_choice(choice))
            confirmed = confirmed or _choice_contains_xml_tool_call(choice)
        await self._open_confirmed(keys, confirmed, raw_request)

    async def wrap_stream(
        self,
        stream: AsyncGenerator[str, None],
        raw_request: Any,
    ) -> AsyncGenerator[str, None]:
        """Forward an SSE stream while observing confirmed tool-call output.

        Args:
            stream: Original vLLM SSE generator.
            raw_request: Starlette request containing optional session headers.

        Yields:
            Original SSE chunks without modification.
        """
        keys: set[str] = set()
        confirmed = False
        streamed_text = ""
        async for chunk in stream:
            payload = chunk.strip()
            if payload.startswith("data: "):
                payload = payload[6:].strip()
            if payload and payload != "[DONE]":
                try:
                    data = json.loads(payload)
                except (TypeError, ValueError):
                    data = {}
                for choice_value in data.get("choices") or []:
                    choice = _as_dict(choice_value)
                    keys.update(_tool_keys_from_choice(choice))
                    delta = _as_dict(choice.get("delta"))
                    streamed_text += str(delta.get("content") or "")
                    streamed_text += str(delta.get("reasoning_content") or "")
                    confirmed = confirmed or (
                        choice.get("finish_reason") == "tool_calls"
                    )
                    confirmed = confirmed or "<tool_call>" in streamed_text
            yield chunk
        await self._open_confirmed(keys, confirmed, raw_request)

    async def _open_confirmed(
        self,
        keys: set[str],
        confirmed: bool,
        raw_request: Any,
    ) -> None:
        if not confirmed:
            return
        session_id = _header(raw_request, _SESSION_HEADER)
        if session_id:
            keys.add(f"session:{session_id}")
        if not keys:
            logger.warning(
                "Tool call confirmed without tool_call_id or %s; "
                "using implicit idle slack",
                _SESSION_HEADER,
            )
            return
        await self._close_keys(keys)
        duration_s = _expected_duration(raw_request, self._default_duration_s)
        try:
            handle = await asyncio.to_thread(
                self._client.begin_tool_call,
                duration_s,
            )
        except Exception:
            logger.exception("Failed to broadcast tool-call write slack")
            return
        with self._lock:
            for key in keys:
                self._handles_by_key[key] = handle

    async def _close_keys(self, keys: set[str]) -> None:
        if not keys:
            return
        with self._lock:
            handles = {
                self._handles_by_key[key]
                for key in keys
                if key in self._handles_by_key
            }
            if handles:
                self._handles_by_key = {
                    key: handle
                    for key, handle in self._handles_by_key.items()
                    if handle not in handles
                }
        for handle in handles:
            try:
                await asyncio.to_thread(self._client.end, handle)
            except Exception:
                # A finite manager window may already have expired. The next
                # request must still proceed; stale end is best-effort.
                logger.warning(
                    "Failed to close expired tool-call write slack",
                    exc_info=True,
                )


_INSTALL_LOCK = threading.Lock()
_INSTALLED = False


def install_vllm_tool_slack_hook() -> bool:
    """Install the chat-completion tool-slack hook once.

    Returns:
        Whether the hook is enabled after this call.
    """
    global _INSTALLED
    enabled = os.getenv("LMCACHE_TOOL_SLACK_HOOK", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not enabled:
        return False
    with _INSTALL_LOCK:
        if _INSTALLED:
            return True
        try:
            # Third Party
            from vllm.entrypoints.openai.chat_completion.serving import (
                OpenAIServingChat,
            )
        except ImportError:
            logger.exception("vLLM chat serving module is unavailable")
            return False

        client = TuttiWriteSlackFanoutClient.from_worker_ports(
            host=os.getenv("LMCACHE_WRITE_SLACK_API_HOST", "127.0.0.1"),
            first_worker_port=int(
                os.getenv("LMCACHE_WRITE_SLACK_FIRST_WORKER_PORT", "7000")
            ),
            worker_count=int(os.getenv("LMCACHE_WRITE_SLACK_WORKER_COUNT", "8")),
            timeout_s=float(os.getenv("LMCACHE_WRITE_SLACK_API_TIMEOUT_SEC", "0.5")),
        )
        tracker = VllmToolSlackTracker(
            client,
            default_duration_s=float(
                os.getenv("LMCACHE_TOOL_SLACK_DEFAULT_SEC", "30")
            ),
        )
        original = OpenAIServingChat.create_chat_completion

        @wraps(original)
        async def create_chat_completion_with_write_slack(
            serving: Any,
            request: Any,
            raw_request: Any = None,
        ) -> Any:
            await tracker.before_request(request, raw_request)
            result = await original(serving, request, raw_request)
            if getattr(request, "stream", False) and hasattr(result, "__aiter__"):
                return tracker.wrap_stream(result, raw_request)
            await tracker.observe_full_response(result, raw_request)
            return result

        OpenAIServingChat.create_chat_completion = (  # type: ignore[method-assign]
            create_chat_completion_with_write_slack
        )
        _INSTALLED = True
        logger.info("Installed vLLM tool-call write-slack hook")
        return True
