# SPDX-License-Identifier: Apache-2.0
"""Agent-frontend client for broadcasting Tutti write slack to TP workers."""

# Standard
import json
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True, slots=True)
class WorkerWriteSlackToken:
    """One worker endpoint and its local write-slack token."""

    endpoint: str
    token: int


@dataclass(frozen=True, slots=True)
class WriteSlackHandle:
    """Tokens for one slack window broadcast to all TP workers."""

    source: str
    worker_tokens: tuple[WorkerWriteSlackToken, ...]


class TuttiWriteSlackFanoutClient:
    """Broadcast tool-call slack lifecycle events to Tutti workers.

    Args:
        worker_endpoints: Internal API roots, one per TP worker.
        timeout_s: Per-worker control request timeout.

    Raises:
        ValueError: If no worker endpoints are supplied or timeout is invalid.
    """

    def __init__(self, worker_endpoints: list[str], timeout_s: float = 0.5) -> None:
        endpoints = [endpoint.rstrip("/") for endpoint in worker_endpoints]
        if not endpoints or any(not endpoint for endpoint in endpoints):
            raise ValueError("worker_endpoints must contain non-empty URLs")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self._worker_endpoints = tuple(endpoints)
        self._timeout_s = timeout_s

    @classmethod
    def from_worker_ports(
        cls,
        *,
        host: str = "127.0.0.1",
        first_worker_port: int = 7000,
        worker_count: int = 8,
        timeout_s: float = 0.5,
    ) -> "TuttiWriteSlackFanoutClient":
        """Build a client for consecutive LMCache worker API ports.

        Args:
            host: Internal API host reachable from the agent frontend.
            first_worker_port: Port belonging to TP worker zero.
            worker_count: Number of TP workers.
            timeout_s: Per-worker control request timeout.

        Returns:
            A configured fan-out client.

        Raises:
            ValueError: If the worker count or first port is invalid.
        """
        if worker_count <= 0 or first_worker_port <= 0:
            raise ValueError("worker_count and first_worker_port must be positive")
        return cls(
            [
                f"http://{host}:{first_worker_port + worker_id}"
                for worker_id in range(worker_count)
            ],
            timeout_s=timeout_s,
        )

    def begin(
        self,
        source: str = "tool_call",
        expected_duration_s: Optional[float] = None,
    ) -> WriteSlackHandle:
        """Open a write-slack window on every worker.

        Partial fan-out is rolled back before an exception is raised.

        Args:
            source: Slack source label.
            expected_duration_s: Optional expected remaining duration.

        Returns:
            Handle containing every worker-local close token.

        Raises:
            RuntimeError: If any worker rejects or times out during begin.
        """
        payload: dict[str, Any] = {"source": source}
        if expected_duration_s is not None:
            payload["expected_duration_s"] = expected_duration_s
        tokens: list[WorkerWriteSlackToken] = []
        errors: list[str] = []

        def open_worker(endpoint: str) -> WorkerWriteSlackToken:
            result = self._request_json(endpoint, "/write_slack/begin", payload)
            return WorkerWriteSlackToken(endpoint, int(result["token"]))

        with ThreadPoolExecutor(max_workers=len(self._worker_endpoints)) as executor:
            futures = {
                endpoint: executor.submit(open_worker, endpoint)
                for endpoint in self._worker_endpoints
            }
            for endpoint, future in futures.items():
                try:
                    tokens.append(future.result())
                except Exception as exc:
                    errors.append(f"{endpoint}: {exc}")
        if errors:
            self._close_tokens(tokens, suppress_errors=True)
            raise RuntimeError("write-slack begin failed: " + "; ".join(errors))
        return WriteSlackHandle(
            source=source,
            worker_tokens=tuple(
                sorted(tokens, key=lambda worker_token: worker_token.endpoint)
            ),
        )

    def begin_tool_call(
        self,
        expected_duration_s: Optional[float] = None,
    ) -> WriteSlackHandle:
        """Open tool-call slack on every worker.

        Args:
            expected_duration_s: Optional expected tool execution duration.

        Returns:
            Handle to close immediately before the continuation request.
        """
        return self.begin("tool_call", expected_duration_s)

    def end(self, handle: WriteSlackHandle) -> None:
        """Close a previously broadcast write-slack window.

        Args:
            handle: Handle returned by :meth:`begin`.

        Raises:
            RuntimeError: If any worker fails to close its token.
        """
        errors = self._close_tokens(list(handle.worker_tokens), suppress_errors=False)
        if errors:
            raise RuntimeError("write-slack end failed: " + "; ".join(errors))

    def configure_background_limit(
        self,
        rate_mib_s: float,
        burst_mib: float,
    ) -> tuple[dict[str, Any], ...]:
        """Set the live non-tool write limit on every worker.

        Args:
            rate_mib_s: Long-run per-worker background rate. Zero disables it.
            burst_mib: Maximum per-worker background burst.

        Returns:
            Updated worker snapshots in endpoint order.

        Raises:
            RuntimeError: If any worker rejects or times out.
        """
        payload = {"rate_mib_s": rate_mib_s, "burst_mib": burst_mib}

        def configure_worker(endpoint: str) -> dict[str, Any]:
            return self._request_json(
                endpoint,
                "/write_slack/background_limit",
                payload,
            )

        results: dict[str, dict[str, Any]] = {}
        errors: list[str] = []
        with ThreadPoolExecutor(max_workers=len(self._worker_endpoints)) as executor:
            futures = {
                endpoint: executor.submit(configure_worker, endpoint)
                for endpoint in self._worker_endpoints
            }
            for endpoint, future in futures.items():
                try:
                    results[endpoint] = future.result()
                except Exception as exc:
                    errors.append(f"{endpoint}: {exc}")
        if errors:
            raise RuntimeError(
                "background write-limit update failed: " + "; ".join(errors)
            )
        return tuple(results[endpoint] for endpoint in sorted(results))

    def _close_tokens(
        self,
        tokens: list[WorkerWriteSlackToken],
        *,
        suppress_errors: bool,
    ) -> list[str]:
        if not tokens:
            return []

        def close_worker(worker_token: WorkerWriteSlackToken) -> None:
            self._request_json(
                worker_token.endpoint,
                "/write_slack/end",
                {"token": worker_token.token},
            )

        errors: list[str] = []
        with ThreadPoolExecutor(max_workers=len(tokens)) as executor:
            futures = {
                token: executor.submit(close_worker, token) for token in tokens
            }
            for token, future in futures.items():
                try:
                    future.result()
                except Exception as exc:
                    errors.append(f"{token.endpoint}: {exc}")
        return [] if suppress_errors else errors

    def _request_json(
        self,
        endpoint: str,
        path: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            endpoint + path,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
                result = json.loads(response.read().decode("utf-8"))
        except (OSError, ValueError, urllib.error.HTTPError) as exc:
            raise RuntimeError(str(exc)) from exc
        if not isinstance(result, dict):
            raise RuntimeError("write-slack endpoint returned a non-object response")
        return result
