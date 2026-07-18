# SPDX-License-Identifier: Apache-2.0
"""NVTX events for the pipelined CSA/HCA prefetch path.

The labels emitted here intentionally use a stable, machine-readable format so
that an Nsight Systems SQLite export can group ranges without depending on
Python function names.  Instrumentation is disabled by default and can be
enabled with ``LMCACHE_CSA_PIPELINE_NVTX=1``.
"""

# Future
from __future__ import annotations

# Standard
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import Iterator, Mapping, Protocol
import os

try:
    # Third Party
    import nvtx
except ImportError:
    nvtx = None  # type: ignore[assignment]


_DOMAIN = "lmcache.csa.pipeline"
_TRUE_VALUES = frozenset(("1", "on", "true", "yes"))


class CsaNvtxEvent(str, Enum):
    """Stable event names used by the CSA/HCA pipeline trace."""

    HCA_LAYER = "hca_layer"
    HCA_ATTENTION = "hca_attention"
    L2_PROXY = "l2_proxy"
    IO_IN_FLIGHT = "io_in_flight"
    IO_SUBMIT = "io_submit"
    IO_DONE = "io_done"
    TARGET_GATE_WAIT = "target_gate_wait"
    TRUE_INDEXER = "true_indexer"


class CsaNvtxBackend(Protocol):
    """Backend contract used to emit explicit NVTX ranges and marks."""

    def start_range(self, message: str, color: str, domain: str) -> object:
        """Start an NVTX range and return its opaque identifier."""

    def end_range(self, range_id: object) -> None:
        """End a range previously returned by :meth:`start_range`."""

    def mark(self, message: str, color: str, domain: str) -> None:
        """Emit an instantaneous NVTX mark."""


class _PythonNvtxBackend:
    def start_range(self, message: str, color: str, domain: str) -> object:
        assert nvtx is not None
        return nvtx.start_range(message=message, color=color, domain=domain)

    def end_range(self, range_id: object) -> None:
        assert nvtx is not None
        nvtx.end_range(range_id)

    def mark(self, message: str, color: str, domain: str) -> None:
        assert nvtx is not None
        nvtx.mark(message=message, color=color, domain=domain)


_EVENT_COLORS = {
    CsaNvtxEvent.HCA_LAYER: "blue",
    CsaNvtxEvent.HCA_ATTENTION: "cyan",
    CsaNvtxEvent.L2_PROXY: "orange",
    CsaNvtxEvent.IO_IN_FLIGHT: "purple",
    CsaNvtxEvent.IO_SUBMIT: "purple",
    CsaNvtxEvent.IO_DONE: "green",
    CsaNvtxEvent.TARGET_GATE_WAIT: "red",
    CsaNvtxEvent.TRUE_INDEXER: "rapids",
}


def _escape_label_value(value: object) -> str:
    return str(value).replace("%", "%25").replace("|", "%7C").replace("=", "%3D")


def _make_label(
    event: CsaNvtxEvent,
    layer_id: int,
    target_layer_id: int | None,
    request_id: str | None,
    operation_id: str | int | None,
    attributes: Mapping[str, object] | None,
) -> str:
    fields: list[tuple[str, object]] = [("event", event.value), ("layer", layer_id)]
    if target_layer_id is not None:
        fields.append(("target", target_layer_id))
    if request_id is not None:
        fields.append(("request", request_id))
    if operation_id is not None:
        fields.append(("op", operation_id))
    if attributes is not None:
        fields.extend(sorted(attributes.items()))
    return "|".join(f"{key}={_escape_label_value(value)}" for key, value in fields)


@dataclass
class CsaNvtxRange:
    """A range handle that can be closed from an asynchronous completion path.

    The handle is idempotent: multiple callers may race to close it, but the
    backend receives exactly one ``end_range`` call.
    """

    _range_id: object
    _backend: CsaNvtxBackend
    _closed: bool = False
    _lock: Lock = field(default_factory=Lock)

    def close(self) -> None:
        """Close this range once; later calls have no effect."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._backend.end_range(self._range_id)


class CsaPipelineNvtx:
    """Emit low-overhead NVTX events for the CSA/HCA prefetch pipeline.

    Args:
        enabled: Whether event emission is enabled. If omitted, the value is
            read from ``LMCACHE_CSA_PIPELINE_NVTX``.
        backend: Optional backend, primarily useful for tests. The installed
            Python ``nvtx`` package is used when omitted.

    Notes:
        Explicit start/end ranges are used so an I/O range may begin on the
        submit thread and finish on a completion thread.
    """

    def __init__(
        self,
        enabled: bool | None = None,
        backend: CsaNvtxBackend | None = None,
    ) -> None:
        requested = (
            os.getenv("LMCACHE_CSA_PIPELINE_NVTX", "0").lower() in _TRUE_VALUES
            if enabled is None
            else enabled
        )
        if backend is None and nvtx is not None:
            backend = _PythonNvtxBackend()
        self.enabled = requested and backend is not None
        self.backend = backend

    def start(
        self,
        event: CsaNvtxEvent,
        *,
        layer_id: int,
        target_layer_id: int | None = None,
        request_id: str | None = None,
        operation_id: str | int | None = None,
        attributes: Mapping[str, object] | None = None,
    ) -> CsaNvtxRange | None:
        """Start an event range.

        Args:
            event: Pipeline event represented by the range.
            layer_id: Source or current transformer layer.
            target_layer_id: Optional predicted/prefetched target layer.
            request_id: Optional request correlation identifier.
            operation_id: Optional I/O or proxy operation identifier.
            attributes: Optional low-cardinality fields appended to the label.

        Returns:
            A closable range handle, or ``None`` when tracing is disabled.
        """

        if not self.enabled or self.backend is None:
            return None
        label = _make_label(
            event,
            layer_id,
            target_layer_id,
            request_id,
            operation_id,
            attributes,
        )
        range_id = self.backend.start_range(
            message=label,
            color=_EVENT_COLORS[event],
            domain=_DOMAIN,
        )
        return CsaNvtxRange(range_id, self.backend)

    def finish(self, handle: CsaNvtxRange | None) -> None:
        """Finish a range returned by :meth:`start`, accepting ``None``."""

        if handle is not None:
            handle.close()

    @contextmanager
    def range(
        self,
        event: CsaNvtxEvent,
        *,
        layer_id: int,
        target_layer_id: int | None = None,
        request_id: str | None = None,
        operation_id: str | int | None = None,
        attributes: Mapping[str, object] | None = None,
    ) -> Iterator[None]:
        """Trace a synchronous pipeline stage as a context manager.

        Args:
            event: Pipeline event represented by the range.
            layer_id: Source or current transformer layer.
            target_layer_id: Optional predicted/prefetched target layer.
            request_id: Optional request correlation identifier.
            operation_id: Optional operation correlation identifier.
            attributes: Optional low-cardinality fields appended to the label.

        Yields:
            Control to the traced code block.
        """

        handle = self.start(
            event,
            layer_id=layer_id,
            target_layer_id=target_layer_id,
            request_id=request_id,
            operation_id=operation_id,
            attributes=attributes,
        )
        try:
            yield
        finally:
            self.finish(handle)

    def mark(
        self,
        event: CsaNvtxEvent,
        *,
        layer_id: int,
        target_layer_id: int | None = None,
        request_id: str | None = None,
        operation_id: str | int | None = None,
        attributes: Mapping[str, object] | None = None,
    ) -> None:
        """Emit an instantaneous pipeline event.

        Args:
            event: Pipeline event represented by the mark.
            layer_id: Source or current transformer layer.
            target_layer_id: Optional predicted/prefetched target layer.
            request_id: Optional request correlation identifier.
            operation_id: Optional operation correlation identifier.
            attributes: Optional low-cardinality fields appended to the label.
        """

        if not self.enabled or self.backend is None:
            return
        label = _make_label(
            event,
            layer_id,
            target_layer_id,
            request_id,
            operation_id,
            attributes,
        )
        self.backend.mark(
            message=label,
            color=_EVENT_COLORS[event],
            domain=_DOMAIN,
        )

    def start_io(
        self,
        *,
        layer_id: int,
        target_layer_id: int,
        operation_id: str | int,
        request_id: str | None = None,
        attributes: Mapping[str, object] | None = None,
    ) -> CsaNvtxRange | None:
        """Mark I/O submission and start its asynchronous in-flight range.

        Args:
            layer_id: Layer that initiated the I/O.
            target_layer_id: Layer that will consume the data.
            operation_id: Identifier shared with the completion event.
            request_id: Optional request correlation identifier.
            attributes: Optional low-cardinality fields appended to the label.

        Returns:
            A handle to pass to :meth:`finish_io`, or ``None`` when disabled.
        """

        self.mark(
            CsaNvtxEvent.IO_SUBMIT,
            layer_id=layer_id,
            target_layer_id=target_layer_id,
            request_id=request_id,
            operation_id=operation_id,
            attributes=attributes,
        )
        return self.start(
            CsaNvtxEvent.IO_IN_FLIGHT,
            layer_id=layer_id,
            target_layer_id=target_layer_id,
            request_id=request_id,
            operation_id=operation_id,
            attributes=attributes,
        )

    def finish_io(
        self,
        handle: CsaNvtxRange | None,
        *,
        layer_id: int,
        target_layer_id: int,
        operation_id: str | int,
        request_id: str | None = None,
        status: str = "ok",
    ) -> None:
        """Finish an in-flight I/O range and emit its completion mark.

        Args:
            handle: Handle returned by :meth:`start_io`.
            layer_id: Layer that initiated the I/O.
            target_layer_id: Layer that consumes the data.
            operation_id: Identifier supplied to :meth:`start_io`.
            request_id: Optional request correlation identifier.
            status: Completion status included in the mark.
        """

        self.finish(handle)
        self.mark(
            CsaNvtxEvent.IO_DONE,
            layer_id=layer_id,
            target_layer_id=target_layer_id,
            request_id=request_id,
            operation_id=operation_id,
            attributes={"status": status},
        )


csa_pipeline_nvtx = CsaPipelineNvtx()
"""Process-wide pipeline tracer used by CSA/HCA integration points."""
