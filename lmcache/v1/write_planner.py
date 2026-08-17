# SPDX-License-Identifier: Apache-2.0
"""Admission planning for background Tutti writes.

The planner is deliberately independent from CUDA and NVMe.  It decides
whether one bounded physical write wave may start; the caller re-evaluates the
decision before every wave so demand reads can run between waves.
"""

# Standard
import threading
import time
from dataclasses import dataclass
from typing import Optional


_MIB = 1024**2


@dataclass(frozen=True, slots=True)
class WriteAdmissionDecision:
    """Result of evaluating one physical write wave.

    Attributes:
        admitted: Whether the wave may start now.
        source: Policy condition that produced the decision.
        retry_after_s: Suggested maximum wait before re-evaluating.
        estimated_wave_s: Estimated device time for the proposed wave.
    """

    admitted: bool
    source: str
    retry_after_s: float
    estimated_wave_s: float


@dataclass(frozen=True, slots=True)
class WritePlanSnapshot:
    """Observable state of the write planner."""

    active_slack_sources: tuple[str, ...]
    compute_requests: int
    read_sensitive_requests: int
    queued_waves: int
    active_waves: int
    queued_bytes: int
    completed_waves: int
    failed_waves: int
    estimated_bandwidth_mib_s: float
    background_rate_mib_s: float = 0.0
    background_burst_mib: float = 0.0
    background_tokens_mib: float = 0.0
    decode_requests: int = 0
    decode_rate_mib_s: float = 0.0
    decode_tokens_mib: float = 0.0


@dataclass(frozen=True, slots=True)
class _SlackWindow:
    source: str
    deadline_s: Optional[float]


class TuttiWritePlanManager:
    """Plan bounded background writes around demand reads and known slack.

    Explicit slack can be supplied by an agent runtime for a tool call.  The
    vLLM adapter also marks model-compute batches that need no external KV
    read.  Without either signal, the planner retains the existing idle-gap
    and maximum-delay behavior.

    Args:
        write_slack_s: Required queue-idle time for implicit admission.
        write_max_delay_s: Maximum wait before speculative readers yield.
        initial_bandwidth_mib_s: Initial write-throughput estimate.
        deadline_guard_s: Safety margin before a finite slack window ends.
        background_rate_mib_s: Write rate outside explicit slack. Zero keeps
            the legacy unlimited behavior.
        background_burst_mib: Maximum byte credit accumulated by the
            background rate limiter. It must be positive when a rate is set.
        unlimited_decode_writes: Whether decode compute admits writes at full
            speed. External KV reads retain the configured safe-rate priority.
        decode_rate_mib_s: Optional bounded write rate during decode. Zero
            keeps decode paused unless ``unlimited_decode_writes`` is enabled.

    Raises:
        ValueError: If any duration is negative or bandwidth is not positive.
    """

    def __init__(
        self,
        write_slack_s: float,
        write_max_delay_s: float,
        initial_bandwidth_mib_s: float = 4096.0,
        deadline_guard_s: float = 0.01,
        background_rate_mib_s: float = 0.0,
        background_burst_mib: float = 8.0,
        unlimited_decode_writes: bool = False,
        decode_rate_mib_s: float = 0.0,
    ) -> None:
        if write_slack_s < 0 or write_max_delay_s < 0 or deadline_guard_s < 0:
            raise ValueError("write planner durations must be non-negative")
        if initial_bandwidth_mib_s <= 0:
            raise ValueError("initial_bandwidth_mib_s must be positive")
        if background_rate_mib_s < 0:
            raise ValueError("background_rate_mib_s must be non-negative")
        if decode_rate_mib_s < 0:
            raise ValueError("decode_rate_mib_s must be non-negative")
        if background_rate_mib_s > 0 and background_burst_mib <= 0:
            raise ValueError(
                "background_burst_mib must be positive when rate limiting is enabled"
            )
        self._write_slack_s = write_slack_s
        self._write_max_delay_s = write_max_delay_s
        self._deadline_guard_s = deadline_guard_s
        self._bandwidth_mib_s = initial_bandwidth_mib_s
        self._background_rate_mib_s = background_rate_mib_s
        self._background_burst_bytes = max(0.0, background_burst_mib * _MIB)
        self._background_tokens_bytes = self._background_burst_bytes
        self._background_last_refill_s: Optional[float] = None
        self._unlimited_decode_writes = unlimited_decode_writes
        self._decode_rate_mib_s = decode_rate_mib_s
        self._decode_tokens_bytes = self._background_burst_bytes
        self._decode_last_refill_s: Optional[float] = None
        self._lock = threading.Lock()
        self._next_token = 1
        self._slack_windows: dict[int, _SlackWindow] = {}
        self._compute_requests: set[str] = set()
        self._read_sensitive_requests: set[str] = set()
        self._decode_request_deadlines: dict[str, float] = {}
        self._queued_waves = 0
        self._active_waves = 0
        self._queued_bytes = 0
        self._completed_waves = 0
        self._failed_waves = 0

    def begin_slack(
        self,
        source: str,
        expected_duration_s: Optional[float] = None,
        *,
        now_s: Optional[float] = None,
    ) -> int:
        """Open an explicit slack window and return its close token.

        Args:
            source: Human-readable source such as ``"tool_call"``.
            expected_duration_s: Optional expected remaining duration.
            now_s: Optional monotonic timestamp used by deterministic tests.

        Returns:
            An opaque token accepted by :meth:`end_slack`.

        Raises:
            ValueError: If the source is empty or duration is not positive.
        """
        normalized_source = source.strip()
        if not normalized_source:
            raise ValueError("slack source must be non-empty")
        if expected_duration_s is not None and expected_duration_s <= 0:
            raise ValueError("expected_duration_s must be positive")
        now = time.perf_counter() if now_s is None else now_s
        deadline = None if expected_duration_s is None else now + expected_duration_s
        with self._lock:
            token = self._next_token
            self._next_token += 1
            self._slack_windows[token] = _SlackWindow(normalized_source, deadline)
        return token

    def end_slack(self, token: int) -> bool:
        """Close an explicit slack window.

        Args:
            token: Token returned by :meth:`begin_slack`.

        Returns:
            ``True`` when an active window was removed.
        """
        with self._lock:
            return self._slack_windows.pop(token, None) is not None

    def configure_background_limit(
        self,
        rate_mib_s: float,
        burst_mib: float,
        *,
        now_s: Optional[float] = None,
    ) -> None:
        """Set the non-slack write limit without restarting the worker.

        Args:
            rate_mib_s: Long-run per-worker rate. Zero disables limiting.
            burst_mib: Maximum background byte credit.
            now_s: Optional monotonic timestamp used by deterministic tests.

        Raises:
            ValueError: If the rate is negative or the burst is not positive.

        Notes:
            Reconfiguration resets available credit to one full burst so a
            previous experiment cannot leak token debt into the next arm.
        """
        if rate_mib_s < 0:
            raise ValueError("rate_mib_s must be non-negative")
        if burst_mib <= 0:
            raise ValueError("burst_mib must be positive")
        now = time.perf_counter() if now_s is None else now_s
        with self._lock:
            self._background_rate_mib_s = rate_mib_s
            self._background_burst_bytes = burst_mib * _MIB
            self._background_tokens_bytes = self._background_burst_bytes
            self._background_last_refill_s = now

    def configure_decode_limit(
        self,
        rate_mib_s: float,
        *,
        now_s: Optional[float] = None,
    ) -> None:
        """Set the decode-phase write limit and reset its burst credit.

        Args:
            rate_mib_s: Per-rank decode write rate. Zero pauses decode writes.
            now_s: Optional monotonic timestamp for deterministic callers.

        Raises:
            ValueError: If the rate is negative.
        """
        if rate_mib_s < 0:
            raise ValueError("rate_mib_s must be non-negative")
        now = time.perf_counter() if now_s is None else now_s
        with self._lock:
            self._decode_rate_mib_s = rate_mib_s
            self._decode_tokens_bytes = self._background_burst_bytes
            self._decode_last_refill_s = now

    def set_compute_slack(self, request_id: str, active: bool) -> bool:
        """Mark whether a request is in compute with no external KV read.

        Args:
            request_id: Serving request identifier.
            active: Whether its compute slack is currently active.

        Returns:
            ``True`` when the active-request set changed.
        """
        with self._lock:
            if active:
                changed = str(request_id) not in self._compute_requests
                self._compute_requests.add(str(request_id))
            else:
                changed = str(request_id) in self._compute_requests
                self._compute_requests.discard(str(request_id))
            return changed

    def set_read_sensitive(self, request_id: str, active: bool) -> bool:
        """Mark a forward that may issue additional external KV reads.

        Args:
            request_id: Serving request identifier.
            active: Whether its read-sensitive forward is active.

        Returns:
            ``True`` when the active-request set changed.
        """
        with self._lock:
            if active:
                changed = str(request_id) not in self._read_sensitive_requests
                self._read_sensitive_requests.add(str(request_id))
            else:
                changed = str(request_id) in self._read_sensitive_requests
                self._read_sensitive_requests.discard(str(request_id))
            return changed

    def set_decode_active(
        self,
        request_id: str,
        active: bool,
        guard_s: float,
        *,
        now_s: Optional[float] = None,
    ) -> bool:
        """Block background writes while a request is decoding.

        The marker expires after ``guard_s`` unless the adapter refreshes it
        on a later decode step. This prevents an aborted or completed request
        from leaving writes paused indefinitely. Explicit tool-call slack may
        override this marker, but external-KV read sensitivity remains strict.

        Args:
            request_id: Serving request identifier.
            active: Whether the request is currently decoding.
            guard_s: Time to retain an active marker without a refresh.
            now_s: Optional monotonic timestamp used by deterministic callers.

        Returns:
            ``True`` when the active marker set changed.

        Raises:
            ValueError: If ``guard_s`` is not positive.
        """
        if guard_s <= 0:
            raise ValueError("decode write guard must be positive")
        normalized_id = str(request_id)
        now = time.perf_counter() if now_s is None else now_s
        with self._lock:
            self._discard_expired_decode_locked(now)
            was_active = normalized_id in self._decode_request_deadlines
            if active:
                self._decode_request_deadlines[normalized_id] = now + guard_s
            else:
                self._decode_request_deadlines.pop(normalized_id, None)
            return was_active != active

    def decide(
        self,
        *,
        now_s: float,
        readers_waiting: int,
        demand_readers_waiting: int,
        last_read_end_s: float,
        writer_wait_started_s: float,
        wave_nbytes: int,
    ) -> WriteAdmissionDecision:
        """Decide whether one physical write wave may start.

        Demand readers always have strict priority.  Speculative readers may
        yield only after the write maximum-delay threshold.

        Args:
            now_s: Current monotonic timestamp.
            readers_waiting: All announced readers.
            demand_readers_waiting: Synchronous demand readers.
            last_read_end_s: Timestamp of the last read or slack boundary.
            writer_wait_started_s: Timestamp when this wave began waiting.
            wave_nbytes: Physical byte count of the wave.

        Returns:
            A complete admission decision.

        Raises:
            ValueError: If counters or byte count are negative.
        """
        if readers_waiting < 0 or demand_readers_waiting < 0 or wave_nbytes < 0:
            raise ValueError("reader counts and wave_nbytes must be non-negative")
        waited_s = max(0.0, now_s - writer_wait_started_s)
        idle_for_s = max(0.0, now_s - last_read_end_s)
        with self._lock:
            estimated_wave_s = (wave_nbytes / _MIB) / self._bandwidth_mib_s
            self._discard_expired_slack_locked(now_s)
            self._discard_expired_decode_locked(now_s)

            # Paused decode always wins, including the short distributed transition
            # where a tool-slack token may not have been closed on every rank
            # yet. The tool gap remains full speed, but it cannot authorize a
            # new wave after model execution has resumed.
            if (
                self._decode_request_deadlines
                and not self._unlimited_decode_writes
                and self._decode_rate_mib_s <= 0
            ):
                nearest_expiry = min(self._decode_request_deadlines.values())
                return self._blocked(
                    "decode_compute",
                    estimated_wave_s,
                    max(0.001, nearest_expiry - now_s),
                )
            # A configured background limit is also the safe overlap budget
            # during external-KV reads.  Keep the legacy strict-priority
            # behavior when no limit is configured: rate=0 historically means
            # "unlimited" outside reads and must never make read contention
            # unlimited by accident.  These checks intentionally precede
            # explicit slack, so one session's tool window cannot override a
            # concurrent session's demand read.
            if demand_readers_waiting > 0:
                return self._read_overlap_admission_locked(
                    now_s,
                    wave_nbytes,
                    "demand_reader",
                    estimated_wave_s,
                )
            if self._read_sensitive_requests:
                return self._read_overlap_admission_locked(
                    now_s,
                    wave_nbytes,
                    "kv_read_compute",
                    estimated_wave_s,
                )
            # In the experimental full-speed decode mode, external reads above
            # still retain the configured safe overlap rate.  With no such
            # read active, decode is an explicit drain window and bypasses the
            # background token bucket just like a tool-call window.
            if self._decode_request_deadlines:
                if self._unlimited_decode_writes:
                    return self._admitted("decode_compute", estimated_wave_s)
                return self._decode_admission_locked(
                    now_s,
                    wave_nbytes,
                    estimated_wave_s,
                )
            if readers_waiting > 0 and waited_s < self._write_max_delay_s:
                remaining = self._write_max_delay_s - waited_s
                return self._blocked("reader", estimated_wave_s, remaining)

            if self._slack_windows:
                required_s = estimated_wave_s + self._deadline_guard_s
                fitting_sources = [
                    window.source
                    for window in self._slack_windows.values()
                    if window.deadline_s is None
                    or window.deadline_s - now_s >= required_s
                ]
                if fitting_sources:
                    return self._admitted(
                        f"slack:{sorted(fitting_sources)[0]}", estimated_wave_s
                    )
                shortest_expiry = min(
                    window.deadline_s
                    for window in self._slack_windows.values()
                    if window.deadline_s is not None
                )
                return self._blocked(
                    "slack_too_short",
                    estimated_wave_s,
                    max(0.001, shortest_expiry - now_s),
                )

            admission_source: Optional[str] = None
            if self._compute_requests:
                admission_source = "compute_no_kv"
            if idle_for_s >= self._write_slack_s:
                admission_source = admission_source or "idle"
            elif waited_s >= self._write_max_delay_s:
                admission_source = admission_source or "max_delay"
            if admission_source is not None:
                return self._background_admission_locked(
                    now_s,
                    wave_nbytes,
                    admission_source,
                    estimated_wave_s,
                )
            retry_after_s = min(
                self._write_slack_s - idle_for_s,
                self._write_max_delay_s - waited_s,
            )
            return self._blocked("waiting", estimated_wave_s, retry_after_s)

    def wave_queued(self, nbytes: int) -> None:
        """Record a wave entering the admission queue.

        Args:
            nbytes: Physical payload bytes queued.
        """
        if nbytes < 0:
            raise ValueError("nbytes must be non-negative")
        with self._lock:
            self._queued_waves += 1
            self._queued_bytes += nbytes

    def wave_started(self) -> None:
        """Record an admitted wave beginning device work."""
        with self._lock:
            self._active_waves += 1

    def wave_finished(self, nbytes: int, duration_s: float, success: bool) -> None:
        """Record wave completion and update the throughput estimate.

        Args:
            nbytes: Physical payload bytes completed.
            duration_s: Time spent in the device-write critical section.
            success: Whether the wave completed successfully.
        """
        if nbytes < 0 or duration_s < 0:
            raise ValueError("nbytes and duration_s must be non-negative")
        with self._lock:
            self._queued_waves = max(0, self._queued_waves - 1)
            self._active_waves = max(0, self._active_waves - 1)
            self._queued_bytes = max(0, self._queued_bytes - nbytes)
            if success:
                self._completed_waves += 1
                if nbytes > 0 and duration_s > 0:
                    measured = (nbytes / _MIB) / duration_s
                    self._bandwidth_mib_s = 0.8 * self._bandwidth_mib_s + 0.2 * measured
            else:
                self._failed_waves += 1

    def snapshot(self) -> WritePlanSnapshot:
        """Return a thread-safe observability snapshot.

        Returns:
            The current queue, slack, completion, and bandwidth state.
        """
        now_s = time.perf_counter()
        with self._lock:
            self._discard_expired_slack_locked(now_s)
            self._discard_expired_decode_locked(now_s)
            return WritePlanSnapshot(
                active_slack_sources=tuple(
                    sorted(window.source for window in self._slack_windows.values())
                ),
                compute_requests=len(self._compute_requests),
                read_sensitive_requests=len(self._read_sensitive_requests),
                decode_requests=len(self._decode_request_deadlines),
                decode_rate_mib_s=self._decode_rate_mib_s,
                decode_tokens_mib=self._decode_tokens_bytes / _MIB,
                queued_waves=self._queued_waves,
                active_waves=self._active_waves,
                queued_bytes=self._queued_bytes,
                completed_waves=self._completed_waves,
                failed_waves=self._failed_waves,
                estimated_bandwidth_mib_s=self._bandwidth_mib_s,
                background_rate_mib_s=self._background_rate_mib_s,
                background_burst_mib=self._background_burst_bytes / _MIB,
                background_tokens_mib=self._background_tokens_bytes / _MIB,
            )

    def _discard_expired_decode_locked(self, now_s: float) -> None:
        """Remove decode markers whose refresh guard has elapsed."""
        expired = [
            request_id
            for request_id, deadline_s in self._decode_request_deadlines.items()
            if deadline_s <= now_s
        ]
        for request_id in expired:
            self._decode_request_deadlines.pop(request_id, None)

    def _background_admission_locked(
        self,
        now_s: float,
        wave_nbytes: int,
        source: str,
        estimated_wave_s: float,
    ) -> WriteAdmissionDecision:
        """Apply the non-slack token bucket to an otherwise admissible wave."""
        if self._background_rate_mib_s <= 0:
            return self._admitted(source, estimated_wave_s)
        if self._background_last_refill_s is None:
            self._background_last_refill_s = now_s
        else:
            elapsed_s = max(0.0, now_s - self._background_last_refill_s)
            refill_bytes = elapsed_s * self._background_rate_mib_s * _MIB
            self._background_tokens_bytes = min(
                self._background_burst_bytes,
                self._background_tokens_bytes + refill_bytes,
            )
            self._background_last_refill_s = max(
                self._background_last_refill_s,
                now_s,
            )
        required_bytes = min(float(wave_nbytes), self._background_burst_bytes)
        if self._background_tokens_bytes >= required_bytes:
            # A wave may exceed the burst because physical writes are atomic at
            # this boundary. The resulting token debt delays later waves and
            # preserves the configured long-run rate without splitting I/O.
            self._background_tokens_bytes -= wave_nbytes
            return self._admitted(f"rate:{source}", estimated_wave_s)
        deficit_bytes = required_bytes - self._background_tokens_bytes
        retry_after_s = deficit_bytes / (self._background_rate_mib_s * _MIB)
        return self._blocked("background_rate", estimated_wave_s, retry_after_s)

    def _read_overlap_admission_locked(
        self,
        now_s: float,
        wave_nbytes: int,
        source: str,
        estimated_wave_s: float,
    ) -> WriteAdmissionDecision:
        """Apply the configured safe rate while an external read is active."""
        if self._background_rate_mib_s <= 0:
            return self._blocked(source, estimated_wave_s, 0.001)
        return self._background_admission_locked(
            now_s,
            wave_nbytes,
            f"read:{source}",
            estimated_wave_s,
        )

    def _decode_admission_locked(
        self,
        now_s: float,
        wave_nbytes: int,
        estimated_wave_s: float,
    ) -> WriteAdmissionDecision:
        """Apply the independent decode-phase rate limiter."""
        if self._decode_last_refill_s is None:
            self._decode_last_refill_s = now_s
        else:
            elapsed_s = max(0.0, now_s - self._decode_last_refill_s)
            self._decode_tokens_bytes = min(
                self._background_burst_bytes,
                self._decode_tokens_bytes + elapsed_s * self._decode_rate_mib_s * _MIB,
            )
            self._decode_last_refill_s = max(self._decode_last_refill_s, now_s)
        required_bytes = min(float(wave_nbytes), self._background_burst_bytes)
        if self._decode_tokens_bytes >= required_bytes:
            self._decode_tokens_bytes -= wave_nbytes
            return self._admitted("rate:decode_compute", estimated_wave_s)
        deficit_bytes = required_bytes - self._decode_tokens_bytes
        return self._blocked(
            "decode_rate",
            estimated_wave_s,
            deficit_bytes / (self._decode_rate_mib_s * _MIB),
        )

    def _discard_expired_slack_locked(self, now_s: float) -> None:
        expired = [
            token
            for token, window in self._slack_windows.items()
            if window.deadline_s is not None and window.deadline_s <= now_s
        ]
        for token in expired:
            del self._slack_windows[token]

    @staticmethod
    def _admitted(source: str, estimated_wave_s: float) -> WriteAdmissionDecision:
        return WriteAdmissionDecision(True, source, 0.0, estimated_wave_s)

    @staticmethod
    def _blocked(
        source: str,
        estimated_wave_s: float,
        retry_after_s: float,
    ) -> WriteAdmissionDecision:
        return WriteAdmissionDecision(
            False,
            source,
            max(0.001, retry_after_s),
            estimated_wave_s,
        )
