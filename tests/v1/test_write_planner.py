# SPDX-License-Identifier: Apache-2.0
"""Tests for bounded-write admission planning."""

# Third Party
import pytest

# First Party
from lmcache.v1.write_planner import (
    TuttiWritePlanManager,
    WriteAdmissionDecision,
)


def _planner() -> TuttiWritePlanManager:
    return TuttiWritePlanManager(
        write_slack_s=0.05,
        write_max_delay_s=2.0,
        initial_bandwidth_mib_s=100.0,
        deadline_guard_s=0.01,
    )


def _decide(
    planner: TuttiWritePlanManager,
    *,
    now_s: float = 10.0,
    readers: int = 0,
    demand_readers: int = 0,
    last_read_end_s: float = 10.0,
    wait_started_s: float = 10.0,
    wave_mib: int = 1,
) -> WriteAdmissionDecision:
    return planner.decide(
        now_s=now_s,
        readers_waiting=readers,
        demand_readers_waiting=demand_readers,
        last_read_end_s=last_read_end_s,
        writer_wait_started_s=wait_started_s,
        wave_nbytes=wave_mib * 1024**2,
    )


def test_demand_reader_blocks_explicit_slack_and_overdue_write() -> None:
    planner = _planner()
    planner.begin_slack("tool_call", now_s=0.0)

    decision = _decide(
        planner,
        readers=1,
        demand_readers=1,
        wait_started_s=0.0,
    )

    assert not decision.admitted
    assert decision.source == "demand_reader"


def test_speculative_reader_yields_only_after_max_delay() -> None:
    planner = _planner()

    blocked = _decide(planner, readers=1, wait_started_s=8.1)
    admitted = _decide(planner, readers=1, wait_started_s=7.9)

    assert not blocked.admitted
    assert blocked.source == "reader"
    assert admitted.admitted
    assert admitted.source == "max_delay"


def test_idle_gap_admits_without_explicit_signal() -> None:
    decision = _decide(_planner(), last_read_end_s=9.94)

    assert decision.admitted
    assert decision.source == "idle"


def test_compute_without_external_kv_admits_immediately() -> None:
    planner = _planner()
    planner.set_compute_slack("request-1", True)

    decision = _decide(planner)

    assert decision.admitted
    assert decision.source == "compute_no_kv"
    assert planner.snapshot().compute_requests == 1


def test_read_sensitive_forward_blocks_global_tool_slack() -> None:
    planner = _planner()
    planner.begin_slack("tool_call", now_s=0.0)
    planner.set_read_sensitive("request-with-kv", True)

    decision = _decide(planner, wait_started_s=0.0)

    assert not decision.admitted
    assert decision.source == "kv_read_compute"
    assert planner.snapshot().read_sensitive_requests == 1


def test_finite_tool_window_checks_estimated_wave_duration() -> None:
    planner = _planner()
    token = planner.begin_slack("tool_call", 0.2, now_s=10.0)

    fitting = _decide(planner, wave_mib=10)
    too_short = _decide(planner, wave_mib=20)

    assert fitting.admitted
    assert fitting.source == "slack:tool_call"
    assert not too_short.admitted
    assert too_short.source == "slack_too_short"
    assert planner.end_slack(token)
    assert not planner.end_slack(token)


def test_expired_window_falls_back_to_idle_policy() -> None:
    planner = _planner()
    planner.begin_slack("tool_call", 0.1, now_s=1.0)

    decision = _decide(planner, now_s=2.0, last_read_end_s=1.0)

    assert decision.admitted
    assert decision.source == "idle"
    assert planner.snapshot().active_slack_sources == ()


def test_snapshot_prunes_expired_window_without_admission() -> None:
    planner = _planner()
    planner.begin_slack("tool_call", 0.1, now_s=-1.0)

    assert planner.snapshot().active_slack_sources == ()


def test_wave_accounting_and_bandwidth_ewma() -> None:
    planner = _planner()
    planner.wave_queued(20 * 1024**2)
    planner.wave_started()
    planner.wave_finished(20 * 1024**2, duration_s=0.1, success=True)

    snapshot = planner.snapshot()
    assert snapshot.queued_waves == 0
    assert snapshot.active_waves == 0
    assert snapshot.queued_bytes == 0
    assert snapshot.completed_waves == 1
    assert snapshot.failed_waves == 0
    assert snapshot.estimated_bandwidth_mib_s == pytest.approx(120.0)
