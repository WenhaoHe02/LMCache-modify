# SPDX-License-Identifier: Apache-2.0
"""Tests for the Tutti write-slack control-plane API."""

# Standard
from unittest.mock import MagicMock

# Third Party
from fastapi.testclient import TestClient
import pytest

# First Party
from lmcache.v1.internal_api_server.api_server import app
from lmcache.v1.write_planner import WritePlanSnapshot


@pytest.fixture
def mock_engine() -> MagicMock:
    engine = MagicMock()
    engine.begin_tutti_write_slack.return_value = 17
    engine.end_tutti_write_slack.return_value = True
    engine.get_tutti_write_plan_snapshot.return_value = WritePlanSnapshot(
        active_slack_sources=("tool_call",),
        compute_requests=0,
        read_sensitive_requests=0,
        queued_waves=2,
        active_waves=1,
        queued_bytes=4096,
        completed_waves=3,
        failed_waves=0,
        estimated_bandwidth_mib_s=1024.0,
    )
    return engine


@pytest.fixture
def write_slack_client(mock_engine: MagicMock):
    adapter = MagicMock()
    adapter.lmcache_engine = mock_engine
    app.state.lmcache_adapter = adapter
    with TestClient(app) as client:
        yield client


def test_begin_write_slack_returns_worker_token(
    write_slack_client: TestClient,
    mock_engine: MagicMock,
) -> None:
    response = write_slack_client.post(
        "/write_slack/begin",
        json={"source": "tool_call", "expected_duration_s": 2.5},
    )

    assert response.status_code == 200
    assert response.json()["token"] == 17
    mock_engine.begin_tutti_write_slack.assert_called_once_with("tool_call", 2.5)


def test_end_write_slack_closes_worker_token(
    write_slack_client: TestClient,
    mock_engine: MagicMock,
) -> None:
    response = write_slack_client.post("/write_slack/end", json={"token": 17})

    assert response.status_code == 200
    assert response.json()["removed"] is True
    mock_engine.end_tutti_write_slack.assert_called_once_with(17)


def test_end_expired_write_slack_is_idempotent(
    write_slack_client: TestClient,
    mock_engine: MagicMock,
) -> None:
    mock_engine.end_tutti_write_slack.return_value = False

    response = write_slack_client.post("/write_slack/end", json={"token": 17})

    assert response.status_code == 200
    assert response.json() == {
        "status": "not_found",
        "token": 17,
        "removed": False,
    }


def test_status_exposes_read_sensitive_state(
    write_slack_client: TestClient,
) -> None:
    response = write_slack_client.get("/write_slack/status")

    assert response.status_code == 200
    assert response.json()["active_slack_sources"] == ["tool_call"]
    assert response.json()["read_sensitive_requests"] == 0
    assert response.json()["queued_waves"] == 2


def test_begin_returns_unavailable_without_worker_engine() -> None:
    adapter = MagicMock()
    adapter.lmcache_engine = None
    app.state.lmcache_adapter = adapter
    with TestClient(app) as client:
        response = client.post("/write_slack/begin", json={})

    assert response.status_code == 503
