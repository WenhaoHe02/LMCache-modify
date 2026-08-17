# SPDX-License-Identifier: Apache-2.0
"""Internal control-plane endpoints for Tutti write slack."""

# Standard
from dataclasses import asdict
from typing import Optional

# Third Party
from fastapi import APIRouter
from pydantic import BaseModel, Field
from starlette.requests import Request
from starlette.responses import JSONResponse

router = APIRouter()


class BeginWriteSlackPayload(BaseModel):
    """Payload for opening a named source of write slack."""

    source: str = Field(default="tool_call", min_length=1)
    expected_duration_s: Optional[float] = Field(default=None, gt=0)


class EndWriteSlackPayload(BaseModel):
    """Payload for closing one worker-local write-slack token."""

    token: int = Field(gt=0)


class BackgroundWriteLimitPayload(BaseModel):
    """Live per-rank rate and burst for writes outside explicit slack."""

    rate_mib_s: float = Field(ge=0)
    burst_mib: float = Field(gt=0)


class DecodeWriteLimitPayload(BaseModel):
    """Live per-rank write rate during decode."""

    rate_mib_s: float = Field(ge=0)


class ReuseWriteStatusPayload(BaseModel):
    """Request whose exact SSD generation may gate its next session turn."""

    request_id: str = Field(min_length=1)


def _get_engine(request: Request):
    adapter = request.app.state.lmcache_adapter
    engine = getattr(adapter, "lmcache_engine", None)
    if engine is None:
        return None, JSONResponse(
            status_code=503,
            content={
                "error": "write_slack API is unavailable",
                "message": "LMCache engine is not configured on this process.",
            },
        )
    return engine, None


@router.post("/write_slack/begin")
async def begin_write_slack(
    payload: BeginWriteSlackPayload,
    request: Request,
) -> JSONResponse:
    """Open one worker-local write-slack window.

    Args:
        payload: Slack source and optional expected duration.
        request: FastAPI request carrying the LMCache manager.

    Returns:
        JSON containing the worker-local close token.
    """
    engine, error = _get_engine(request)
    if error is not None:
        return error
    try:
        token = engine.begin_tutti_write_slack(
            payload.source,
            payload.expected_duration_s,
        )
        if token is None:
            return JSONResponse(
                status_code=503,
                content={
                    "error": "Tutti write planner is unavailable",
                    "message": "Tutti loader is not active on this worker.",
                },
            )
        return JSONResponse(
            content={
                "status": "success",
                "source": payload.source,
                "token": token,
            }
        )
    except (TypeError, ValueError) as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})


@router.post("/write_slack/end")
async def end_write_slack(
    payload: EndWriteSlackPayload,
    request: Request,
) -> JSONResponse:
    """Close one worker-local write-slack window.

    Args:
        payload: Worker-local token returned by the begin endpoint.
        request: FastAPI request carrying the LMCache manager.

    Returns:
        JSON indicating whether the token was active.
    """
    engine, error = _get_engine(request)
    if error is not None:
        return error
    removed = bool(engine.end_tutti_write_slack(payload.token))
    return JSONResponse(
        content={
            "status": "success" if removed else "not_found",
            "token": payload.token,
            "removed": removed,
        },
    )


@router.get("/write_slack/status")
async def get_write_slack_status(request: Request) -> JSONResponse:
    """Return one worker's write-planner status.

    Args:
        request: FastAPI request carrying the LMCache manager.

    Returns:
        JSON containing queue, slack, completion, and bandwidth state.
    """
    engine, error = _get_engine(request)
    if error is not None:
        return error
    snapshot = engine.get_tutti_write_plan_snapshot()
    if snapshot is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Tutti write planner is unavailable"},
        )
    return JSONResponse(content={"status": "success", **asdict(snapshot)})


@router.post("/write_slack/background_limit")
async def configure_background_write_limit(
    payload: BackgroundWriteLimitPayload,
    request: Request,
) -> JSONResponse:
    """Update one worker's write limit outside explicit slack.

    Args:
        payload: Per-rank rate and burst in MiB units.
        request: FastAPI request carrying the LMCache manager.

    Returns:
        JSON containing the updated planner state.
    """
    engine, error = _get_engine(request)
    if error is not None:
        return error
    snapshot = engine.configure_tutti_background_write_limit(
        payload.rate_mib_s,
        payload.burst_mib,
    )
    if snapshot is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Tutti write planner is unavailable"},
        )
    return JSONResponse(content={"status": "success", **asdict(snapshot)})


@router.post("/write_slack/decode_limit")
async def configure_decode_write_limit(
    payload: DecodeWriteLimitPayload,
    request: Request,
) -> JSONResponse:
    """Update one worker's decode-phase write rate."""
    engine, error = _get_engine(request)
    if error is not None:
        return error
    snapshot = engine.configure_tutti_decode_write_limit(payload.rate_mib_s)
    if snapshot is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Tutti write planner is unavailable"},
        )
    return JSONResponse(content={"status": "success", **asdict(snapshot)})


@router.post("/write_slack/reuse_ready")
async def get_reuse_write_status(
    payload: ReuseWriteStatusPayload,
    request: Request,
) -> JSONResponse:
    """Return request-scoped SSD publication readiness.

    Args:
        payload: Full serving request ID or OpenAI response-ID prefix.
        request: FastAPI request carrying the LMCache manager.

    Returns:
        JSON status for the exact generation written by that request.
    """
    engine, error = _get_engine(request)
    if error is not None:
        return error
    try:
        status = engine.get_tutti_reuse_write_status(payload.request_id)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    return JSONResponse(content={"status": "success", **status})
