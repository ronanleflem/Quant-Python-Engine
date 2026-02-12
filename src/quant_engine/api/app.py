"""In-memory orchestration helpers and their FastAPI wrappers.

The synchronous helpers keep the test suite light-weight while the
FastAPI application exposes the same capabilities over HTTP for local
development.
"""
from __future__ import annotations

import json
import os
import time
import threading
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from sqlalchemy import create_engine, text

from ..core import spec as spec_module
from ..core.spec import Spec
from ..levels.runner import run_levels_build, run_levels_fill
from ..levels.repo import select_levels as repo_select_levels
from ..levels.schemas import LevelsBuildSpec
from ..optimize.runner import run as run_optimisation
from ..io import ids
from ..persistence import db
from ..stats import runner as stats_runner
from ..stats import conditions as stats_conditions
from ..stats.estimators import freq_with_wilson
from ..seasonality import runner as seasonality_runner
from ..seasonality.optimize import run_optimization as seasonality_run_optimization
from ..filters import list_filter_types
from . import schemas
from .run_request_input import validate_run_request_input
from .validation_errors import (
    ApiValidationException,
    normalize_fastapi_errors,
    normalize_pydantic_errors,
    single_validation_error,
)
from .metrics import METRICS

JOB_TYPE_OPTIMIZATION = "optimization"
JOB_TYPE_STATS = "stats"
JOB_TYPE_LEVELS_BUILD = "levels_build"
JOB_TYPE_LEVELS_FILL = "levels_fill"
JOB_TYPE_SEASONALITY_RUN = "seasonality_run"
JOB_TYPE_SEASONALITY_OPTIMIZE = "seasonality_optimize"
JOB_TYPE_CANONICAL_RUN = "canonical_run"

JOB_STATUS_PENDING = "pending"
JOB_STATUS_RUNNING = "running"
JOB_STATUS_COMPLETED = "completed"
JOB_STATUS_FAILED = "failed"
JOB_STATUS_QUEUED = "QUEUED"
JOB_STATUS_RUNNING_CANONICAL = "RUNNING"
JOB_STATUS_SUCCEEDED = "SUCCEEDED"
JOB_STATUS_FAILED_CANONICAL = "FAILED"
JOB_STATUS_CANCELED = "CANCELED"

JOB_STATUSES = {
    JOB_STATUS_PENDING,
    JOB_STATUS_RUNNING,
    JOB_STATUS_COMPLETED,
    JOB_STATUS_FAILED,
    JOB_STATUS_QUEUED,
    JOB_STATUS_RUNNING_CANONICAL,
    JOB_STATUS_SUCCEEDED,
    JOB_STATUS_FAILED_CANONICAL,
    JOB_STATUS_CANCELED,
}
JOB_RESULT_WITH_ID = {
    JOB_TYPE_STATS,
    JOB_TYPE_LEVELS_BUILD,
    JOB_TYPE_LEVELS_FILL,
    JOB_TYPE_SEASONALITY_RUN,
    JOB_TYPE_SEASONALITY_OPTIMIZE,
    JOB_TYPE_CANONICAL_RUN,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalize_payload(payload: Any) -> Any:
    if is_dataclass(payload):
        return asdict(payload)
    return payload


def _serialize_payload(payload: Any) -> str | None:
    if payload is None:
        return None
    try:
        return json.dumps(_normalize_payload(payload), default=str)
    except TypeError:
        return json.dumps(str(payload))


def _decode_json(payload: Any) -> Any:
    if payload in (None, ""):
        return None
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return payload
    return payload


def _init_job(
    job_id: str,
    job_type: str,
    payload: Any | None,
    *,
    status: str = JOB_STATUS_PENDING,
    max_attempts: int | None = None,
    timeout_seconds: int | None = None,
) -> None:
    with db.session() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO api_jobs(
                job_id,
                job_type,
                status,
                payload_json,
                attempts,
                max_attempts,
                timeout_seconds,
                progress_json,
                cancel_requested,
                canceled_at,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                job_type,
                status,
                _serialize_payload(payload),
                0,
                max_attempts,
                timeout_seconds,
                None,
                0,
                None,
                _utc_now(),
                _utc_now(),
            ),
        )


def _update_job_status(job_id: str, status: str, *, error: str | None = None) -> None:
    if status not in JOB_STATUSES:
        raise ValueError(f"Unknown job status: {status}")
    now = _utc_now()
    normalized = status.lower()
    started_at = now if normalized == "running" else None
    finished_at = now if normalized in {"completed", "failed", "canceled"} else None
    if status in {JOB_STATUS_SUCCEEDED, JOB_STATUS_FAILED_CANONICAL, JOB_STATUS_CANCELED}:
        finished_at = now
    canceled_at = now if normalized == "canceled" or status == JOB_STATUS_CANCELED else None
    with db.session() as conn:
        conn.execute(
            """
            UPDATE api_jobs
            SET status = ?,
                error_message = COALESCE(?, error_message),
                started_at = COALESCE(?, started_at),
                finished_at = COALESCE(?, finished_at),
                canceled_at = COALESCE(?, canceled_at),
                updated_at = ?
            WHERE job_id = ?
            """,
            (status, error, started_at, finished_at, canceled_at, now, job_id),
        )


def _update_job_result(job_id: str, result: Any, *, status: str = JOB_STATUS_COMPLETED) -> None:
    if status not in JOB_STATUSES:
        raise ValueError(f"Unknown job status: {status}")
    with db.session() as conn:
        conn.execute(
            """
            UPDATE api_jobs
            SET status = ?,
                result_json = ?,
                finished_at = ?,
                updated_at = ?
            WHERE job_id = ?
            """,
            (status, _serialize_payload(result), _utc_now(), _utc_now(), job_id),
        )


def _get_job(job_id: str) -> Dict[str, Any] | None:
    with db.session() as conn:
        row = conn.execute(
            """
            SELECT job_id, job_type, status, payload_json, result_json, error_message,
                   attempts, max_attempts, timeout_seconds, progress_json,
                   cancel_requested, canceled_at,
                   created_at, started_at, finished_at, updated_at
            FROM api_jobs
            WHERE job_id = ?
            """,
            (job_id,),
        ).fetchone()
        if not row:
            return None
        payload = dict(row)
        payload["payload"] = _decode_json(payload.pop("payload_json", None))
        payload["result"] = _decode_json(payload.pop("result_json", None))
        payload["progress"] = _decode_json(payload.pop("progress_json", None))
        return payload


def _latest_job(job_type: str) -> Dict[str, Any] | None:
    with db.session() as conn:
        row = conn.execute(
            """
            SELECT job_id, job_type, status, payload_json, result_json, error_message,
                   attempts, max_attempts, timeout_seconds, progress_json,
                   cancel_requested, canceled_at,
                   created_at, started_at, finished_at, updated_at
            FROM api_jobs
            WHERE job_type = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (job_type,),
        ).fetchone()
        if not row:
            return None
        payload = dict(row)
        payload["payload"] = _decode_json(payload.pop("payload_json", None))
        payload["result"] = _decode_json(payload.pop("result_json", None))
        payload["progress"] = _decode_json(payload.pop("progress_json", None))
        return payload


def _enqueue_job(
    job_type: str,
    payload: Any | None,
    *,
    status: str = JOB_STATUS_PENDING,
    max_attempts: int | None = None,
    timeout_seconds: int | None = None,
) -> str:
    job_id = ids.generate_id()
    _init_job(
        job_id,
        job_type,
        payload=payload,
        status=status,
        max_attempts=max_attempts,
        timeout_seconds=timeout_seconds,
    )
    return job_id


def _build_job_result(job_type: str, job_id: str, result: Any) -> Any:
    if job_type == JOB_TYPE_STATS:
        payload: Dict[str, Any] = _stats_payload_from_df(result)
        payload["job_id"] = job_id
        return payload
    if job_type in JOB_RESULT_WITH_ID:
        if isinstance(result, dict):
            payload = dict(result)
        else:
            payload = {"result": result}
        payload["job_id"] = job_id
        return payload
    return result


def _job_error_payload(code: str, message: str) -> Dict[str, Any]:
    return {"error": {"code": code, "message": message}}


def _update_job_error_result(job_id: str, payload: Dict[str, Any], *, status: str) -> None:
    if status not in JOB_STATUSES:
        raise ValueError(f"Unknown job status: {status}")
    now = _utc_now()
    with db.session() as conn:
        conn.execute(
            """
            UPDATE api_jobs
            SET status = ?,
                result_json = ?,
                finished_at = ?,
                updated_at = ?
            WHERE job_id = ?
            """,
            (status, _serialize_payload(payload), now, now, job_id),
        )


def _requeue_job(job_id: str, *, error: str | None = None, queued_status: str = JOB_STATUS_QUEUED) -> None:
    now = _utc_now()
    with db.session() as conn:
        conn.execute(
            """
            UPDATE api_jobs
            SET status = ?,
                error_message = COALESCE(?, error_message),
                started_at = NULL,
                finished_at = NULL,
                updated_at = ?
            WHERE job_id = ?
            """,
            (queued_status, error, now, job_id),
        )


def _mark_canceled(job_id: str, *, message: str = "Run canceled") -> None:
    _update_job_status(job_id, JOB_STATUS_CANCELED, error=message)
    _update_job_error_result(job_id, _job_error_payload("canceled", message), status=JOB_STATUS_CANCELED)


def request_job_cancel(job_id: str, *, message: str = "Run canceled") -> bool:
    """Request cancellation of a job. Returns True when the job is canceled immediately."""
    with db.session() as conn:
        row = conn.execute(
            "SELECT status FROM api_jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        if not row:
            return False
        status = str(row["status"])
        if status in {JOB_STATUS_QUEUED, JOB_STATUS_PENDING}:
            conn.execute(
                """
                UPDATE api_jobs
                SET status = ?, cancel_requested = 1, canceled_at = ?, finished_at = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (JOB_STATUS_CANCELED, _utc_now(), _utc_now(), _utc_now(), job_id),
            )
            return True
        conn.execute(
            "UPDATE api_jobs SET cancel_requested = 1, updated_at = ? WHERE job_id = ?",
            (_utc_now(), job_id),
        )
    return False


def update_job_progress(job_id: str, progress: Dict[str, Any]) -> None:
    with db.session() as conn:
        conn.execute(
            """
            UPDATE api_jobs
            SET progress_json = ?, updated_at = ?
            WHERE job_id = ?
            """,
            (_serialize_payload(progress), _utc_now(), job_id),
        )


def _run_job_payload(job_type: str, payload: Any | None) -> Any:
    if job_type == JOB_TYPE_OPTIMIZATION:
        spec_payload = payload["spec"] if isinstance(payload, dict) and "spec" in payload else payload
        spec_obj = spec_module.spec_from_dict(spec_payload)
        return run_optimisation(spec_obj)
    if job_type == JOB_TYPE_STATS:
        spec_obj = schemas.StatsSpec.model_validate(payload or {})
        return stats_runner.run_stats(spec_obj)
    if job_type == JOB_TYPE_LEVELS_BUILD:
        spec_obj = LevelsBuildSpec.model_validate(payload or {})
        return run_levels_build(spec_obj)
    if job_type == JOB_TYPE_LEVELS_FILL:
        spec_obj = LevelsBuildSpec.model_validate(payload or {})
        return run_levels_fill(spec_obj)
    if job_type == JOB_TYPE_SEASONALITY_RUN:
        spec_obj = schemas.SeasonalitySpec.model_validate(payload or {})
        return seasonality_runner.run(spec_obj)
    if job_type == JOB_TYPE_SEASONALITY_OPTIMIZE:
        spec_obj = schemas.SeasonalitySpec.model_validate(payload or {})
        return seasonality_run_optimization(spec_obj)
    if job_type == JOB_TYPE_CANONICAL_RUN:
        # Canonical /runs requests are queued first; execution wiring is added incrementally.
        if isinstance(payload, dict):
            request = payload.get("request", {})
            if isinstance(request, dict):
                return {"accepted": True, "spec_type": request.get("spec_type")}
        return {"accepted": True}
    raise ValueError(f"Unknown job type: {job_type}")


def _execute_job(
    job_id: str,
    job_type: str,
    payload: Any | None,
    *,
    set_running: bool = True,
    running_status: str = JOB_STATUS_RUNNING,
    success_status: str = JOB_STATUS_COMPLETED,
    failure_status: str = JOB_STATUS_FAILED,
) -> Any:
    if set_running:
        _update_job_status(job_id, running_status)
    try:
        result = _run_job_payload(job_type, payload)
    except Exception as exc:  # pragma: no cover - defensive
        _update_job_status(job_id, failure_status, error=str(exc))
        raise
    payload_out = _build_job_result(job_type, job_id, result)
    _update_job_result(job_id, payload_out, status=success_status)
    return payload_out


def _run_job(job_type: str, payload: Any | None) -> tuple[str, Any]:
    job_id = _enqueue_job(job_type, payload=payload)
    result = _execute_job(job_id, job_type, payload, set_running=True)
    return job_id, result


def _claim_next_job(
    job_type: str | None = None,
    *,
    statuses: Sequence[str] | None = None,
    running_status: str = JOB_STATUS_RUNNING,
) -> Dict[str, Any] | None:
    if statuses is None:
        statuses = (JOB_STATUS_PENDING,)
    with db.session() as conn:
        conn.execute("BEGIN IMMEDIATE")
        placeholders = ", ".join(["?"] * len(statuses))
        query = f"SELECT job_id FROM api_jobs WHERE status IN ({placeholders})"
        params: List[Any] = []
        params.extend(list(statuses))
        if job_type:
            query += " AND job_type = ?"
            params.append(job_type)
        query += " AND (cancel_requested IS NULL OR cancel_requested = 0)"
        query += " ORDER BY created_at ASC LIMIT 1"
        row = conn.execute(query, params).fetchone()
        if not row:
            return None
        job_id = row["job_id"]
        now = _utc_now()
        update_placeholders = ", ".join(["?"] * len(statuses))
        conn.execute(
            f"""
            UPDATE api_jobs
            SET status = ?,
                started_at = ?,
                updated_at = ?,
                attempts = COALESCE(attempts, 0) + 1
            WHERE job_id = ? AND status IN ({update_placeholders})
            """,
            (running_status, now, now, job_id, *statuses),
        )
    job = _get_job(job_id)
    if job is None or job.get("status") != running_status:
        return None
    return job


def run_next_job(
    job_type: str | None = None,
    *,
    statuses: Sequence[str] | None = None,
    running_status: str = JOB_STATUS_RUNNING,
    success_status: str = JOB_STATUS_COMPLETED,
    failure_status: str = JOB_STATUS_FAILED,
) -> Dict[str, Any] | None:
    """Claim and execute the next queued job (used by async workers)."""

    job = _claim_next_job(job_type, statuses=statuses, running_status=running_status)
    if not job:
        return None
    return _execute_job(
        job_id=job["job_id"],
        job_type=job["job_type"],
        payload=job.get("payload"),
        set_running=False,
        running_status=running_status,
        success_status=success_status,
        failure_status=failure_status,
    )


def submit(spec: Spec) -> schemas.SubmitResponse:
    job_id, _ = _run_job(JOB_TYPE_OPTIMIZATION, payload={"spec": asdict(spec)})
    return schemas.SubmitResponse(id=job_id)


def submit_async(spec: Spec) -> schemas.StatusResponse:
    job_id = _enqueue_job(JOB_TYPE_OPTIMIZATION, payload={"spec": asdict(spec)})
    return schemas.StatusResponse(status="pending", id=job_id)


def status(job_id: str) -> schemas.StatusResponse:
    job = _get_job(job_id)
    if not job:
        return schemas.StatusResponse(status="unknown")
    return schemas.StatusResponse(status=job["status"])


def result(job_id: str) -> schemas.ResultResponse:
    job = _get_job(job_id)
    if not job:
        return schemas.ResultResponse(result=None)
    return schemas.ResultResponse(result=job.get("result"))


def enqueue_run_request(payload: Dict[str, Any]) -> schemas.RunEnqueueResponse:
    """Validate and enqueue a canonical run request."""

    try:
        parsed = validate_run_request_input(payload)
    except ValidationError as exc:
        raise ApiValidationException(normalize_pydantic_errors(exc)) from exc

    canonical_payload = parsed.model_dump(mode="json")
    request_id = _normalize_request_id(canonical_payload.get("request_id"))
    reused = False
    if request_id is not None:
        existing = _get_job(request_id)
        if existing and existing.get("job_type") == JOB_TYPE_CANONICAL_RUN:
            reused = True
            status = _external_job_status(str(existing.get("status", "pending")))
            return schemas.RunEnqueueResponse(run_id=request_id, status=status, reused=True)
        if existing and existing.get("job_type") != JOB_TYPE_CANONICAL_RUN:
            raise ApiValidationException(
                single_validation_error("request_id", "conflict", "request_id already exists")
            )
    else:
        request_id = ids.generate_id()

    max_attempts, timeout_seconds = _canonical_job_defaults()
    _init_job(
        request_id,
        JOB_TYPE_CANONICAL_RUN,
        payload={"request": canonical_payload},
        status=JOB_STATUS_QUEUED,
        max_attempts=max_attempts,
        timeout_seconds=timeout_seconds,
    )
    status = JOB_STATUS_QUEUED
    return schemas.RunEnqueueResponse(run_id=request_id, status=status, reused=reused)


def _normalize_request_id(request_id: Any) -> str | None:
    if request_id is None:
        return None
    value = str(request_id).strip()
    return value or None


def _external_job_status(status: str) -> str:
    normalized = str(status or "").strip()
    if not normalized:
        return JOB_STATUS_QUEUED
    legacy_map = {
        JOB_STATUS_PENDING: JOB_STATUS_QUEUED,
        JOB_STATUS_RUNNING: JOB_STATUS_RUNNING_CANONICAL,
        JOB_STATUS_COMPLETED: JOB_STATUS_SUCCEEDED,
        JOB_STATUS_FAILED: JOB_STATUS_FAILED_CANONICAL,
    }
    if normalized in legacy_map:
        return legacy_map[normalized]
    upper = normalized.upper()
    if upper in {
        JOB_STATUS_QUEUED,
        JOB_STATUS_RUNNING_CANONICAL,
        JOB_STATUS_SUCCEEDED,
        JOB_STATUS_FAILED_CANONICAL,
        JOB_STATUS_CANCELED,
    }:
        return upper
    return normalized


def _json_error(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"code": code, "message": message})


def _is_terminal_status(status: str) -> bool:
    return status in {JOB_STATUS_SUCCEEDED, JOB_STATUS_FAILED_CANONICAL, JOB_STATUS_CANCELED}


def _canonical_run_payload(job: Dict[str, Any]) -> Dict[str, Any]:
    status = _external_job_status(job.get("status", ""))
    payload: Dict[str, Any] = {
        "run_id": job.get("job_id"),
        "request_id": job.get("job_id"),
        "requestId": job.get("job_id"),
        "status": status,
        "job_type": job.get("job_type"),
        "created_at": job.get("created_at"),
        "updated_at": job.get("updated_at"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "attempts": job.get("attempts"),
        "max_attempts": job.get("max_attempts"),
        "timeout_seconds": job.get("timeout_seconds"),
        "progress": job.get("progress"),
    }
    if job.get("error_message"):
        payload["error"] = {"code": "execution_error", "message": job.get("error_message")}
    return payload


def _db_ready() -> tuple[bool, str | None]:
    try:
        with db.session() as conn:
            conn.execute("SELECT 1")
        return True, None
    except Exception as exc:  # pragma: no cover - defensive
        return False, str(exc)


def _get_correlation_id(request: Request) -> str | None:
    return (
        request.headers.get("x-correlation-id")
        or request.headers.get("x-correlationid")
        or request.headers.get("x-request-id")
    )


def _log_event(event: Dict[str, Any]) -> None:
    try:
        print(json.dumps(event, separators=(",", ":")), flush=True)
    except Exception:
        pass


def _canonical_job_defaults() -> tuple[int | None, int | None]:
    max_attempts_raw = os.getenv("QE_CANONICAL_MAX_ATTEMPTS", "").strip()
    timeout_raw = os.getenv("QE_CANONICAL_TIMEOUT_SECONDS", "").strip()
    max_attempts = int(max_attempts_raw) if max_attempts_raw.isdigit() else 3
    timeout_seconds = int(timeout_raw) if timeout_raw.isdigit() else None
    return max_attempts, timeout_seconds


# ---------------------------------------------------------------------------
# Statistics endpoints (synchronous MVP)


def stats_run(spec: schemas.StatsSpec) -> schemas.StatusResponse:
    """Execute a statistics run synchronously and store the result."""

    job_id, _ = _run_job(JOB_TYPE_STATS, payload=spec.model_dump(mode="json"))
    return schemas.StatusResponse(status="completed", id=job_id)


def stats_run_async(spec: schemas.StatsSpec) -> schemas.StatusResponse:
    """Queue a statistics run to be processed by an async worker."""

    job_id = _enqueue_job(JOB_TYPE_STATS, payload=spec.model_dump(mode="json"))
    return schemas.StatusResponse(status="pending", id=job_id)


def _stats_payload_from_df(df: Any) -> Dict[str, Any]:
    try:
        import pandas as pd  # type: ignore
    except Exception:  # pragma: no cover - pandas is an install dependency
        payload_df = df
    else:
        payload_df = df.where(pd.notna(df), None)
    return {
        "columns": list(payload_df.columns),
        "rows": payload_df.to_dict(orient="records"),
    }


def stats_result() -> schemas.ResultResponse:
    """Return the last statistics result if available."""

    job = _latest_job(JOB_TYPE_STATS)
    if not job or not job.get("result"):
        return schemas.ResultResponse(result=None)
    return schemas.ResultResponse(result=job.get("result"))


def stats_condition_types() -> List[str]:
    """Return the list of supported condition factory names."""

    return stats_conditions.list_condition_types()


def filters_list() -> List[str]:
    """Return the list of supported filter identifiers."""

    return list_filter_types()


# ---------------------------------------------------------------------------
# Levels helpers


LEVELS_TABLE = "marketdata.levels"


def _resolve_levels_engine():
    url = os.environ.get("QE_LEVELS_MYSQL_URL") or os.environ.get("QE_MARKETDATA_MYSQL_URL")
    if not url:
        raise HTTPException(status_code=500, detail="MySQL URL not configured for levels module")
    return create_engine(url)


def _parse_iso_ts(value: str | None, field: str) -> datetime | None:
    if value in (None, ""):
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:  # pragma: no cover - defensive validation
        raise HTTPException(status_code=400, detail=f"Invalid datetime for {field}") from exc
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def _format_ts(value: datetime | None) -> str | None:
    if value is None:
        return None
    dt = value
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def _serialise_row(row) -> Dict[str, Any]:
    if hasattr(row, "_mapping"):
        data = dict(row._mapping)
    elif isinstance(row, dict):
        data = dict(row)
    else:  # pragma: no cover - legacy tuples from DB-API
        keys = getattr(row, "keys", None)
        if callable(keys):
            data = dict(zip(keys(), row))
        else:
            data = dict(row)
    for key in ("anchor_ts", "valid_from_ts", "valid_to_ts"):
        if key in data:
            data[key] = _format_ts(data.get(key))
    return data


def _fetch_levels(
    symbol: str,
    level_type: str | None = None,
    start: str | None = None,
    end: str | None = None,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    engine = _resolve_levels_engine()
    clauses = ["symbol = :symbol"]
    params: Dict[str, Any] = {"symbol": symbol, "limit": limit}
    if level_type:
        clauses.append("level_type = :level_type")
        params["level_type"] = level_type
    start_dt = _parse_iso_ts(start, "from")
    if start_dt is not None:
        clauses.append("anchor_ts >= :start")
        params["start"] = start_dt
    end_dt = _parse_iso_ts(end, "to")
    if end_dt is not None:
        clauses.append("anchor_ts <= :end")
        params["end"] = end_dt
    query = (
        f"SELECT symbol, timeframe, level_type, price, price_lo, price_hi, anchor_ts, "
        f"valid_from_ts, valid_to_ts, params_hash, source FROM {LEVELS_TABLE} "
        "WHERE " + " AND ".join(clauses) + " ORDER BY anchor_ts DESC LIMIT :limit"
    )
    with engine.connect() as conn:
        rows = conn.execute(text(query), params).fetchall()
    return [_serialise_row(row) for row in rows]


def _distance_to_price(price: float, payload: Dict[str, Any]) -> float:
    price_val = payload.get("price")
    if price_val is not None:
        return abs(float(price_val) - price)
    lo = payload.get("price_lo")
    hi = payload.get("price_hi")
    if lo is None or hi is None:
        return float("inf")
    lo_f = float(lo)
    hi_f = float(hi)
    if lo_f <= price <= hi_f:
        return 0.0
    if price < lo_f:
        return lo_f - price
    return price - hi_f


def _nearest_levels(
    symbol: str,
    price: float,
    level_type: str | None = None,
    window: float | None = None,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    fetch_limit = max(limit * 5, 200)
    engine = _resolve_levels_engine()
    clauses = ["symbol = :symbol"]
    params: Dict[str, Any] = {"symbol": symbol, "limit": fetch_limit}
    if level_type:
        clauses.append("level_type = :level_type")
        params["level_type"] = level_type
    if window is not None and window > 0:
        params["price_lo"] = price - window
        params["price_hi"] = price + window
        clauses.append(
            "((price IS NOT NULL AND price BETWEEN :price_lo AND :price_hi) OR "
            "(price_lo IS NOT NULL AND price_hi IS NOT NULL AND price_hi >= :price_lo AND price_lo <= :price_hi))"
        )
    query = (
        f"SELECT symbol, timeframe, level_type, price, price_lo, price_hi, anchor_ts, "
        f"valid_from_ts, valid_to_ts, params_hash, source FROM {LEVELS_TABLE} "
        "WHERE " + " AND ".join(clauses) + " ORDER BY anchor_ts DESC LIMIT :limit"
    )
    with engine.connect() as conn:
        rows = conn.execute(text(query), params).fetchall()
    payloads = [_serialise_row(row) for row in rows]
    for payload in payloads:
        distance = _distance_to_price(price, payload)
        payload["distance"] = distance
    filtered = [p for p in payloads if p.get("distance", float("inf")) != float("inf")]
    filtered.sort(key=lambda p: p.get("distance", float("inf")))
    return filtered[:limit]


def levels_build(spec: LevelsBuildSpec) -> Dict[str, Any]:
    """Execute a levels build request synchronously."""

    _, payload = _run_job(JOB_TYPE_LEVELS_BUILD, payload=spec.model_dump(mode="json"))
    return payload


def levels_fill(spec: LevelsBuildSpec) -> Dict[str, Any]:
    """Refresh fills for GAP and FVG levels."""

    _, payload = _run_job(JOB_TYPE_LEVELS_FILL, payload=spec.model_dump(mode="json"))
    return payload


def levels_build_async(spec: LevelsBuildSpec) -> schemas.StatusResponse:
    """Queue a levels build request for async processing."""

    job_id = _enqueue_job(JOB_TYPE_LEVELS_BUILD, payload=spec.model_dump(mode="json"))
    return schemas.StatusResponse(status="pending", id=job_id)


def levels_fill_async(spec: LevelsBuildSpec) -> schemas.StatusResponse:
    """Queue a levels fill request for async processing."""

    job_id = _enqueue_job(JOB_TYPE_LEVELS_FILL, payload=spec.model_dump(mode="json"))
    return schemas.StatusResponse(status="pending", id=job_id)


def levels_list(
    symbol: str,
    level_type: str | None = None,
    start: str | None = None,
    end: str | None = None,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    """Return persisted levels filtered by the provided criteria."""

    return _fetch_levels(symbol=symbol, level_type=level_type, start=start, end=end, limit=limit)


def levels_search(
    symbol: str,
    level_types: Sequence[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    """Return levels filtered by type list using the repository helper."""

    engine = _resolve_levels_engine()
    df = repo_select_levels(
        engine,
        LEVELS_TABLE,
        symbol=symbol,
        level_types=list(level_types or []),
        active_only=False,
        start=start,
        end=end,
        limit=limit,
    )
    if df.empty:
        return []
    rows = df.sort_values("anchor_ts", ascending=False).to_dict(orient="records")
    return [_serialise_row(row) for row in rows]


def levels_active(
    symbol: str,
    level_types: Optional[List[str]] | None = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    """Return currently active levels filtered by type."""

    engine = _resolve_levels_engine()
    df = repo_select_levels(
        engine,
        LEVELS_TABLE,
        symbol=symbol,
        level_types=level_types or [],
        active_only=True,
        start=start,
        end=end,
        limit=limit,
    )
    if df.empty:
        return []
    rows = df.to_dict(orient="records")
    return [_serialise_row(row) for row in rows]


def levels_nearest(
    symbol: str,
    price: float,
    level_type: str | None = None,
    window: float | None = None,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """Return the nearest levels around a target price."""

    return _nearest_levels(symbol=symbol, price=price, level_type=level_type, window=window, limit=limit)


# ---------------------------------------------------------------------------
# Seasonality endpoints


def seasonality_run(spec: schemas.SeasonalitySpec) -> schemas.ResultResponse:
    """Execute a seasonality run synchronously and return its summary."""

    _, payload = _run_job(JOB_TYPE_SEASONALITY_RUN, payload=spec.model_dump(mode="json"))
    return schemas.ResultResponse(result=payload)


def seasonality_optimize(spec: schemas.SeasonalitySpec) -> schemas.ResultResponse:
    """Launch the seasonality optimisation loop and return its outcome."""

    _, payload = _run_job(JOB_TYPE_SEASONALITY_OPTIMIZE, payload=spec.model_dump(mode="json"))
    return schemas.ResultResponse(result=payload)


def seasonality_run_async(spec: schemas.SeasonalitySpec) -> schemas.StatusResponse:
    """Queue a seasonality run for async processing."""

    job_id = _enqueue_job(JOB_TYPE_SEASONALITY_RUN, payload=spec.model_dump(mode="json"))
    return schemas.StatusResponse(status="pending", id=job_id)


def seasonality_optimize_async(spec: schemas.SeasonalitySpec) -> schemas.StatusResponse:
    """Queue a seasonality optimisation for async processing."""

    job_id = _enqueue_job(JOB_TYPE_SEASONALITY_OPTIMIZE, payload=spec.model_dump(mode="json"))
    return schemas.StatusResponse(status="pending", id=job_id)


def list_seasonality_profiles(
    symbol: str | None = None,
    timeframe: str | None = None,
    dim: str | None = None,
    measure: str | None = None,
    spec_id: str | None = None,
    dataset_id: str | None = None,
    metrics: Sequence[str] | None = None,
    page: int = 1,
    page_size: int = 50,
) -> List[Dict[str, Any]]:
    """Return paginated seasonality profiles from persistence."""

    offset = (page - 1) * page_size
    with db.session() as conn:
        query = (
            "SELECT id, symbol, timeframe, dim, bin, measure, score, n, baseline, lift, "
            "metrics, start, end, spec_id, dataset_id, created_at FROM seasonality_profiles"
        )
        params: List[Any] = []
        clauses: List[str] = []
        if symbol:
            clauses.append("symbol = ?")
            params.append(symbol)
        if timeframe:
            clauses.append("timeframe = ?")
            params.append(timeframe)
        if dim:
            clauses.append("dim = ?")
            params.append(dim)
        if measure:
            clauses.append("measure = ?")
            params.append(measure)
        if spec_id:
            clauses.append("spec_id = ?")
            params.append(spec_id)
        if dataset_id:
            clauses.append("dataset_id = ?")
            params.append(dataset_id)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([page_size, offset])
        rows = conn.execute(query, params).fetchall()
        metrics_filter = {m.strip() for m in (metrics or []) if m.strip()}
        results: List[Dict[str, Any]] = []
        for row in rows:
            payload = dict(row)
            metrics_raw = payload.get("metrics")
            metrics_map: Dict[str, Any]
            if metrics_raw in (None, ""):
                metrics_map = {}
            elif isinstance(metrics_raw, str):
                try:
                    metrics_map = json.loads(metrics_raw)
                except json.JSONDecodeError:
                    metrics_map = {}
            else:
                metrics_map = dict(metrics_raw)
            if metrics_filter:
                include = True
                for metric_name in metrics_filter:
                    if metrics_map.get(metric_name) is None:
                        include = False
                        break
                if not include:
                    continue
            payload["metrics"] = metrics_map
            results.append(payload)
        return results


def _decode_best_summary(value: Any) -> Any:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:  # pragma: no cover - defensive
            return value
    return value


def list_seasonality_runs(
    status: str | None = None,
    spec_id: str | None = None,
    dataset_id: str | None = None,
    page: int = 1,
    page_size: int = 50,
) -> List[Dict[str, Any]]:
    """Return paginated seasonality runs."""

    offset = (page - 1) * page_size
    with db.session() as conn:
        query = (
            "SELECT run_id, spec_id, dataset_id, out_dir, status, best_summary, created_at "
            "FROM seasonality_runs"
        )
        params: List[Any] = []
        clauses: List[str] = []
        if status:
            clauses.append("status = ?")
            params.append(status)
        if spec_id:
            clauses.append("spec_id = ?")
            params.append(spec_id)
        if dataset_id:
            clauses.append("dataset_id = ?")
            params.append(dataset_id)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([page_size, offset])
        rows = conn.execute(query, params).fetchall()
        out: List[Dict[str, Any]] = []
        for row in rows:
            payload = dict(row)
            payload["best_summary"] = _decode_best_summary(payload.get("best_summary"))
            out.append(payload)
        return out


def get_seasonality_run(run_id: str) -> Dict[str, Any] | None:
    """Return a single seasonality run if available."""

    with db.session() as conn:
        row = conn.execute(
            "SELECT run_id, spec_id, dataset_id, out_dir, status, best_summary, created_at "
            "FROM seasonality_runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if not row:
            return None
        payload = dict(row)
        payload["best_summary"] = _decode_best_summary(payload.get("best_summary"))
        return payload


# ---------------------------------------------------------------------------
# Read-only endpoints backed by the SQLite persistence layer


def list_runs(
    status: str | None = None,
    symbol: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    page: int = 1,
    page_size: int = 50,
) -> List[Dict[str, Any]]:
    """Return paginated runs."""

    offset = (page - 1) * page_size
    with db.session() as conn:
        query = "SELECT run_id, status, objective, out_dir, started_at, finished_at FROM experiment_runs"
        params: List[Any] = []
        clauses: List[str] = []
        if status:
            clauses.append("status = ?")
            params.append(status)
        if date_from:
            clauses.append("started_at >= ?")
            params.append(date_from)
        if date_to:
            clauses.append("started_at <= ?")
            params.append(date_to)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY started_at DESC LIMIT ? OFFSET ?"
        params.extend([page_size, offset])
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]


def get_run(run_id: str) -> Dict[str, Any] | None:
    """Return a single run with aggregated metrics."""

    with db.session() as conn:
        cur = conn.execute(
            "SELECT * FROM experiment_runs WHERE run_id = ?", (run_id,)
        )
        run = cur.fetchone()
        if not run:
            return None
        aggregated = {
            r["metric_name"]: r["metric_value"]
            for r in conn.execute(
                "SELECT metric_name, metric_value FROM run_metrics WHERE run_id = ? AND fold IS NULL",
                (run_id,),
            )
        }
        folds: Dict[int, Dict[str, float]] = {}
        for r in conn.execute(
            "SELECT fold, metric_name, metric_value FROM run_metrics WHERE run_id = ? AND fold IS NOT NULL",
            (run_id,),
        ):
            folds.setdefault(r["fold"], {})[r["metric_name"]] = r["metric_value"]
        return {"run": dict(run), "metrics": {"aggregated": aggregated, "folds": folds}}


def get_run_trials(
    run_id: str,
    order_by: str = "objective_value.desc",
    page: int = 1,
    page_size: int = 50,
) -> List[Dict[str, Any]]:
    """Return leaderboard of trials for a run."""

    offset = (page - 1) * page_size
    field, _, direction = order_by.partition(".")
    direction = "DESC" if direction.lower() == "desc" else "ASC"
    allowed_fields = {
        "objective_value",
        "n_trades",
        "max_dd",
        "sharpe",
        "sortino",
        "cagr",
        "hit_rate",
        "avg_r",
    }
    if field not in allowed_fields:
        field = "objective_value"
    with db.session() as conn:
        query = (
            f"SELECT trial_number, params_json, objective_value, status, n_trades, max_dd, "
            f"sharpe, sortino, cagr, hit_rate, avg_r FROM trials WHERE run_id = ? "
            f"ORDER BY {field} {direction} LIMIT ? OFFSET ?"
        )
        rows = conn.execute(query, (run_id, page_size, offset)).fetchall()
        return [dict(row) for row in rows]


def get_run_metrics(run_id: str) -> Dict[str, Any]:
    """Return metrics for a run (aggregated and per fold)."""

    with db.session() as conn:
        aggregated = {
            r["metric_name"]: r["metric_value"]
            for r in conn.execute(
                "SELECT metric_name, metric_value FROM run_metrics WHERE run_id = ? AND fold IS NULL",
                (run_id,),
            )
        }
        folds: Dict[int, Dict[str, float]] = {}
        for r in conn.execute(
            "SELECT fold, metric_name, metric_value FROM run_metrics WHERE run_id = ? AND fold IS NOT NULL",
            (run_id,),
        ):
            folds.setdefault(r["fold"], {})[r["metric_name"]] = r["metric_value"]
    return {"aggregated": aggregated, "folds": folds}


# ---------------------------------------------------------------------------
# Market statistics read endpoints


def list_stats(
    symbol: str | None = None,
    timeframe: str | None = None,
    event: str | None = None,
    condition_name: str | None = None,
    target: str | None = None,
    split: str | None = None,
    min_n: int | None = None,
    significant_only: bool = False,
    method: str = "freq",
    alpha: float = 0.05,
    page: int = 1,
    page_size: int = 50,
) -> List[Dict[str, Any]]:
    """Return statistics rows filtered and ordered according to parameters."""

    with db.session() as conn:
        query = "SELECT * FROM market_stats"
        params: List[Any] = []
        clauses: List[str] = []
        if symbol:
            clauses.append("symbol = ?")
            params.append(symbol)
        if timeframe:
            clauses.append("timeframe = ?")
            params.append(timeframe)
        if event:
            clauses.append("event = ?")
            params.append(event)
        if condition_name:
            clauses.append("condition_name = ?")
            params.append(condition_name)
        if target:
            clauses.append("target = ?")
            params.append(target)
        if split:
            clauses.append("split = ?")
            params.append(split)
        if min_n is not None:
            clauses.append("n >= ?")
            params.append(min_n)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)

        rows = conn.execute(query, params).fetchall()

    out = [dict(r) for r in rows]

    for r in out:
        if "lift_freq" not in r and "lift" in r:
            r["lift_freq"] = r.get("lift")
        if "lift_bayes" not in r:
            r["lift_bayes"] = r.get("lift_freq")

    if significant_only:
        out = [
            r
            for r in out
            if r.get("significant")
            or (r.get("q_value") is not None and r["q_value"] <= alpha)
        ]

    key = "lift_bayes" if method == "bayes" else "lift_freq"
    out.sort(key=lambda r: r.get(key, 0), reverse=True)

    start = (page - 1) * page_size
    end = start + page_size
    return out[start:end]


def stats_summary(
    symbol: str | None = None,
    timeframe: str | None = None,
    event: str | None = None,
) -> List[Dict[str, Any]]:
    with db.session() as conn:
        query = (
            "SELECT condition_name, condition_value, target, SUM(n) as n, "
            "SUM(successes) as successes FROM market_stats"
        )
        params: List[Any] = []
        clauses: List[str] = []
        if symbol:
            clauses.append("symbol = ?")
            params.append(symbol)
        if timeframe:
            clauses.append("timeframe = ?")
            params.append(timeframe)
        if event:
            clauses.append("event = ?")
            params.append(event)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " GROUP BY condition_name, condition_value, target"
        rows = conn.execute(query, params).fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            n = int(r["n"])
            successes = int(r["successes"])
            p_hat, ci_low, ci_high = freq_with_wilson(successes, n)
            out.append(
                {
                    "condition_name": r["condition_name"],
                    "condition_value": r["condition_value"],
                    "target": r["target"],
                    "n": n,
                    "successes": successes,
                    "p_hat": p_hat,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )
        return out


def stats_heatmap(
    symbol: str,
    timeframe: str,
    event: str,
    target: str,
    condition_name: str,
) -> List[Dict[str, Any]]:
    """Return heatmap-style bins for a condition."""

    base_query = (
        "SELECT condition_value as bin, p_hat, ci_low, ci_high, n, lift "
        "FROM market_stats WHERE symbol = ? AND timeframe = ? AND event = ? "
        "AND target = ? AND condition_name = ?"
    )
    params = [symbol, timeframe, event, target, condition_name]
    with db.session() as conn:
        rows = conn.execute(base_query + " AND split = 'test'", params).fetchall()
        if not rows:
            rows = conn.execute(base_query, params).fetchall()
    out = [dict(r) for r in rows]

    def sort_key(r: Dict[str, Any]):
        try:
            return float(r["bin"])
        except (TypeError, ValueError):
            return r["bin"]

    out.sort(key=sort_key)
    return out


def stats_top(
    symbol: str,
    timeframe: str,
    k: int = 10,
    method: str = "freq",
    significant_only: bool = False,
) -> List[Dict[str, Any]]:
    """Return top-k patterns ordered by lift for the chosen method."""

    base_query = "SELECT * FROM market_stats WHERE symbol = ? AND timeframe = ?"
    params = [symbol, timeframe]
    with db.session() as conn:
        rows = conn.execute(base_query + " AND split = 'test'", params).fetchall()
        if not rows:
            rows = conn.execute(base_query, params).fetchall()

    data = [dict(r) for r in rows]

    for r in data:
        if "lift_freq" not in r and "lift" in r:
            r["lift_freq"] = r.get("lift")
        if "lift_bayes" not in r:
            r["lift_bayes"] = r.get("lift_freq")

    if significant_only:
        data = [
            r
            for r in data
            if r.get("significant") or (r.get("q_value") is not None and r["q_value"] <= 0.05)
        ]

    key = "lift_bayes" if method == "bayes" else "lift_freq"
    rows_sorted = sorted(data, key=lambda r: abs(r.get(key, 0)), reverse=True)[:k]
    return rows_sorted




fastapi_app = FastAPI(title="Quant Engine API", version="0.1.0")


@fastapi_app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.monotonic()
    correlation_id = _get_correlation_id(request)
    response: Response
    try:
        response = await call_next(request)
    except Exception as exc:  # pragma: no cover - defensive
        duration_ms = (time.monotonic() - start) * 1000.0
        METRICS.record(
            endpoint=request.url.path,
            method=request.method,
            status_code=500,
            duration_ms=duration_ms,
        )
        event_payload = {
            "event": "http_request",
            "path": request.url.path,
            "method": request.method,
            "status": 500,
            "duration_ms": round(duration_ms, 2),
            "correlation_id": correlation_id,
            "error": str(exc),
        }
        threading.Thread(target=_log_event, args=(event_payload,), daemon=True).start()
        raise
    duration_ms = (time.monotonic() - start) * 1000.0
    METRICS.record(
        endpoint=request.url.path,
        method=request.method,
        status_code=response.status_code,
        duration_ms=duration_ms,
    )
    request_id = getattr(request.state, "request_id", None)
    if correlation_id:
        response.headers["X-Correlation-Id"] = correlation_id
    if request_id:
        response.headers["X-Request-Id"] = request_id
    event_payload = {
        "event": "http_request",
        "path": request.url.path,
        "method": request.method,
        "status": response.status_code,
        "duration_ms": round(duration_ms, 2),
        "correlation_id": correlation_id,
        "request_id": request_id,
    }
    threading.Thread(target=_log_event, args=(event_payload,), daemon=True).start()
    return response


@fastapi_app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    errors = normalize_fastapi_errors(exc.errors())
    event_payload = {
        "event": "validation_error",
        "path": str(request.url.path),
        "method": request.method,
        "status": 422,
        "correlation_id": _get_correlation_id(request),
        "errors": errors,
    }
    threading.Thread(target=_log_event, args=(event_payload,), daemon=True).start()
    return JSONResponse(status_code=422, content={"errors": errors})


@fastapi_app.exception_handler(ApiValidationException)
async def api_validation_exception_handler(
    request: Request, exc: ApiValidationException
) -> JSONResponse:
    event_payload = {
        "event": "validation_error",
        "path": str(request.url.path),
        "method": request.method,
        "status": 422,
        "correlation_id": _get_correlation_id(request),
        "errors": exc.errors,
    }
    threading.Thread(target=_log_event, args=(event_payload,), daemon=True).start()
    return JSONResponse(status_code=422, content={"errors": exc.errors})


@fastapi_app.get('/healthz', response_model=Dict[str, Any])
def healthz_endpoint() -> Dict[str, Any]:
    """Liveness probe."""

    return {"status": "ok", "ts": _utc_now()}


@fastapi_app.get('/readyz', response_model=Dict[str, Any])
def readyz_endpoint() -> Dict[str, Any]:
    """Readiness probe (checks DB connectivity)."""

    ok, error = _db_ready()
    if not ok:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "code": "db_unavailable", "message": error},
        )
    return {"status": "ready", "ts": _utc_now()}


@fastapi_app.get('/metrics', response_model=Dict[str, Any])
def metrics_endpoint() -> Dict[str, Any]:
    """Return in-memory metrics snapshot."""

    return METRICS.snapshot()


@fastapi_app.post('/submit', response_model=schemas.SubmitResponse)
def submit_endpoint(payload: Dict[str, Any]) -> schemas.SubmitResponse:
    """HTTP endpoint wrapping :func:`submit`."""

    try:
        spec_obj = spec_module.spec_from_dict(payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise ApiValidationException(
            single_validation_error("spec", "invalid_spec", f"Invalid spec: {exc}")
        ) from exc
    except ValidationError as exc:
        raise ApiValidationException(normalize_pydantic_errors(exc)) from exc
    return submit(spec_obj)


@fastapi_app.post('/submit/async', response_model=schemas.StatusResponse)
def submit_async_endpoint(payload: Dict[str, Any]) -> schemas.StatusResponse:
    """Queue an optimisation run and return a pending job id."""

    try:
        spec_obj = spec_module.spec_from_dict(payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise ApiValidationException(
            single_validation_error("spec", "invalid_spec", f"Invalid spec: {exc}")
        ) from exc
    except ValidationError as exc:
        raise ApiValidationException(normalize_pydantic_errors(exc)) from exc
    return submit_async(spec_obj)


@fastapi_app.post('/runs', response_model=schemas.RunEnqueueResponse)
def runs_submit_endpoint(payload: Dict[str, Any], request: Request) -> schemas.RunEnqueueResponse:
    """Validate and enqueue a canonical run request."""

    response = enqueue_run_request(payload)
    request.state.request_id = response.run_id
    return response


@fastapi_app.post('/runs/{run_id}/cancel', response_model=schemas.StatusResponse)
def runs_cancel_endpoint(run_id: str) -> schemas.StatusResponse:
    """Request cancellation for a canonical run."""

    job = _get_job(run_id)
    if not job:
        return _json_error(404, "not_found", "Run not found")
    status = _external_job_status(job.get("status", ""))
    if status in {JOB_STATUS_SUCCEEDED, JOB_STATUS_FAILED_CANONICAL}:
        return _json_error(409, "already_finished", "Run already finished")
    request_job_cancel(run_id)
    job = _get_job(run_id) or job
    status = _external_job_status(job.get("status", ""))
    return schemas.StatusResponse(status=status, id=run_id)


@fastapi_app.get('/status/{job_id}', response_model=schemas.StatusResponse)
def status_endpoint(job_id: str) -> schemas.StatusResponse:
    """Return the status for a submitted job."""

    return status(job_id)


@fastapi_app.get('/result/{job_id}', response_model=schemas.ResultResponse)
def result_endpoint(job_id: str) -> schemas.ResultResponse:
    """Return the optimisation result for a job if available."""

    return result(job_id)


@fastapi_app.get('/stats/conditions', response_model=List[str])
def stats_conditions_endpoint() -> List[str]:
    """Return the list of supported condition factories."""

    return stats_condition_types()


@fastapi_app.get('/filters/list', response_model=List[str])
def filters_list_endpoint() -> List[str]:
    """Return the list of available filters."""

    return filters_list()


@fastapi_app.post('/stats/run', response_model=schemas.StatusResponse)
def stats_run_endpoint(spec: schemas.StatsSpec) -> schemas.StatusResponse:
    """Kick off a statistics computation synchronously."""

    return stats_run(spec)


@fastapi_app.post('/stats/run/async', response_model=schemas.StatusResponse)
def stats_run_async_endpoint(spec: schemas.StatsSpec) -> schemas.StatusResponse:
    """Queue a statistics computation for async processing."""

    return stats_run_async(spec)


@fastapi_app.get('/stats/result', response_model=schemas.ResultResponse)
def stats_result_endpoint() -> schemas.ResultResponse:
    """Return the result of the last statistics computation."""

    return stats_result()


@fastapi_app.post('/levels/build', response_model=Dict[str, Any])
def levels_build_endpoint(spec: LevelsBuildSpec) -> Dict[str, Any]:
    """Trigger a synchronous levels build run."""

    return levels_build(spec)


@fastapi_app.post('/levels/build/async', response_model=schemas.StatusResponse)
def levels_build_async_endpoint(spec: LevelsBuildSpec) -> schemas.StatusResponse:
    """Queue a levels build run for async processing."""

    return levels_build_async(spec)


@fastapi_app.post('/levels/fill', response_model=Dict[str, Any])
def levels_fill_endpoint(spec: LevelsBuildSpec) -> Dict[str, Any]:
    """Refresh fills for active FVG/GAP levels."""

    return levels_fill(spec)


@fastapi_app.post('/levels/fill/async', response_model=schemas.StatusResponse)
def levels_fill_async_endpoint(spec: LevelsBuildSpec) -> schemas.StatusResponse:
    """Queue a levels fill run for async processing."""

    return levels_fill_async(spec)


@fastapi_app.get('/levels', response_model=List[Dict[str, Any]])
def levels_list_endpoint(
    symbol: str,
    level_type: Optional[str] = None,
    from_: Optional[str] = Query(None, alias="from"),
    to: Optional[str] = Query(None, alias="to"),
    limit: int = Query(200, ge=1, le=1000),
) -> List[Dict[str, Any]]:
    """Return persisted levels."""

    return levels_list(symbol=symbol, level_type=level_type, start=from_, end=to, limit=limit)


@fastapi_app.get('/levels/active', response_model=List[Dict[str, Any]])
def levels_active_endpoint(
    symbol: str,
    level_type: Optional[List[str]] = Query(None),
    from_: Optional[str] = Query(None, alias="from"),
    to: Optional[str] = Query(None, alias="to"),
    limit: int = Query(200, ge=1, le=10000),
) -> List[Dict[str, Any]]:
    """Return currently active levels."""

    return levels_active(symbol=symbol, level_types=level_type, start=from_, end=to, limit=limit)


@fastapi_app.get('/levels/search', response_model=List[Dict[str, Any]])
def levels_search_endpoint(
    symbol: str,
    type_: Optional[str] = Query(None, alias="type"),
    from_: Optional[str] = Query(None, alias="from"),
    to: Optional[str] = Query(None, alias="to"),
    limit: int = Query(200, ge=1, le=10000),
) -> List[Dict[str, Any]]:
    """Search persisted levels by a comma separated type filter."""

    level_types: List[str] | None = None
    if type_:
        level_types = [t.strip().upper() for t in type_.split(",") if t.strip()]
    return levels_search(symbol=symbol, level_types=level_types, start=from_, end=to, limit=limit)


@fastapi_app.get('/levels/nearest', response_model=List[Dict[str, Any]])
def levels_nearest_endpoint(
    symbol: str,
    price: float = Query(..., description="Reference price used to rank levels"),
    level_type: Optional[str] = None,
    window: Optional[float] = Query(None, description="Optional +/- window to pre-filter levels"),
    limit: int = Query(20, ge=1, le=200),
) -> List[Dict[str, Any]]:
    """Return levels closest to the requested price."""

    return levels_nearest(symbol=symbol, price=price, level_type=level_type, window=window, limit=limit)


@fastapi_app.get('/stats', response_model=List[Dict[str, Any]])
def stats_list_endpoint(
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    event: Optional[str] = None,
    condition_name: Optional[str] = None,
    target: Optional[str] = None,
    split: Optional[str] = None,
    min_n: Optional[int] = Query(None, ge=0),
    significant_only: bool = False,
    method: str = Query('freq', pattern='^(freq|bayes)$'),
    alpha: float = Query(0.05, ge=0.0, le=1.0),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
) -> List[Dict[str, Any]]:
    """List persisted statistics with optional filters."""

    return list_stats(
        symbol=symbol,
        timeframe=timeframe,
        event=event,
        condition_name=condition_name,
        target=target,
        split=split,
        min_n=min_n,
        significant_only=significant_only,
        method=method,
        alpha=alpha,
        page=page,
        page_size=page_size,
    )


@fastapi_app.get('/stats/summary', response_model=List[Dict[str, Any]])
def stats_summary_endpoint(
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    event: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return aggregated statistics summary."""

    return stats_summary(symbol=symbol, timeframe=timeframe, event=event)


@fastapi_app.get('/stats/heatmap', response_model=List[Dict[str, Any]])
def stats_heatmap_endpoint(
    symbol: str,
    timeframe: str,
    event: str,
    target: str,
    condition_name: str,
) -> List[Dict[str, Any]]:
    """Return heatmap-style bins for a condition."""

    return stats_heatmap(symbol, timeframe, event, target, condition_name)


@fastapi_app.get('/stats/top', response_model=List[Dict[str, Any]])
def stats_top_endpoint(
    symbol: str,
    timeframe: str,
    k: int = Query(10, ge=1, le=100),
    method: str = Query('freq', pattern='^(freq|bayes)$'),
    significant_only: bool = False,
) -> List[Dict[str, Any]]:
    """Return top patterns ordered by lift."""

    return stats_top(symbol, timeframe, k=k, method=method, significant_only=significant_only)


@fastapi_app.post('/seasonality/run', response_model=schemas.ResultResponse)
def seasonality_run_endpoint(spec: schemas.SeasonalitySpec) -> schemas.ResultResponse:
    """Execute a seasonality run synchronously."""

    return seasonality_run(spec)


@fastapi_app.post('/seasonality/run/async', response_model=schemas.StatusResponse)
def seasonality_run_async_endpoint(spec: schemas.SeasonalitySpec) -> schemas.StatusResponse:
    """Queue a seasonality run for async processing."""

    return seasonality_run_async(spec)


@fastapi_app.post('/seasonality/optimize', response_model=schemas.ResultResponse)
def seasonality_optimize_endpoint(spec: schemas.SeasonalitySpec) -> schemas.ResultResponse:
    """Execute the seasonality optimisation loop."""

    return seasonality_optimize(spec)


@fastapi_app.post('/seasonality/optimize/async', response_model=schemas.StatusResponse)
def seasonality_optimize_async_endpoint(spec: schemas.SeasonalitySpec) -> schemas.StatusResponse:
    """Queue a seasonality optimisation for async processing."""

    return seasonality_optimize_async(spec)


@fastapi_app.get('/seasonality/profiles', response_model=List[Dict[str, Any]])
def seasonality_profiles_endpoint(
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    dim: Optional[str] = None,
    measure: Optional[str] = None,
    spec_id: Optional[str] = None,
    dataset_id: Optional[str] = None,
    metrics: Optional[List[str]] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
) -> List[Dict[str, Any]]:
    """List persisted seasonality profiles."""

    return list_seasonality_profiles(
        symbol=symbol,
        timeframe=timeframe,
        dim=dim,
        measure=measure,
        spec_id=spec_id,
        dataset_id=dataset_id,
        metrics=metrics,
        page=page,
        page_size=page_size,
    )


@fastapi_app.get('/seasonality/runs', response_model=List[Dict[str, Any]])
def seasonality_runs_endpoint(
    status: Optional[str] = None,
    spec_id: Optional[str] = None,
    dataset_id: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
) -> List[Dict[str, Any]]:
    """List persisted seasonality runs."""

    return list_seasonality_runs(
        status=status,
        spec_id=spec_id,
        dataset_id=dataset_id,
        page=page,
        page_size=page_size,
    )


@fastapi_app.get('/seasonality/runs/{run_id}', response_model=Dict[str, Any])
def seasonality_run_detail_endpoint(run_id: str) -> Dict[str, Any]:
    """Return details for a specific seasonality run."""

    payload = get_seasonality_run(run_id)
    if payload is None:
        raise HTTPException(status_code=404, detail='Run not found')
    return payload


@fastapi_app.get('/runs', response_model=List[Dict[str, Any]])
def runs_list_endpoint(
    status: Optional[str] = None,
    symbol: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
) -> List[Dict[str, Any]]:
    """List optimisation runs."""

    return list_runs(
        status=status,
        symbol=symbol,
        date_from=date_from,
        date_to=date_to,
        page=page,
        page_size=page_size,
    )


@fastapi_app.get('/runs/{run_id}', response_model=Dict[str, Any])
def run_detail_endpoint(run_id: str) -> Dict[str, Any]:
    """Return a single run and aggregated metrics."""

    job = _get_job(run_id)
    if job and job.get("job_type") == JOB_TYPE_CANONICAL_RUN:
        return _canonical_run_payload(job)
    payload = get_run(run_id)
    if payload is None:
        raise HTTPException(status_code=404, detail='Run not found')
    return payload


@fastapi_app.get('/runs/{run_id}/result', response_model=Dict[str, Any])
def run_result_endpoint(run_id: str) -> Dict[str, Any]:
    """Return the result for a canonical run."""

    job = _get_job(run_id)
    if not job or job.get("job_type") != JOB_TYPE_CANONICAL_RUN:
        return _json_error(404, "not_found", "Run not found")
    status = _external_job_status(job.get("status", ""))
    payload: Dict[str, Any] = {
        "run_id": job.get("job_id"),
        "request_id": job.get("job_id"),
        "requestId": job.get("job_id"),
        "status": status,
    }
    if not _is_terminal_status(status):
        payload["message"] = "Result not available yet"
        return payload
    result = job.get("result")
    if result is not None:
        payload["result"] = result
    if job.get("error_message"):
        payload["error"] = {"code": "execution_error", "message": job.get("error_message")}
    if isinstance(result, dict) and "error" in result:
        payload["error"] = result.get("error")
    return payload


@fastapi_app.get('/runs/{run_id}/trials', response_model=List[Dict[str, Any]])
def run_trials_endpoint(
    run_id: str,
    order_by: str = 'objective_value.desc',
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
) -> List[Dict[str, Any]]:
    """Return leaderboard of trials for a run."""

    return get_run_trials(run_id, order_by=order_by, page=page, page_size=page_size)


@fastapi_app.get('/runs/{run_id}/metrics', response_model=Dict[str, Any])
def run_metrics_endpoint(run_id: str) -> Dict[str, Any]:
    """Return metrics for a run."""

    return get_run_metrics(run_id)


app = fastapi_app
