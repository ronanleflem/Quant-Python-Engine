"""Async worker for processing queued API jobs."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, Optional

from . import app as api_app


def _int_env(name: str, default: Optional[int]) -> Optional[int]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    text = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


@dataclass(frozen=True)
class WorkerConfig:
    job_type: str = api_app.JOB_TYPE_CANONICAL_RUN
    poll_interval: float = 1.0
    stale_after_seconds: Optional[int] = None
    heartbeat_seconds: float = 30.0
    statuses: tuple[str, ...] = (api_app.JOB_STATUS_QUEUED,)
    running_status: str = api_app.JOB_STATUS_RUNNING_CANONICAL
    success_status: str = api_app.JOB_STATUS_SUCCEEDED
    failure_status: str = api_app.JOB_STATUS_FAILED_CANONICAL
    queued_status: str = api_app.JOB_STATUS_QUEUED

    @staticmethod
    def from_env() -> "WorkerConfig":
        return WorkerConfig(
            poll_interval=_float_env("QE_WORKER_POLL_SECONDS", 1.0),
            stale_after_seconds=_int_env("QE_CANONICAL_STALE_SECONDS", None),
            heartbeat_seconds=_float_env("QE_WORKER_HEARTBEAT_SECONDS", 30.0),
        )


def recover_stale_jobs(
    *,
    job_type: str = api_app.JOB_TYPE_CANONICAL_RUN,
    running_status: str = api_app.JOB_STATUS_RUNNING_CANONICAL,
    queued_status: str = api_app.JOB_STATUS_QUEUED,
    failure_status: str = api_app.JOB_STATUS_FAILED_CANONICAL,
    stale_after_seconds: Optional[int] = None,
) -> int:
    """Requeue or fail jobs stuck in RUNNING past the configured timeout."""
    if stale_after_seconds is None:
        return 0
    now = datetime.now(timezone.utc)
    recovered = 0
    with api_app.db.session() as conn:
        rows = conn.execute(
            """
            SELECT job_id, started_at, updated_at, attempts, max_attempts, timeout_seconds, cancel_requested
            FROM api_jobs
            WHERE status = ? AND job_type = ?
            """,
            (running_status, job_type),
        ).fetchall()
    for row in rows:
        started_at = _parse_timestamp(row["started_at"]) or _parse_timestamp(row["updated_at"])
        if started_at is None:
            continue
        timeout_seconds = row["timeout_seconds"] or stale_after_seconds
        if timeout_seconds is None:
            continue
        if (now - started_at).total_seconds() <= timeout_seconds:
            continue
        job_id = row["job_id"]
        if row["cancel_requested"]:
            api_app._mark_canceled(job_id, message="Run canceled (stale)")
            recovered += 1
            continue
        attempts = int(row["attempts"] or 0)
        max_attempts = row["max_attempts"]
        if max_attempts is not None and attempts >= max_attempts:
            api_app._update_job_status(job_id, failure_status, error="Timeout")
            api_app._update_job_error_result(
                job_id,
                api_app._job_error_payload("timeout", "Timeout"),
                status=failure_status,
            )
        else:
            api_app._requeue_job(job_id, error="Timeout", queued_status=queued_status)
        recovered += 1
    return recovered


def process_next_job(
    *,
    job_type: str = api_app.JOB_TYPE_CANONICAL_RUN,
    statuses: Iterable[str] = (api_app.JOB_STATUS_QUEUED,),
    running_status: str = api_app.JOB_STATUS_RUNNING_CANONICAL,
    success_status: str = api_app.JOB_STATUS_SUCCEEDED,
    failure_status: str = api_app.JOB_STATUS_FAILED_CANONICAL,
    queued_status: str = api_app.JOB_STATUS_QUEUED,
    stale_after_seconds: Optional[int] = None,
    on_started: Optional[Callable[[str], None]] = None,
) -> Optional[Dict[str, Any]]:
    recover_stale_jobs(
        job_type=job_type,
        running_status=running_status,
        queued_status=queued_status,
        failure_status=failure_status,
        stale_after_seconds=stale_after_seconds,
    )
    job = api_app._claim_next_job(
        job_type,
        statuses=tuple(statuses),
        running_status=running_status,
    )
    if not job:
        return None
    job_id = str(job["job_id"])
    if on_started is not None:
        on_started(job_id)
    if job.get("cancel_requested"):
        api_app._mark_canceled(job_id)
        return None
    attempts = int(job.get("attempts") or 0)
    max_attempts = job.get("max_attempts")
    if max_attempts is not None and attempts > max_attempts:
        api_app._update_job_status(job_id, failure_status, error="Max attempts exceeded")
        api_app._update_job_error_result(
            job_id,
            api_app._job_error_payload("max_attempts", "Max attempts exceeded"),
            status=failure_status,
        )
        return None
    try:
        result = api_app._run_job_payload(job.get("job_type"), job.get("payload"))
    except Exception as exc:
        message = str(exc)
        if max_attempts is not None and attempts >= max_attempts:
            api_app._update_job_status(job_id, failure_status, error=message)
            api_app._update_job_error_result(
                job_id,
                api_app._job_error_payload("execution_error", message),
                status=failure_status,
            )
        else:
            api_app._requeue_job(job_id, error=message, queued_status=queued_status)
        return None
    if api_app._get_job(job_id).get("cancel_requested"):
        api_app._mark_canceled(job_id)
        return None
    payload_out = api_app._build_job_result(job.get("job_type"), job_id, result)
    api_app._update_job_result(job_id, payload_out, status=success_status)
    return payload_out


def run_worker(config: WorkerConfig, *, once: bool = False) -> None:
    last_heartbeat = time.monotonic()
    while True:
        result = process_next_job(
            job_type=config.job_type,
            statuses=config.statuses,
            running_status=config.running_status,
            success_status=config.success_status,
            failure_status=config.failure_status,
            queued_status=config.queued_status,
            stale_after_seconds=config.stale_after_seconds,
        )
        heartbeat_interval = max(0.0, float(config.heartbeat_seconds))
        if heartbeat_interval > 0:
            now = time.monotonic()
            if now - last_heartbeat >= heartbeat_interval:
                try:
                    print(
                        {
                            "event": "worker_heartbeat",
                            "job_type": config.job_type,
                            "ts": api_app._utc_now(),
                        }
                    )
                except Exception:
                    pass
                last_heartbeat = now
        if once:
            return
        if result is None:
            time.sleep(config.poll_interval)
