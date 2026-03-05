from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Mapping

from fastapi import HTTPException

from .. import schemas


RunLookup = Callable[[str], Dict[str, Any] | None]
StatusMapper = Callable[[str], str]
JsonError = Callable[[int, str, str], Any]
TerminalStatusPredicate = Callable[[str], bool]


def cancel_run(
    run_id: str,
    *,
    get_job: RunLookup,
    external_job_status: StatusMapper,
    request_job_cancel: Callable[[str], bool],
    json_error: JsonError,
) -> schemas.StatusResponse | Any:
    """Request cancellation for a canonical run while preserving API response semantics."""

    job = get_job(run_id)
    if not job:
        return json_error(404, "not_found", "Run not found")

    status = external_job_status(job.get("status", ""))
    if status in {"SUCCEEDED", "FAILED"}:
        return json_error(409, "already_finished", "Run already finished")

    request_job_cancel(run_id)
    job = get_job(run_id) or job
    status = external_job_status(job.get("status", ""))
    return schemas.StatusResponse(status=status, id=run_id)


def run_detail(
    run_id: str,
    *,
    get_job: RunLookup,
    get_legacy_run: RunLookup,
    canonical_run_payload: Callable[[Dict[str, Any]], Dict[str, Any]],
) -> Dict[str, Any]:
    """Return canonical payload for canonical runs, fallback to legacy run payload otherwise."""

    job = get_job(run_id)
    if job and job.get("job_type") == "canonical_run":
        return canonical_run_payload(job)

    payload = get_legacy_run(run_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return payload


def run_result(
    run_id: str,
    *,
    get_job: RunLookup,
    external_job_status: StatusMapper,
    is_terminal_status: TerminalStatusPredicate,
    json_error: JsonError,
) -> Dict[str, Any] | Any:
    """Return canonical run result payload while keeping compatibility for pending states."""

    job = get_job(run_id)
    if not job or job.get("job_type") != "canonical_run":
        return json_error(404, "not_found", "Run not found")

    status = external_job_status(job.get("status", ""))
    payload: Dict[str, Any] = {
        "run_id": job.get("job_id"),
        "request_id": job.get("job_id"),
        "requestId": job.get("job_id"),
        "status": status,
    }
    if not is_terminal_status(status):
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


def run_artifacts(
    run_id: str,
    *,
    get_job: RunLookup,
    schema_version: str,
) -> Dict[str, Any]:
    """Return artifact listing for a canonical run output directory."""

    job = get_job(run_id)
    if not job or job.get("job_type") != "canonical_run":
        raise HTTPException(status_code=404, detail="Run not found")

    payload = job.get("payload") if isinstance(job.get("payload"), Mapping) else {}
    request = payload.get("request") if isinstance(payload.get("request"), Mapping) else {}
    output = request.get("output") if isinstance(request.get("output"), Mapping) else {}
    out_dir_raw = output.get("out_dir")
    out_dir = Path(str(out_dir_raw)).expanduser() if isinstance(out_dir_raw, str) and out_dir_raw.strip() else None

    files: list[dict[str, Any]] = []
    if out_dir and out_dir.exists() and out_dir.is_dir():
        for item in sorted(out_dir.iterdir()):
            if item.is_file():
                files.append({"name": item.name, "path": str(item), "size_bytes": item.stat().st_size})

    file_names = {entry["name"] for entry in files}
    contract_artifacts = {
        "best_plausible_passive_ex_ante": {
            "json": "best_plausible_passive_ex_ante.json" if "best_plausible_passive_ex_ante.json" in file_names else None,
            "parquet": "best_plausible_passive_ex_ante.parquet" if "best_plausible_passive_ex_ante.parquet" in file_names else None,
        }
    }
    return {
        "run_id": run_id,
        "schema_version": schema_version,
        "out_dir": str(out_dir) if out_dir is not None else None,
        "files": files,
        "contract_artifacts": contract_artifacts,
        "audit_trail": {
            "run_manifest": "run_manifest.json" if "run_manifest.json" in file_names else None,
            "checksums": "checksums.txt" if "checksums.txt" in file_names else None,
        },
    }
