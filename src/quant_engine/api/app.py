"""Tiny in-memory API resembling the FastAPI specification.

The implementation exposes three functions ``submit``, ``status`` and
``result`` that mimic the behaviour of REST endpoints.  They can be called
synchronously which keeps the tests lightweight while preserving the public
contract of the original design.
"""
from __future__ import annotations

from typing import Dict, Any

from ..core.spec import Spec
from ..optimize.runner import run as run_optimisation
from ..io import ids
from . import schemas

_jobs: Dict[str, Dict[str, Any]] = {}


def submit(spec: Spec) -> schemas.SubmitResponse:
    job_id = ids.generate_id()
    result = run_optimisation(spec)
    _jobs[job_id] = {"status": "completed", "result": result}
    return schemas.SubmitResponse(id=job_id)


def status(job_id: str) -> schemas.StatusResponse:
    job = _jobs.get(job_id)
    if not job:
        return schemas.StatusResponse(status="unknown")
    return schemas.StatusResponse(status=job["status"])


def result(job_id: str) -> schemas.ResultResponse:
    job = _jobs.get(job_id)
    if not job:
        return schemas.ResultResponse(result=None)
    return schemas.ResultResponse(result=job.get("result"))

