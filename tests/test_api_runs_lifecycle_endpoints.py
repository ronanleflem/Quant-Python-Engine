import pytest
from fastapi.testclient import TestClient

from quant_engine.api import app as api_app
from quant_engine.config import reset_settings_cache


def _canonical_payload() -> dict:
    return {
        "spec_type": "backtest",
        "catalog_version": "v1",
        "data": {
            "symbol": "EURUSD",
            "timeframe": "M1",
            "start_date": "2025-01-01",
            "end_date": "2025-01-02",
        },
        "signal": {"type": "ema_cross", "fast": 9, "slow": 21},
    }


def _setup_db(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.setenv("DB_SQLITE_PATH", str(tmp_path / "quant.db"))
    reset_settings_cache()
    return TestClient(api_app.fastapi_app)


@pytest.mark.parametrize(
    "status",
    [
        api_app.JOB_STATUS_QUEUED,
        api_app.JOB_STATUS_RUNNING_CANONICAL,
        api_app.JOB_STATUS_SUCCEEDED,
        api_app.JOB_STATUS_FAILED_CANONICAL,
        api_app.JOB_STATUS_CANCELED,
    ],
)
def test_run_detail_statuses(tmp_path, monkeypatch, status: str) -> None:
    client = _setup_db(tmp_path, monkeypatch)
    response = api_app.enqueue_run_request(_canonical_payload())
    run_id = response.run_id

    if status == api_app.JOB_STATUS_RUNNING_CANONICAL:
        api_app._update_job_status(run_id, api_app.JOB_STATUS_RUNNING_CANONICAL)
    elif status == api_app.JOB_STATUS_SUCCEEDED:
        api_app._update_job_result(run_id, {"ok": True}, status=api_app.JOB_STATUS_SUCCEEDED)
    elif status == api_app.JOB_STATUS_FAILED_CANONICAL:
        api_app._update_job_status(run_id, api_app.JOB_STATUS_FAILED_CANONICAL, error="boom")
        api_app._update_job_error_result(
            run_id,
            api_app._job_error_payload("execution_error", "boom"),
            status=api_app.JOB_STATUS_FAILED_CANONICAL,
        )
    elif status == api_app.JOB_STATUS_CANCELED:
        api_app.request_job_cancel(run_id)

    resp = client.get(f"/runs/{run_id}")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["run_id"] == run_id
    assert payload["status"] == status


def test_run_result_non_terminal(tmp_path, monkeypatch) -> None:
    client = _setup_db(tmp_path, monkeypatch)
    response = api_app.enqueue_run_request(_canonical_payload())

    resp = client.get(f"/runs/{response.run_id}/result")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == api_app.JOB_STATUS_QUEUED
    assert "Result not available yet" in payload["message"]


def test_run_result_terminal_success(tmp_path, monkeypatch) -> None:
    client = _setup_db(tmp_path, monkeypatch)
    response = api_app.enqueue_run_request(_canonical_payload())
    api_app._update_job_result(response.run_id, {"ok": True}, status=api_app.JOB_STATUS_SUCCEEDED)

    resp = client.get(f"/runs/{response.run_id}/result")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == api_app.JOB_STATUS_SUCCEEDED
    assert payload["result"]["ok"] is True


def test_run_result_terminal_failure(tmp_path, monkeypatch) -> None:
    client = _setup_db(tmp_path, monkeypatch)
    response = api_app.enqueue_run_request(_canonical_payload())
    api_app._update_job_status(response.run_id, api_app.JOB_STATUS_FAILED_CANONICAL, error="boom")
    api_app._update_job_error_result(
        response.run_id,
        api_app._job_error_payload("execution_error", "boom"),
        status=api_app.JOB_STATUS_FAILED_CANONICAL,
    )

    resp = client.get(f"/runs/{response.run_id}/result")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == api_app.JOB_STATUS_FAILED_CANONICAL
    assert payload["error"]["message"] == "boom"


def test_cancel_idempotent(tmp_path, monkeypatch) -> None:
    client = _setup_db(tmp_path, monkeypatch)
    response = api_app.enqueue_run_request(_canonical_payload())
    run_id = response.run_id

    first = client.post(f"/runs/{run_id}/cancel")
    second = client.post(f"/runs/{run_id}/cancel")

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["status"] == api_app.JOB_STATUS_CANCELED
    assert second.json()["status"] == api_app.JOB_STATUS_CANCELED


def test_cancel_terminal_conflict(tmp_path, monkeypatch) -> None:
    client = _setup_db(tmp_path, monkeypatch)
    response = api_app.enqueue_run_request(_canonical_payload())
    api_app._update_job_result(response.run_id, {"ok": True}, status=api_app.JOB_STATUS_SUCCEEDED)

    resp = client.post(f"/runs/{response.run_id}/cancel")

    assert resp.status_code == 409
    payload = resp.json()
    assert payload["code"] == "already_finished"


def test_run_result_not_found(tmp_path, monkeypatch) -> None:
    client = _setup_db(tmp_path, monkeypatch)

    resp = client.get("/runs/missing_run/result")
    assert resp.status_code == 404
    payload = resp.json()
    assert payload["code"] == "not_found"


def test_cancel_not_found(tmp_path, monkeypatch) -> None:
    client = _setup_db(tmp_path, monkeypatch)

    resp = client.post("/runs/missing_run/cancel")
    assert resp.status_code == 404
    payload = resp.json()
    assert payload["code"] == "not_found"
