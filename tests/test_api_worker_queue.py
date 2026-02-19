import threading
import time

from quant_engine.api import app as api_app
from quant_engine.api import worker as worker_module
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


def test_worker_processes_job_success(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DB_SQLITE_PATH", str(tmp_path / "quant.db"))
    reset_settings_cache()

    def _fake_run(job_type, payload):
        return {"ok": True}

    monkeypatch.setattr(api_app, "_run_job_payload", _fake_run)

    response = api_app.enqueue_run_request(_canonical_payload())

    result = worker_module.process_next_job()

    assert result is not None
    job = api_app._get_job(response.run_id)
    assert job["status"] == api_app.JOB_STATUS_SUCCEEDED


def test_worker_failure_marks_failed(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DB_SQLITE_PATH", str(tmp_path / "quant.db"))
    monkeypatch.setenv("QE_CANONICAL_MAX_ATTEMPTS", "1")
    reset_settings_cache()

    def _boom(job_type, payload):
        raise ValueError("boom")

    monkeypatch.setattr(api_app, "_run_job_payload", _boom)

    response = api_app.enqueue_run_request(_canonical_payload())

    result = worker_module.process_next_job()

    assert result is None
    job = api_app._get_job(response.run_id)
    assert job["status"] == api_app.JOB_STATUS_FAILED_CANONICAL
    assert job["result"]["error"]["message"] == "boom"


def test_worker_cancel_before_execution(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DB_SQLITE_PATH", str(tmp_path / "quant.db"))
    reset_settings_cache()

    response = api_app.enqueue_run_request(_canonical_payload())

    assert api_app.request_job_cancel(response.run_id) is True

    worker_module.process_next_job()

    job = api_app._get_job(response.run_id)
    assert job["status"] == api_app.JOB_STATUS_CANCELED


def test_worker_cancel_during_execution(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DB_SQLITE_PATH", str(tmp_path / "quant.db"))
    reset_settings_cache()

    def _slow(job_type, payload):
        time.sleep(0.1)
        return {"ok": True}

    monkeypatch.setattr(api_app, "_run_job_payload", _slow)

    response = api_app.enqueue_run_request(_canonical_payload())

    def _cancel_later(job_id: str) -> None:
        def _cancel():
            time.sleep(0.02)
            api_app.request_job_cancel(job_id)

        threading.Thread(target=_cancel, daemon=True).start()

    worker_module.process_next_job(on_started=_cancel_later)

    job = api_app._get_job(response.run_id)
    assert job["status"] == api_app.JOB_STATUS_CANCELED


def test_recover_stale_jobs_requeues(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DB_SQLITE_PATH", str(tmp_path / "quant.db"))
    reset_settings_cache()

    response = api_app.enqueue_run_request(_canonical_payload())
    with api_app.db.session() as conn:
        conn.execute(
            """
            UPDATE api_jobs
            SET status = ?, started_at = ?, attempts = 1
            WHERE job_id = ?
            """,
            (
                api_app.JOB_STATUS_RUNNING_CANONICAL,
                "2020-01-01T00:00:00Z",
                response.run_id,
            ),
        )

    recovered = worker_module.recover_stale_jobs(stale_after_seconds=1)

    assert recovered == 1
    job = api_app._get_job(response.run_id)
    assert job["status"] == api_app.JOB_STATUS_QUEUED


def test_worker_marks_canonical_backtest_not_implemented(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DB_SQLITE_PATH", str(tmp_path / "quant.db"))
    reset_settings_cache()
    payload = _canonical_payload()
    payload["filters"] = {"filters": [{"id": "trend", "params": {"min": 1}}]}
    payload["strategy"] = {"name": "demo", "params": {"tp_sl": {"dynamic_sl": {"enabled": True}}}}
    response = api_app.enqueue_run_request(payload)

    result = worker_module.process_next_job()

    assert result is None
    job = api_app._get_job(response.run_id)
    assert job["status"] == api_app.JOB_STATUS_FAILED_CANONICAL
    error = job["result"]["error"]
    assert error["code"] == "not_implemented_feature"
    assert error["message"] == "Feature not implemented for canonical backtest run"
    fields = {item["field"] for item in error["details"]}
    assert "signal" in fields
    assert "filters.filters" in fields
    assert "strategy.name" in fields
    assert "strategy.params.tp_sl" in fields
