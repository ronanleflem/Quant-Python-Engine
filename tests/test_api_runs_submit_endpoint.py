from fastapi.testclient import TestClient

from quant_engine.api import app as api_app
from quant_engine.config import reset_settings_cache


def _canonical_backtest_payload(request_id: str | None = None) -> dict:
    payload = {
        "spec_type": "backtest",
        "catalog_version": "v1",
        "data": {
            "symbol": "EURUSD",
            "timeframe": "M1",
            "start_date": "2025-01-01",
            "end_date": "2025-01-31",
        },
        "signal": {"type": "ema_cross", "fast": 9, "slow": 21},
    }
    if request_id is not None:
        payload["request_id"] = request_id
    return payload


def test_runs_submit_enqueues_request(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DB_SQLITE_PATH", str(tmp_path / "quant.db"))
    reset_settings_cache()
    client = TestClient(api_app.fastapi_app)

    response = client.post("/runs", json=_canonical_backtest_payload())

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "QUEUED"
    assert payload["reused"] is False
    assert isinstance(payload["run_id"], str)
    assert payload["run_id"]


def test_runs_submit_reuses_existing_request_id(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DB_SQLITE_PATH", str(tmp_path / "quant.db"))
    reset_settings_cache()
    client = TestClient(api_app.fastapi_app)
    req_id = "run_req_001"

    first = client.post("/runs", json=_canonical_backtest_payload(request_id=req_id))
    second = client.post("/runs", json=_canonical_backtest_payload(request_id=req_id))

    assert first.status_code == 200
    assert second.status_code == 200
    first_payload = first.json()
    second_payload = second.json()
    assert first_payload["run_id"] == req_id
    assert second_payload["run_id"] == req_id
    assert first_payload["reused"] is False
    assert second_payload["reused"] is True


def test_runs_submit_returns_422_with_normalized_errors(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DB_SQLITE_PATH", str(tmp_path / "quant.db"))
    reset_settings_cache()
    client = TestClient(api_app.fastapi_app)

    response = client.post("/runs", json={"catalog_version": "v1"})

    assert response.status_code == 422
    payload = response.json()
    assert "errors" in payload
    assert payload["errors"]
    assert set(payload["errors"][0].keys()) == {"field", "code", "message"}
