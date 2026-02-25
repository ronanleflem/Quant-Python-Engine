import pytest
from fastapi.testclient import TestClient

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


def _canonical_dca_payload() -> dict:
    return {
        "spec_type": "dca",
        "catalog_version": "v1",
        "data": {
            "symbol": "BTCUSD",
            "timeframe": "H1",
            "start_date": "2025-01-01",
            "end_date": "2025-01-02",
        },
        "strategy": {
            "type": "dca_equity",
            "grid": [],
            "params": {
                "grid": [{"dd": -5.0, "weight": 1.0}],
                "execution_mode": "bar_close",
                "drawdown_reference": "ATH",
            },
        },
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


def test_run_result_returns_not_implemented_error_details(tmp_path, monkeypatch) -> None:
    client = _setup_db(tmp_path, monkeypatch)
    payload = _canonical_payload()
    payload["filters"] = {"filters": [{"id": "trend", "params": {"min": 1}}]}
    payload["strategy"] = {"name": "demo", "params": {"tpSl": {"dynamic_sl": {"enabled": True}}}}
    response = api_app.enqueue_run_request(payload)

    worker_module.process_next_job()

    resp = client.get(f"/runs/{response.run_id}/result")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == api_app.JOB_STATUS_FAILED_CANONICAL
    assert body["error"]["code"] == "not_implemented_feature"
    fields = {item["field"] for item in body["error"]["details"]}
    assert "strategy.name" in fields


def test_run_result_returns_canonical_backtest_success_payload(tmp_path, monkeypatch) -> None:
    client = _setup_db(tmp_path, monkeypatch)

    def _fake_run_backtest(spec):
        return {"trades": [], "metrics": {"n_trades": 0}, "symbol": spec["data"]["symbol"]}

    monkeypatch.setattr(api_app.backtest_runner, "run_backtest_from_spec", _fake_run_backtest)
    response = api_app.enqueue_run_request(_canonical_payload())
    worker_module.process_next_job()

    resp = client.get(f"/runs/{response.run_id}/result")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == api_app.JOB_STATUS_SUCCEEDED
    assert body["result"]["accepted"] is True
    assert body["result"]["spec_type"] == "backtest"
    assert body["result"]["result"]["symbol"] == "EURUSD"


def test_run_result_returns_canonical_dca_success_payload(tmp_path, monkeypatch) -> None:
    client = _setup_db(tmp_path, monkeypatch)

    def _fake_backtest(spec):
        return {"result": {"counts": {"BTCUSD": 2}}, "payload": {"run": {"status": "ok"}}}

    monkeypatch.setattr(api_app.strategies_runner, "run_backtest_with_payload", _fake_backtest)
    response = api_app.enqueue_run_request(_canonical_dca_payload())
    worker_module.process_next_job()

    resp = client.get(f"/runs/{response.run_id}/result")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == api_app.JOB_STATUS_SUCCEEDED
    assert payload["result"]["accepted"] is True
    assert payload["result"]["spec_type"] == "dca"
    assert payload["result"]["result"]["counts"]["BTCUSD"] == 2


def test_run_result_returns_canonical_dca_unwired_tp_sl_error(tmp_path, monkeypatch) -> None:
    client = _setup_db(tmp_path, monkeypatch)
    payload = _canonical_dca_payload()
    payload["strategy"]["params"]["tp_sl"] = "tp_custom"
    response = api_app.enqueue_run_request(payload)

    worker_module.process_next_job()

    resp = client.get(f"/runs/{response.run_id}/result")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == api_app.JOB_STATUS_FAILED_CANONICAL
    assert body["error"]["code"] == "not_implemented_feature"
    fields = {item["field"] for item in body["error"]["details"]}
    assert "strategy.params.tp_sl" in fields


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


def test_run_detail_not_found(tmp_path, monkeypatch) -> None:
    client = _setup_db(tmp_path, monkeypatch)

    resp = client.get("/runs/missing_run")
    assert resp.status_code == 404


def test_runs_capabilities_returns_dca_runtime_matrix(tmp_path, monkeypatch) -> None:
    client = _setup_db(tmp_path, monkeypatch)

    resp = client.get("/runs/capabilities", params={"spec_type": "dca"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["spec_type"] == "dca"
    assert body["catalog_version"] == "2026-02-02"
    assert "fields" in body
    assert "supported" in body["fields"]
    assert "accepted_but_not_wired" in body["fields"]
    assert "universe" in body["fields"]["supported"]
    assert "data.currency" in body["fields"]["supported"]
    assert "strategy.params.asset_class" in body["fields"]["supported"]
    assert "strategy.params.tp_sl" in body["fields"]["supported"]
    assert "performance.stress_tests" in body["fields"]["accepted_but_not_wired"]
    assert body["presets"]["supported"]["strategy.grid"] == ["grid_balanced"]
    assert "grid_conservative" in body["presets"]["not_supported"]["strategy.grid"]
    assert "grid_aggressive" in body["presets"]["not_supported"]["strategy.grid"]
    assert "filters" in body
    assert "supported_ids" in body["filters"]
    assert "ema_slope" in body["filters"]["supported_ids"]
    assert "trend" in body["filters"]["supported_ids"]
    assert body["filters"]["rules_modes"] == ["hard", "soft"]
    assert "runtime_rules" in body
    assert body["runtime_rules"]["multi_symbol"]["execution_scope"] == "per_symbol"
    assert "missing_data_behavior" in body["runtime_rules"]["multi_symbol"]
    assert "legacy_dca" in body
    assert "universe" in body["legacy_dca"]["fields"]["supported"]
    assert "universe" in body["legacy_dca"]["fields"]["not_in_canonical"]
    assert "strategy.strategy_id" in body["legacy_dca"]["fields"]["not_in_canonical"]
    assert "canonical_passthrough_supported" in body["legacy_dca"]["fields"]
    assert "strategy.params.asset_class" in body["legacy_dca"]["fields"]["canonical_passthrough_supported"]
    assert body["resolution"]["dca_symbol_source_priority"] == ["universe", "data.symbol"]
    assert body["deprecations"]["data.symbol"]["status"] == "deprecated"
    assert body["deprecations"]["data.symbol"]["recommended_replacement"] == "universe[]"


def test_runs_capabilities_rejects_unknown_spec_type(tmp_path, monkeypatch) -> None:
    client = _setup_db(tmp_path, monkeypatch)

    resp = client.get("/runs/capabilities", params={"spec_type": "unknown_type"})

    assert resp.status_code == 422
    body = resp.json()
    assert body["errors"][0]["field"] == "spec_type"
    assert body["errors"][0]["code"] == "unsupported_spec_type"


def test_runs_capabilities_returns_market_stats_runtime_matrix(tmp_path, monkeypatch) -> None:
    client = _setup_db(tmp_path, monkeypatch)

    resp = client.get("/runs/capabilities", params={"spec_type": "market_stats"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["spec_type"] == "market_stats"
    assert "data.asset_class" in body["fields"]["supported"]
    assert "data.currency" in body["fields"]["supported"]
    assert "data.symbols" in body["fields"]["supported"]
    assert "stats.validation" in body["fields"]["supported"]
    assert body["runtime_rules"]["symbol_resolution"] == "data.symbols has priority over data.symbol"
    assert body["runtime_rules"]["execution_status"] == "partially_wired"


def test_runs_capabilities_returns_seasonality_runtime_matrix(tmp_path, monkeypatch) -> None:
    client = _setup_db(tmp_path, monkeypatch)

    resp = client.get("/runs/capabilities", params={"spec_type": "seasonality"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["spec_type"] == "seasonality"
    assert "data.asset_class" in body["fields"]["supported"]
    assert "data.currency" in body["fields"]["supported"]
    assert "data.symbols" in body["fields"]["supported"]
    assert "seasonality.profile" in body["fields"]["supported"]
    assert body["runtime_rules"]["symbol_resolution"] == "data.symbols has priority over data.symbol"
    assert body["runtime_rules"]["execution_status"] == "partially_wired"


def test_runs_capabilities_returns_backtest_runtime_matrix(tmp_path, monkeypatch) -> None:
    client = _setup_db(tmp_path, monkeypatch)

    resp = client.get("/runs/capabilities", params={"spec_type": "backtest"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["spec_type"] == "backtest"
    assert body["catalog_version"] == "2026-02-02"
    assert "signal" in body["fields"]["supported"]
    assert "data.currency" in body["fields"]["supported"]
    assert "filters.rules" in body["fields"]["supported"]
    assert "strategy.params.tp_sl" in body["fields"]["supported"]
    assert "strategy.name" in body["fields"]["accepted_but_not_wired"]
    assert body["runtime_rules"]["execution_status"] == "partially_wired"
    assert body["runtime_rules"]["data_source_resolution"]["mode"] == "auto_when_no_explicit_source"
    assert body["runtime_rules"]["data_source_resolution"]["order"] == ["delta", "mysql", "java"]


def test_run_result_returns_canonical_backtest_unwired_tp_sl_error(tmp_path, monkeypatch) -> None:
    client = _setup_db(tmp_path, monkeypatch)
    payload = _canonical_payload()
    payload["strategy"] = {"params": {"tp_sl": "tp_2_sl_1"}}
    response = api_app.enqueue_run_request(payload)

    worker_module.process_next_job()

    resp = client.get(f"/runs/{response.run_id}/result")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == api_app.JOB_STATUS_FAILED_CANONICAL
    assert body["error"]["code"] == "not_implemented_feature"
    fields = {item["field"] for item in body["error"]["details"]}
    assert "strategy.params.tp_sl" in fields
