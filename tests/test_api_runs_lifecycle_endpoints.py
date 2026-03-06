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
    captured_spec = {}

    def _fake_run_backtest(spec):
        captured_spec.update(spec)
        return {"trades": [], "metrics": {"n_trades": 0}, "symbol": spec["data"]["symbol"]}

    monkeypatch.setattr(api_app.backtest_runner, "run_backtest_from_spec", _fake_run_backtest)
    payload = _canonical_payload()
    payload["performance"] = {
        "initial_capital": 25_000,
        "risk_free_rate_pct": 2.0,
        "stress_tests": {
            "enabled": True,
            "source": "returns",
            "method": "block_bootstrap",
            "n_sims": 250,
            "block_size": 8,
            "overlapping": True,
        },
    }
    response = api_app.enqueue_run_request(payload)
    worker_module.process_next_job()

    resp = client.get(f"/runs/{response.run_id}/result")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == api_app.JOB_STATUS_SUCCEEDED
    assert body["result"]["accepted"] is True
    assert body["result"]["spec_type"] == "backtest"
    assert body["result"]["result"]["symbol"] == "EURUSD"
    assert captured_spec.get("performance", {}).get("initial_capital") == 25_000
    assert captured_spec.get("performance", {}).get("risk_free_pct") == 2.0
    assert captured_spec.get("performance", {}).get("stress_tests", {}).get("monte_carlo", {}).get("n_sims") == 250


def test_run_result_returns_canonical_dca_success_payload(tmp_path, monkeypatch) -> None:
    client = _setup_db(tmp_path, monkeypatch)
    captured_spec = {}

    def _fake_backtest(spec):
        captured_spec.update(spec)
        return {"result": {"counts": {"BTCUSD": 2}}, "payload": {"run": {"status": "ok"}}}

    monkeypatch.setattr(api_app.strategies_runner, "run_backtest_with_payload", _fake_backtest)
    payload = _canonical_dca_payload()
    payload["performance"] = {
        "initial_capital": 11_000,
        "capital_per_unit": 125,
        "stress_tests": {
            "enabled": True,
            "method": "bootstrap",
            "n_sims": 120,
            "block_size": 5,
            "output": {"mode": "summary"},
        },
    }
    response = api_app.enqueue_run_request(payload)
    worker_module.process_next_job()

    resp = client.get(f"/runs/{response.run_id}/result")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == api_app.JOB_STATUS_SUCCEEDED
    assert payload["result"]["accepted"] is True
    assert payload["result"]["spec_type"] == "dca"
    assert payload["result"]["result"]["counts"]["BTCUSD"] == 2
    assert captured_spec.get("performance", {}).get("initial_capital") == 11_000
    assert captured_spec.get("performance", {}).get("capital_per_unit") == 125
    assert captured_spec.get("performance", {}).get("stress_tests", {}).get("monte_carlo", {}).get("n_sims") == 120


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
    assert "performance.stress_tests" in body["fields"]["supported"]
    assert body["presets"]["supported"]["strategy.grid"] == ["grid_balanced"]
    assert "tp_sl.trailing(percent)" in body["presets"]["supported"]["strategy.params.tp_sl"]
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
    assert body["fields"]["accepted_but_not_wired"] == ["data.session", "data.include_weekends"]
    assert body["runtime_rules"]["symbol_resolution"] == "data.symbols has priority over data.symbol"
    assert body["runtime_rules"]["execution_status"] == "wired"
    assert body["stats_pack_catalog"]


def test_runs_capabilities_returns_seasonality_runtime_matrix(tmp_path, monkeypatch) -> None:
    client = _setup_db(tmp_path, monkeypatch)

    resp = client.get("/runs/capabilities", params={"spec_type": "seasonality"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["spec_type"] == "seasonality"
    assert "data.asset_class" in body["fields"]["supported"]
    assert "data.currency" in body["fields"]["supported"]
    assert "data.symbols" in body["fields"]["supported"]
    assert "data.start_date" in body["fields"]["supported"]
    assert "data.end_date" in body["fields"]["supported"]
    assert "seasonality.profile" in body["fields"]["supported"]
    assert "seasonality.execution" in body["fields"]["supported"]
    assert "seasonality.risk" in body["fields"]["supported"]
    assert "seasonality.tp_sl" in body["fields"]["supported"]
    assert "seasonality.execution" in body["fields"]["accepted_but_not_wired"]
    assert "seasonality.risk" in body["fields"]["accepted_but_not_wired"]
    assert "seasonality.tp_sl" in body["fields"]["accepted_but_not_wired"]
    assert body["runtime_rules"]["symbol_resolution"] == "data.symbols has priority over data.symbol"
    assert body["runtime_rules"]["execution_status"] == "partially_wired"
    assert "accepted_but_not_wired" in body["runtime_rules"]["failure_mode"]


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
    assert "signal.type=ema_cross" in body["support_matrix"]["supported"]
    assert "signal.type!=ema_cross" in body["support_matrix"]["not_supported_runtime"]
    assert "legacy ta4j fields outside canonical request model" in body["support_matrix"]["not_supported_contract"]
    assert body["signaling"]["unsupported_contract"]["http_status"] == 422
    assert body["signaling"]["unsupported_runtime"]["error"]["code"] == "not_implemented_feature"
    assert body["signaling"]["unsupported_runtime"]["error"]["details_reason"] == "accepted_but_not_wired"
    assert body["signaling"]["execution_error"]["error"]["code"] == "execution_error"


def test_runs_capabilities_returns_stress_tests_runtime_matrix(tmp_path, monkeypatch) -> None:
    client = _setup_db(tmp_path, monkeypatch)

    resp = client.get("/runs/capabilities", params={"spec_type": "stress_tests"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["spec_type"] == "stress_tests"
    assert "data.base_run_id" in body["fields"]["supported"]
    assert "performance.stress_tests" in body["fields"]["supported"]
    assert body["runtime_rules"]["execution_status"] == "wired"
    assert "trades_completed table" in body["runtime_rules"]["trade_source_priority"]


def test_runs_capabilities_returns_dca_support_matrix_and_signaling(tmp_path, monkeypatch) -> None:
    client = _setup_db(tmp_path, monkeypatch)

    resp = client.get("/runs/capabilities", params={"spec_type": "dca"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["spec_type"] == "dca"
    assert "strategy.grid=grid_balanced or strategy.params.grid[]" in body["support_matrix"]["supported"]
    assert "strategy.grid in {grid_conservative,grid_aggressive}" in body["support_matrix"]["not_supported_runtime"]
    assert "legacy ta4j fields outside canonical request model" in body["support_matrix"]["not_supported_contract"]
    assert body["signaling"]["unsupported_contract"]["http_status"] == 422
    assert body["signaling"]["unsupported_runtime"]["status"] == api_app.JOB_STATUS_FAILED_CANONICAL
    assert body["signaling"]["unsupported_runtime"]["error"]["code"] == "not_implemented_feature"
    assert body["signaling"]["execution_error"]["error"]["code"] == "execution_error"


@pytest.mark.parametrize(
    ("spec_type", "target_spec_type"),
    [
        ("optimize_backtest", "backtest"),
        ("optimize_dca", "dca"),
    ],
)
def test_runs_capabilities_returns_optimization_runtime_matrix(
    tmp_path,
    monkeypatch,
    spec_type: str,
    target_spec_type: str,
) -> None:
    client = _setup_db(tmp_path, monkeypatch)

    resp = client.get("/runs/capabilities", params={"spec_type": spec_type})

    assert resp.status_code == 200
    body = resp.json()
    assert body["spec_type"] == spec_type
    assert "optimization.search_space" in body["fields"]["supported"]
    assert "optimization.objective.metric" in body["fields"]["supported"]
    assert "optimization.budget.max_trials" in body["fields"]["supported"]
    assert "output" in body["fields"]["accepted_but_not_wired"]
    assert body["runtime_rules"]["execution_status"] == "wired"
    assert body["runtime_rules"]["target_spec_type"] == target_spec_type


def test_run_result_returns_canonical_stress_tests_success_payload(tmp_path, monkeypatch) -> None:
    client = _setup_db(tmp_path, monkeypatch)
    base_run_id = "base_run_1"

    base_result = {
        "accepted": True,
        "spec_type": "dca",
        "payload": {
            "trades": [
                {
                    "symbol": "BTC",
                    "entryTimeUtc": "2024-01-01T00:00:00Z",
                    "exitTimeUtc": "2024-01-02T00:00:00Z",
                    "grossPnl": 10.0,
                    "meta": {"r_multiple": 1.2},
                },
                {
                    "symbol": "BTC",
                    "entryTimeUtc": "2024-01-03T00:00:00Z",
                    "exitTimeUtc": "2024-01-04T00:00:00Z",
                    "grossPnl": -5.0,
                    "meta": {"r_multiple": -0.8},
                },
                {
                    "symbol": "BTC",
                    "entryTimeUtc": "2024-01-05T00:00:00Z",
                    "exitTimeUtc": "2024-01-06T00:00:00Z",
                    "grossPnl": 8.0,
                    "meta": {"r_multiple": 0.7},
                },
            ]
        },
    }
    api_app._init_job(base_run_id, api_app.JOB_TYPE_CANONICAL_RUN, payload={"request": {"spec_type": "dca"}})
    api_app._update_job_result(base_run_id, base_result, status=api_app.JOB_STATUS_SUCCEEDED)

    payload = {
        "spec_type": "stress_tests",
        "catalog_version": "v1",
        "data": {"base_run_id": base_run_id},
        "performance": {
            "stress_tests": {
                "enabled": True,
                "method": "bootstrap",
                "n_sims": 12,
                "seed": 42,
            }
        },
    }
    response = api_app.enqueue_run_request(payload)
    worker_module.process_next_job()

    resp = client.get(f"/runs/{response.run_id}/result")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == api_app.JOB_STATUS_SUCCEEDED
    assert body["result"]["accepted"] is True
    assert body["result"]["spec_type"] == "stress_tests"
    assert body["result"]["base_run_id"] == base_run_id
    assert body["result"]["result"]["source"]["trades_count"] == 3
    assert "monte_carlo" in body["result"]["result"]["stress_tests"]


def test_run_result_returns_canonical_optimization_success_payload(tmp_path, monkeypatch) -> None:
    client = _setup_db(tmp_path, monkeypatch)
    observed = {}

    def _fake_optimize_backtest(spec):
        observed["spec"] = spec
        return {
            "trials_path": "",
            "summary": "",
            "best": {"params": {"signal.fast": 10}, "objective": 1.23},
            "total_trials": 2,
        }

    monkeypatch.setattr(api_app.optimize_variants, "run_backtest_optimization", _fake_optimize_backtest)

    payload = {
        "spec_type": "optimize_backtest",
        "catalog_version": "v1",
        "optimization": {
            "base_spec": _canonical_payload(),
            "search_space": {"signal.fast": {"type": "int", "min": 5, "max": 20}},
            "objective": {"metric": "sharpe", "direction": "max"},
            "budget": {"max_trials": 3},
        },
    }
    response = api_app.enqueue_run_request(payload)

    worker_module.process_next_job()

    resp = client.get(f"/runs/{response.run_id}/result")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == api_app.JOB_STATUS_SUCCEEDED
    assert body["result"]["accepted"] is True
    assert body["result"]["spec_type"] == "optimize_backtest"
    assert body["result"]["result"]["objective"]["metric"] == "sharpe"
    assert body["result"]["result"]["best"]["score"] == 1.23
    assert body["result"]["result"]["summary"]["total_trials"] == 0
    assert observed["spec"]["optimization"]["method"] == "random"
    assert observed["spec"]["optimization"]["max_trials"] == 3
    assert observed["spec"]["optimization"]["search_space"]["signal.fast"]["step"] == 1


def test_run_result_returns_canonical_optimization_mixed_trials_payload(tmp_path, monkeypatch) -> None:
    client = _setup_db(tmp_path, monkeypatch)

    def _fake_optimize_dca(_spec):
        return {
            "trials_path": "trials.json",
            "summary": "summary.json",
            "best": {"params": {"strategy.params.grid[0].dd": -10}, "objective": 0.42},
            "total_trials": 3,
        }

    monkeypatch.setattr(api_app.optimize_variants, "run_strategy_optimization", _fake_optimize_dca)
    monkeypatch.setattr(
        api_app,
        "_canonical_optimization_trials",
        lambda _result: [
            {"trial_id": 1, "score": 0.42, "status": "SUCCEEDED", "params": {"a": 1}},
            {"trial_id": 2, "score": None, "status": "FAILED", "params": {"a": 2}},
            {"trial_id": 3, "score": 0.11, "status": "SUCCEEDED", "params": {"a": 3}},
        ],
    )

    payload = {
        "spec_type": "optimize_dca",
        "catalog_version": "v1",
        "optimization": {
            "base_spec": _canonical_dca_payload(),
            "search_space": {"strategy.params.grid[0].dd": {"type": "int", "min": -15, "max": -5}},
            "objective": {"metric": "sharpe", "direction": "max"},
            "budget": {"max_trials": 3},
        },
    }
    response = api_app.enqueue_run_request(payload)
    worker_module.process_next_job()

    resp = client.get(f"/runs/{response.run_id}/result")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == api_app.JOB_STATUS_SUCCEEDED
    summary = body["result"]["result"]["summary"]
    assert summary["total_trials"] == 3
    assert summary["succeeded_trials"] == 2
    assert summary["failed_trials"] == 1


def test_run_result_returns_validation_error_for_canonical_optimization_base_run_mismatch(tmp_path, monkeypatch) -> None:
    client = _setup_db(tmp_path, monkeypatch)
    base_run_id = "base_backtest_1"
    base_request = _canonical_payload()
    api_app._init_job(base_run_id, api_app.JOB_TYPE_CANONICAL_RUN, payload={"request": base_request})
    api_app._update_job_result(base_run_id, {"accepted": True}, status=api_app.JOB_STATUS_SUCCEEDED)

    payload = {
        "spec_type": "optimize_dca",
        "catalog_version": "v1",
        "optimization": {
            "base_run_id": base_run_id,
            "search_space": {"strategy.params.grid[0].dd": {"type": "int", "min": -15, "max": -5}},
            "objective": {"metric": "sharpe", "direction": "max"},
            "budget": {"max_trials": 3},
        },
    }
    response = api_app.enqueue_run_request(payload)
    worker_module.process_next_job()

    resp = client.get(f"/runs/{response.run_id}/result")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == api_app.JOB_STATUS_FAILED_CANONICAL
    assert body["error"]["code"] == "validation_error"
    assert "requires base spec_type=dca" in body["error"]["message"]


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
