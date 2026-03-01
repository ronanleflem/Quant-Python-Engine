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
            "currency": "USD",
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
            "currency": "USDT",
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


def _canonical_dca_payload_universe() -> dict:
    payload = _canonical_dca_payload()
    payload["universe"] = [
        {"symbol": "ETHUSDT", "asset_class": "CRYPTO"},
        {"symbol": "BTCUSDT", "asset_class": "CRYPTO", "currency": "USDC"},
    ]
    return payload


def _canonical_market_stats_payload() -> dict:
    return {
        "spec_type": "market_stats",
        "catalog_version": "v1",
        "data": {
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "path": "tests/data/ohlcv_ts.csv",
            "lookback": 200,
            "stats_pack": "Volatility",
            "session": "Full",
            "include_weekends": True,
            "asset_class": "CRYPTO",
            "currency": "USDT",
        },
        "stats": {
            "event": {"id": "always_true", "params": {}},
            "condition": {"id": "day_of_week", "params": {}},
            "target": {"id": "up_next_bar", "params": {}},
            "validation": {"train_months": 6, "test_months": 2, "folds": 2, "embargo_days": 0},
        },
        "output": {"out_dir": "runs/stats_canonical"},
        "persistence": {"enabled": False},
    }


def _canonical_market_stats_payload_with_symbols_priority() -> dict:
    payload = _canonical_market_stats_payload()
    payload["data"]["symbol"] = "BTCUSDT"
    payload["data"]["symbols"] = ["ETHUSDT", "BTCUSDT"]
    return payload


def _canonical_seasonality_payload() -> dict:
    return {
        "spec_type": "seasonality",
        "catalog_version": "v1",
        "data": {
            "symbol": "SPY",
            "timeframe": "1d",
            "path": "tests/data/ohlcv_ts.csv",
            "window": "Monthly",
            "start_year": 2022,
            "end_year": 2024,
            "asset_class": "EQUITY",
            "currency": "USD",
        },
        "seasonality": {
            "profile": {"id": "by_hour", "measure": "avg_return", "ret_horizon": 5, "min_samples_bin": 50, "params": {}},
            "signal": {"method": "zscore", "threshold": 1.2, "topk": 5, "dims": ["hour"], "combine": "mean"},
            "compute": {"max_trials": 20, "search_space": {}},
            "execution": {"risk_model": "fixed_fraction", "tp_sl": "tp_2_sl_1"},
        },
        "output": {"out_dir": "runs/seasonality_canonical"},
        "persistence": {"enabled": False},
    }


def _canonical_seasonality_payload_with_symbols_priority() -> dict:
    payload = _canonical_seasonality_payload()
    payload["data"]["symbol"] = "SPY"
    payload["data"]["symbols"] = ["QQQ", "SPY"]
    return payload


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
    assert "strategy.name" in fields


def test_worker_processes_canonical_backtest_with_runner(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DB_SQLITE_PATH", str(tmp_path / "quant.db"))
    reset_settings_cache()
    observed = {}

    def _fake_run_backtest(spec):
        observed["spec"] = spec
        return {"trades": [], "metrics": {"n_trades": 0}}

    monkeypatch.setattr(api_app.backtest_runner, "run_backtest_from_spec", _fake_run_backtest)

    response = api_app.enqueue_run_request(_canonical_payload())
    result = worker_module.process_next_job()

    assert result is not None
    job = api_app._get_job(response.run_id)
    assert job["status"] == api_app.JOB_STATUS_SUCCEEDED
    assert observed["spec"]["data"]["symbol"] == "EURUSD"
    assert observed["spec"]["data"]["start"] == "2025-01-01"
    assert observed["spec"]["signal"]["type"] == "ema_cross"
    assert observed["spec"]["signal"]["params"]["fast"] == 9
    assert observed["spec"]["signal"]["params"]["slow"] == 21
    assert observed["spec"]["data"]["currency"] == "USD"
    assert observed["spec"]["data"]["delta_quotes"] == "USD"
    assert observed["spec"]["data"]["mysql_env"] == "QE_MARKETDATA_MYSQL_URL"


def test_worker_maps_canonical_backtest_filters_and_tp_sl(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DB_SQLITE_PATH", str(tmp_path / "quant.db"))
    reset_settings_cache()
    observed = {}
    payload = _canonical_payload()
    payload["filters"] = {
        "filters": [{"id": "ema_slope", "params": {"period": 20}}],
        "rules": [{"id": "momentum_alignment", "mode": "soft", "weight": 0.6}],
        "rules_config": {"min_score": 60, "min_score_pct": 70},
    }
    payload["strategy"] = {
        "params": {
            "asset_class": "CRYPTO",
            "tp_sl": {
                "atr_window": 14,
                "atr_k": 2.0,
                "r_mult": 1.5,
                "slippage_bps": 5,
                "fee_bps": 2,
            }
        }
    }

    def _fake_run_backtest(spec):
        observed["spec"] = spec
        return {"trades": [], "metrics": {"n_trades": 0}}

    monkeypatch.setattr(api_app.backtest_runner, "run_backtest_from_spec", _fake_run_backtest)

    response = api_app.enqueue_run_request(payload)
    result = worker_module.process_next_job()

    assert result is not None
    job = api_app._get_job(response.run_id)
    assert job["status"] == api_app.JOB_STATUS_SUCCEEDED
    assert observed["spec"]["filters"][0]["type"] == "ema_slope"
    assert observed["spec"]["filter_rules"][0]["type"] == "momentum_alignment"
    assert observed["spec"]["filter_rules_config"]["min_score"] == 60
    assert observed["spec"]["strategy"]["asset_class"] == "CRYPTO"
    assert observed["spec"]["tpsl"]["atr_window"] == 14
    assert observed["spec"]["tpsl"]["r_mult"] == 1.5


def test_worker_marks_canonical_backtest_not_implemented_for_unwired_tp_sl(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DB_SQLITE_PATH", str(tmp_path / "quant.db"))
    reset_settings_cache()
    payload = _canonical_payload()
    payload["strategy"] = {"params": {"tp_sl": "tp_2_sl_1"}}
    response = api_app.enqueue_run_request(payload)

    result = worker_module.process_next_job()

    assert result is None
    job = api_app._get_job(response.run_id)
    assert job["status"] == api_app.JOB_STATUS_FAILED_CANONICAL
    error = job["result"]["error"]
    assert error["code"] == "not_implemented_feature"
    fields = {item["field"] for item in error["details"]}
    assert "strategy.params.tp_sl" in fields


def test_worker_processes_canonical_dca_with_strategy_runner(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DB_SQLITE_PATH", str(tmp_path / "quant.db"))
    reset_settings_cache()

    observed = {}

    def _fake_backtest(spec):
        observed["spec"] = spec
        return {"result": {"counts": {"BTCUSD": 3}}, "payload": {"run": {"status": "ok"}}}

    monkeypatch.setattr(api_app.strategies_runner, "run_backtest_with_payload", _fake_backtest)

    response = api_app.enqueue_run_request(_canonical_dca_payload())
    result = worker_module.process_next_job()

    assert result is not None
    job = api_app._get_job(response.run_id)
    assert job["status"] == api_app.JOB_STATUS_SUCCEEDED
    assert observed["spec"]["strategy"]["type"] == "dca_equity"
    assert observed["spec"]["universe"][0]["symbol"] == "BTCUSD"
    assert job["result"]["accepted"] is True
    assert job["result"]["spec_type"] == "dca"
    assert job["result"]["result"]["counts"]["BTCUSD"] == 3


def test_worker_processes_canonical_market_stats_with_runner(tmp_path, monkeypatch) -> None:
    import pandas as pd

    monkeypatch.setenv("DB_SQLITE_PATH", str(tmp_path / "quant.db"))
    reset_settings_cache()

    observed = {}

    def _fake_run_stats(spec):
        observed["spec"] = spec
        return pd.DataFrame(
            [
                {
                    "symbol": "BTCUSDT",
                    "event": "always_true",
                    "condition_name": "day_of_week",
                    "condition_value": "1",
                    "target": "up_next_bar",
                    "n": 10,
                    "successes": 6,
                    "p_hat": 0.6,
                }
            ]
        )

    monkeypatch.setattr(api_app.stats_runner, "run_stats", _fake_run_stats)

    response = api_app.enqueue_run_request(_canonical_market_stats_payload())
    result = worker_module.process_next_job()

    assert result is not None
    job = api_app._get_job(response.run_id)
    assert job["status"] == api_app.JOB_STATUS_SUCCEEDED
    assert job["result"]["accepted"] is True
    assert job["result"]["spec_type"] == "market_stats"
    assert job["result"]["result"]["columns"]
    assert job["result"]["result"]["rows"]
    assert observed["spec"].data.symbols == ["BTCUSDT"]
    assert observed["spec"].data.dataset_path == "tests/data/ohlcv_ts.csv"
    assert observed["spec"].validation is not None
    assert observed["spec"].validation.train_months == 6


def test_worker_processes_canonical_seasonality_with_runner(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DB_SQLITE_PATH", str(tmp_path / "quant.db"))
    reset_settings_cache()

    observed = {}

    def _fake_run_seasonality(spec):
        observed["spec"] = spec
        return {"summary": {"n_profiles": 2}, "profiles": []}

    monkeypatch.setattr(api_app.seasonality_runner, "run", _fake_run_seasonality)

    response = api_app.enqueue_run_request(_canonical_seasonality_payload())
    result = worker_module.process_next_job()

    assert result is not None
    job = api_app._get_job(response.run_id)
    assert job["status"] == api_app.JOB_STATUS_SUCCEEDED
    assert job["result"]["accepted"] is True
    assert job["result"]["spec_type"] == "seasonality"
    assert job["result"]["result"]["summary"]["n_profiles"] == 2
    assert observed["spec"].data.symbols == ["SPY"]
    assert observed["spec"].data.dataset_path == "tests/data/ohlcv_ts.csv"
    assert observed["spec"].profile.by_hour is True
    assert observed["spec"].signal.method == "threshold"
    assert observed["spec"].signal.combine == "sum"


def test_worker_uses_symbols_over_symbol_for_canonical_market_stats(tmp_path, monkeypatch) -> None:
    import pandas as pd

    monkeypatch.setenv("DB_SQLITE_PATH", str(tmp_path / "quant.db"))
    reset_settings_cache()
    observed = {}

    def _fake_run_stats(spec):
        observed["spec"] = spec
        return pd.DataFrame(
            [
                {
                    "symbol": "ETHUSDT",
                    "event": "always_true",
                    "condition_name": "day_of_week",
                    "condition_value": "1",
                    "target": "up_next_bar",
                    "n": 10,
                    "successes": 6,
                    "p_hat": 0.6,
                }
            ]
        )

    monkeypatch.setattr(api_app.stats_runner, "run_stats", _fake_run_stats)

    response = api_app.enqueue_run_request(_canonical_market_stats_payload_with_symbols_priority())
    result = worker_module.process_next_job()

    assert result is not None
    job = api_app._get_job(response.run_id)
    assert job["status"] == api_app.JOB_STATUS_SUCCEEDED
    assert observed["spec"].data.symbols == ["ETHUSDT", "BTCUSDT"]


def test_worker_uses_explicit_dates_over_lookback_for_canonical_market_stats(tmp_path, monkeypatch) -> None:
    import pandas as pd

    monkeypatch.setenv("DB_SQLITE_PATH", str(tmp_path / "quant.db"))
    reset_settings_cache()
    observed = {}
    payload = _canonical_market_stats_payload()
    payload["data"]["start_date"] = "2022-01-01T00:00:00.000Z"
    payload["data"]["end_date"] = "2024-12-31T00:00:00.000Z"

    def _fake_run_stats(spec):
        observed["spec"] = spec
        return pd.DataFrame(
            [
                {
                    "symbol": "BTCUSDT",
                    "event": "always_true",
                    "condition_name": "day_of_week",
                    "condition_value": "1",
                    "target": "up_next_bar",
                    "n": 10,
                    "successes": 6,
                    "p_hat": 0.6,
                }
            ]
        )

    monkeypatch.setattr(api_app.stats_runner, "run_stats", _fake_run_stats)

    response = api_app.enqueue_run_request(payload)
    result = worker_module.process_next_job()

    assert result is not None
    job = api_app._get_job(response.run_id)
    assert job["status"] == api_app.JOB_STATUS_SUCCEEDED
    assert observed["spec"].data.start == "2022-01-01T00:00:00+00:00"
    assert observed["spec"].data.end == "2024-12-31T00:00:00+00:00"


def test_worker_uses_symbols_over_symbol_for_canonical_seasonality(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DB_SQLITE_PATH", str(tmp_path / "quant.db"))
    reset_settings_cache()
    observed = {}

    def _fake_run_seasonality(spec):
        observed["spec"] = spec
        return {"summary": {"n_profiles": 2}, "profiles": []}

    monkeypatch.setattr(api_app.seasonality_runner, "run", _fake_run_seasonality)

    response = api_app.enqueue_run_request(_canonical_seasonality_payload_with_symbols_priority())
    result = worker_module.process_next_job()

    assert result is not None
    job = api_app._get_job(response.run_id)
    assert job["status"] == api_app.JOB_STATUS_SUCCEEDED
    assert observed["spec"].data.symbols == ["QQQ", "SPY"]


def test_worker_uses_explicit_dates_over_years_for_canonical_seasonality(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DB_SQLITE_PATH", str(tmp_path / "quant.db"))
    reset_settings_cache()
    observed = {}
    payload = _canonical_seasonality_payload()
    payload["data"]["start_year"] = 2010
    payload["data"]["end_year"] = 2011
    payload["data"]["start_date"] = "2022-01-01T00:00:00.000Z"
    payload["data"]["end_date"] = "2024-12-31T00:00:00.000Z"

    def _fake_run_seasonality(spec):
        observed["spec"] = spec
        return {"summary": {"n_profiles": 2}, "profiles": []}

    monkeypatch.setattr(api_app.seasonality_runner, "run", _fake_run_seasonality)

    response = api_app.enqueue_run_request(payload)
    result = worker_module.process_next_job()

    assert result is not None
    job = api_app._get_job(response.run_id)
    assert job["status"] == api_app.JOB_STATUS_SUCCEEDED
    assert observed["spec"].data.start == "2022-01-01T00:00:00+00:00"
    assert observed["spec"].data.end == "2024-12-31T00:00:00+00:00"


def test_worker_uses_universe_over_data_symbol_for_canonical_dca(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DB_SQLITE_PATH", str(tmp_path / "quant.db"))
    reset_settings_cache()
    payload = _canonical_dca_payload_universe()
    observed = {}

    def _fake_backtest(spec):
        observed["spec"] = spec
        return {"result": {"counts": {"ETHUSDT": 2}}, "payload": {"run": {"status": "ok"}}}

    monkeypatch.setattr(api_app.strategies_runner, "run_backtest_with_payload", _fake_backtest)

    response = api_app.enqueue_run_request(payload)
    result = worker_module.process_next_job()

    assert result is not None
    job = api_app._get_job(response.run_id)
    assert job["status"] == api_app.JOB_STATUS_SUCCEEDED
    symbols = [item["symbol"] for item in observed["spec"]["universe"]]
    assert symbols == ["ETHUSDT", "BTCUSDT"]
    assert "BTCUSD" not in symbols
    by_symbol = {item["symbol"]: item for item in observed["spec"]["universe"]}
    assert by_symbol["ETHUSDT"]["currency"] == "USDT"
    assert by_symbol["BTCUSDT"]["currency"] == "USDC"
    assert observed["spec"]["data"]["currency"] == "USDT"
    assert observed["spec"]["data"]["delta_quotes"] == "USDT"


def test_worker_logs_deprecation_warning_when_using_data_symbol_fallback(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("DB_SQLITE_PATH", str(tmp_path / "quant.db"))
    reset_settings_cache()
    payload = _canonical_dca_payload()

    def _fake_backtest(spec):
        return {"result": {"counts": {"BTCUSD": 1}}, "payload": {"run": {"status": "ok"}}}

    monkeypatch.setattr(api_app.strategies_runner, "run_backtest_with_payload", _fake_backtest)

    response = api_app.enqueue_run_request(payload)
    result = worker_module.process_next_job()

    assert result is not None
    job = api_app._get_job(response.run_id)
    assert job["status"] == api_app.JOB_STATUS_SUCCEEDED
    captured = capsys.readouterr()
    assert "canonical_dca_deprecation_warning" in captured.out
    assert '"field":"data.symbol"' in captured.out


def test_worker_processes_canonical_dca_multisymbol_universe_with_real_runner(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DB_SQLITE_PATH", str(tmp_path / "quant.db"))
    reset_settings_cache()
    payload = {
        "spec_type": "dca",
        "catalog_version": "v1",
        "data": {
            "timeframe": "1D",
            "start_date": "2025-01-01",
            "end_date": "2025-01-20",
            "path": "tests/data/ohlcv_ts.csv",
        },
        "universe": [
            {"symbol": "SPY", "asset_class": "EQUITY"},
            {"symbol": "QQQ", "asset_class": "EQUITY"},
        ],
        "strategy": {
            "type": "dca_equity",
            "grid": [],
            "params": {
                "asset_class": "EQUITY",
                "drawdown_reference": "ATH",
                "execution_mode": "bar_close",
                "grid": [{"dd": -5.0, "weight": 1.0}],
                "require_crossing": False,
            },
        },
    }

    response = api_app.enqueue_run_request(payload)
    result = worker_module.process_next_job()

    assert result is not None
    job = api_app._get_job(response.run_id)
    assert job["status"] == api_app.JOB_STATUS_SUCCEEDED
    counts = job["result"]["result"]["counts"]
    assert set(counts.keys()) == {"SPY", "QQQ"}


def test_worker_processes_canonical_dca_with_tp_sl_preset(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DB_SQLITE_PATH", str(tmp_path / "quant.db"))
    reset_settings_cache()
    payload = _canonical_dca_payload()
    payload["strategy"]["params"]["tp_sl"] = "tp_2_sl_1"
    observed = {}

    def _fake_backtest(spec):
        observed["spec"] = spec
        return {"result": {"counts": {"BTCUSD": 1}}, "payload": {"run": {"status": "ok"}}}

    monkeypatch.setattr(api_app.strategies_runner, "run_backtest_with_payload", _fake_backtest)

    response = api_app.enqueue_run_request(payload)
    result = worker_module.process_next_job()

    assert result is not None
    job = api_app._get_job(response.run_id)
    assert job["status"] == api_app.JOB_STATUS_SUCCEEDED
    tp_sl = observed["spec"]["strategy"]["params"]["tp_sl"]
    assert tp_sl["mode"] == "per_grid_max_dd"
    assert tp_sl["rules"][0]["tp_pct"] == 2.0
    assert tp_sl["sl_dd"] == -1.0


def test_worker_processes_canonical_dca_with_explicit_tp_sl_object(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DB_SQLITE_PATH", str(tmp_path / "quant.db"))
    reset_settings_cache()
    payload = _canonical_dca_payload()
    payload["strategy"]["params"]["tp_sl"] = {
        "enabled": True,
        "mode": "rule_based",
        "tp": {"type": "percent", "value": 2.5},
        "sl": {"type": "percent", "value": 1.5},
        "break_even": {"enabled": True, "trigger_pct": 0.8},
    }
    observed = {}

    def _fake_backtest(spec):
        observed["spec"] = spec
        return {"result": {"counts": {"BTCUSD": 1}}, "payload": {"run": {"status": "ok"}}}

    monkeypatch.setattr(api_app.strategies_runner, "run_backtest_with_payload", _fake_backtest)

    response = api_app.enqueue_run_request(payload)
    result = worker_module.process_next_job()

    assert result is not None
    job = api_app._get_job(response.run_id)
    assert job["status"] == api_app.JOB_STATUS_SUCCEEDED
    tp_sl = observed["spec"]["strategy"]["params"]["tp_sl"]
    assert tp_sl["mode"] == "per_grid_max_dd"
    assert tp_sl["rules"][0]["tp_pct"] == 2.5
    assert tp_sl["rules"][0]["be_pct"] == 0.8
    assert tp_sl["sl_dd"] == -1.5


def test_worker_processes_canonical_dca_with_explicit_tp_sl_trailing_object(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DB_SQLITE_PATH", str(tmp_path / "quant.db"))
    reset_settings_cache()
    payload = _canonical_dca_payload()
    payload["strategy"]["params"]["tp_sl"] = {
        "enabled": True,
        "mode": "rule_based",
        "tp": {"type": "percent", "value": 2.5},
        "sl": {"type": "percent", "value": 1.5},
        "trailing": {"enabled": True, "type": "percent", "value": 1.0, "trigger_pct": 1.2},
    }
    observed = {}

    def _fake_backtest(spec):
        observed["spec"] = spec
        return {"result": {"counts": {"BTCUSD": 1}}, "payload": {"run": {"status": "ok"}}}

    monkeypatch.setattr(api_app.strategies_runner, "run_backtest_with_payload", _fake_backtest)

    response = api_app.enqueue_run_request(payload)
    result = worker_module.process_next_job()

    assert result is not None
    job = api_app._get_job(response.run_id)
    assert job["status"] == api_app.JOB_STATUS_SUCCEEDED
    tp_sl = observed["spec"]["strategy"]["params"]["tp_sl"]
    assert tp_sl["trailing"]["type"] == "percent"
    assert tp_sl["trailing"]["value"] == 1.0
    assert tp_sl["trailing"]["trigger_pct"] == 1.2


def test_worker_marks_canonical_dca_not_implemented_for_invalid_trailing_tp_sl(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DB_SQLITE_PATH", str(tmp_path / "quant.db"))
    reset_settings_cache()
    payload = _canonical_dca_payload()
    payload["strategy"]["params"]["tp_sl"] = {
        "enabled": True,
        "mode": "rule_based",
        "tp": {"type": "percent", "value": 2.5},
        "sl": {"type": "percent", "value": 1.5},
        "trailing": {"enabled": True, "type": "atr", "value": 1.0},
    }

    response = api_app.enqueue_run_request(payload)
    result = worker_module.process_next_job()

    assert result is not None
    job = api_app._get_job(response.run_id)
    assert job["status"] == api_app.JOB_STATUS_FAILED
    error = job["result"]["error"]
    assert error["code"] == "not_implemented_feature"
    fields = [item["field"] for item in error["details"]]
    assert "strategy.params.tp_sl" in fields


def test_worker_processes_canonical_dca_with_grid_preset_and_rolling_high(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DB_SQLITE_PATH", str(tmp_path / "quant.db"))
    reset_settings_cache()
    payload = _canonical_dca_payload()
    payload["strategy"]["grid"] = ["grid_balanced"]
    payload["strategy"]["params"].pop("grid", None)
    payload["strategy"]["params"]["drawdown_reference"] = "rolling_high"
    observed = {}

    def _fake_backtest(spec):
        observed["spec"] = spec
        return {"result": {"counts": {"BTCUSD": 1}}, "payload": {"run": {"status": "ok"}}}

    monkeypatch.setattr(api_app.strategies_runner, "run_backtest_with_payload", _fake_backtest)

    response = api_app.enqueue_run_request(payload)
    result = worker_module.process_next_job()

    assert result is not None
    job = api_app._get_job(response.run_id)
    assert job["status"] == api_app.JOB_STATUS_SUCCEEDED
    params = observed["spec"]["strategy"]["params"]
    assert isinstance(params["grid"], list)
    assert len(params["grid"]) == 3
    assert params["drawdown_reference"] == "90D"


def test_worker_maps_canonical_filters_id_to_internal_type(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DB_SQLITE_PATH", str(tmp_path / "quant.db"))
    reset_settings_cache()
    payload = _canonical_dca_payload()
    payload["filters"] = {"filters": [{"id": "ema_slope", "params": {"period": 20}}]}
    observed = {}

    def _fake_backtest(spec):
        observed["spec"] = spec
        return {"result": {"counts": {"BTCUSD": 1}}, "payload": {"run": {"status": "ok"}}}

    monkeypatch.setattr(api_app.strategies_runner, "run_backtest_with_payload", _fake_backtest)

    response = api_app.enqueue_run_request(payload)
    result = worker_module.process_next_job()

    assert result is not None
    job = api_app._get_job(response.run_id)
    assert job["status"] == api_app.JOB_STATUS_SUCCEEDED
    assert observed["spec"]["filters"][0]["type"] == "ema_slope"
    assert observed["spec"]["filters"][0]["params"]["period"] == 20


def test_worker_maps_canonical_filter_rules_id_to_internal_type(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DB_SQLITE_PATH", str(tmp_path / "quant.db"))
    reset_settings_cache()
    payload = _canonical_dca_payload()
    payload["filters"] = {
        "filters": [],
        "rules": [{"id": "momentum_alignment", "mode": "soft", "weight": 0.6}],
        "rules_config": {"min_score": 60, "min_score_pct": 70},
    }
    observed = {}

    def _fake_backtest(spec):
        observed["spec"] = spec
        return {"result": {"counts": {"BTCUSD": 1}}, "payload": {"run": {"status": "ok"}}}

    monkeypatch.setattr(api_app.strategies_runner, "run_backtest_with_payload", _fake_backtest)

    response = api_app.enqueue_run_request(payload)
    result = worker_module.process_next_job()

    assert result is not None
    job = api_app._get_job(response.run_id)
    assert job["status"] == api_app.JOB_STATUS_SUCCEEDED
    assert observed["spec"]["filter_rules"][0]["type"] == "momentum_alignment"
    assert observed["spec"]["filter_rules"][0]["mode"] == "soft"
    assert observed["spec"]["filter_rules"][0]["weight"] == 0.6


def test_worker_marks_canonical_dca_not_implemented_for_unwired_fields(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DB_SQLITE_PATH", str(tmp_path / "quant.db"))
    reset_settings_cache()
    payload = _canonical_dca_payload()
    payload["strategy"]["grid"] = ["grid_custom"]
    payload["strategy"]["params"].pop("grid", None)
    payload["strategy"]["params"]["execution_mode"] = "limit"
    payload["strategy"]["params"]["drawdown_reference"] = "ATH"
    payload["strategy"]["params"]["tp_sl"] = "tp_custom"

    response = api_app.enqueue_run_request(payload)
    result = worker_module.process_next_job()

    assert result is None
    job = api_app._get_job(response.run_id)
    assert job["status"] == api_app.JOB_STATUS_FAILED_CANONICAL
    error = job["result"]["error"]
    assert error["code"] == "not_implemented_feature"
    assert error["message"] == "Feature not implemented for canonical dca run"
    fields = {item["field"] for item in error["details"]}
    assert "strategy.grid" in fields
    assert "strategy.params.tp_sl" in fields
    assert "strategy.params.execution_mode" in fields
    assert "strategy.params.drawdown_reference" not in fields
