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


def _canonical_dca_payload(request_id: str | None = None) -> dict:
    payload = {
        "spec_type": "dca",
        "catalog_version": "v1",
        "data": {
            "symbol": "BTCUSD",
            "timeframe": "H1",
            "start_date": "2025-01-01",
            "end_date": "2025-01-31",
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
    if request_id is not None:
        payload["request_id"] = request_id
    return payload


def _canonical_dca_payload_universe_only() -> dict:
    return {
        "spec_type": "dca",
        "catalog_version": "v1",
        "data": {
            "timeframe": "H1",
            "start_date": "2025-01-01",
            "end_date": "2025-01-31",
        },
        "universe": [{"symbol": "BTCUSDT", "asset_class": "CRYPTO"}],
        "strategy": {
            "type": "dca_equity",
            "grid": [],
            "params": {
                "asset_class": "CRYPTO",
                "grid": [{"dd": -5.0, "weight": 1.0}],
                "execution_mode": "bar_close",
                "drawdown_reference": "ATH",
            },
        },
    }


def _canonical_market_stats_payload_symbols_only() -> dict:
    return {
        "spec_type": "market_stats",
        "catalog_version": "v1",
        "data": {
            "symbols": ["BTCUSDT", "ETHUSDT"],
            "timeframe": "1h",
            "path": "tests/data/ohlcv_ts.csv",
        },
        "stats": {
            "event": {"id": "always_true", "params": {}},
            "condition": {"id": "day_of_week", "params": {}},
            "target": {"id": "up_next_bar", "params": {}},
        },
    }


def _canonical_market_stats_payload_k_consecutive() -> dict:
    return {
        "spec_type": "market_stats",
        "catalog_version": "v1",
        "data": {
            "symbols": ["BTCUSDT"],
            "timeframe": "1d",
            "path": "tests/data/ohlcv_ts.csv",
        },
        "stats": {
            "event": {"id": "k_consecutive", "params": {"k": 2, "direction": "up"}},
            "condition": {"id": "day_of_week", "params": {}},
            "target": {"id": "up_next_bar", "params": {}},
        },
    }


def _canonical_seasonality_payload_symbols_only() -> dict:
    return {
        "spec_type": "seasonality",
        "catalog_version": "v1",
        "data": {
            "symbols": ["SPY", "QQQ"],
            "timeframe": "1d",
            "path": "tests/data/ohlcv_ts.csv",
        },
        "seasonality": {
            "profile": {"id": "by_hour"},
            "signal": {"method": "threshold"},
        },
    }


def _canonical_optimize_backtest_payload() -> dict:
    return {
        "spec_type": "optimize_backtest",
        "catalog_version": "v1",
        "optimization": {
            "base_spec": _canonical_backtest_payload(),
            "search_space": {
                "signal.fast": {"type": "int", "min": 5, "max": 20},
                "signal.slow": {"type": "int", "min": 21, "max": 80},
            },
            "objective": {"metric": "sharpe", "direction": "max"},
            "budget": {"max_trials": 10, "seed": 7},
        },
    }


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


def test_runs_submit_enqueues_dca_universe_only_request(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DB_SQLITE_PATH", str(tmp_path / "quant.db"))
    reset_settings_cache()
    client = TestClient(api_app.fastapi_app)

    response = client.post("/runs", json=_canonical_dca_payload_universe_only())

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "QUEUED"


def test_runs_submit_enqueues_dca_request(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DB_SQLITE_PATH", str(tmp_path / "quant.db"))
    reset_settings_cache()
    client = TestClient(api_app.fastapi_app)

    response = client.post("/runs", json=_canonical_dca_payload())

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "QUEUED"
    assert payload["reused"] is False
    assert isinstance(payload["run_id"], str)
    assert payload["run_id"]


def test_runs_submit_enqueues_market_stats_symbols_only_request(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DB_SQLITE_PATH", str(tmp_path / "quant.db"))
    reset_settings_cache()
    client = TestClient(api_app.fastapi_app)

    response = client.post("/runs", json=_canonical_market_stats_payload_symbols_only())

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "QUEUED"


def test_runs_submit_enqueues_seasonality_symbols_only_request(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DB_SQLITE_PATH", str(tmp_path / "quant.db"))
    reset_settings_cache()
    client = TestClient(api_app.fastapi_app)

    response = client.post("/runs", json=_canonical_seasonality_payload_symbols_only())

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "QUEUED"


def test_runs_submit_enqueues_optimize_backtest_request(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DB_SQLITE_PATH", str(tmp_path / "quant.db"))
    reset_settings_cache()
    client = TestClient(api_app.fastapi_app)

    response = client.post("/runs", json=_canonical_optimize_backtest_payload())

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "QUEUED"


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


def test_runs_submit_rejects_top_level_screening_field(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DB_SQLITE_PATH", str(tmp_path / "quant.db"))
    reset_settings_cache()
    client = TestClient(api_app.fastapi_app)
    payload = _canonical_backtest_payload()
    payload["screening"] = {"enabled": True}

    response = client.post("/runs", json=payload)

    assert response.status_code == 422
    errors = response.json()["errors"]
    assert any(err["field"] == "backtest.screening" for err in errors)


def test_runs_submit_rejects_market_stats_missing_k_consecutive_params(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DB_SQLITE_PATH", str(tmp_path / "quant.db"))
    reset_settings_cache()
    client = TestClient(api_app.fastapi_app)
    payload = _canonical_market_stats_payload_k_consecutive()
    payload["stats"]["event"]["params"] = {"k": 2}

    response = client.post("/runs", json=payload)

    assert response.status_code == 422
    errors = response.json()["errors"]
    assert any(err["field"] == "market_stats.stats.event.params.direction" for err in errors)


def test_runs_submit_rejects_market_stats_invalid_htf_trend_params(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DB_SQLITE_PATH", str(tmp_path / "quant.db"))
    reset_settings_cache()
    client = TestClient(api_app.fastapi_app)
    payload = _canonical_market_stats_payload_k_consecutive()
    payload["stats"]["condition"] = {
        "id": "htf_trend",
        "params": {"tf_multiplier": 0, "ema_period": 0},
    }

    response = client.post("/runs", json=payload)

    assert response.status_code == 422
    errors = response.json()["errors"]
    fields = {err["field"] for err in errors}
    assert "market_stats.stats.condition.params.tf_multiplier" in fields
    assert "market_stats.stats.condition.params.ema_period" in fields


def test_runs_submit_rejects_optimize_backtest_empty_search_space(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DB_SQLITE_PATH", str(tmp_path / "quant.db"))
    reset_settings_cache()
    client = TestClient(api_app.fastapi_app)
    payload = _canonical_optimize_backtest_payload()
    payload["optimization"]["search_space"] = {}

    response = client.post("/runs", json=payload)

    assert response.status_code == 422
    errors = response.json()["errors"]
    assert errors


def test_submit_run_accepts_currency_strength_features_block(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_SQLITE_PATH", str(tmp_path / "quant.db"))
    reset_settings_cache()
    client = TestClient(api_app.fastapi_app)

    payload = _canonical_backtest_payload()
    payload["features"] = {
        "currency_strength": {
            "enabled": True,
            "lookback": 72,
            "majors": ["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"],
        }
    }

    resp = client.post("/runs", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "queued"
