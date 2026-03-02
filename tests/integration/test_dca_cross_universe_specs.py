from __future__ import annotations

import sys
import types

import pandas as pd

if "pymysql" not in sys.modules:
    pymysql_mod = types.ModuleType("pymysql")
    cursors_mod = types.ModuleType("pymysql.cursors")

    class _DictCursor:  # pragma: no cover - import shim
        pass

    cursors_mod.DictCursor = _DictCursor
    pymysql_mod.cursors = cursors_mod
    sys.modules["pymysql"] = pymysql_mod
    sys.modules["pymysql.cursors"] = cursors_mod

from quant_engine.strategies import runner as strategies_runner


def _ohlc() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=40, freq="D", tz="UTC")
    close = pd.Series([100.0, 102.0, 101.0, 98.0, 95.0] * 8, dtype=float)
    return pd.DataFrame(
        {
            "ts": idx,
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1000,
        }
    )


def _spec(asset_class: str) -> dict:
    return {
        "strategy": {
            "strategy_id": f"DCA_{asset_class}",
            "type": "dca_equity",
            "params": {
                "asset_class": asset_class,
                "drawdown_reference": "ATH",
                "grid": [{"dd": -2.0, "weight": 1.0}, {"dd": -4.0, "weight": 1.0}],
                "tp_sl": {"enabled": False},
                "require_crossing": False,
            },
        },
        "data": {"timeframe": "1D", "start": "2024-01-01", "end": "2024-02-29"},
        "universe": [{"symbol": "TEST", "asset_class": asset_class}],
        "performance": {"initial_capital": 10000, "capital_per_unit": 100},
    }


def test_cross_universe_adapter_non_regression(monkeypatch) -> None:
    monkeypatch.delenv("DB_DSN", raising=False)
    captured_data_specs: list[dict] = []

    def _fake_fetch(symbol: str, asset_class: str, data_spec: dict, instrument: dict) -> pd.DataFrame:
        captured_data_specs.append(dict(data_spec))
        return _ohlc()

    monkeypatch.setattr(strategies_runner, "_fetch_ohlc_for_symbol", _fake_fetch)

    cases = {
        "ETF": "xetra_business_days",
        "EQUITY": "nyse_business_days",
        "CRYPTO": "24x7",
    }
    for asset_class, expected_calendar in cases.items():
        result = strategies_runner.run_backtest_with_payload(_spec(asset_class))
        run_extra = result["payload"]["run"].get("extra", {})
        assert run_extra.get("universe_rules_version") == "asset-universe-rules-v1"
        assert captured_data_specs[-1]["calendar"] == expected_calendar