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


def _minimal_spec() -> dict:
    return {
        "strategy": {
            "strategy_id": "DCA_ROLLING_E2E",
            "type": "dca_equity",
            "params": {
                "asset_class": "EQUITY",
                "drawdown_reference": "ATH",
                "grid": [{"dd": -2.0, "weight": 1.0}, {"dd": -4.0, "weight": 1.0}],
                "tp_sl": {
                    "enabled": True,
                    "mode": "per_grid_max_dd",
                    "sl_dd": -50.0,
                    "rules": [
                        {"max_dd_reached": -2.0, "tp_pct": 0.5},
                        {"max_dd_reached": -4.0, "tp_pct": 0.5},
                    ],
                },
                "require_crossing": False,
            },
        },
        "data": {"timeframe": "1D", "start": "2020-01-01", "end": "2025-12-31"},
        "universe": [{"symbol": "TEST", "asset_class": "EQUITY"}],
        "performance": {
            "initial_capital": 10_000,
            "capital_per_unit": 100,
            "rolling_windows": {"years": [3], "step_months": 12},
        },
    }


def _toy_ohlc() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=365 * 6, freq="D", tz="UTC")
    closes: list[float] = []
    price = 100.0
    for i, _ in enumerate(dates):
        phase = i % 30
        if phase < 10:
            price *= 0.996
        elif phase < 20:
            price *= 1.004
        else:
            price *= 1.0005
        closes.append(price)

    close = pd.Series(closes, dtype=float)
    return pd.DataFrame(
        {
            "ts": dates,
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1_000,
        }
    )


def test_dca_rolling_pipeline_exposes_regimes_and_underperformance(monkeypatch) -> None:
    monkeypatch.delenv("DB_DSN", raising=False)

    def _fake_fetch(symbol: str, asset_class: str, data_spec: dict, instrument: dict) -> pd.DataFrame:
        return _toy_ohlc()

    monkeypatch.setattr(strategies_runner, "_fetch_ohlc_for_symbol", _fake_fetch)
    strategies_runner._OHLC_CACHE.clear()

    result = strategies_runner.run_backtest_with_payload(_minimal_spec())

    rolling = result["payload"]["run"]["extra"]["rolling_windows"]
    assert rolling["series"]
    assert rolling["series"][0]["window_years"] == 3
    assert rolling["series"][0]["regime"] in {"bull", "bear", "sideways"}

    regime_definition = rolling["regime_definition"]
    assert regime_definition["version"] == "v1"
    assert regime_definition["bull_min_return_pct"] is not None
    assert regime_definition["bear_max_return_pct"] is not None

    underperformance = rolling["underperformance"]
    assert underperformance["duration_windows"] is not None
    assert underperformance["severity_pct_points"] is not None
