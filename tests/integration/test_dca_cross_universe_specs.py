from __future__ import annotations

import sys
import types

import pandas as pd
import pytest

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


def _spec(
    asset_class: str,
    *,
    start: str = "2024-01-01",
    timezone: str | None = None,
    universe_overrides: dict | None = None,
) -> dict:
    data_spec: dict = {"timeframe": "1D", "start": start, "end": "2024-02-29"}
    if timezone is not None:
        data_spec["timezone"] = timezone

    instrument = {"symbol": "TEST", "asset_class": asset_class}
    if universe_overrides:
        instrument["universe"] = dict(universe_overrides)

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
        "data": data_spec,
        "universe": [instrument],
        "performance": {"initial_capital": 10000, "capital_per_unit": 100},
    }


@pytest.mark.parametrize(
    ("asset_class", "start", "timezone", "universe_overrides", "expected_calendar", "expected_timezone"),
    [
        ("ETF", "2024-01-01", "UTC", None, "xetra_business_days", "UTC"),
        (
            "ETF",
            "2024-01-01",
            "UTC",
            {"timezone": "Europe/Paris"},
            "xetra_business_days",
            "Europe/Paris",
        ),
        ("EQUITY", "2024-01-06", "UTC", None, "nyse_business_days", "UTC"),
        (
            "EQUITY",
            "2024-01-06",
            "UTC",
            {"timezone": "Europe/Paris"},
            "nyse_business_days",
            "Europe/Paris",
        ),
        ("CRYPTO", "2024-01-06", "UTC", None, "24x7", "UTC"),
        (
            "CRYPTO",
            "2024-01-06",
            "UTC",
            {"timezone": "Europe/Paris"},
            "24x7",
            "Europe/Paris",
        ),
    ],
)
def test_cross_universe_adapter_non_regression(
    monkeypatch,
    asset_class: str,
    start: str,
    timezone: str,
    universe_overrides: dict | None,
    expected_calendar: str,
    expected_timezone: str,
) -> None:
    monkeypatch.delenv("DB_DSN", raising=False)
    captured_data_specs: list[dict] = []
    captured_contexts: list[dict] = []

    def _fake_fetch(symbol: str, asset_class: str, data_spec: dict, instrument: dict) -> pd.DataFrame:
        captured_data_specs.append(dict(data_spec))
        return _ohlc()

    class _StubStrategy:
        def backtest(self, df: pd.DataFrame, context: dict) -> list:
            captured_contexts.append(dict(context))
            return []

    def _fake_create_strategy(strategy_type: str, strategy_id: str, params: dict) -> _StubStrategy:
        return _StubStrategy()

    monkeypatch.setattr(strategies_runner, "_fetch_ohlc_for_symbol", _fake_fetch)
    monkeypatch.setattr(strategies_runner, "create_strategy", _fake_create_strategy)

    result = strategies_runner.run_backtest_with_payload(
        _spec(asset_class, start=start, timezone=timezone, universe_overrides=universe_overrides)
    )

    run_extra = result["payload"]["run"].get("extra", {})
    assert run_extra.get("universe_rules_version") == "asset-universe-rules-v1"
    assert captured_data_specs[-1]["calendar"] == expected_calendar
    assert captured_contexts[-1]["universe_rules"]["calendar"] == expected_calendar
    assert captured_contexts[-1]["universe_rules"]["timezone"] == expected_timezone
