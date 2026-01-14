from __future__ import annotations

from pathlib import Path

import pytest

from quant_engine.backtest import runner as backtest_runner
from quant_engine.strategies import runner as strategies_runner


SPEC_DIR = Path("specs/tests")


def test_missing_bars_backtest_runs() -> None:
    spec = backtest_runner.load_backtest_spec(SPEC_DIR / "backtest_missing_bars.json")
    result = backtest_runner.run_backtest_from_spec(spec)
    assert result.get("payload")


def test_naive_timestamps_are_localized() -> None:
    spec = backtest_runner.load_backtest_spec(SPEC_DIR / "backtest_naive_timestamps.json")
    result = backtest_runner.run_backtest_from_spec(spec)
    trades = result.get("payload", {}).get("trades", [])
    if trades:
        ts_entry = trades[0].get("ts_entry")
        assert "Z" in ts_entry or "+" in ts_entry


def test_missing_ohlc_column_raises(monkeypatch) -> None:
    spec = {
        "strategy": {
            "strategy_id": "BT_MISSING_COLS",
            "type": "dca_equity",
            "params": {"asset_class": "EQUITY", "grid": [{"dd": -5.0, "weight": 1.0}]},
        },
        "data": {
            "source": "csv",
            "path": "tests/data/ohlcv_missing_cols.csv",
            "timeframe": "1m",
            "start": "2025-01-01",
            "end": "2025-01-01",
        },
        "universe": [{"symbol": "EURUSD", "asset_class": "FX"}],
    }
    with pytest.raises(Exception, match="Missing OHLC columns|ts"):
        strategies_runner.run_backtest_from_spec(spec)


def test_bad_timestamp_raises() -> None:
    spec = backtest_runner.load_backtest_spec(SPEC_DIR / "backtest_bad_timestamp.json")
    with pytest.raises(ValueError, match="Invalid isoformat|Invalid"):
        backtest_runner.run_backtest_from_spec(spec)


def test_nan_ohlc_raises() -> None:
    spec = backtest_runner.load_backtest_spec(SPEC_DIR / "backtest_nan_ohlc.json")
    with pytest.raises(ValueError, match="NaN OHLC value"):
        backtest_runner.run_backtest_from_spec(spec)


def test_missing_symbol_fallback() -> None:
    spec = backtest_runner.load_backtest_spec(SPEC_DIR / "backtest_missing_symbol.json")
    result = backtest_runner.run_backtest_from_spec(spec)
    run = result.get("payload", {}).get("run", {})
    assert run.get("symbol") == "EURUSD"
