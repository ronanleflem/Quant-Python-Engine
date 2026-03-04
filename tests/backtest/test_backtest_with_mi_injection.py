from __future__ import annotations

from pathlib import Path
import sys
import types

import pandas as pd

pymysql_module = types.ModuleType("pymysql")
cursors_module = types.ModuleType("pymysql.cursors")
cursors_module.DictCursor = object
pymysql_module.cursors = cursors_module
sys.modules.setdefault("pymysql", pymysql_module)
sys.modules.setdefault("pymysql.cursors", cursors_module)

from quant_engine.backtest import runner as backtest_runner


SPEC_DIR = Path("specs/tests")


class DummyMarketIntelligenceService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def build_snapshot(self, symbol: str, ohlcv: pd.DataFrame) -> dict:
        self.calls.append((symbol, len(ohlcv)))
        return {
            "symbol": symbol,
            "features": {"regime": "trend"},
            "bars": len(ohlcv),
        }


def test_backtest_injects_market_intelligence_service(monkeypatch) -> None:
    spec = backtest_runner.load_backtest_spec(SPEC_DIR / "backtest_csv_basic.json")
    observed: dict = {}
    original_build_signal = backtest_runner._build_signal

    def wrapped_build_signal(spec_input, rows, features=None):
        observed["features"] = features
        return original_build_signal(spec_input, rows, features)

    monkeypatch.setattr(backtest_runner, "_build_signal", wrapped_build_signal)
    mi_service = DummyMarketIntelligenceService()

    result = backtest_runner.run_backtest_from_spec(spec, mi_service=mi_service)

    assert result["symbol"] == "EURUSD"
    assert mi_service.calls
    assert observed["features"]["features"]["regime"] == "trend"


def test_backtest_uses_null_mi_adapter_by_default(monkeypatch) -> None:
    spec = backtest_runner.load_backtest_spec(SPEC_DIR / "backtest_csv_basic.json")
    observed: dict = {}
    original_build_signal = backtest_runner._build_signal

    def wrapped_build_signal(spec_input, rows, features=None):
        observed["features"] = features
        return original_build_signal(spec_input, rows, features)

    monkeypatch.setattr(backtest_runner, "_build_signal", wrapped_build_signal)

    result = backtest_runner.run_backtest_from_spec(spec)

    assert result["symbol"] == "EURUSD"
    assert observed["features"]["symbol"] == "EURUSD"
    assert observed["features"]["features"] == {}
