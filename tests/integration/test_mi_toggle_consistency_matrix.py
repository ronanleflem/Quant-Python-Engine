from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional
import sys
import types

import pandas as pd
import pytest

pymysql_module = types.ModuleType("pymysql")
cursors_module = types.ModuleType("pymysql.cursors")
cursors_module.DictCursor = object
pymysql_module.cursors = cursors_module
sys.modules.setdefault("pymysql", pymysql_module)
sys.modules.setdefault("pymysql.cursors", cursors_module)

from quant_engine.backtest import runner as backtest_runner
from quant_engine.config import reset_settings_cache
from quant_engine.strategies.base import StrategySignal
from quant_engine.strategies import runner as strategies_runner


class _DummyMIService:
    def __init__(self) -> None:
        self.calls: List[tuple[str, int]] = []

    def build_snapshot(self, symbol: str, ohlcv: pd.DataFrame) -> Dict[str, Any]:
        self.calls.append((symbol, len(ohlcv)))
        return {"symbol": symbol, "features": {"mi_mode": "on"}, "ohlcv_rows": len(ohlcv)}


class _FeatureAwareDummyStrategy:
    def __init__(self, strategy_id: str, params: Dict[str, Any]) -> None:
        self.strategy_id = strategy_id
        self.params = params
        self.rows: list[tuple[pd.Timestamp, Optional[Mapping[str, Any]]]] = []

    def on_bar(
        self,
        bar: pd.Series,
        context: Dict[str, Any],
        features_row: Optional[Mapping[str, Any]] = None,
    ) -> List[StrategySignal]:
        self.rows.append((pd.Timestamp(bar.name), features_row))
        return []


def _build_rows() -> list[dict[str, Any]]:
    return [
        {
            "timestamp": "2024-01-01T00:00:00Z",
            "symbol": "EURUSD",
            "open": 1.1,
            "high": 1.2,
            "low": 1.0,
            "close": 1.15,
            "volume": 100,
        },
        {
            "timestamp": "2024-01-02T00:00:00Z",
            "symbol": "EURUSD",
            "open": 1.2,
            "high": 1.3,
            "low": 1.1,
            "close": 1.25,
            "volume": 110,
        },
    ]


@pytest.mark.parametrize(
    ("env_value", "spec_enabled", "expected_enabled"),
    [
        (None, None, True),
        ("true", None, True),
        ("false", None, False),
        ("false", True, True),
        ("true", False, False),
    ],
)
def test_mi_toggle_consistency_matrix_backtest(monkeypatch, env_value, spec_enabled, expected_enabled) -> None:
    monkeypatch.setattr(backtest_runner, "_load_rows", lambda *_args, **_kwargs: (_build_rows(), None))

    observed: dict[str, Any] = {}
    original_build_signal = backtest_runner._build_signal

    def wrapped_build_signal(spec_input, rows, features=None):
        observed["features"] = features
        return original_build_signal(spec_input, rows, features)

    monkeypatch.setattr(backtest_runner, "_build_signal", wrapped_build_signal)

    if env_value is None:
        monkeypatch.delenv("MARKET_INTELLIGENCE_ENABLED", raising=False)
    else:
        monkeypatch.setenv("MARKET_INTELLIGENCE_ENABLED", env_value)
    reset_settings_cache()

    spec: Dict[str, Any] = {
        "data": {"source": "csv", "path": "unused.csv", "symbol": "EURUSD", "start": "2024-01-01", "end": "2024-01-02"},
        "strategy": {"asset_class": "FX"},
        "signal": {"type": "ema_cross", "params": {"fast": 1, "slow": 2}},
    }
    if spec_enabled is not None:
        spec["market_intelligence"] = {"enabled": spec_enabled}

    mi_service = _DummyMIService()
    backtest_runner.run_backtest_from_spec(spec, mi_service=mi_service)

    if expected_enabled:
        assert mi_service.calls
        assert observed["features"]["features"]["mi_mode"] == "on"
    else:
        assert mi_service.calls == []
        assert observed["features"]["features"] == {}


@pytest.mark.parametrize(
    ("env_value", "spec_enabled", "expected_enabled"),
    [
        (None, None, True),
        ("true", None, True),
        ("false", None, False),
        ("false", True, True),
        ("true", False, False),
    ],
)
def test_mi_toggle_consistency_matrix_strategies(monkeypatch, env_value, spec_enabled, expected_enabled) -> None:
    observed: Dict[str, Any] = {}

    def _factory(strategy_type: str, strategy_id: str, params: Dict[str, Any]):
        strategy = _FeatureAwareDummyStrategy(strategy_id=strategy_id, params=params)
        observed["strategy"] = strategy
        return strategy

    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"], utc=True),
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "volume": [1000.0, 1100.0],
            "feat_regime": ["trend", "range"],
        }
    )

    monkeypatch.setattr(strategies_runner, "create_strategy", _factory)
    monkeypatch.setattr(
        strategies_runner,
        "_fetch_ohlc_for_symbol",
        lambda symbol, asset_class, data_spec, instrument_spec: df.copy(),
    )

    if env_value is None:
        monkeypatch.delenv("MARKET_INTELLIGENCE_ENABLED", raising=False)
    else:
        monkeypatch.setenv("MARKET_INTELLIGENCE_ENABLED", env_value)
    reset_settings_cache()

    spec: Dict[str, Any] = {
        "strategy": {"type": "dummy", "strategy_id": "s1", "params": {}},
        "universe": [{"symbol": "SPY", "asset_class": "ETF"}],
        "data": {"source": "csv", "path": "unused.csv"},
    }
    if spec_enabled is not None:
        spec["market_intelligence"] = {"enabled": spec_enabled}

    strategies_runner.run_backtest_from_spec(spec)

    if expected_enabled:
        assert observed["strategy"].rows[-1][1] == {"feat_regime": "range"}
    else:
        assert all(features is None for _, features in observed["strategy"].rows)
