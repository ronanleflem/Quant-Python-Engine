from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional
import sys
import types

import pandas as pd

pymysql_module = types.ModuleType("pymysql")
cursors_module = types.ModuleType("pymysql.cursors")
cursors_module.DictCursor = object
pymysql_module.cursors = cursors_module
sys.modules.setdefault("pymysql", pymysql_module)
sys.modules.setdefault("pymysql.cursors", cursors_module)

from quant_engine.strategies.base import StrategySignal
from quant_engine.strategies import runner as strategies_runner


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
        ts = pd.Timestamp(bar.name)
        self.rows.append((ts, features_row))
        return []


class _LegacyDummyStrategy:
    def __init__(self, strategy_id: str, params: Dict[str, Any]) -> None:
        self.strategy_id = strategy_id
        self.params = params

    def backtest(self, ohlc: pd.DataFrame, context: Dict[str, Any]) -> List[StrategySignal]:
        return []


def _build_spec(strategy_type: str) -> Dict[str, Any]:
    return {
        "strategy": {"type": strategy_type, "strategy_id": "s1", "params": {}},
        "universe": [{"symbol": "SPY", "asset_class": "ETF"}],
        "data": {"source": "csv", "path": "unused.csv"},
    }


def _build_ohlc_with_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts": pd.to_datetime([
                "2024-01-01T00:00:00Z",
                "2024-01-02T00:00:00Z",
            ], utc=True),
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "volume": [1000.0, 1100.0],
            "feat_regime": ["trend", "range"],
            "feat_score": [0.75, 0.35],
        }
    )


def test_runner_passes_timestamp_aligned_features_row(monkeypatch) -> None:
    observed: Dict[str, Any] = {}

    def _factory(strategy_type: str, strategy_id: str, params: Dict[str, Any]):
        strat = _FeatureAwareDummyStrategy(strategy_id=strategy_id, params=params)
        observed["strategy"] = strat
        return strat

    monkeypatch.setattr(strategies_runner, "create_strategy", _factory)
    monkeypatch.setattr(
        strategies_runner,
        "_fetch_ohlc_for_symbol",
        lambda symbol, asset_class, data_spec, instrument_spec: _build_ohlc_with_features(),
    )

    strategies_runner.run_backtest_from_spec(_build_spec("dummy_feature_aware"))

    strategy = observed["strategy"]
    assert len(strategy.rows) == 2
    first_ts, first_features = strategy.rows[0]
    second_ts, second_features = strategy.rows[1]
    assert first_ts == pd.Timestamp("2024-01-01T00:00:00Z")
    assert second_ts == pd.Timestamp("2024-01-02T00:00:00Z")
    assert first_features == {"feat_regime": "trend", "feat_score": 0.75}
    assert second_features == {"feat_regime": "range", "feat_score": 0.35}


def test_runner_keeps_backward_compat_without_on_bar(monkeypatch) -> None:
    monkeypatch.setattr(
        strategies_runner,
        "create_strategy",
        lambda strategy_type, strategy_id, params: _LegacyDummyStrategy(strategy_id=strategy_id, params=params),
    )
    monkeypatch.setattr(
        strategies_runner,
        "_fetch_ohlc_for_symbol",
        lambda symbol, asset_class, data_spec, instrument_spec: _build_ohlc_with_features(),
    )

    result = strategies_runner.run_backtest_from_spec(_build_spec("dummy_legacy"))

    assert result["counts"]["SPY"] == 0
