from __future__ import annotations

from typing import Any, Dict

import sys
import types

pymysql_module = types.ModuleType("pymysql")
cursors_module = types.ModuleType("pymysql.cursors")
cursors_module.DictCursor = object
pymysql_module.cursors = cursors_module
sys.modules.setdefault("pymysql", pymysql_module)
sys.modules.setdefault("pymysql.cursors", cursors_module)

from quant_engine.api import app as api_app


MI_CONTRACT = {
    "enabled": True,
    "provider": "dummy",
    "labels": ["regime", "liquidity"],
}


def _optimize_backtest_request() -> Dict[str, Any]:
    return {
        "spec_type": "optimize_backtest",
        "catalog_version": "v1",
        "market_intelligence": dict(MI_CONTRACT),
        "optimization": {
            "base_spec": {
                "spec_type": "backtest",
                "catalog_version": "v1",
                "market_intelligence": dict(MI_CONTRACT),
                "data": {
                    "symbol": "EURUSD",
                    "timeframe": "M1",
                    "start_date": "2025-01-01",
                    "end_date": "2025-01-02",
                },
                "signal": {"type": "ema_cross", "fast": 9, "slow": 21},
            },
            "search_space": {"signal.fast": {"type": "int", "min": 5, "max": 6}},
            "objective": {"metric": "sharpe", "direction": "max"},
            "budget": {"max_trials": 1, "seed": 7},
        },
    }


def _optimize_dca_request() -> Dict[str, Any]:
    return {
        "spec_type": "optimize_dca",
        "catalog_version": "v1",
        "mi": dict(MI_CONTRACT),
        "optimization": {
            "base_spec": {
                "spec_type": "dca",
                "catalog_version": "v1",
                "mi": dict(MI_CONTRACT),
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
            },
            "search_space": {"strategy.params.grid[0].dd": {"type": "int", "min": -6, "max": -5}},
            "objective": {"metric": "sharpe", "direction": "max"},
            "budget": {"max_trials": 1, "seed": 9},
        },
    }


def test_optimize_backtest_and_dca_align_mi_contract_and_metadata(monkeypatch) -> None:
    observed: Dict[str, Dict[str, Any]] = {}

    def _fake_backtest(spec: Dict[str, Any]) -> Dict[str, Any]:
        observed["backtest"] = spec
        return {
            "trials_path": "",
            "summary": "",
            "best": {"params": {"signal.fast": 5}, "objective": 1.0},
            "metadata": {"mi": {"enabled": True, "contract": dict(spec.get("market_intelligence") or {})}},
        }

    def _fake_dca(spec: Dict[str, Any]) -> Dict[str, Any]:
        observed["dca"] = spec
        return {
            "trials_path": "",
            "summary": "",
            "best": {"params": {"strategy.params.grid[0].dd": -5}, "objective": 0.5},
            "metadata": {"mi": {"enabled": True, "contract": dict(spec.get("mi") or {})}},
        }

    monkeypatch.setattr(api_app.optimize_variants, "run_backtest_optimization", _fake_backtest)
    monkeypatch.setattr(api_app.optimize_variants, "run_strategy_optimization", _fake_dca)

    backtest_result = api_app._canonical_optimization_from_request(_optimize_backtest_request())
    dca_result = api_app._canonical_optimization_from_request(_optimize_dca_request())

    assert observed["backtest"]["market_intelligence"] == MI_CONTRACT
    assert observed["dca"]["mi"] == MI_CONTRACT

    assert backtest_result["result"]["metadata"] == {"mi": {"enabled": True, "contract": MI_CONTRACT}}
    assert dca_result["result"]["metadata"] == {"mi": {"enabled": True, "contract": MI_CONTRACT}}
