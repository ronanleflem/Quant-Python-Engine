from __future__ import annotations

from pathlib import Path
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


def _optimize_backtest_request(csv_path: Path) -> Dict[str, Any]:
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
                    "end_date": "2025-01-01",
                    "path": str(csv_path),
                },
                "signal": {"type": "ema_cross", "fast": 2, "slow": 5},
            },
            "search_space": {"signal.params.fast": {"type": "int", "min": 2, "max": 2}},
            "objective": {"metric": "sharpe", "direction": "max"},
            "budget": {"max_trials": 1, "seed": 7},
        },
    }


def _optimize_dca_request(csv_path: Path) -> Dict[str, Any]:
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
                    "symbol": "SPY",
                    "timeframe": "1D",
                    "start_date": "2025-01-01",
                    "end_date": "2025-01-20",
                    "path": str(csv_path),
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
            "search_space": {"strategy.params.grid[0].dd": {"type": "int", "min": -5, "max": -5}},
            "objective": {"metric": "sharpe", "direction": "max"},
            "budget": {"max_trials": 1, "seed": 9},
        },
    }


def test_optimize_backtest_and_dca_mi_e2e_contract_metadata_and_shape(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    backtest_request = _optimize_backtest_request(Path(__file__).resolve().parents[1] / "data" / "ohlcv.csv")
    dca_request = _optimize_dca_request(Path(__file__).resolve().parents[1] / "data" / "ohlcv_ts.csv")

    backtest_result = api_app._canonical_optimization_from_request(backtest_request)
    dca_result = api_app._canonical_optimization_from_request(dca_request)

    backtest_payload = backtest_result["result"]
    dca_payload = dca_result["result"]

    invariant_keys = {"objective", "best", "trials", "summary", "source", "artifacts", "metadata"}
    assert invariant_keys.issubset(backtest_payload.keys())
    assert invariant_keys.issubset(dca_payload.keys())

    for payload in (backtest_payload, dca_payload):
        assert payload["best"] is not None
        assert isinstance(payload["trials"], list)
        assert payload["summary"]["total_trials"] == len(payload["trials"])
        assert payload["summary"]["succeeded_trials"] + payload["summary"]["failed_trials"] == len(payload["trials"])
        assert set(payload["metadata"]["mi"].keys()) == {"enabled", "contract"}

    backtest_mi = backtest_payload["metadata"]["mi"]
    dca_mi = dca_payload["metadata"]["mi"]
    assert backtest_mi["enabled"] is True
    assert dca_mi["enabled"] is True
    assert backtest_mi["contract"] == MI_CONTRACT
    assert dca_mi["contract"] == MI_CONTRACT
    assert backtest_mi == dca_mi
