from __future__ import annotations

from pathlib import Path
import os

import pytest

from quant_engine.optimize import variants as optimize_variants


LARGE_SOURCE = Path(
    "specs/examples/data/forex/EURUSD_20250101_20250601_1min.csv"
)

if not os.getenv("QE_PERF_TRACE"):
    os.environ["QE_PERF_TRACE"] = "1"


@pytest.fixture(scope="session")
def large_csv_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if not LARGE_SOURCE.exists():
        pytest.skip(f"Missing large dataset: {LARGE_SOURCE}")
    out_dir = tmp_path_factory.mktemp("large_csv_opt")
    out_path = out_dir / "eurusd_large_converted.csv"
    if out_path.exists():
        return out_path

    import pandas as pd

    df = pd.read_csv(
        LARGE_SOURCE,
        usecols=["Timestamp", "Open", "High", "Low", "Close", "Volume"],
    )
    df = df.rename(
        columns={
            "Timestamp": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    df["symbol"] = "EURUSD"
    df["ts"] = df["timestamp"]
    df.to_csv(out_path, index=False)
    return out_path


def _backtest_base_spec(csv_path: Path) -> dict:
    return {
        "strategy": {"strategy_id": "OPT_BT_LARGE", "asset_class": "FX"},
        "data": {
            "dataset_path": str(csv_path),
            "symbols": ["EURUSD"],
            "timeframe": "1m",
            "start": "2025-01-01",
            "end": "2025-06-01",
        },
        "signal": {
            "type": "ema_cross",
            "params": {"fast": 2, "slow": 10, "require_crossing": False},
        },
        "tpsl": {"atr_window": 14, "atr_k": 1.0, "r_mult": 2.0},
        "performance": {"initial_capital": 10000},
        "persistence": {"enabled": False},
    }


def _strategy_base_spec(csv_path: Path) -> dict:
    return {
        "strategy": {
            "strategy_id": "OPT_STRAT_LARGE",
            "type": "dca_equity",
            "params": {
                "asset_class": "EQUITY",
                "drawdown_reference": "ATH",
                "grid": [
                    {"dd": -1.0, "weight": 1.0},
                    {"dd": -6.0, "weight": 1.0},
                ],
                "tp_sl": {"enabled": False},
                "require_crossing": False,
            },
        },
        "data": {
            "source": "csv",
            "path": str(csv_path),
            "timeframe": "1m",
            "start": "2025-01-01",
            "end": "2025-06-01",
        },
        "universe": [{"symbol": "EURUSD", "asset_class": "EQUITY"}],
        "performance": {"initial_capital": 10000, "capital_per_unit": 100},
    }


def _screening_cfg(enabled: bool) -> dict:
    if not enabled:
        return {"enabled": False}
    return {
        "enabled": True,
        "max_bars": 5000,
        "max_trades": 50,
        "max_seconds": 2.0,
        "pruning": {
            "enabled": True,
            "max_drawdown_pct": 1.0,
            "min_signals_after_bars": {"bars": 1000, "min_signals": 2},
        },
    }


@pytest.mark.slow
def test_optimize_backtest_grid_large_no_screening(tmp_path: Path, large_csv_path: Path) -> None:
    spec = _backtest_base_spec(large_csv_path)
    spec["optimization"] = {
        "method": "grid",
        "objective": "sharpe",
        "search_space": {
            "signal.params.fast": [1, 2, 3, 4, 5],
            "signal.params.slow": [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        },
        "cache_features": {"enabled": True, "max_items": 8, "ttl_seconds": 600},
        "screening": _screening_cfg(False),
    }
    result = optimize_variants.run_backtest_optimization(spec, out_dir=tmp_path / "opt_bt_grid_no_screen")
    assert result.get("total_trials") == 50


@pytest.mark.slow
def test_optimize_backtest_grid_large_screening(tmp_path: Path, large_csv_path: Path) -> None:
    spec = _backtest_base_spec(large_csv_path)
    spec["optimization"] = {
        "method": "grid",
        "objective": "sharpe",
        "search_space": {
            "signal.params.fast": [1, 2, 3, 4, 5],
            "signal.params.slow": [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        },
        "cache_features": {"enabled": True, "max_items": 8, "ttl_seconds": 600},
        "screening": _screening_cfg(True),
    }
    result = optimize_variants.run_backtest_optimization(spec, out_dir=tmp_path / "opt_bt_grid_screen")
    assert result.get("total_trials") == 50


@pytest.mark.slow
def test_optimize_backtest_random_large_no_screening(tmp_path: Path, large_csv_path: Path) -> None:
    spec = _backtest_base_spec(large_csv_path)
    spec["optimization"] = {
        "method": "random",
        "objective": "sharpe",
        "max_trials": 50,
        "seed": 42,
        "search_space": {
            "signal.params.fast": [1, 2, 3, 4, 5],
            "signal.params.slow": [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        },
        "cache_features": {"enabled": True, "max_items": 8, "ttl_seconds": 600},
        "screening": _screening_cfg(False),
    }
    result = optimize_variants.run_backtest_optimization(spec, out_dir=tmp_path / "opt_bt_rand_no_screen")
    assert result.get("total_trials") == 50


@pytest.mark.slow
def test_optimize_backtest_random_large_screening(tmp_path: Path, large_csv_path: Path) -> None:
    spec = _backtest_base_spec(large_csv_path)
    spec["optimization"] = {
        "method": "random",
        "objective": "sharpe",
        "max_trials": 50,
        "seed": 42,
        "search_space": {
            "signal.params.fast": [1, 2, 3, 4, 5],
            "signal.params.slow": [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        },
        "cache_features": {"enabled": True, "max_items": 8, "ttl_seconds": 600},
        "screening": _screening_cfg(True),
    }
    result = optimize_variants.run_backtest_optimization(spec, out_dir=tmp_path / "opt_bt_rand_screen")
    assert result.get("total_trials") == 50


@pytest.mark.slow
def test_optimize_strategy_grid_large_no_screening(tmp_path: Path, large_csv_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("DB_DSN", raising=False)
    spec = _strategy_base_spec(large_csv_path)
    spec["optimization"] = {
        "method": "grid",
        "objective": "sharpe",
        "search_space": {
            "strategy.params.grid[0].dd": [-1.0, -2.0, -3.0, -4.0, -5.0],
            "strategy.params.grid[1].dd": [-6.0, -7.0, -8.0, -9.0, -10.0, -11.0, -12.0, -13.0, -14.0, -15.0],
        },
        "cache_features": {"enabled": True, "max_items": 8, "ttl_seconds": 600},
        "screening": _screening_cfg(False),
    }
    result = optimize_variants.run_strategy_optimization(spec, out_dir=tmp_path / "opt_strat_grid_no_screen")
    assert result.get("total_trials") == 50


@pytest.mark.slow
def test_optimize_strategy_grid_large_screening(tmp_path: Path, large_csv_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("DB_DSN", raising=False)
    spec = _strategy_base_spec(large_csv_path)
    spec["optimization"] = {
        "method": "grid",
        "objective": "sharpe",
        "search_space": {
            "strategy.params.grid[0].dd": [-1.0, -2.0, -3.0, -4.0, -5.0],
            "strategy.params.grid[1].dd": [-6.0, -7.0, -8.0, -9.0, -10.0, -11.0, -12.0, -13.0, -14.0, -15.0],
        },
        "cache_features": {"enabled": True, "max_items": 8, "ttl_seconds": 600},
        "screening": _screening_cfg(True),
    }
    result = optimize_variants.run_strategy_optimization(spec, out_dir=tmp_path / "opt_strat_grid_screen")
    assert result.get("total_trials") == 50


@pytest.mark.slow
def test_optimize_strategy_random_large_no_screening(tmp_path: Path, large_csv_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("DB_DSN", raising=False)
    spec = _strategy_base_spec(large_csv_path)
    spec["optimization"] = {
        "method": "random",
        "objective": "sharpe",
        "max_trials": 50,
        "seed": 7,
        "search_space": {
            "strategy.params.grid[0].dd": [-1.0, -2.0, -3.0, -4.0, -5.0],
            "strategy.params.grid[1].dd": [-6.0, -7.0, -8.0, -9.0, -10.0, -11.0, -12.0, -13.0, -14.0, -15.0],
        },
        "cache_features": {"enabled": True, "max_items": 8, "ttl_seconds": 600},
        "screening": _screening_cfg(False),
    }
    result = optimize_variants.run_strategy_optimization(spec, out_dir=tmp_path / "opt_strat_rand_no_screen")
    assert result.get("total_trials") == 50


@pytest.mark.slow
def test_optimize_strategy_random_large_screening(tmp_path: Path, large_csv_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("DB_DSN", raising=False)
    spec = _strategy_base_spec(large_csv_path)
    spec["optimization"] = {
        "method": "random",
        "objective": "sharpe",
        "max_trials": 50,
        "seed": 7,
        "search_space": {
            "strategy.params.grid[0].dd": [-1.0, -2.0, -3.0, -4.0, -5.0],
            "strategy.params.grid[1].dd": [-6.0, -7.0, -8.0, -9.0, -10.0, -11.0, -12.0, -13.0, -14.0, -15.0],
        },
        "cache_features": {"enabled": True, "max_items": 8, "ttl_seconds": 600},
        "screening": _screening_cfg(True),
    }
    result = optimize_variants.run_strategy_optimization(spec, out_dir=tmp_path / "opt_strat_rand_screen")
    assert result.get("total_trials") == 50
