from __future__ import annotations

from pathlib import Path

from quant_engine.backtest import engine as backtest_engine

from quant_engine.optimize import variants as optimize_variants


SPEC_DIR = Path("specs/tests")


def _assert_opt_result(result: dict, expected_trials: int | None = None) -> None:
    assert result.get("best") is not None
    assert result.get("summary")
    assert result.get("trials_path")
    if expected_trials is not None:
        assert result.get("total_trials") == expected_trials


def test_optimize_backtest_screening(tmp_path) -> None:
    spec = optimize_variants.backtest_runner.load_backtest_spec(
        SPEC_DIR / "optimize_backtest_screening.json"
    )
    result = optimize_variants.run_backtest_optimization(spec, out_dir=tmp_path / "opt_bt_screen")
    _assert_opt_result(result, expected_trials=4)


def test_optimize_strategy_screening(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("DB_DSN", raising=False)
    spec = optimize_variants.strategy_runner.load_strategy_spec(
        SPEC_DIR / "optimize_strategy_screening.json"
    )
    result = optimize_variants.run_strategy_optimization(spec, out_dir=tmp_path / "opt_strat_screen")
    _assert_opt_result(result, expected_trials=2)


def test_pruning_max_drawdown_pct_triggers() -> None:
    rows = [
        {
            "timestamp": "2025-01-01T00:00:00",
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 100,
            "symbol": "EURUSD",
        },
        {
            "timestamp": "2025-01-01T00:01:00",
            "open": 1.0,
            "high": 1.0,
            "low": 0.98,
            "close": 0.99,
            "volume": 100,
            "symbol": "EURUSD",
        },
        {
            "timestamp": "2025-01-01T00:02:00",
            "open": 0.98,
            "high": 0.99,
            "low": 0.97,
            "close": 0.98,
            "volume": 100,
            "symbol": "EURUSD",
        },
        {
            "timestamp": "2025-01-01T00:03:00",
            "open": 0.98,
            "high": 0.99,
            "low": 0.97,
            "close": 0.98,
            "volume": 100,
            "symbol": "EURUSD",
        },
    ]
    signals = [1, 1, 1, 1]
    atr_values = [0.01] * len(rows)
    trades, equity, _summary = backtest_engine.run(
        rows,
        signals,
        atr_values,
        atr_mult=1.0,
        r_mult=2.0,
        pruning={"enabled": True, "max_drawdown_pct": 1.0},
    )
    assert trades
    assert len(equity) < len(rows)


def test_pruning_min_signals_after_bars_triggers() -> None:
    rows = [
        {
            "timestamp": "2025-01-01T00:00:00",
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 100,
            "symbol": "EURUSD",
        },
        {
            "timestamp": "2025-01-01T00:01:00",
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 100,
            "symbol": "EURUSD",
        },
        {
            "timestamp": "2025-01-01T00:02:00",
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 100,
            "symbol": "EURUSD",
        },
        {
            "timestamp": "2025-01-01T00:03:00",
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 100,
            "symbol": "EURUSD",
        },
        {
            "timestamp": "2025-01-01T00:04:00",
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 100,
            "symbol": "EURUSD",
        },
        {
            "timestamp": "2025-01-01T00:05:00",
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 100,
            "symbol": "EURUSD",
        },
    ]
    signals = [0] * len(rows)
    atr_values = [0.01] * len(rows)
    _trades, equity, _summary = backtest_engine.run(
        rows,
        signals,
        atr_values,
        atr_mult=1.0,
        r_mult=2.0,
        pruning={
            "enabled": True,
            "min_signals_after_bars": {"bars": 3, "min_signals": 2},
        },
    )
    assert len(equity) < len(rows)
