from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_engine.optimize import variants as optimize_variants
from quant_engine.filters.utils import apply_filter_stack
from quant_engine.backtest import runner as backtest_runner


SPEC_DIR = Path("specs/tests")


def _assert_opt_result(result: dict, expected_trials: int | None = None) -> None:
    assert result.get("best") is not None
    assert result.get("summary")
    assert result.get("trials_path")
    if expected_trials is not None:
        assert result.get("total_trials") == expected_trials


def test_optimize_backtest_filters(tmp_path) -> None:
    spec = optimize_variants.backtest_runner.load_backtest_spec(
        SPEC_DIR / "optimize_backtest_filters.json"
    )
    result = optimize_variants.run_backtest_optimization(spec, out_dir=tmp_path / "opt_bt_filters")
    _assert_opt_result(result, expected_trials=8)


def test_optimize_strategy_filters(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("DB_DSN", raising=False)
    spec = optimize_variants.strategy_runner.load_strategy_spec(
        SPEC_DIR / "optimize_strategy_filters.json"
    )
    result = optimize_variants.run_strategy_optimization(spec, out_dir=tmp_path / "opt_strat_filters")
    _assert_opt_result(result, expected_trials=4)


def test_filter_reduces_bar_count() -> None:
    df = pd.read_csv("tests/data/ohlcv.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp")
    filters_spec = [{"type": "day_of_week", "params": {"blocked_days": [2]}}]
    mask = apply_filter_stack(df, filters_spec, strict=True)
    assert int(mask.sum()) < len(df)


def test_require_crossing_changes_signal_mask() -> None:
    signal = [1, 1, 1]
    mask = [True, False, False]
    crossing = backtest_runner._apply_filter_mask(signal, mask, require_crossing=True)
    non_crossing = backtest_runner._apply_filter_mask(signal, mask, require_crossing=False)
    assert crossing != non_crossing
    assert crossing == [1, 0, 0]
    assert non_crossing == [1, 1, 1]
