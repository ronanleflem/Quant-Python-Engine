import numpy as np
import pandas as pd

from quant_engine.filters import (
    bos_filter,
    liquidity_sweep_filter,
    mss_filter,
)


def _make_structure_df(rows: int = 240) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=rows, freq="1min", tz="UTC")
    pattern = np.array([0.0, 0.01, 0.02, 0.035, 0.015, -0.005, -0.025, -0.015, 0.005, 0.025, 0.0, -0.03])
    cycles = rows // len(pattern)
    close_values = []
    level = 1.05
    for cycle in range(cycles + 1):
        scale = 1 + 0.15 * cycle
        cycle_vals = level + scale * pattern
        close_values.extend(cycle_vals)
        level += 0.01 * scale
    close = np.array(close_values[:rows])
    high = close + 0.003
    low = close - 0.003
    df = pd.DataFrame({"open": close, "high": high, "low": low, "close": close}, index=index)

    lookback = 30
    ref_high = df["high"].rolling(window=lookback, min_periods=1).max().shift(1)
    ref_low = df["low"].rolling(window=lookback, min_periods=1).min().shift(1)

    for idx in (40, 80, 120, 160, 200):
        ref = float(ref_high.iloc[idx])
        if np.isnan(ref):
            continue
        df.iloc[idx, df.columns.get_loc("high")] = ref + 0.004
        df.iloc[idx, df.columns.get_loc("close")] = ref - 0.001
        df.iloc[idx, df.columns.get_loc("open")] = ref - 0.0015

    for idx in (55, 95, 135, 175, 215):
        ref = float(ref_low.iloc[idx])
        if np.isnan(ref):
            continue
        df.iloc[idx, df.columns.get_loc("low")] = ref - 0.004
        df.iloc[idx, df.columns.get_loc("close")] = ref + 0.001
        df.iloc[idx, df.columns.get_loc("open")] = ref + 0.0015

    return df


def test_liquidity_sweep_filters_detect_events() -> None:
    df = _make_structure_df()
    series_high = liquidity_sweep_filter(df, side="high", lookback=30, require_close_back_in=True)
    series_low = liquidity_sweep_filter(df, side="low", lookback=30, require_close_back_in=True)

    assert isinstance(series_high, pd.Series)
    assert isinstance(series_low, pd.Series)
    assert series_high.index.equals(df.index)
    assert series_low.index.equals(df.index)
    assert series_high.dtype == bool
    assert series_low.dtype == bool
    assert series_high.sum() > 0
    assert series_low.sum() > 0


def test_bos_filters_return_boolean_series() -> None:
    df = _make_structure_df()
    bos_up = bos_filter(df, direction="up", use_levels=False, left=2, right=2)
    bos_down = bos_filter(df, direction="down", use_levels=False, left=2, right=2)

    assert isinstance(bos_up, pd.Series)
    assert isinstance(bos_down, pd.Series)
    assert bos_up.index.equals(df.index)
    assert bos_down.index.equals(df.index)
    assert bos_up.dtype == bool
    assert bos_down.dtype == bool
    assert bos_up.sum() > 0
    assert bos_down.sum() > 0


def test_mss_filter_detects_structure_shift() -> None:
    df = _make_structure_df()
    mss = mss_filter(df, use_levels=False, window=50, left=2, right=2)

    assert isinstance(mss, pd.Series)
    assert mss.index.equals(df.index)
    assert mss.dtype == bool
    assert mss.sum() > 0
