"""Smoke tests for volatility and trend filters."""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant_engine.filters import adx_filter, atr_filter, ema_slope_filter


def _make_sample_df(rows: int = 256) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    index = pd.date_range("2024-01-01", periods=rows, freq="1h", tz="UTC")
    trend = np.linspace(1.0, 1.1, rows)
    noise = rng.normal(scale=0.001, size=rows)
    close = trend + noise
    high = close + 0.003 + rng.normal(scale=0.0005, size=rows)
    low = close - 0.003 - rng.normal(scale=0.0005, size=rows)
    data = pd.DataFrame({"high": high, "low": low, "close": close}, index=index)
    return data


def test_filters_return_boolean_series() -> None:
    df = _make_sample_df()
    filters = [
        (adx_filter, {"window": 14, "thresh": 20.0}),
        (atr_filter, {"window": 14, "min_mult": 0.5, "max_mult": 2.0}),
        (ema_slope_filter, {"window": 32, "slope_thresh": 0.0}),
    ]
    for func, kwargs in filters:
        series = func(df, **kwargs)
        assert isinstance(series, pd.Series)
        assert series.index.equals(df.index)
        assert len(series) == len(df)
        assert series.dtype == bool
