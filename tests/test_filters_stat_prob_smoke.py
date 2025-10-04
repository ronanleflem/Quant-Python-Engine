"""Smoke tests for statistical & probabilistic filters."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_engine.filters.stat_prob import (
    entropy_window_filter,
    hurst_regime_filter,
    k_consecutive_filter,
    seasonality_bin_filter,
)


def _make_ohlc(count: int = 720) -> pd.DataFrame:
    rng = np.random.default_rng(1234)
    index = pd.date_range("2024-01-01", periods=count, freq="min", tz="UTC")
    close = 1.10 + rng.normal(scale=0.0005, size=count).cumsum()
    open_ = close - rng.normal(scale=0.0002, size=count)
    high = np.maximum(open_, close) + np.abs(rng.normal(scale=0.0003, size=count))
    low = np.minimum(open_, close) - np.abs(rng.normal(scale=0.0003, size=count))
    volume = rng.integers(1_000, 5_000, size=count)
    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )
    return df


def test_k_consecutive_filter_detects_sequences() -> None:
    df = _make_ohlc()
    seq_idx = df.index[50:53]
    df.loc[seq_idx, "open"] = 1.0
    df.loc[seq_idx, "close"] = [1.2, 1.3, 1.4]

    result = k_consecutive_filter(df, k=3, direction="up", use_body=True)

    assert result.index.equals(df.index)
    assert result.dtype == bool
    assert result.loc[seq_idx[-1]]


def test_seasonality_bin_filter_hour_allowed_only() -> None:
    df = _make_ohlc()
    mask = seasonality_bin_filter(df, mode="hour", allowed=[8, 9])

    assert mask.index.equals(df.index)
    assert mask.dtype == bool
    allowed_hours = set(df.index[mask].hour)
    assert allowed_hours <= {8, 9}
    blocked_hours = set(df.index[~mask].hour)
    assert blocked_hours.intersection({8, 9}) == set()


def test_hurst_regime_filter_returns_boolean_series() -> None:
    df = _make_ohlc()
    result = hurst_regime_filter(df, window=128, min_h=0.0, max_h=1.0)

    assert result.index.equals(df.index)
    assert result.dtype == bool
    assert not result.isna().any()
    assert result.iloc[200:400].all()  # windows fully populated should be True within wide range


def test_entropy_window_filter_has_true_and_false() -> None:
    df = _make_ohlc()
    trend_idx = df.index[120:220]
    df.loc[trend_idx, "open"] = 1.0
    df.loc[trend_idx, "close"] = np.linspace(1.01, 1.20, len(trend_idx))

    result = entropy_window_filter(df, window=64, max_entropy=0.9)

    assert result.index.equals(df.index)
    assert result.dtype == bool
    unique_values = set(result.unique().tolist())
    assert unique_values.issubset({True, False})
    assert unique_values == {True, False}
