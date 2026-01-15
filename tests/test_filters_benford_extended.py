from __future__ import annotations

import pandas as pd

from quant_engine.filters.benford import benford_law_filter


def _make_df() -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=10, freq="1min", tz="UTC")
    data = {
        "open": [1.0 + i * 0.01 for i in range(len(idx))],
        "high": [1.1 + i * 0.01 for i in range(len(idx))],
        "low": [0.9 + i * 0.01 for i in range(len(idx))],
        "close": [1.0 + i * 0.01 for i in range(len(idx))],
        "volume": [100 + i for i in range(len(idx))],
    }
    return pd.DataFrame(data, index=idx)


def test_benford_atr_series_type() -> None:
    df = _make_df()
    mask = benford_law_filter(
        df,
        window=3,
        series_type="atr",
        atr_window=3,
        metric="mad",
        mad_threshold=1.0,
    )
    assert len(mask) == len(df)
    assert mask.dtype == bool


def test_benford_wick_series_type() -> None:
    df = _make_df()
    mask = benford_law_filter(
        df,
        window=3,
        series_type="wick",
        metric="mad",
        mad_threshold=1.0,
    )
    assert len(mask) == len(df)
    assert mask.dtype == bool
