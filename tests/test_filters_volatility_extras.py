from __future__ import annotations

import pandas as pd

from quant_engine.filters.volatility import volatility_filter


def test_volatility_bollinger_width() -> None:
    idx = pd.date_range("2025-01-01", periods=10, freq="min", tz="UTC")
    close = pd.Series([1.0] * 10, index=idx)
    df = pd.DataFrame(
        {
            "high": close + 0.01,
            "low": close - 0.01,
            "close": close,
        },
        index=idx,
    )
    mask = volatility_filter(df, window=3, max_entropy=1.0, bb_window=5, bb_max_width=0.5)
    assert mask.iloc[-1] == True


def test_volatility_vix_hv_thresholds() -> None:
    idx = pd.date_range("2025-01-01", periods=30, freq="min", tz="UTC")
    close = pd.Series([1.0 + 0.001 * i for i in range(len(idx))], index=idx)
    df = pd.DataFrame(
        {
            "high": close + 0.01,
            "low": close - 0.01,
            "close": close,
        },
        index=idx,
    )
    mask = volatility_filter(
        df,
        hv_window=10,
        max_hv=10.0,
        vix_window=10,
        max_vix=10.0,
    )
    assert mask.iloc[-1] == True
