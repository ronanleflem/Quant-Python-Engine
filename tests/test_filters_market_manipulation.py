from __future__ import annotations

import pandas as pd

from quant_engine.filters.market_manipulation import market_manipulation_filter


def _make_df(close: list[float], high: list[float], low: list[float]) -> pd.DataFrame:
    n = len(close)
    idx = pd.date_range("2025-01-01", periods=n, freq="min", tz="UTC")
    return pd.DataFrame(
        {
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": [100.0] * n,
        },
        index=idx,
    )


def test_market_manipulation_returns_mask() -> None:
    close = [1.0, 1.01, 1.0, 1.02, 1.01, 1.0]
    high = [c + 0.01 for c in close]
    low = [c - 0.01 for c in close]
    df = _make_df(close, high, low)
    mask = market_manipulation_filter(df, window=3, atr_window=3, atr_mult=2.0)
    assert isinstance(mask, pd.Series)
    assert mask.index.equals(df.index)
    assert mask.dtype == bool


def test_market_manipulation_blocks_on_vol_spike() -> None:
    close = [1.0] * 10 + [1.2]
    high = [1.01] * 10 + [1.6]
    low = [0.99] * 10 + [0.5]
    df = _make_df(close, high, low)
    mask = market_manipulation_filter(
        df,
        window=5,
        entropy_threshold=0.95,
        kurtosis_threshold=10.0,
        atr_window=3,
        atr_mult=1.5,
        require_all=False,
    )
    assert mask.iloc[-1] == False


def test_market_manipulation_require_all() -> None:
    close = [1.0, 1.01, 1.0, 1.01, 1.0, 1.01, 1.0, 1.01, 1.0, 1.01]
    high = [c + 0.01 for c in close]
    low = [c - 0.01 for c in close]
    df = _make_df(close, high, low)
    mask_any = market_manipulation_filter(
        df,
        window=5,
        entropy_threshold=0.8,
        kurtosis_threshold=99.0,
        atr_window=3,
        atr_mult=99.0,
        require_all=False,
    )
    mask_all = market_manipulation_filter(
        df,
        window=5,
        entropy_threshold=0.8,
        kurtosis_threshold=99.0,
        atr_window=3,
        atr_mult=99.0,
        require_all=True,
    )
    assert mask_any.iloc[-1] == False
    assert mask_all.iloc[-1] == True
