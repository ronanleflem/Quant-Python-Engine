from __future__ import annotations

import pandas as pd

from quant_engine.filters.lt_confluence import lower_timeframe_confluence_filter


def _make_df() -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=10, freq="min", tz="UTC")
    close = pd.Series([1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09], index=idx)
    df = pd.DataFrame(
        {
            "open": close - 0.001,
            "high": close + 0.002,
            "low": close - 0.002,
            "close": close,
            "volume": [100] * len(close),
            "buy_volume": [120] * len(close),
            "sell_volume": [80] * len(close),
        },
        index=idx,
    )
    return df


def test_lower_timeframe_confluence_scores() -> None:
    df = _make_df()
    mask = lower_timeframe_confluence_filter(
        df,
        side="long",
        momentum_period=2,
        min_momentum=0.0,
        adx_window=3,
        adx_thresh=5.0,
        vwap_max_dev=0.02,
        vwap_side="any",
        delta_min=10.0,
        min_score=3,
        allow_if_missing=False,
    )
    assert mask.iloc[-1] == True


def test_lower_timeframe_confluence_require_all() -> None:
    df = _make_df()
    mask = lower_timeframe_confluence_filter(
        df,
        side="short",
        momentum_period=2,
        min_momentum=0.0,
        adx_window=3,
        adx_thresh=5.0,
        vwap_max_dev=0.02,
        vwap_side="any",
        delta_min=10.0,
        require_all=True,
        allow_if_missing=False,
    )
    assert mask.iloc[-1] == False
