from __future__ import annotations

import pandas as pd

from quant_engine.filters.macro_cot_oi import macro_cot_oi_filter


def test_macro_cot_oi_allows_missing_when_configured() -> None:
    idx = pd.date_range("2025-01-01", periods=3, freq="min", tz="UTC")
    df = pd.DataFrame({"close": [1.0, 1.01, 1.0]}, index=idx)
    mask = macro_cot_oi_filter(df, allow_if_missing=True)
    assert mask.all()


def test_macro_cot_oi_bullish_thresholds() -> None:
    idx = pd.date_range("2025-01-01", periods=3, freq="min", tz="UTC")
    df = pd.DataFrame(
        {
            "cot_bias": [0.2, 0.5, 0.8],
            "oi_change": [0.1, 0.3, 0.6],
        },
        index=idx,
    )
    mask = macro_cot_oi_filter(
        df,
        cot_bias_threshold=0.4,
        oi_change_threshold=0.2,
        side="bull",
        require_all=True,
        allow_if_missing=False,
    )
    assert mask.iloc[0] == False
    assert mask.iloc[-1] == True


def test_macro_cot_oi_bearish_any() -> None:
    idx = pd.date_range("2025-01-01", periods=3, freq="min", tz="UTC")
    df = pd.DataFrame(
        {
            "cot_bias": [-0.6, -0.2, 0.1],
            "oi_change": [-0.1, -0.4, 0.2],
        },
        index=idx,
    )
    mask = macro_cot_oi_filter(
        df,
        cot_bias_threshold=0.5,
        oi_change_threshold=0.3,
        side="bear",
        require_all=False,
        allow_if_missing=False,
    )
    assert mask.iloc[0] == True
    assert mask.iloc[-1] == False
