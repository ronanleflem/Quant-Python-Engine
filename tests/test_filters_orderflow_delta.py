from __future__ import annotations

import pandas as pd

from quant_engine.filters.orderflow import orderflow_delta_filter


def test_orderflow_delta_allows_missing_when_configured() -> None:
    idx = pd.date_range("2025-01-01", periods=3, freq="min", tz="UTC")
    df = pd.DataFrame({"close": [1.0, 1.01, 1.0]}, index=idx)
    mask = orderflow_delta_filter(df, allow_if_missing=True)
    assert mask.all()


def test_orderflow_delta_blocks_on_negative_delta() -> None:
    idx = pd.date_range("2025-01-01", periods=4, freq="min", tz="UTC")
    df = pd.DataFrame(
        {
            "buy_volume": [100, 90, 80, 70],
            "sell_volume": [120, 110, 90, 85],
        },
        index=idx,
    )
    mask = orderflow_delta_filter(df, min_delta=5.0, allow_if_missing=False)
    assert mask.any() == False


def test_orderflow_delta_ratio_side_sell() -> None:
    idx = pd.date_range("2025-01-01", periods=4, freq="min", tz="UTC")
    df = pd.DataFrame(
        {
            "buy_volume": [100, 90, 80, 70],
            "sell_volume": [200, 150, 120, 110],
            "volume": [300, 240, 200, 180],
        },
        index=idx,
    )
    mask = orderflow_delta_filter(
        df,
        mode="ratio",
        min_ratio=0.2,
        side="sell",
        allow_if_missing=False,
    )
    assert mask.iloc[-1] == True
