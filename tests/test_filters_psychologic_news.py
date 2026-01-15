from __future__ import annotations

import pandas as pd

from quant_engine.filters.psychologic_news import psychologic_and_news_filter


def test_psychologic_news_blackout_window() -> None:
    idx = pd.date_range("2025-01-01 12:00:00", periods=5, freq="min", tz="UTC")
    df = pd.DataFrame({"close": [1, 2, 3, 4, 5]}, index=idx)
    mask = psychologic_and_news_filter(
        df,
        news_times=["2025-01-01T12:02:00Z"],
        pre_minutes=1,
        post_minutes=1,
        allow_if_missing=False,
    )
    assert mask.iloc[1] == False
    assert mask.iloc[2] == False
    assert mask.iloc[3] == False


def test_psychologic_news_column() -> None:
    idx = pd.date_range("2025-01-01 12:00:00", periods=3, freq="min", tz="UTC")
    df = pd.DataFrame({"news_hit": [False, True, False]}, index=idx)
    mask = psychologic_and_news_filter(df, news_col="news_hit", allow_if_missing=False)
    assert mask.iloc[1] == False
