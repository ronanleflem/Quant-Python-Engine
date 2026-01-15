"""Psychologic & news filter."""
from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd


def psychologic_and_news_filter(
    df: pd.DataFrame,
    *,
    news_times: Optional[Iterable[str]] = None,
    news_col: Optional[str] = None,
    pre_minutes: int = 30,
    post_minutes: int = 30,
    allow_if_missing: bool = True,
) -> pd.Series:
    """Return True when outside news blackout windows."""
    if not isinstance(df.index, pd.DatetimeIndex):
        if allow_if_missing:
            return pd.Series(True, index=df.index)
        raise ValueError("psychologic_and_news_filter requires a DatetimeIndex")

    if news_col and news_col in df.columns:
        series = df[news_col].astype(bool)
        return (~series).reindex(df.index).fillna(True)

    if not news_times:
        return pd.Series(True, index=df.index) if allow_if_missing else pd.Series(False, index=df.index)

    idx = df.index.tz_convert("UTC") if df.index.tz is not None else df.index.tz_localize("UTC")
    events = pd.to_datetime(list(news_times), utc=True, errors="coerce").dropna()
    if events.empty:
        return pd.Series(True, index=df.index) if allow_if_missing else pd.Series(False, index=df.index)

    pre = pd.Timedelta(minutes=int(pre_minutes))
    post = pd.Timedelta(minutes=int(post_minutes))
    mask = pd.Series(True, index=idx)
    for ts in events:
        window_mask = (idx >= ts - pre) & (idx <= ts + post)
        mask &= ~window_mask
    return mask.reindex(df.index).fillna(True)


__all__ = ["psychologic_and_news_filter"]
