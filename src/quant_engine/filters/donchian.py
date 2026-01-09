"""Donchian channels breakout filter."""
from __future__ import annotations

import pandas as pd


def donchian_channels_filter(
    df: pd.DataFrame,
    window: int = 20,
    direction: str = "up",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pd.Series:
    """Return True on Donchian breakout in the requested direction."""
    for col in (high_col, low_col, close_col):
        if col not in df.columns:
            raise ValueError(f"DataFrame missing column: {col}")
    highs = df[high_col].astype(float)
    lows = df[low_col].astype(float)
    close = df[close_col].astype(float)
    upper = highs.rolling(window, min_periods=window).max().shift(1)
    lower = lows.rolling(window, min_periods=window).min().shift(1)
    if direction.lower() == "down":
        return (close < lower).fillna(False)
    return (close > upper).fillna(False)


__all__ = ["donchian_channels_filter"]
