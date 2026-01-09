"""Trend direction filter."""
from __future__ import annotations

import pandas as pd


def trend_filter(
    df: pd.DataFrame,
    direction: str = "up",
    lookback: int = 20,
    method: str = "hhhl",
    ema_fast: int = 20,
    ema_slow: int = 50,
    price_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
) -> pd.Series:
    """Return True when trend conditions are met."""
    for col in (price_col, high_col, low_col):
        if col not in df.columns:
            raise ValueError(f"DataFrame missing column: {col}")
    close = df[price_col].astype(float)
    high = df[high_col].astype(float)
    low = df[low_col].astype(float)
    direction_key = direction.strip().lower()

    if method == "ema":
        fast = close.ewm(span=int(ema_fast), adjust=False).mean()
        slow = close.ewm(span=int(ema_slow), adjust=False).mean()
        if direction_key == "down":
            return (fast < slow).fillna(False)
        return (fast > slow).fillna(False)

    swing_high = high.rolling(int(lookback), min_periods=int(lookback)).max().shift(1)
    swing_low = low.rolling(int(lookback), min_periods=int(lookback)).min().shift(1)
    hh = high > swing_high
    hl = low > swing_low
    lh = high < swing_high
    ll = low < swing_low

    if direction_key == "down":
        return (lh & ll).fillna(False)
    return (hh & hl).fillna(False)


__all__ = ["trend_filter"]
