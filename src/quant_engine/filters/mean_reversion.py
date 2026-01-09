"""Mean reversion probability filter."""
from __future__ import annotations

import pandas as pd


def mean_reversion_probability_filter(
    df: pd.DataFrame,
    window: int = 20,
    bb_mult: float = 2.0,
    keltner_mult: float = 1.5,
    atr_window: int = 14,
    direction: str = "long",
    price_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
) -> pd.Series:
    """Return True when price is outside both BB and Keltner channels."""
    for col in (price_col, high_col, low_col):
        if col not in df.columns:
            raise ValueError(f"DataFrame missing column: {col}")
    close = df[price_col].astype(float)
    high = df[high_col].astype(float)
    low = df[low_col].astype(float)

    sma = close.rolling(window, min_periods=window).mean()
    std = close.rolling(window, min_periods=window).std(ddof=0)
    bb_upper = sma + float(bb_mult) * std
    bb_lower = sma - float(bb_mult) * std

    prev_close = close.shift(1)
    tr_components = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1)
    tr = tr_components.max(axis=1)
    atr = tr.ewm(alpha=1.0 / float(atr_window), adjust=False).mean()
    keltner_mid = close.ewm(span=int(window), adjust=False).mean()
    keltner_upper = keltner_mid + float(keltner_mult) * atr
    keltner_lower = keltner_mid - float(keltner_mult) * atr

    if direction.lower() == "short":
        cond = (close > bb_upper) & (close > keltner_upper)
    else:
        cond = (close < bb_lower) & (close < keltner_lower)
    return cond.fillna(False)


__all__ = ["mean_reversion_probability_filter"]
