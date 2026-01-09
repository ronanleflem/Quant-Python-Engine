"""Indicator-driven rules."""
from __future__ import annotations

import numpy as np
import pandas as pd


def atr_rising_filter(
    df: pd.DataFrame,
    window: int = 14,
    lookback: int = 1,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pd.Series:
    """Return True when ATR is rising vs previous bar or lookback mean."""
    for col in (high_col, low_col, close_col):
        if col not in df.columns:
            raise ValueError(f"DataFrame missing column: {col}")
    high = df[high_col].astype(float)
    low = df[low_col].astype(float)
    close = df[close_col].astype(float)
    prev_close = close.shift(1)
    tr_components = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    )
    tr = tr_components.max(axis=1)
    atr = tr.ewm(alpha=1.0 / float(window), adjust=False).mean()
    if lookback <= 1:
        return (atr > atr.shift(1)).fillna(False)
    base = atr.rolling(int(lookback), min_periods=int(lookback)).mean()
    return (atr > base).fillna(False)


def linear_regression_macd_cross_filter(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    lookback: int = 20,
    direction: str = "bullish",
    price_col: str = "close",
) -> pd.Series:
    """Predict MACD cross using linear regression on MACD-signal diff."""
    if price_col not in df.columns:
        raise ValueError(f"DataFrame missing column: {price_col}")
    close = df[price_col].astype(float)
    ema_fast = close.ewm(span=int(fast), adjust=False).mean()
    ema_slow = close.ewm(span=int(slow), adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=int(signal), adjust=False).mean()
    diff = (macd - sig).astype(float)

    def _predict(arr: np.ndarray) -> float:
        if len(arr) < 2:
            return float(arr[-1]) if len(arr) else 0.0
        x = np.arange(len(arr), dtype=float)
        slope, intercept = np.polyfit(x, arr.astype(float), 1)
        return float(slope * (len(arr)) + intercept)

    pred = diff.rolling(int(lookback), min_periods=int(lookback)).apply(_predict, raw=True)
    if direction.lower() == "bearish":
        cond = (diff >= 0) & (pred < 0)
    else:
        cond = (diff <= 0) & (pred > 0)
    return cond.fillna(False)


__all__ = [
    "atr_rising_filter",
    "linear_regression_macd_cross_filter",
]
