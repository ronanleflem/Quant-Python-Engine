"""Entry-style filters derived from common indicators."""
from __future__ import annotations

import pandas as pd


def rsi_entry_filter(
    df: pd.DataFrame,
    window: int = 14,
    threshold: float = 30.0,
    direction: str = "below",
    price_col: str = "close",
) -> pd.Series:
    """Return True when RSI crosses below/above a threshold."""
    if price_col not in df.columns:
        raise ValueError(f"DataFrame missing column: {price_col}")
    close = df[price_col].astype(float)
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / float(window), adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / float(window), adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    if direction.lower() == "above":
        return (rsi >= float(threshold)).fillna(False)
    return (rsi <= float(threshold)).fillna(False)


def macd_entry_filter(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    direction: str = "bullish",
    price_col: str = "close",
) -> pd.Series:
    """Return True on MACD cross in the requested direction."""
    if price_col not in df.columns:
        raise ValueError(f"DataFrame missing column: {price_col}")
    close = df[price_col].astype(float)
    ema_fast = close.ewm(span=int(fast), adjust=False).mean()
    ema_slow = close.ewm(span=int(slow), adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=int(signal), adjust=False).mean()
    if direction.lower() == "bearish":
        cross = (macd < sig) & (macd.shift(1) >= sig.shift(1))
    else:
        cross = (macd > sig) & (macd.shift(1) <= sig.shift(1))
    return cross.fillna(False)


def volume_above_average_filter(
    df: pd.DataFrame,
    window: int = 20,
    multiplier: float = 1.0,
    volume_col: str = "volume",
) -> pd.Series:
    """Return True when volume is above rolling average."""
    if volume_col not in df.columns:
        raise ValueError(f"DataFrame missing column: {volume_col}")
    volume = df[volume_col].astype(float)
    avg = volume.rolling(window, min_periods=window).mean()
    return (volume >= avg * float(multiplier)).fillna(False)


__all__ = ["rsi_entry_filter", "macd_entry_filter", "volume_above_average_filter"]
