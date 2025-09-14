"""Event definitions for statistics runs."""
from __future__ import annotations

import pandas as pd


def k_consecutive(df: pd.DataFrame, *, k: int, direction: str) -> pd.Series:
    """Return ``True`` when ``k`` consecutive bars close in ``direction``.

    Parameters
    ----------
    df:
        DataFrame containing at least ``open`` and ``close`` columns.
    k:
        Number of consecutive bars to check.
    direction:
        ``"up"`` for positive closes, ``"down"`` for negative closes.
    """

    up = df["close"] > df["open"]
    down = df["open"] > df["close"]
    series = up if direction == "up" else down
    return series.rolling(k).sum().eq(k).fillna(False)


def shock_atr(df: pd.DataFrame, *, mult: float, window: int) -> pd.Series:
    """True if the current true range exceeds ``mult`` times the ATR."""

    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr = tr.rolling(window).mean()
    return (tr > mult * atr).fillna(False)


def breakout_hhll(df: pd.DataFrame, *, lookback: int, type: str) -> pd.Series:
    """Detect breakout of higher highs or lower lows.

    Parameters
    ----------
    df:
        OHLC DataFrame.
    lookback:
        Number of bars to look back when computing the reference high/low.
    type:
        ``"hh"`` to detect new highs, ``"ll"`` for new lows.
    """

    if type == "hh":
        ref = df["high"].shift(1).rolling(lookback, min_periods=1).max()
        return (df["high"] > ref).fillna(False)
    if type == "ll":
        ref = df["low"].shift(1).rolling(lookback, min_periods=1).min()
        return (df["low"] < ref).fillna(False)
    raise ValueError("type must be 'hh' or 'll'")

