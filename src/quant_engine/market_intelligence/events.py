"""Event feature functions for market intelligence."""
from __future__ import annotations

import pandas as pd


def k_consecutive(df: pd.DataFrame, *, k: int, direction: str) -> pd.Series:
    """Return ``True`` when ``k`` consecutive bars close in ``direction``."""

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
    """Detect breakout of higher highs or lower lows."""

    if type == "hh":
        ref = df["high"].shift(1).rolling(lookback, min_periods=1).max()
        return (df["high"] > ref).fillna(False)
    if type == "ll":
        ref = df["low"].shift(1).rolling(lookback, min_periods=1).min()
        return (df["low"] < ref).fillna(False)
    raise ValueError("type must be 'hh' or 'll'")


def bullish_candle(df: pd.DataFrame) -> pd.Series:
    """True when close > open."""
    return (df["close"] > df["open"]).fillna(False)


def bearish_candle(df: pd.DataFrame) -> pd.Series:
    """True when close < open."""
    return (df["close"] < df["open"]).fillna(False)


def bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    """True on bullish engulfing pattern."""
    prev_open = df["open"].shift(1)
    prev_close = df["close"].shift(1)
    prev_bear = prev_close < prev_open
    curr_bull = df["close"] > df["open"]
    engulf = (df["close"] >= prev_open) & (df["open"] <= prev_close)
    return (prev_bear & curr_bull & engulf).fillna(False)


def bearish_engulfing(df: pd.DataFrame) -> pd.Series:
    """True on bearish engulfing pattern."""
    prev_open = df["open"].shift(1)
    prev_close = df["close"].shift(1)
    prev_bull = prev_close > prev_open
    curr_bear = df["close"] < df["open"]
    engulf = (df["open"] >= prev_close) & (df["close"] <= prev_open)
    return (prev_bull & curr_bear & engulf).fillna(False)


def bullish_streak(df: pd.DataFrame, *, k: int = 3) -> pd.Series:
    """True when k consecutive bullish candles occur."""
    return k_consecutive(df, k=k, direction="up")


def bearish_streak(df: pd.DataFrame, *, k: int = 3) -> pd.Series:
    """True when k consecutive bearish candles occur."""
    return k_consecutive(df, k=k, direction="down")


def gap_up(df: pd.DataFrame) -> pd.Series:
    """True when current low is above previous high."""
    prev_high = df["high"].shift(1)
    return (df["low"] > prev_high).fillna(False)


def gap_down(df: pd.DataFrame) -> pd.Series:
    """True when current high is below previous low."""
    prev_low = df["low"].shift(1)
    return (df["high"] < prev_low).fillna(False)


def always_true(df: pd.DataFrame) -> pd.Series:
    """Always-true event for frequency calculations."""
    return pd.Series(True, index=df.index)
