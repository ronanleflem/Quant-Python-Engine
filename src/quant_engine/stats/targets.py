"""Target metrics for statistics runs."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def up_next_bar(df: pd.DataFrame, **params: Any) -> pd.Series:
    """True if the next bar closes higher than the current bar."""

    next_close = df["close"].shift(-1)
    res = (next_close > df["close"]).astype("boolean")
    return res.mask(next_close.isna())


def continuation_n(df: pd.DataFrame, *, n: int, direction: str) -> pd.Series:
    """Continuation of the move after ``n`` bars in ``direction``."""

    future = df["close"].shift(-n)
    if direction == "up":
        res = (future > df["close"]).astype("boolean")
    else:
        res = (future < df["close"]).astype("boolean")
    return res.mask(future.isna())


def time_to_reversal(df: pd.DataFrame, *, max_horizon: int) -> pd.Series:
    """Number of bars until price movement reverses direction."""

    close = df["close"].to_numpy()
    n = len(close)
    out = np.full(n, np.nan)
    for i in range(n - 1):
        initial = np.sign(close[i + 1] - close[i])
        if initial == 0 or np.isnan(initial):
            continue
        for j in range(1, max_horizon + 1):
            if i + j >= n:
                break
            step = np.sign(close[i + j] - close[i + j - 1])
            if step == -initial and step != 0:
                out[i] = j
                break
            if j == max_horizon:
                out[i] = max_horizon
    return pd.Series(out).astype("Int64")


def is_bullish(df: pd.DataFrame) -> pd.Series:
    """True when close > open."""
    return (df["close"] > df["open"]).astype("boolean")


def is_bearish(df: pd.DataFrame) -> pd.Series:
    """True when close < open."""
    return (df["close"] < df["open"]).astype("boolean")


def next_bullish(df: pd.DataFrame) -> pd.Series:
    """True when next candle is bullish."""
    next_close = df["close"].shift(-1)
    next_open = df["open"].shift(-1)
    res = (next_close > next_open).astype("boolean")
    return res.mask(next_close.isna())


def next_bearish(df: pd.DataFrame) -> pd.Series:
    """True when next candle is bearish."""
    next_close = df["close"].shift(-1)
    next_open = df["open"].shift(-1)
    res = (next_close < next_open).astype("boolean")
    return res.mask(next_close.isna())


def body_ratio(df: pd.DataFrame) -> pd.Series:
    """Body size divided by total candle range."""
    body = (df["close"] - df["open"]).abs()
    rng = (df["high"] - df["low"]).replace(0.0, np.nan)
    return (body / rng).astype(float)


def upper_wick_ratio(df: pd.DataFrame) -> pd.Series:
    """Upper wick size divided by total candle range."""
    top = df[["open", "close"]].max(axis=1)
    upper = (df["high"] - top).clip(lower=0.0)
    rng = (df["high"] - df["low"]).replace(0.0, np.nan)
    return (upper / rng).astype(float)


def lower_wick_ratio(df: pd.DataFrame) -> pd.Series:
    """Lower wick size divided by total candle range."""
    bottom = df[["open", "close"]].min(axis=1)
    lower = (bottom - df["low"]).clip(lower=0.0)
    rng = (df["high"] - df["low"]).replace(0.0, np.nan)
    return (lower / rng).astype(float)


def candle_std(df: pd.DataFrame, *, window: int = 20) -> pd.Series:
    """Rolling standard deviation of candle body size."""
    body = (df["close"] - df["open"]).abs()
    return body.rolling(window, min_periods=window).std(ddof=0)


def candle_zscore(df: pd.DataFrame, *, window: int = 20) -> pd.Series:
    """Z-score of candle body size."""
    body = (df["close"] - df["open"]).abs()
    mean = body.rolling(window, min_periods=window).mean()
    std = body.rolling(window, min_periods=window).std(ddof=0)
    return (body - mean) / std.replace(0.0, np.nan)


def breakout_high_first(
    df: pd.DataFrame,
    *,
    lookback: int = 20,
    horizon: int = 10,
) -> pd.Series:
    """True if breakout above prior high happens before breakout below prior low."""
    highs = df["high"].astype(float)
    lows = df["low"].astype(float)
    ref_high = highs.shift(1).rolling(lookback, min_periods=1).max()
    ref_low = lows.shift(1).rolling(lookback, min_periods=1).min()
    n = len(df)
    out = np.full(n, np.nan)
    for i in range(n):
        if i + 1 >= n:
            continue
        hi_level = ref_high.iloc[i]
        lo_level = ref_low.iloc[i]
        hi_first = None
        lo_first = None
        for j in range(1, horizon + 1):
            if i + j >= n:
                break
            if hi_first is None and highs.iloc[i + j] > hi_level:
                hi_first = j
            if lo_first is None and lows.iloc[i + j] < lo_level:
                lo_first = j
            if hi_first is not None or lo_first is not None:
                if hi_first is not None and lo_first is not None:
                    break
        if hi_first is None and lo_first is None:
            continue
        if lo_first is None or (hi_first is not None and hi_first < lo_first):
            out[i] = 1.0
        else:
            out[i] = 0.0
    return pd.Series(out)


def breakout_low_first(
    df: pd.DataFrame,
    *,
    lookback: int = 20,
    horizon: int = 10,
) -> pd.Series:
    """True if breakout below prior low happens before breakout above prior high."""
    res = breakout_high_first(df, lookback=lookback, horizon=horizon)
    return res.where(res.isna(), 1.0 - res)


def retracement_probability(
    df: pd.DataFrame,
    *,
    lookback: int = 20,
    horizon: int = 10,
    direction: str = "up",
) -> pd.Series:
    """True when price retraces to the breakout level within horizon."""
    highs = df["high"].astype(float)
    lows = df["low"].astype(float)
    close = df["close"].astype(float)
    ref_high = highs.shift(1).rolling(lookback, min_periods=1).max()
    ref_low = lows.shift(1).rolling(lookback, min_periods=1).min()
    n = len(df)
    out = np.full(n, np.nan)
    direction_key = direction.lower()
    for i in range(n):
        if i + 1 >= n:
            continue
        if direction_key == "down":
            breakout = close.iloc[i] < ref_low.iloc[i]
            level = ref_low.iloc[i]
            if not breakout:
                continue
            retraced = any(highs.iloc[i + 1 : i + 1 + horizon] >= level)
        else:
            breakout = close.iloc[i] > ref_high.iloc[i]
            level = ref_high.iloc[i]
            if not breakout:
                continue
            retraced = any(lows.iloc[i + 1 : i + 1 + horizon] <= level)
        out[i] = 1.0 if retraced else 0.0
    return pd.Series(out)

