"""Condition feature functions for market intelligence."""
from __future__ import annotations

import numpy as np
import pandas as pd


def htf_trend(df: pd.DataFrame, *, tf_multiplier: int, ema_period: int) -> pd.Series:
    """Return higher time frame trend as ``"up"`` or ``"down"``."""

    groups = np.arange(len(df)) // tf_multiplier
    htf_close = df["close"].groupby(groups).last()
    ema = htf_close.ewm(span=ema_period, adjust=False).mean()
    trend_per_group = pd.Series(
        np.where(ema.diff() > 0, "up", "down"), index=htf_close.index
    )
    return pd.Series(groups).map(trend_per_group).astype("category")


def vol_tertile(df: pd.DataFrame, *, window: int) -> pd.Series:
    """Classify current ATR into tertiles across the sample."""

    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr = tr.rolling(window).mean()
    q1, q2 = atr.quantile([1 / 3, 2 / 3])
    if q1 == q2:
        tertiles = pd.Series(["mid"] * len(atr), index=atr.index)
    else:
        tertiles = pd.cut(atr, [-np.inf, q1, q2, np.inf], labels=["low", "mid", "high"])
    return tertiles.astype("category")


def session(df: pd.DataFrame, *, col: str = "session_id") -> pd.Series:
    """Return the session label as a categorical series."""

    return df[col].astype("category")


def hour_bin(df: pd.DataFrame) -> pd.Series:
    """Return hour-of-day bins (0-23) from timestamp column."""
    ts = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    return ts.dt.hour.astype("Int64")


def day_of_week(df: pd.DataFrame) -> pd.Series:
    """Return day-of-week bins (0=Mon..6=Sun)."""
    ts = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    return ts.dt.dayofweek.astype("Int64")


def month_of_year(df: pd.DataFrame) -> pd.Series:
    """Return month-of-year bins (1-12)."""
    ts = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    return ts.dt.month.astype("Int64")


def session_from_ts(df: pd.DataFrame) -> pd.Series:
    """Return a coarse session label from timestamp if session_id is missing."""
    ts = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    hours = ts.dt.hour
    session_id = pd.Series("unknown", index=df.index)
    session_id[(hours >= 23) | (hours < 7)] = "asia"
    session_id[(hours >= 7) & (hours < 15)] = "london"
    session_id[(hours >= 13) & (hours < 21)] = "newyork"
    return session_id.astype("category")
