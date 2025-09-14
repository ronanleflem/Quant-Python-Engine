"""Regime condition helpers for statistics runs."""
from __future__ import annotations

import numpy as np
import pandas as pd


def htf_trend(df: pd.DataFrame, *, tf_multiplier: int, ema_period: int) -> pd.Series:
    """Return higher time frame trend as ``"up"`` or ``"down"``.

    The dataset is downsampled by ``tf_multiplier`` and an EMA of ``ema_period``
    is computed on the resulting series.  Trend labels are then forward filled
    to the original frequency.
    """

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

