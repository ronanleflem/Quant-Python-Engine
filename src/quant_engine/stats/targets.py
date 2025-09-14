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

