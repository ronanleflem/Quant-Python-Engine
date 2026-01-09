"""Shared helpers for Java parity filters."""
from __future__ import annotations

from typing import Callable

import pandas as pd


def rolling_apply(values: pd.Series, window: int, fn: Callable[[list[float]], float]) -> pd.Series:
    if window <= 1:
        raise ValueError("window must be > 1")
    return values.rolling(window, min_periods=window).apply(lambda arr: fn(arr.tolist()), raw=True).reindex(values.index)
