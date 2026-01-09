"""Psychologic drawdown filters."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .common_filters import rolling_apply


def psychologic_ulcer_filter(
    df: pd.DataFrame,
    window: int = 60,
    max_ulcer: float = 5.0,
    price_col: str = "close",
) -> pd.Series:
    """Return True when Ulcer Index is below max_ulcer."""
    if price_col not in df.columns:
        raise ValueError(f"DataFrame missing column: {price_col}")
    series = df[price_col].astype(float)

    def _ulcer(arr: list[float]) -> float:
        if len(arr) < 2:
            return 0.0
        data = np.array(arr, dtype=float)
        peak = np.maximum.accumulate(data)
        dd = (data - peak) / np.where(peak == 0, np.nan, peak) * 100.0
        dd = np.nan_to_num(dd, nan=0.0)
        return float(math.sqrt((dd**2).mean()))

    ulcer = rolling_apply(series, window, _ulcer)
    return (ulcer < float(max_ulcer)).fillna(False)


__all__ = ["psychologic_ulcer_filter"]
