"""Stationarity-inspired filter."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common_filters import rolling_apply


def stationarity_filter(
    df: pd.DataFrame,
    window: int = 120,
    max_abs_autocorr: float = 0.7,
    price_col: str = "close",
) -> pd.Series:
    """Return True when lag-1 autocorrelation is below threshold."""
    if price_col not in df.columns:
        raise ValueError(f"DataFrame missing column: {price_col}")
    series = df[price_col].astype(float)
    returns = series.diff().fillna(0.0)

    def _acf1(arr: list[float]) -> float:
        if len(arr) < 3:
            return 1.0
        data = np.array(arr, dtype=float)
        x = data[:-1]
        y = data[1:]
        if x.std() == 0 or y.std() == 0:
            return 1.0
        return float(np.corrcoef(x, y)[0, 1])

    acf1 = rolling_apply(returns, window, _acf1)
    return (acf1.abs() < float(max_abs_autocorr)).fillna(False)


__all__ = ["stationarity_filter"]
