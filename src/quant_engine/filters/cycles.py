"""Cycle detection filter."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common_filters import rolling_apply


def cycles_filter(
    df: pd.DataFrame,
    window: int = 120,
    max_lag: int = 50,
    max_r2: float = 0.5,
    price_col: str = "close",
) -> pd.Series:
    """Return True when dominant autocorrelation is weak (low R^2)."""
    if price_col not in df.columns:
        raise ValueError(f"DataFrame missing column: {price_col}")
    series = df[price_col].astype(float)
    returns = series.diff().fillna(0.0)

    def _max_r2(arr: list[float]) -> float:
        if len(arr) < 5:
            return 1.0
        max_abs = 0.0
        max_l = min(int(max_lag), len(arr) - 2)
        if max_l < 2:
            return 1.0
        data = np.array(arr, dtype=float)
        for lag in range(2, max_l + 1):
            x = data[:-lag]
            y = data[lag:]
            if x.std() == 0 or y.std() == 0:
                continue
            corr = np.corrcoef(x, y)[0, 1]
            if np.isnan(corr):
                continue
            max_abs = max(max_abs, abs(float(corr)))
        return float(max_abs**2)

    r2 = rolling_apply(returns, window, _max_r2)
    return (r2 < float(max_r2)).fillna(False)


__all__ = ["cycles_filter"]
