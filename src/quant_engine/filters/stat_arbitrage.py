"""Statistical arbitrage filters."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common_filters import rolling_apply


def statistical_arbitrage_filter(
    df: pd.DataFrame,
    window: int = 120,
    omega_thresh: float = 1.0,
    info_thresh: float = 0.0,
    price_col: str = "close",
    require_info: bool = False,
) -> pd.Series:
    """Return True when omega ratio (and optionally info ratio) exceeds thresholds."""
    if price_col not in df.columns:
        raise ValueError(f"DataFrame missing column: {price_col}")
    series = df[price_col].astype(float)
    returns = series.pct_change().fillna(0.0)

    def _omega(arr: list[float]) -> float:
        data = np.array(arr, dtype=float)
        pos = data[data > 0].sum()
        neg = abs(data[data < 0].sum())
        if neg == 0:
            return float("inf") if pos > 0 else 0.0
        return float(pos / neg)

    def _info(arr: list[float]) -> float:
        data = np.array(arr, dtype=float)
        std = data.std(ddof=0)
        if std == 0:
            return 0.0
        return float(data.mean() / std)

    omega = rolling_apply(returns, window, _omega)
    info = rolling_apply(returns, window, _info)
    cond = omega > float(omega_thresh)
    if require_info:
        cond &= info > float(info_thresh)
    return cond.fillna(False)


__all__ = ["statistical_arbitrage_filter"]
