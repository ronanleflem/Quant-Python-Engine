"""Orderflow / volume delta filter."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def orderflow_delta_filter(
    df: pd.DataFrame,
    *,
    delta_col: Optional[str] = None,
    buy_col: str = "buy_volume",
    sell_col: str = "sell_volume",
    volume_col: str = "volume",
    mode: str = "delta",
    window: int = 20,
    min_delta: Optional[float] = None,
    min_ratio: Optional[float] = None,
    zscore_thresh: Optional[float] = None,
    side: str = "buy",
    allow_if_missing: bool = True,
    require_positive: bool = True,
) -> pd.Series:
    """Return True when orderflow delta conditions are satisfied."""
    mode_key = str(mode).strip().lower()
    side_key = str(side).strip().lower()

    delta_series: Optional[pd.Series] = None
    if delta_col and delta_col in df.columns:
        delta_series = df[delta_col].astype(float)
    elif buy_col in df.columns and sell_col in df.columns:
        delta_series = df[buy_col].astype(float) - df[sell_col].astype(float)
    elif allow_if_missing:
        return pd.Series(True, index=df.index)
    else:
        raise ValueError("orderflow_delta_filter missing delta or buy/sell columns")

    if delta_series is None:
        return pd.Series(True, index=df.index)

    if mode_key == "ratio":
        if volume_col not in df.columns:
            return pd.Series(True, index=df.index) if allow_if_missing else pd.Series(False, index=df.index)
        vol = df[volume_col].astype(float).replace(0.0, np.nan)
        score = delta_series / vol
    else:
        score = delta_series

    if side_key in {"sell", "short", "down", "negative"}:
        score = -score

    cond = pd.Series(True, index=df.index)
    if require_positive:
        cond &= score > 0
    if min_delta is not None and mode_key == "delta":
        cond &= score >= float(min_delta)
    if min_ratio is not None and mode_key == "ratio":
        cond &= score >= float(min_ratio)
    if zscore_thresh is not None:
        mean = score.rolling(window, min_periods=max(3, window // 3)).mean()
        std = score.rolling(window, min_periods=max(3, window // 3)).std(ddof=0)
        zscore = (score - mean) / std.replace(0.0, np.nan)
        cond &= zscore >= float(zscore_thresh)

    return cond.fillna(False).astype(bool)


__all__ = ["orderflow_delta_filter"]
