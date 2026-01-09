"""Liquidity filters inspired by Java."""
from __future__ import annotations

import numpy as np
import pandas as pd


def liquidity_cmf_filter(
    df: pd.DataFrame,
    window: int = 20,
    threshold: float = 0.2,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    volume_col: str = "volume",
) -> pd.Series:
    """Return True when absolute Chaikin Money Flow exceeds threshold."""
    for col in (high_col, low_col, close_col, volume_col):
        if col not in df.columns:
            raise ValueError(f"DataFrame missing column: {col}")
    high = df[high_col].astype(float)
    low = df[low_col].astype(float)
    close = df[close_col].astype(float)
    volume = df[volume_col].astype(float)
    denom = (high - low).replace(0.0, np.nan)
    mfm = ((close - low) - (high - close)) / denom
    mfv = mfm * volume
    cmf = mfv.rolling(window, min_periods=window).sum() / volume.rolling(window, min_periods=window).sum()
    return (cmf.abs() > float(threshold)).fillna(False)


__all__ = ["liquidity_cmf_filter"]
