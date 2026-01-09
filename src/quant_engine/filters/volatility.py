"""Volatility filter inspired by Java."""
from __future__ import annotations

import numpy as np
import pandas as pd


def volatility_filter(
    df: pd.DataFrame,
    window: int = 60,
    max_entropy: float = 0.7,
    atr_window: int | None = None,
    max_atr_pct: float | None = None,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pd.Series:
    """Return True when entropy (and optional ATR pct) are below thresholds."""
    for col in (high_col, low_col, close_col):
        if col not in df.columns:
            raise ValueError(f"DataFrame missing column: {col}")
    close = df[close_col].astype(float)
    direction = np.sign(close.diff().fillna(0.0))
    up_prob = (direction > 0).astype(int).rolling(window, min_periods=max(8, window // 4)).mean()
    eps = 1e-12
    entropy = -(up_prob * np.log2(up_prob + eps) + (1 - up_prob) * np.log2(1 - up_prob + eps))
    cond = entropy <= float(max_entropy)

    if atr_window is not None and max_atr_pct is not None:
        high = df[high_col].astype(float)
        low = df[low_col].astype(float)
        prev_close = close.shift(1)
        tr_components = pd.concat(
            [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
        )
        tr = tr_components.max(axis=1)
        atr = tr.ewm(alpha=1.0 / float(atr_window), adjust=False).mean()
        atr_pct = atr / close.replace(0.0, np.nan)
        cond &= atr_pct <= float(max_atr_pct)

    return cond.fillna(False)


__all__ = ["volatility_filter"]
