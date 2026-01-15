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
    bb_window: int | None = None,
    bb_max_width: float | None = None,
    hv_window: int | None = None,
    max_hv: float | None = None,
    vix_window: int | None = None,
    max_vix: float | None = None,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pd.Series:
    """Return True when entropy (and optional ATR/BB/HV/VIX) are below thresholds."""
    for col in (high_col, low_col, close_col):
        if col not in df.columns:
            raise ValueError(f"DataFrame missing column: {col}")
    close = df[close_col].astype(float)
    direction = np.sign(close.diff().fillna(0.0))
    min_periods = min(window, max(2, max(8, window // 4)))
    up_prob = (direction > 0).astype(int).rolling(window, min_periods=min_periods).mean()
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

    if bb_window is not None and bb_max_width is not None:
        mean = close.rolling(bb_window, min_periods=bb_window).mean()
        std = close.rolling(bb_window, min_periods=bb_window).std(ddof=0)
        upper = mean + 2 * std
        lower = mean - 2 * std
        width = (upper - lower) / mean.replace(0.0, np.nan)
        cond &= width <= float(bb_max_width)

    if hv_window is not None and max_hv is not None:
        log_ret = np.log(close / close.shift(1).replace(0.0, np.nan))
        hv = log_ret.rolling(hv_window, min_periods=hv_window).std(ddof=0) * np.sqrt(hv_window)
        cond &= hv <= float(max_hv)

    if vix_window is not None and max_vix is not None:
        log_ret = np.log(close / close.shift(1).replace(0.0, np.nan))
        vix = log_ret.rolling(vix_window, min_periods=vix_window).std(ddof=0) * np.sqrt(252)
        cond &= vix <= float(max_vix)

    return cond.fillna(False)


__all__ = ["volatility_filter"]
