"""Institutional bias filter (EMA/VWAP + optional macro data)."""
from __future__ import annotations

import pandas as pd


def _compute_vwap(df: pd.DataFrame, price_col: str, volume_col: str) -> pd.Series:
    if volume_col in df.columns:
        pv = (df[price_col] * df[volume_col]).astype(float)
        vv = df[volume_col].astype(float).replace(0.0, pd.NA)
        return (pv.cumsum() / vv.cumsum()).reindex(df.index)
    # fallback: cumulative mean price
    return df[price_col].astype(float).expanding().mean()


def biais_institutional_filter(
    df: pd.DataFrame,
    ema_fast: int = 50,
    ema_slow: int = 200,
    price_col: str = "close",
    volume_col: str = "volume",
    vwap_side: str = "above",
    cot_col: str | None = None,
    oi_col: str | None = None,
    cot_bias_threshold: float = 0.0,
    oi_min_change: float | None = None,
) -> pd.Series:
    """Return True when EMA/VWAP align with optional macro filters."""
    if price_col not in df.columns:
        raise ValueError(f"DataFrame missing column: {price_col}")
    close = df[price_col].astype(float)
    ema_f = close.ewm(span=int(ema_fast), adjust=False).mean()
    ema_s = close.ewm(span=int(ema_slow), adjust=False).mean()
    vwap = _compute_vwap(df, price_col, volume_col)

    cond = ema_f > ema_s
    cond &= close > ema_s
    if vwap_side.lower() == "below":
        cond &= close <= vwap
    else:
        cond &= close >= vwap

    if cot_col and cot_col in df.columns:
        cot_values = df[cot_col].astype(float)
        cond &= cot_values >= float(cot_bias_threshold)

    if oi_col and oi_col in df.columns and oi_min_change is not None:
        oi_values = df[oi_col].astype(float)
        cond &= oi_values.diff().fillna(0.0) >= float(oi_min_change)

    return cond.fillna(False)


__all__ = ["biais_institutional_filter"]
