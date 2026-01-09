"""Market regime identification."""
from __future__ import annotations

import pandas as pd

from .volatility_trend import adx_filter


def market_regime_filter(
    df: pd.DataFrame,
    regime: str = "trend",
    adx_window: int = 14,
    adx_thresh: float = 25.0,
    atr_window: int = 14,
    max_atr_pct: float = 0.01,
    bb_window: int = 20,
    bb_width_thresh: float = 0.02,
    price_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
) -> pd.Series:
    """Return True when the detected regime matches the target."""
    for col in (price_col, high_col, low_col):
        if col not in df.columns:
            raise ValueError(f"DataFrame missing column: {col}")
    close = df[price_col].astype(float)
    high = df[high_col].astype(float)
    low = df[low_col].astype(float)

    adx = adx_filter(df, window=adx_window, thresh=adx_thresh)

    prev_close = close.shift(1)
    tr_components = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    )
    tr = tr_components.max(axis=1)
    atr = tr.ewm(alpha=1.0 / float(atr_window), adjust=False).mean()
    atr_pct = atr / close.replace(0.0, pd.NA)

    sma = close.rolling(bb_window, min_periods=bb_window).mean()
    std = close.rolling(bb_window, min_periods=bb_window).std(ddof=0)
    bb_width = (2 * std) / sma.replace(0.0, pd.NA)

    regime_key = regime.strip().lower()
    if regime_key in {"compression", "squeeze"}:
        return (bb_width <= float(bb_width_thresh)).fillna(False)
    if regime_key in {"range", "ranging"}:
        return (~adx & (atr_pct <= float(max_atr_pct))).fillna(False)
    return adx.fillna(False)


__all__ = ["market_regime_filter"]
