"""EMA structure filter."""
from __future__ import annotations

import pandas as pd


def ema_structure_filter(
    df: pd.DataFrame,
    ema_fast: int = 20,
    ema_slow: int = 50,
    ema_long: int = 200,
    require_close_above_slow: bool = True,
    require_fast_rising: bool = True,
    price_col: str = "close",
) -> pd.Series:
    """Return True when EMA structure suggests a bullish regime."""
    if price_col not in df.columns:
        raise ValueError(f"DataFrame missing column: {price_col}")
    close = df[price_col].astype(float)
    fast = close.ewm(span=int(ema_fast), adjust=False).mean()
    slow = close.ewm(span=int(ema_slow), adjust=False).mean()
    long = close.ewm(span=int(ema_long), adjust=False).mean() if ema_long else None

    cond = fast > slow
    if long is not None:
        cond &= slow > long
    if require_close_above_slow:
        cond &= close > slow
    if require_fast_rising:
        cond &= fast.diff() > 0
    return cond.fillna(False)


__all__ = ["ema_structure_filter"]
