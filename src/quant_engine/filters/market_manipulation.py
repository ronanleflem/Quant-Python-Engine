"""Market manipulation/anomaly filter."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def _directional_entropy(series: pd.Series, window: int) -> pd.Series:
    direction = np.sign(series.astype(float).diff())
    min_periods = min(window, max(2, max(8, window // 4)))
    up_prob = (direction > 0).astype(int).rolling(window, min_periods=min_periods).mean()
    eps = 1e-12
    return -(up_prob * np.log2(up_prob + eps) + (1 - up_prob) * np.log2(1 - up_prob + eps))


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr_components = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    )
    return tr_components.max(axis=1)


def market_manipulation_filter(
    df: pd.DataFrame,
    *,
    window: int = 60,
    entropy_threshold: Optional[float] = 0.9,
    kurtosis_threshold: Optional[float] = 5.0,
    atr_window: Optional[int] = 14,
    atr_mult: Optional[float] = 2.5,
    require_all: bool = True,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pd.Series:
    """Return True unless entropy/kurtosis/volatility indicate manipulation."""
    for col in (high_col, low_col, close_col):
        if col not in df.columns:
            raise ValueError(f"DataFrame missing column: {col}")

    close = df[close_col].astype(float)
    entropy = _directional_entropy(close, window)

    returns = close.diff()
    kurtosis = returns.rolling(window, min_periods=window).apply(
        lambda arr: float(pd.Series(arr).kurtosis()),
        raw=True,
    )

    if atr_window is not None and atr_mult is not None:
        high = df[high_col].astype(float)
        low = df[low_col].astype(float)
        tr = _true_range(high, low, close)
        atr = tr.ewm(alpha=1.0 / float(atr_window), adjust=False).mean()
        vol_spike = tr > float(atr_mult) * atr
    else:
        vol_spike = pd.Series(False, index=df.index)

    flags = []
    if entropy_threshold is not None:
        flags.append(entropy >= float(entropy_threshold))
    if kurtosis_threshold is not None:
        flags.append(kurtosis >= float(kurtosis_threshold))
    if atr_mult is not None and atr_window is not None:
        flags.append(vol_spike)

    if not flags:
        return pd.Series(True, index=df.index)

    if require_all:
        anomaly = flags[0]
        for flag in flags[1:]:
            anomaly &= flag
    else:
        anomaly = flags[0]
        for flag in flags[1:]:
            anomaly |= flag

    return (~anomaly).reindex(df.index).fillna(True)


__all__ = ["market_manipulation_filter"]
