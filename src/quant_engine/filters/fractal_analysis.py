"""Fractal statistics filter."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common_filters import rolling_apply


def _hurst_rs(series: pd.Series, window: int) -> pd.Series:
    log_price = np.log(series.astype(float))
    returns = log_price.diff().dropna()
    hurst = pd.Series(np.nan, index=series.index)
    for end in range(window, len(returns) + 1):
        segment = returns.iloc[end - window : end]
        if segment.std(ddof=0) == 0:
            h_value = np.nan
        else:
            cumulative = segment.cumsum() - segment.mean() * np.arange(1, window + 1)
            r_val = cumulative.max() - cumulative.min()
            s_val = segment.std(ddof=0)
            if s_val == 0:
                h_value = np.nan
            else:
                rs = r_val / s_val
                h_value = np.log(rs) / np.log(window)
        hurst.loc[segment.index[-1]] = h_value
    return hurst.ffill()


def fractal_analysis_filter(
    df: pd.DataFrame,
    window: int = 64,
    min_h: float = 0.4,
    max_h: float = 0.6,
    max_abs_skew: float | None = None,
    max_kurtosis: float | None = None,
    price_col: str = "close",
) -> pd.Series:
    """Return True when Hurst (and optional skew/kurtosis) are within bounds."""
    if price_col not in df.columns:
        raise ValueError(f"DataFrame missing column: {price_col}")
    price = df[price_col].astype(float)
    hurst = _hurst_rs(price, window)
    cond = (hurst >= float(min_h)) & (hurst <= float(max_h))

    if max_abs_skew is not None or max_kurtosis is not None:
        returns = price.diff().fillna(0.0)
        if max_abs_skew is not None:
            skew = rolling_apply(returns, window, lambda arr: float(pd.Series(arr).skew()))
            cond &= skew.abs() <= float(max_abs_skew)
        if max_kurtosis is not None:
            kurt = rolling_apply(returns, window, lambda arr: float(pd.Series(arr).kurtosis()))
            cond &= kurt <= float(max_kurtosis)

    return cond.fillna(False)


__all__ = ["fractal_analysis_filter"]
