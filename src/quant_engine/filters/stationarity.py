"""Stationarity-inspired filter."""
from __future__ import annotations

from importlib import util as importlib_util
import numpy as np
import pandas as pd

from .common_filters import rolling_apply

_STATS_MODELS_AVAILABLE = importlib_util.find_spec("statsmodels") is not None
if _STATS_MODELS_AVAILABLE:  # pragma: no cover - optional dependency
    from statsmodels.tsa.stattools import adfuller, kpss  # type: ignore


def _adf_pvalue(series: pd.Series) -> float:
    return float(adfuller(series, autolag="AIC")[1])


def _kpss_pvalue(series: pd.Series) -> float:
    return float(kpss(series, nlags="auto")[1])


def stationarity_filter(
    df: pd.DataFrame,
    window: int = 120,
    max_abs_autocorr: float = 0.7,
    price_col: str = "close",
    method: str = "acf",
    adf_pvalue: float = 0.05,
    kpss_pvalue: float = 0.05,
    allow_if_missing: bool = True,
) -> pd.Series:
    """Return True when the stationarity test passes."""
    if price_col not in df.columns:
        if allow_if_missing:
            return pd.Series(True, index=df.index)
        raise ValueError(f"DataFrame missing column: {price_col}")
    series = df[price_col].astype(float)
    returns = series.diff().fillna(0.0)
    method_key = str(method).strip().lower()

    def _acf1(arr: list[float]) -> float:
        if len(arr) < 3:
            return 1.0
        data = np.array(arr, dtype=float)
        x = data[:-1]
        y = data[1:]
        if x.std() == 0 or y.std() == 0:
            return 1.0
        return float(np.corrcoef(x, y)[0, 1])

    if method_key == "acf":
        acf1 = rolling_apply(returns, window, _acf1)
        return (acf1.abs() < float(max_abs_autocorr)).fillna(False)

    if not _STATS_MODELS_AVAILABLE:
        return pd.Series(True, index=df.index) if allow_if_missing else pd.Series(False, index=df.index)

    def _roll(series: pd.Series, fn):
        return series.rolling(window, min_periods=window).apply(
            lambda arr: fn(pd.Series(arr)),
            raw=False,
        )

    if method_key == "adf":
        pvals = _roll(returns, _adf_pvalue)
        return (pvals <= float(adf_pvalue)).fillna(False)
    if method_key == "kpss":
        pvals = _roll(returns, _kpss_pvalue)
        return (pvals >= float(kpss_pvalue)).fillna(False)
    if method_key == "both":
        adf_vals = _roll(returns, _adf_pvalue)
        kpss_vals = _roll(returns, _kpss_pvalue)
        return ((adf_vals <= float(adf_pvalue)) & (kpss_vals >= float(kpss_pvalue))).fillna(False)
    raise ValueError("stationarity_filter method must be 'acf', 'adf', 'kpss', or 'both'")


__all__ = ["stationarity_filter"]
