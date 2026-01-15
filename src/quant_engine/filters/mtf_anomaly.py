"""Multi-timeframe anomaly filter."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .benford import _benford_mad, _build_benford_series
from .common_filters import rolling_apply


def _normalize_rule(timeframe: str) -> str:
    raw = str(timeframe).strip().lower()
    if raw.endswith("min"):
        raw = raw[:-3] + "m"
    if raw.endswith("mins"):
        raw = raw[:-4] + "m"
    if raw.endswith("minute"):
        raw = raw[:-6] + "m"
    if raw.endswith("minutes"):
        raw = raw[:-7] + "m"
    if raw.endswith("m"):
        return raw[:-1] + "min"
    if raw.endswith("h"):
        return raw[:-1] + "H"
    if raw.endswith("d"):
        return raw[:-1] + "D"
    return raw


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex")
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    return idx


def _resample_ohlcv(
    df: pd.DataFrame,
    rule: str,
    *,
    open_col: str,
    high_col: str,
    low_col: str,
    close_col: str,
    volume_col: str,
) -> pd.DataFrame:
    agg = {
        open_col: "first",
        high_col: "max",
        low_col: "min",
        close_col: "last",
    }
    if volume_col in df.columns:
        agg[volume_col] = "sum"
    resampled = df.resample(rule).agg(agg)
    return resampled.dropna(subset=[open_col, high_col, low_col, close_col])


def _entropy_series(series: pd.Series, window: int) -> pd.Series:
    direction = np.sign(series.astype(float).diff())
    min_periods = min(window, max(2, max(8, window // 4)))
    up_prob = (direction > 0).astype(int).rolling(window, min_periods=min_periods).mean()
    eps = 1e-12
    entropy = -(up_prob * np.log2(up_prob + eps) + (1 - up_prob) * np.log2(1 - up_prob + eps))
    return entropy


def mtf_anomaly_filter(
    df: pd.DataFrame,
    *,
    higher_timeframe: str = "15m",
    base_window: int = 128,
    higher_window: int = 64,
    metric: str = "benford_mad",
    series_type: str = "range",
    mad_threshold: float = 0.006,
    entropy_threshold: float = 0.9,
    atr_window: int = 14,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    volume_col: str = "volume",
    require_both: bool = True,
) -> pd.Series:
    """Return True unless anomalies are detected on both timeframes."""

    idx = _ensure_datetime_index(df)
    df_local = df.copy()
    df_local.index = idx

    metric_key = str(metric).strip().lower()
    base_series = _build_benford_series(
        df_local,
        series_type=series_type,
        price_col=close_col,
        open_col=open_col,
        high_col=high_col,
        low_col=low_col,
        volume_col=volume_col,
        atr_window=atr_window,
    )
    if metric_key == "benford_mad":
        base_score = rolling_apply(base_series, base_window, _benford_mad)
    elif metric_key == "entropy":
        base_score = _entropy_series(base_series, base_window)
    else:
        raise ValueError("mtf_anomaly_filter metric must be 'benford_mad' or 'entropy'")

    if higher_timeframe:
        rule = _normalize_rule(higher_timeframe)
        df_high = _resample_ohlcv(
            df_local,
            rule,
            open_col=open_col,
            high_col=high_col,
            low_col=low_col,
            close_col=close_col,
            volume_col=volume_col,
        )
        if df_high.empty or len(df_high) < higher_window:
            high_score = pd.Series(np.nan, index=df_local.index)
        else:
            high_series = _build_benford_series(
                df_high,
                series_type=series_type,
                price_col=close_col,
                open_col=open_col,
                high_col=high_col,
                low_col=low_col,
                volume_col=volume_col,
                atr_window=atr_window,
            )
            if metric_key == "benford_mad":
                high_score = rolling_apply(high_series, higher_window, _benford_mad)
            else:
                high_score = _entropy_series(high_series, higher_window)
            high_score = high_score.reindex(df_local.index, method="ffill")
    else:
        high_score = pd.Series(np.nan, index=df_local.index)

    if metric_key == "benford_mad":
        threshold = float(mad_threshold)
    else:
        threshold = float(entropy_threshold)
    anom_base = base_score > threshold
    anom_high = high_score > threshold
    if require_both:
        mask = ~(anom_base & anom_high)
    else:
        mask = ~(anom_base | anom_high)
    return mask.reindex(df_local.index).fillna(True)


__all__ = ["mtf_anomaly_filter"]
