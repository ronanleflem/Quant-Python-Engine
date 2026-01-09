"""Benford law anomaly filter."""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

from .common_filters import rolling_apply


def _benford_mse(values: Iterable[float]) -> float:
    digits = []
    for value in values:
        v = abs(float(value))
        if v == 0:
            continue
        v = math.floor(v)
        if v == 0:
            continue
        digits.append(int(str(v)[0]))
    if not digits:
        return 1.0
    counts = np.bincount(digits, minlength=10).astype(float)
    counts[0] = 0.0
    dist = counts / counts.sum()
    expected = np.array([0.0] + [math.log10(1 + 1 / d) for d in range(1, 10)], dtype=float)
    return float(((dist - expected) ** 2).mean())


def _first_digit_counts(values: Iterable[float]) -> np.ndarray:
    digits = []
    for value in values:
        v = abs(float(value))
        if v == 0:
            continue
        v = math.floor(v)
        if v == 0:
            continue
        digits.append(int(str(v)[0]))
    counts = np.bincount(digits, minlength=10).astype(float)
    counts[0] = 0.0
    return counts


def _benford_expected() -> np.ndarray:
    return np.array([0.0] + [math.log10(1 + 1 / d) for d in range(1, 10)], dtype=float)


def _benford_mad(values: Iterable[float]) -> float:
    counts = _first_digit_counts(values)
    if counts.sum() == 0:
        return 1.0
    dist = counts / counts.sum()
    expected = _benford_expected()
    return float(np.abs(dist - expected).mean())


def _benford_chi2(values: Iterable[float]) -> float:
    counts = _first_digit_counts(values)
    total = counts.sum()
    if total == 0:
        return float("inf")
    expected = _benford_expected() * total
    expected = np.where(expected == 0, np.nan, expected)
    chi2 = np.nansum(((counts - expected) ** 2) / expected)
    return float(chi2)


def _build_benford_series(
    df: pd.DataFrame,
    *,
    series_type: str,
    price_col: str,
    open_col: str,
    high_col: str,
    low_col: str,
    volume_col: str,
) -> pd.Series:
    stype = series_type.lower()
    if stype in {"range", "hl"}:
        return (df[high_col].astype(float) - df[low_col].astype(float)).abs()
    if stype in {"body", "oc"}:
        return (df[price_col].astype(float) - df[open_col].astype(float)).abs()
    if stype in {"delta_range", "range_delta"}:
        rng = (df[high_col].astype(float) - df[low_col].astype(float)).abs()
        return rng.diff().abs().fillna(0.0)
    if stype in {"volume", "vol"}:
        return df[volume_col].astype(float).abs()
    if stype in {"returns", "close_diff"}:
        return df[price_col].astype(float).diff().abs().fillna(0.0)
    return df[price_col].astype(float).abs()


def benford_law_filter(
    df: pd.DataFrame,
    window: int = 100,
    series_type: str = "range",
    metric: str = "mad",
    mad_threshold: float = 0.006,
    chi2_threshold: float = 25.0,
    price_col: str = "close",
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    volume_col: str = "volume",
) -> pd.Series:
    """Return True when Benford anomaly metrics remain below thresholds."""
    required = {price_col}
    stype = series_type.lower()
    if stype in {"range", "hl", "delta_range", "range_delta"}:
        required |= {high_col, low_col}
    if stype in {"body", "oc"}:
        required |= {open_col}
    if stype in {"volume", "vol"}:
        required |= {volume_col}
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing column(s): {', '.join(missing)}")

    series = _build_benford_series(
        df,
        series_type=series_type,
        price_col=price_col,
        open_col=open_col,
        high_col=high_col,
        low_col=low_col,
        volume_col=volume_col,
    )

    metric_key = metric.lower()
    if metric_key in {"chi2", "chi_square"}:
        chi2 = rolling_apply(series, window, _benford_chi2)
        return (chi2 <= float(chi2_threshold)).fillna(False)
    if metric_key in {"both", "mad_chi2"}:
        mad = rolling_apply(series, window, _benford_mad)
        chi2 = rolling_apply(series, window, _benford_chi2)
        return (mad <= float(mad_threshold)) & (chi2 <= float(chi2_threshold))

    mad = rolling_apply(series, window, _benford_mad)
    return (mad <= float(mad_threshold)).fillna(False)


__all__ = ["benford_law_filter"]
