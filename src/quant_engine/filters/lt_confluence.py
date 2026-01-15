"""Lower timeframe confluence filter."""
from __future__ import annotations

from math import ceil
from typing import Optional

import numpy as np
import pandas as pd


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    components = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    )
    return components.max(axis=1)


def _adx_value(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    up_move = high.diff()
    down_move = low.shift(1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = _true_range(high, low, close)
    tr_smoothed = tr.ewm(alpha=1 / float(window), adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=high.index).ewm(alpha=1 / float(window), adjust=False).mean() / tr_smoothed
    minus_di = 100 * pd.Series(minus_dm, index=high.index).ewm(alpha=1 / float(window), adjust=False).mean() / tr_smoothed
    denominator = (plus_di + minus_di).replace(0, np.nan)
    dx = (plus_di.subtract(minus_di).abs() / denominator) * 100.0
    return dx.ewm(alpha=1 / float(window), adjust=False).mean()


def _intraday_vwap(df: pd.DataFrame, price_col: str, volume_col: str) -> pd.Series:
    grp = df.index.normalize()
    if volume_col not in df.columns:
        cum_price = df.groupby(grp)[price_col].cumsum()
        counts = df.groupby(grp).cumcount() + 1
        return (cum_price / counts).reindex(df.index)
    pv = (df[price_col] * df[volume_col]).groupby(grp).cumsum()
    vv = df.groupby(grp)[volume_col].cumsum().replace(0, np.nan)
    return (pv / vv).reindex(df.index)


def lower_timeframe_confluence_filter(
    df: pd.DataFrame,
    *,
    side: str = "long",
    momentum_period: int = 10,
    min_momentum: Optional[float] = 0.0,
    adx_window: int = 14,
    adx_thresh: Optional[float] = 20.0,
    vwap_max_dev: Optional[float] = 0.005,
    vwap_side: str = "any",
    vwap_price_col: str = "close",
    vwap_volume_col: str = "volume",
    ema_fast: int = 20,
    ema_slow: int = 50,
    ema_long: int = 200,
    delta_col: Optional[str] = None,
    buy_col: str = "buy_volume",
    sell_col: str = "sell_volume",
    volume_col: str = "volume",
    delta_min: Optional[float] = None,
    delta_ratio: Optional[float] = None,
    min_score: Optional[int] = 3,
    min_score_pct: Optional[float] = None,
    require_all: bool = False,
    allow_if_missing: bool = True,
    close_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
) -> pd.Series:
    """Return True when lower timeframe signals align with the requested side."""
    side_key = str(side).strip().lower()
    if close_col not in df.columns:
        if allow_if_missing:
            return pd.Series(True, index=df.index)
        raise ValueError(f"DataFrame missing column: {close_col}")

    close = df[close_col].astype(float)
    high = df[high_col].astype(float) if high_col in df.columns else None
    low = df[low_col].astype(float) if low_col in df.columns else None

    components: list[pd.Series] = []

    if min_momentum is not None and momentum_period > 0:
        prev = close.shift(momentum_period)
        roc = (close - prev) / prev.replace(0.0, np.nan)
        if side_key in {"short", "bear", "down"}:
            components.append(roc <= -float(min_momentum))
        else:
            components.append(roc >= float(min_momentum))

    if adx_thresh is not None and high is not None and low is not None:
        adx = _adx_value(high, low, close, adx_window)
        components.append(adx >= float(adx_thresh))

    if vwap_max_dev is not None:
        vwap = _intraday_vwap(df, vwap_price_col, vwap_volume_col)
        dev = (close - vwap) / vwap.replace(0.0, np.nan)
        max_dev = float(vwap_max_dev)
        side_check = str(vwap_side).strip().lower()
        if side_check == "above":
            components.append((dev >= 0) & (dev.abs() <= max_dev))
        elif side_check == "below":
            components.append((dev <= 0) & (dev.abs() <= max_dev))
        else:
            components.append(dev.abs() <= max_dev)

    if ema_fast and ema_slow and ema_long:
        ema_f = close.ewm(span=int(ema_fast), adjust=False).mean()
        ema_s = close.ewm(span=int(ema_slow), adjust=False).mean()
        ema_l = close.ewm(span=int(ema_long), adjust=False).mean()
        if side_key in {"short", "bear", "down"}:
            components.append((ema_f < ema_s) & (ema_s < ema_l))
        else:
            components.append((ema_f > ema_s) & (ema_s > ema_l))

    delta_series: Optional[pd.Series] = None
    if delta_col and delta_col in df.columns:
        delta_series = df[delta_col].astype(float)
    elif buy_col in df.columns and sell_col in df.columns:
        delta_series = df[buy_col].astype(float) - df[sell_col].astype(float)

    if delta_series is not None:
        delta_score = delta_series
        if delta_ratio is not None:
            if volume_col in df.columns:
                delta_score = delta_score / df[volume_col].astype(float).replace(0.0, np.nan)
            elif not allow_if_missing:
                return pd.Series(False, index=df.index)
            else:
                return pd.Series(True, index=df.index)
        if side_key in {"short", "bear", "down"}:
            delta_score = -delta_score
        if delta_ratio is not None:
            components.append(delta_score >= float(delta_ratio))
        elif delta_min is not None:
            components.append(delta_score >= float(delta_min))
        else:
            components.append(delta_score > 0)
    elif not allow_if_missing and (delta_min is not None or delta_ratio is not None):
        return pd.Series(False, index=df.index)

    if not components:
        return pd.Series(True, index=df.index) if allow_if_missing else pd.Series(False, index=df.index)

    if require_all:
        combined = components[0]
        for comp in components[1:]:
            combined &= comp
        return combined.reindex(df.index).fillna(False).astype(bool)

    if min_score_pct is not None:
        required = max(1, ceil(len(components) * float(min_score_pct)))
    else:
        required = min(int(min_score or 1), len(components))
    score = sum(comp.fillna(False).astype(int) for comp in components)
    return (score >= required).reindex(df.index).fillna(False).astype(bool)


__all__ = ["lower_timeframe_confluence_filter"]
