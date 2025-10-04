"""Volatility and trend filters used to pre-screen trading opportunities."""
from __future__ import annotations

from importlib import util as importlib_util
from typing import Iterable

import numpy as np
import pandas as pd

_TALIB_AVAILABLE = importlib_util.find_spec("talib") is not None
if _TALIB_AVAILABLE:  # pragma: no cover - optional dependency
    import talib  # type: ignore


_REQUIRED_OHLC_COLUMNS = ("high", "low", "close")


def _ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {', '.join(missing)}")


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    components = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    tr = components.max(axis=1)
    return tr


def _wilder_smoothing(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(alpha=1 / float(window), adjust=False).mean()



def adx_filter(df: pd.DataFrame, window: int = 14, thresh: float = 20.0) -> pd.Series:
    """Return a boolean mask where the Average Directional Index exceeds ``thresh``."""

    _ensure_columns(df, _REQUIRED_OHLC_COLUMNS)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    if _TALIB_AVAILABLE:  # pragma: no cover - optional dependency
        adx_values = talib.ADX(high.values, low.values, close.values, timeperiod=window)
        adx = pd.Series(adx_values, index=df.index, dtype=float)
    else:
        up_move = high.diff()
        down_move = low.shift(1) - low
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        tr = _true_range(high, low, close)
        tr_smoothed = _wilder_smoothing(tr, window)
        plus_di = 100 * _wilder_smoothing(pd.Series(plus_dm, index=df.index), window) / tr_smoothed
        minus_di = 100 * _wilder_smoothing(pd.Series(minus_dm, index=df.index), window) / tr_smoothed
        denominator = (plus_di + minus_di).replace(0, np.nan)
        dx = (plus_di.subtract(minus_di).abs() / denominator) * 100.0
        adx = _wilder_smoothing(dx, window)

    mask = (adx > float(thresh)).fillna(False)
    return mask.astype(bool)



def atr_filter(
    df: pd.DataFrame,
    window: int = 14,
    min_mult: float = 0.5,
    max_mult: float = 2.0,
) -> pd.Series:
    """Return a boolean mask keeping assets where ATR/close is within a band."""

    _ensure_columns(df, _REQUIRED_OHLC_COLUMNS)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    if _TALIB_AVAILABLE:  # pragma: no cover - optional dependency
        atr_values = talib.ATR(high.values, low.values, close.values, timeperiod=window)
        atr = pd.Series(atr_values, index=df.index, dtype=float)
    else:
        tr = _true_range(high, low, close)
        atr = _wilder_smoothing(tr, window)

    close_safe = close.replace(0, np.nan)
    ratio = atr / close_safe
    min_ratio = float(min_mult) / 100.0
    max_ratio = float(max_mult) / 100.0
    mask = ratio.between(min_ratio, max_ratio)
    return mask.fillna(False).astype(bool)



def ema_slope_filter(df: pd.DataFrame, window: int = 50, slope_thresh: float = 0.0) -> pd.Series:
    """Return ``True`` when the EMA slope is above ``slope_thresh``."""

    if "close" not in df.columns:
        raise ValueError("DataFrame is missing required column: close")
    close = df["close"].astype(float)

    if _TALIB_AVAILABLE:  # pragma: no cover - optional dependency
        ema_values = talib.EMA(close.values, timeperiod=window)
        ema = pd.Series(ema_values, index=df.index, dtype=float)
    else:
        ema = close.ewm(span=window, adjust=False).mean()

    slope = ema.diff()
    mask = slope > float(slope_thresh)
    return mask.fillna(False).astype(bool)
