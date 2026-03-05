"""Legacy event definitions for statistics runs.

Deprecated wrappers around market-intelligence event feature functions.
"""
from __future__ import annotations

import warnings

import pandas as pd

from quant_engine.market_intelligence import events as mi_events


def _warn_deprecated(func_name: str) -> None:
    warnings.warn(
        (
            f"quant_engine.stats.events.{func_name} is deprecated and will be removed "
            "in a future release; use quant_engine.market_intelligence.events instead."
        ),
        DeprecationWarning,
        stacklevel=2,
    )


def k_consecutive(df: pd.DataFrame, *, k: int, direction: str) -> pd.Series:
    _warn_deprecated("k_consecutive")
    return mi_events.k_consecutive(df, k=k, direction=direction)


def shock_atr(df: pd.DataFrame, *, mult: float, window: int) -> pd.Series:
    _warn_deprecated("shock_atr")
    return mi_events.shock_atr(df, mult=mult, window=window)


def breakout_hhll(df: pd.DataFrame, *, lookback: int, type: str) -> pd.Series:
    _warn_deprecated("breakout_hhll")
    return mi_events.breakout_hhll(df, lookback=lookback, type=type)


def bullish_candle(df: pd.DataFrame) -> pd.Series:
    _warn_deprecated("bullish_candle")
    return mi_events.bullish_candle(df)


def bearish_candle(df: pd.DataFrame) -> pd.Series:
    _warn_deprecated("bearish_candle")
    return mi_events.bearish_candle(df)


def bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    _warn_deprecated("bullish_engulfing")
    return mi_events.bullish_engulfing(df)


def bearish_engulfing(df: pd.DataFrame) -> pd.Series:
    _warn_deprecated("bearish_engulfing")
    return mi_events.bearish_engulfing(df)


def bullish_streak(df: pd.DataFrame, *, k: int = 3) -> pd.Series:
    _warn_deprecated("bullish_streak")
    return mi_events.bullish_streak(df, k=k)


def bearish_streak(df: pd.DataFrame, *, k: int = 3) -> pd.Series:
    _warn_deprecated("bearish_streak")
    return mi_events.bearish_streak(df, k=k)


def gap_up(df: pd.DataFrame) -> pd.Series:
    _warn_deprecated("gap_up")
    return mi_events.gap_up(df)


def gap_down(df: pd.DataFrame) -> pd.Series:
    _warn_deprecated("gap_down")
    return mi_events.gap_down(df)


def always_true(df: pd.DataFrame) -> pd.Series:
    _warn_deprecated("always_true")
    return mi_events.always_true(df)
