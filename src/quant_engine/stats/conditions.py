"""Regime condition helpers for statistics runs."""
from __future__ import annotations

from functools import lru_cache
from typing import Dict

import numpy as np
import pandas as pd

from ..levels import helpers as level_helpers
from ..levels import repo as levels_repo


LEVELS_TABLE = "marketdata.levels"


def _safe_select_levels(symbol: str, level_type: str) -> pd.DataFrame:
    try:
        engine = levels_repo.get_engine()
    except Exception:
        return pd.DataFrame()
    try:
        df_levels = levels_repo.select_levels(
            engine,
            LEVELS_TABLE,
            symbol=symbol,
            level_types=[level_type],
            active_only=False,
            limit=100000,
        )
    except Exception:
        return pd.DataFrame()
    return df_levels


@lru_cache(maxsize=128)
def _cached_levels(symbol: str, level_type: str) -> pd.DataFrame:
    return _safe_select_levels(symbol, level_type)


def htf_trend(df: pd.DataFrame, *, tf_multiplier: int, ema_period: int) -> pd.Series:
    """Return higher time frame trend as ``"up"`` or ``"down"``.

    The dataset is downsampled by ``tf_multiplier`` and an EMA of ``ema_period``
    is computed on the resulting series.  Trend labels are then forward filled
    to the original frequency.
    """

    groups = np.arange(len(df)) // tf_multiplier
    htf_close = df["close"].groupby(groups).last()
    ema = htf_close.ewm(span=ema_period, adjust=False).mean()
    trend_per_group = pd.Series(
        np.where(ema.diff() > 0, "up", "down"), index=htf_close.index
    )
    return pd.Series(groups).map(trend_per_group).astype("category")


def vol_tertile(df: pd.DataFrame, *, window: int) -> pd.Series:
    """Classify current ATR into tertiles across the sample."""

    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr = tr.rolling(window).mean()
    q1, q2 = atr.quantile([1 / 3, 2 / 3])
    if q1 == q2:
        tertiles = pd.Series(["mid"] * len(atr), index=atr.index)
    else:
        tertiles = pd.cut(atr, [-np.inf, q1, q2, np.inf], labels=["low", "mid", "high"])
    return tertiles.astype("category")


def session(df: pd.DataFrame, *, col: str = "session_id") -> pd.Series:
    """Return the session label as a categorical series."""

    return df[col].astype("category")


def _apply_per_symbol(
    df: pd.DataFrame,
    level_type: str,
    func,
    default: pd.Series,
) -> pd.Series:
    if df.empty:
        return default
    parts: Dict[int, object] = {}
    for symbol, group in df.groupby("symbol"):
        levels = _cached_levels(symbol, level_type).copy()
        series = func(group, levels)
        parts.update({idx: series.loc[idx] for idx in series.index})
    result = default.copy()
    for idx, value in parts.items():
        result.loc[idx] = value
    return result


def in_zone_level(level_type: str, tolerance: float = 0.0):
    """Wrapper returning a boolean series when price trades inside a level."""

    def _inner(df: pd.DataFrame) -> pd.Series:
        base = pd.Series(False, index=df.index, dtype="boolean")

        def _compute(group: pd.DataFrame, levels: pd.DataFrame) -> pd.Series:
            if levels.empty:
                return pd.Series(False, index=group.index, dtype="boolean")
            series = level_helpers.in_zone(group, levels, level_type, tolerance=tolerance)
            return series.reindex(group.index, fill_value=False).astype("boolean")

        return _apply_per_symbol(df, level_type, _compute, base)

    return _inner


def distance_to_level(
    level_type: str,
    side: str = "mid",
    thresh: float | None = None,
):
    """Return distance to a level or a boolean mask if ``thresh`` is provided."""

    def _inner(df: pd.DataFrame) -> pd.Series:
        if thresh is None:
            base = pd.Series(np.nan, index=df.index, dtype="float64")
        else:
            base = pd.Series(False, index=df.index, dtype="boolean")

        def _compute(group: pd.DataFrame, levels: pd.DataFrame) -> pd.Series:
            if levels.empty:
                if thresh is None:
                    return pd.Series(np.nan, index=group.index, dtype="float64")
                return pd.Series(False, index=group.index, dtype="boolean")
            distances = level_helpers.distance_to(group, levels, level_type, side=side)
            distances = distances.reindex(group.index)
            if thresh is None:
                return distances
            mask = (distances <= float(thresh)).fillna(False)
            return mask.astype("boolean")

        return _apply_per_symbol(df, level_type, _compute, base)

    return _inner


def touched_level_since(level_type: str, bars: int = 1):
    """Return a boolean series when a level has been touched within ``bars`` bars."""

    def _inner(df: pd.DataFrame) -> pd.Series:
        base = pd.Series(False, index=df.index, dtype="boolean")

        def _compute(group: pd.DataFrame, levels: pd.DataFrame) -> pd.Series:
            if levels.empty:
                return pd.Series(False, index=group.index, dtype="boolean")
            touched = level_helpers.touched_since(group, levels, level_type, bars=bars)
            return touched.reindex(group.index, fill_value=False).astype("boolean")

        return _apply_per_symbol(df, level_type, _compute, base)

    return _inner

