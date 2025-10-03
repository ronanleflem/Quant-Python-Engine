"""Regime condition helpers for statistics runs."""
from __future__ import annotations

from collections.abc import Callable
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from quant_engine.levels import helpers as lvl_helpers
from quant_engine.levels import repo as lvl_repo


LEVELS_TABLE = "marketdata.levels"
_LEVELS_CACHE: Dict[Tuple[str, Optional[pd.Timestamp], Optional[pd.Timestamp], Tuple[str, ...]], pd.DataFrame] = {}


def _normalise_timestamp(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def _normalise_levels(levels_df: pd.DataFrame) -> pd.DataFrame:
    if levels_df.empty:
        return levels_df
    levels = levels_df.copy()
    for col in ("anchor_ts", "ts", "valid_from_ts", "valid_to_ts"):
        if col in levels.columns:
            levels[col] = _normalise_timestamp(levels[col])
    sort_cols: list[str] = [col for col in ("anchor_ts", "valid_from_ts") if col in levels.columns]
    if sort_cols:
        levels.sort_values(sort_cols, inplace=True)
    levels.reset_index(drop=True, inplace=True)
    return levels


def _window_key(df_symbol: pd.DataFrame, level_types: Iterable[str]) -> Tuple[str, Optional[pd.Timestamp], Optional[pd.Timestamp], Tuple[str, ...]]:
    if df_symbol.empty:
        symbol = str(df_symbol.get("symbol", ""))
        return symbol, None, None, tuple(sorted(level_types))
    symbol_series = df_symbol.get("symbol")
    if symbol_series is None or symbol_series.empty:
        symbol = ""
    else:
        symbol = str(symbol_series.iloc[0])
    ts_col = df_symbol.get("ts")
    if ts_col is None:
        start_ts = end_ts = None
    else:
        ts_normalised = _normalise_timestamp(ts_col).dropna()
        start_ts = ts_normalised.min() if not ts_normalised.empty else None
        end_ts = ts_normalised.max() if not ts_normalised.empty else None
    key = (symbol, start_ts, end_ts, tuple(sorted(level_types)))
    return key


def _load_levels_for(df_symbol: pd.DataFrame, level_types: list[str]) -> pd.DataFrame:
    """Load and cache levels for a symbol over the dataframe window."""

    key = _window_key(df_symbol, level_types)
    cached = _LEVELS_CACHE.get(key)
    if cached is not None:
        return cached

    symbol = key[0]
    if not symbol:
        _LEVELS_CACHE[key] = pd.DataFrame()
        return _LEVELS_CACHE[key]

    start, end = key[1], key[2]
    try:
        engine = lvl_repo.get_engine()
    except Exception:
        _LEVELS_CACHE[key] = pd.DataFrame()
        return _LEVELS_CACHE[key]

    try:
        levels_df = lvl_repo.select_levels(
            engine,
            LEVELS_TABLE,
            symbol=symbol,
            level_types=list(level_types),
            active_only=False,
            start=start.isoformat() if isinstance(start, pd.Timestamp) else start,
            end=end.isoformat() if isinstance(end, pd.Timestamp) else end,
        )
    except Exception:
        levels_df = pd.DataFrame()

    levels_df = _normalise_levels(levels_df)
    _LEVELS_CACHE[key] = levels_df
    return levels_df


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


def in_zone_level(level_type: str, tolerance: float = 0.0) -> Callable[[pd.DataFrame, Optional[pd.DataFrame]], pd.Series]:
    """Build a callable returning a boolean mask when price trades inside a level."""

    def _inner(df_symbol: pd.DataFrame, levels_df: Optional[pd.DataFrame] = None) -> pd.Series:
        if df_symbol.empty:
            return pd.Series(False, index=df_symbol.index, dtype="boolean")
        levels = levels_df
        if levels is None:
            levels = _load_levels_for(df_symbol, [level_type])
        series = lvl_helpers.in_zone(df_symbol, levels, level_type, tolerance=tolerance)
        return series.reindex(df_symbol.index, fill_value=False).astype("boolean")

    return _inner


def distance_to_level(
    level_type: str,
    side: str = "mid",
    thresh: float | None = None,
) -> Callable[[pd.DataFrame, Optional[pd.DataFrame]], pd.Series]:
    """Build a callable returning distance (or mask) to the requested level."""

    def _inner(df_symbol: pd.DataFrame, levels_df: Optional[pd.DataFrame] = None) -> pd.Series:
        if df_symbol.empty:
            if thresh is None:
                return pd.Series(dtype="float64", index=df_symbol.index)
            return pd.Series(False, index=df_symbol.index, dtype="boolean")
        levels = levels_df
        if levels is None:
            levels = _load_levels_for(df_symbol, [level_type])
        distances = lvl_helpers.distance_to(df_symbol, levels, level_type, side=side)
        distances = distances.reindex(df_symbol.index)
        if thresh is None:
            return distances
        mask = (distances <= float(thresh)).fillna(False)
        return mask.astype("boolean")

    return _inner


def touched_level_since(level_type: str, bars: int = 1) -> Callable[[pd.DataFrame, Optional[pd.DataFrame]], pd.Series]:
    """Build a callable returning True if a level was touched in the lookback window."""

    def _inner(df_symbol: pd.DataFrame, levels_df: Optional[pd.DataFrame] = None) -> pd.Series:
        if df_symbol.empty:
            return pd.Series(False, index=df_symbol.index, dtype="boolean")
        levels = levels_df
        if levels is None:
            levels = _load_levels_for(df_symbol, [level_type])
        touched = lvl_helpers.touched_since(df_symbol, levels, level_type, bars=bars)
        return touched.reindex(df_symbol.index, fill_value=False).astype("boolean")

    return _inner


def list_condition_types() -> list[str]:
    """Return the list of available condition factory names."""

    supported: list[str] = []
    for name, obj in globals().items():
        if name.startswith("_") or name == "list_condition_types":
            continue
        if callable(obj):
            supported.append(name)
    return sorted(set(supported))

