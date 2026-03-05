"""Regime condition helpers for statistics runs."""
from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd

from quant_engine.levels import helpers as lvl_helpers
from quant_engine.levels import repo as lvl_repo
from quant_engine.market_intelligence import conditions as mi_conditions


LEVELS_TABLE = "marketdata.levels"
_LEVELS_CACHE: Dict[Tuple[str, Optional[pd.Timestamp], Optional[pd.Timestamp], Tuple[str, ...]], pd.DataFrame] = {}
_DEPRECATION_TARGET_VERSION = "v0.15.0"
_DEPRECATION_TARGET_DATE = "2026-04-30"


def _warn_deprecated(func_name: str) -> None:
    warnings.warn(
        (
            f"quant_engine.stats.conditions.{func_name} is deprecated and will be removed "
            f"in {_DEPRECATION_TARGET_VERSION} (target date: {_DEPRECATION_TARGET_DATE}); "
            "use quant_engine.market_intelligence.conditions instead."
        ),
        DeprecationWarning,
        stacklevel=2,
    )


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
    """Return higher time frame trend as ``"up"`` or ``"down"``."""

    _warn_deprecated("htf_trend")
    return mi_conditions.htf_trend(df, tf_multiplier=tf_multiplier, ema_period=ema_period)


def vol_tertile(df: pd.DataFrame, *, window: int) -> pd.Series:
    """Classify current ATR into tertiles across the sample."""

    _warn_deprecated("vol_tertile")
    return mi_conditions.vol_tertile(df, window=window)


def session(df: pd.DataFrame, *, col: str = "session_id") -> pd.Series:
    """Return the session label as a categorical series."""

    _warn_deprecated("session")
    return mi_conditions.session(df, col=col)


def hour_bin(df: pd.DataFrame) -> pd.Series:
    """Return hour-of-day bins (0-23) from timestamp column."""

    _warn_deprecated("hour_bin")
    return mi_conditions.hour_bin(df)


def day_of_week(df: pd.DataFrame) -> pd.Series:
    """Return day-of-week bins (0=Mon..6=Sun)."""

    _warn_deprecated("day_of_week")
    return mi_conditions.day_of_week(df)


def month_of_year(df: pd.DataFrame) -> pd.Series:
    """Return month-of-year bins (1-12)."""

    _warn_deprecated("month_of_year")
    return mi_conditions.month_of_year(df)


def session_from_ts(df: pd.DataFrame) -> pd.Series:
    """Return a coarse session label from timestamp if session_id is missing."""

    _warn_deprecated("session_from_ts")
    return mi_conditions.session_from_ts(df)


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
