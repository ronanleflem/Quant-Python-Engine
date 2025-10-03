"""Helper utilities to consume detected levels within analytics pipelines."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _normalise_timestamp(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def _prepare_levels(levels_df: pd.DataFrame) -> pd.DataFrame:
    levels = levels_df.copy()
    if levels.empty:
        return levels
    for col in ("ts", "anchor_ts", "valid_from_ts", "valid_to_ts"):
        if col in levels.columns:
            levels[col] = _normalise_timestamp(levels[col])
    if "valid_from_ts" not in levels.columns:
        levels["valid_from_ts"] = levels.get("anchor_ts")
    levels["effective_from"] = levels["valid_from_ts"].fillna(levels.get("anchor_ts"))
    return levels


def join_levels(
    df: pd.DataFrame,
    levels_df: pd.DataFrame,
    how: str = "asof",
    on: str = "ts",
    by: str = "symbol",
) -> pd.DataFrame:
    """Join OHLCV bars with the most recent active levels."""

    if df.empty:
        return df.copy()
    left = df.copy()
    left[on] = _normalise_timestamp(left[on])
    if levels_df.empty:
        return left
    if how != "asof":
        raise ValueError("Only 'asof' joins are currently supported")

    right = _prepare_levels(levels_df)
    right_cols = [col for col in right.columns if col not in {by, "effective_from"}]
    left["__orig_idx"] = left.index

    if by in left.columns and by in right.columns:
        left_sorted = left.sort_values([by, on])
        right_sorted = right.sort_values([by, "effective_from"])
        merged = pd.merge_asof(
            left_sorted,
            right_sorted,
            left_on=on,
            right_on="effective_from",
            by=by,
            direction="backward",
        )
    else:
        left_sorted = left.sort_values(on)
        right_sorted = right.sort_values("effective_from")
        merged = pd.merge_asof(
            left_sorted,
            right_sorted,
            left_on=on,
            right_on="effective_from",
            direction="backward",
        )

    if "valid_from_ts" in merged.columns or "valid_to_ts" in merged.columns:
        valid_from = merged.get("valid_from_ts")
        if valid_from is None:
            valid_from = merged.get("anchor_ts")
        valid_to = merged.get("valid_to_ts")
        ts_series = merged[on]
        active_mask = pd.Series(True, index=merged.index)
        if valid_from is not None:
            active_mask &= ts_series >= valid_from.fillna(pd.Timestamp.min.tz_localize("UTC"))
        if valid_to is not None:
            active_mask &= valid_to.isna() | (ts_series <= valid_to)
        inactive = ~active_mask
        if inactive.any():
            cols_to_clear = [col for col in right_cols if col in merged.columns]
            merged.loc[inactive, cols_to_clear] = np.nan

    merged.sort_values("__orig_idx", inplace=True)
    merged.set_index("__orig_idx", inplace=True)
    merged.index.name = None
    merged.drop(columns=[col for col in ["effective_from"] if col in merged.columns], inplace=True)
    return merged


def _filter_levels(levels_df: pd.DataFrame, level_type: str) -> pd.DataFrame:
    if levels_df.empty:
        return levels_df
    return levels_df[levels_df["level_type"] == level_type].copy()


def distance_to(
    df: pd.DataFrame,
    levels_df: pd.DataFrame,
    level_type: str,
    side: str = "mid",
) -> pd.Series:
    """Return absolute price distance from the close to the requested level."""

    if df.empty:
        return pd.Series(dtype="float64", index=df.index)
    filtered = _filter_levels(levels_df, level_type)
    enriched = join_levels(df, filtered)
    if enriched.empty or "close" not in enriched.columns:
        return pd.Series(dtype="float64", index=enriched.index)
    close = enriched["close"].astype(float)
    distances = pd.Series(np.nan, index=enriched.index, dtype="float64")
    level_col = enriched.get("level_type")
    if level_col is None:
        return distances
    level_mask = level_col.fillna("") == level_type
    if not level_mask.any():
        return distances
    price = enriched.get("price")
    price_lo = enriched.get("price_lo")
    price_hi = enriched.get("price_hi")
    if side == "mid":
        mid = price
        if mid is None:
            mid = pd.Series(np.nan, index=enriched.index)
        if price_lo is not None and price_hi is not None:
            zone_mid = (price_lo + price_hi) / 2.0
            mid = mid.fillna(zone_mid)
        distances.loc[level_mask] = (close - mid).abs()[level_mask]
    elif side == "edge":
        diff_price = pd.Series(np.nan, index=enriched.index)
        if price_lo is not None and price_hi is not None:
            lower = (close - price_lo).abs()
            upper = (close - price_hi).abs()
            edge_dist = pd.concat([lower, upper], axis=1).min(axis=1)
            diff_price = edge_dist
        if price is not None:
            diff_price = diff_price.fillna((close - price).abs())
        distances.loc[level_mask] = diff_price[level_mask]
    else:
        raise ValueError("side must be 'mid' or 'edge'")
    return distances


def in_zone(
    df: pd.DataFrame,
    levels_df: pd.DataFrame,
    level_type: str,
    tolerance: float = 0.0,
) -> pd.Series:
    """Return a boolean Series indicating whether the close trades within the zone."""

    if df.empty:
        return pd.Series(dtype="boolean", index=df.index)
    filtered = _filter_levels(levels_df, level_type)
    enriched = join_levels(df, filtered)
    result = pd.Series(False, index=enriched.index, dtype="boolean")
    if enriched.empty or "close" not in enriched.columns:
        return result
    level_col = enriched.get("level_type")
    if level_col is None:
        return result
    level_mask = level_col.fillna("") == level_type
    if not level_mask.any():
        return result
    close = enriched["close"].astype(float)
    price = enriched.get("price")
    price_lo = enriched.get("price_lo")
    price_hi = enriched.get("price_hi")
    tol = float(tolerance)
    if price_lo is not None and price_hi is not None:
        lower = price_lo - tol
        upper = price_hi + tol
        zone_mask = (close >= lower) & (close <= upper)
        result.loc[level_mask] = zone_mask[level_mask]
    elif price is not None:
        point_mask = (close - price).abs() <= tol
        result.loc[level_mask] = point_mask[level_mask]
    return result


def touched_since(
    df: pd.DataFrame,
    levels_df: pd.DataFrame,
    level_type: str,
    bars: int = 1,
) -> pd.Series:
    """Return True when the zone has been touched within the lookback window."""

    bars = max(int(bars), 1)
    hits = in_zone(df, levels_df, level_type)
    if hits.empty:
        return hits
    rolled = hits.astype(int).rolling(window=bars, min_periods=1).max()
    return rolled.astype(bool).astype("boolean")


__all__ = ["join_levels", "distance_to", "in_zone", "touched_since"]

