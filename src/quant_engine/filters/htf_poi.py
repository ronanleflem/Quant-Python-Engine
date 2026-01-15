"""High-timeframe zones / POI filter."""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

try:
    from quant_engine.levels import repo as lvl_repo
    from quant_engine.levels import helpers as lvl_helpers
except Exception:
    lvl_repo = None
    lvl_helpers = None


def _to_level_types(level_types: Optional[Iterable[str]]) -> list[str]:
    if not level_types:
        return ["FVG_HTF"]
    return [str(item).strip().upper() for item in level_types if str(item).strip()]


def htf_poi_filter(
    df: pd.DataFrame,
    *,
    level_types: Optional[Iterable[str]] = None,
    symbol: Optional[str] = None,
    mode: str = "in_zone",
    tolerance: float = 0.0,
    max_distance: Optional[float] = None,
    distance_unit: str = "abs",
    require_all: bool = False,
    allow_if_missing: bool = True,
    allow_if_empty: bool = True,
    close_col: str = "close",
) -> pd.Series:
    """Return True when price interacts with active HTF zones."""
    if close_col not in df.columns:
        raise ValueError(f"DataFrame missing column: {close_col}")

    if lvl_repo is None or lvl_helpers is None or symbol is None:
        if allow_if_missing:
            return pd.Series(True, index=df.index)
        raise ValueError("levels repository not available or symbol missing")

    lvl_types = _to_level_types(level_types)
    if not lvl_types:
        return pd.Series(True, index=df.index)

    try:
        engine = lvl_repo.get_engine() if hasattr(lvl_repo, "get_engine") else None
        table_fqn = "marketdata.levels"
        levels = lvl_repo.select_levels(
            engine=engine,
            table_fqn=table_fqn,
            symbol=symbol,
            level_types=lvl_types,
            active_only=True,
            start=df.index.min().isoformat(),
            end=df.index.max().isoformat(),
            limit=100000,
        )
    except Exception:
        if allow_if_missing:
            return pd.Series(True, index=df.index)
        raise
    if not isinstance(levels, pd.DataFrame) or levels.empty:
        return pd.Series(True, index=df.index) if allow_if_empty else pd.Series(False, index=df.index)

    mode_key = str(mode).strip().lower()
    close = df[close_col].astype(float)
    pieces: list[pd.Series] = []

    for level_type in lvl_types:
        if mode_key == "in_zone":
            mask = lvl_helpers.in_zone(df, levels, level_type, tolerance=float(tolerance))
        elif mode_key == "distance":
            if max_distance is None:
                raise ValueError("max_distance is required when mode='distance'")
            dist = lvl_helpers.distance_to(df, levels, level_type, side="edge")
            if distance_unit == "pct":
                dist = dist / close.replace(0.0, np.nan)
            mask = dist <= float(max_distance)
        else:
            raise ValueError("mode must be 'in_zone' or 'distance'")
        pieces.append(mask.reindex(df.index).fillna(False).astype(bool))

    if not pieces:
        return pd.Series(True, index=df.index)

    if require_all:
        combined = pieces[0]
        for piece in pieces[1:]:
            combined &= piece
    else:
        combined = pieces[0]
        for piece in pieces[1:]:
            combined |= piece
    return combined.reindex(df.index).fillna(False)


__all__ = ["htf_poi_filter"]
