"""End-to-end orchestration for levels build operations."""
from __future__ import annotations

from typing import Dict

import pandas as pd

from ..core.dataset import load_ohlcv
from .builders import build_levels
from .detectors import fill_fvgs, fill_gaps
from .repo import ensure_table, get_engine, select_levels, upsert_levels, upsert_valid_to_ts
from .schemas import LevelsBuildSpec


def run_levels_build(spec: LevelsBuildSpec) -> Dict[str, object]:
    """Load data, compute requested levels and persist them."""

    ohlcv = load_ohlcv(spec.data)
    records = build_levels(spec, ohlcv)
    table_fqn = f"{spec.output_schema}.{spec.output_table}" if spec.output_schema else spec.output_table
    engine = get_engine()
    ensure_table(engine, table_fqn)
    result = upsert_levels(engine, table_fqn, records) if spec.upsert else {"inserted": 0, "updated": 0}
    payload = {
        "table": table_fqn,
        "count": len(records),
        **result,
    }
    return payload


def run_levels_fill(spec: LevelsBuildSpec) -> Dict[str, object]:
    """Refresh ``valid_to_ts`` for active FVG and GAP levels."""

    ohlcv = load_ohlcv(spec.data)
    if ohlcv.empty:
        return {"updated": 0, "checked": 0}
    df = ohlcv.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)

    do_fill_fvg = any(target.type.upper() == "FILL_FVG" for target in spec.targets)
    do_fill_gap = any(target.type.upper() == "FILL_GAP" for target in spec.targets)
    if not (do_fill_fvg or do_fill_gap):
        return {"updated": 0, "checked": 0}

    table_fqn = f"{spec.output_schema}.{spec.output_table}" if spec.output_schema else spec.output_table
    engine = get_engine()
    ensure_table(engine, table_fqn)

    total_checked = 0
    pending_updates: list[pd.DataFrame] = []
    for symbol in spec.symbols:
        sym_df = df[df["symbol"] == symbol]
        if sym_df.empty:
            continue
        level_types: list[str] = []
        if do_fill_fvg:
            level_types.append("FVG")
        if do_fill_gap:
            level_types.extend(["GAP_D", "GAP_W"])
        if not level_types:
            continue
        levels = select_levels(
            engine,
            table_fqn,
            symbol=symbol,
            level_types=list(dict.fromkeys(level_types)),
            active_only=True,
            start=spec.range_start,
            end=spec.range_end,
            limit=10000,
        )
        if levels.empty:
            continue
        total_checked += int(len(levels))
        if do_fill_fvg:
            fvgs_active = levels[levels["level_type"] == "FVG"]
            if not fvgs_active.empty:
                fvgs_filled = fill_fvgs(sym_df, fvgs_active)
                fvgs_updates = fvgs_filled[fvgs_filled["valid_to_ts"].notna()]
                if not fvgs_updates.empty:
                    pending_updates.append(fvgs_updates)
        if do_fill_gap:
            gaps_active = levels[levels["level_type"].isin(["GAP_D", "GAP_W"])]
            if not gaps_active.empty:
                gaps_filled = fill_gaps(sym_df, gaps_active)
                gaps_updates = gaps_filled[gaps_filled["valid_to_ts"].notna()]
                if not gaps_updates.empty:
                    pending_updates.append(gaps_updates)
    if pending_updates:
        updates_df = pd.concat(pending_updates, ignore_index=True)
        updated_count = upsert_valid_to_ts(engine, table_fqn, updates_df)
    else:
        updated_count = 0
    return {"updated": int(updated_count), "checked": int(total_checked)}


__all__ = ["run_levels_build", "run_levels_fill"]
