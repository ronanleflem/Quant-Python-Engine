"""Macro COT/OI integration filter."""
from __future__ import annotations

from typing import Optional

import pandas as pd


def macro_cot_oi_filter(
    df: pd.DataFrame,
    *,
    cot_col: str = "cot_bias",
    oi_col: str = "oi_change",
    cot_bias_threshold: Optional[float] = None,
    oi_change_threshold: Optional[float] = None,
    side: str = "bull",
    require_all: bool = True,
    allow_if_missing: bool = True,
) -> pd.Series:
    """Return True when macro COT/OI signals align with the requested side."""
    side_key = str(side).strip().lower()
    if cot_col not in df.columns and oi_col not in df.columns:
        return pd.Series(True, index=df.index) if allow_if_missing else pd.Series(False, index=df.index)

    if cot_col in df.columns:
        cot = df[cot_col].astype(float)
    else:
        cot = pd.Series(pd.NA, index=df.index, dtype="float64")
    if oi_col in df.columns:
        oi = df[oi_col].astype(float)
    else:
        oi = pd.Series(pd.NA, index=df.index, dtype="float64")

    flags = []
    if cot_bias_threshold is not None and cot_col in df.columns:
        thresh = float(cot_bias_threshold)
        if side_key in {"bear", "short", "down"}:
            flags.append(cot <= -thresh)
        else:
            flags.append(cot >= thresh)
    if oi_change_threshold is not None and oi_col in df.columns:
        thresh = float(oi_change_threshold)
        if side_key in {"bear", "short", "down"}:
            flags.append(oi <= -thresh)
        else:
            flags.append(oi >= thresh)

    if not flags:
        return pd.Series(True, index=df.index) if allow_if_missing else pd.Series(False, index=df.index)

    if require_all:
        cond = flags[0]
        for flag in flags[1:]:
            cond &= flag
    else:
        cond = flags[0]
        for flag in flags[1:]:
            cond |= flag
    return cond.fillna(False).astype(bool)


__all__ = ["macro_cot_oi_filter"]
