"""ICT Point of Interest (POI) filter."""
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
        return []
    return [str(item).strip().upper() for item in level_types if str(item).strip()]


def _ifvg_mask(
    df: pd.DataFrame,
    *,
    lookback: int,
    tolerance: float,
    invalidate_on_fill: bool,
    fill_threshold: float,
    fill_count: int,
    high_col: str,
    low_col: str,
    close_col: str,
) -> pd.Series:
    high = df[high_col].astype(float).to_numpy()
    low = df[low_col].astype(float).to_numpy()
    close = df[close_col].astype(float).to_numpy()
    n = len(df)
    mask = np.zeros(n, dtype=bool)

    bullish_zones: list[tuple[int, float, float]] = []
    bearish_zones: list[tuple[int, float, float]] = []
    ifvg_zones: list[dict] = []

    for i in range(2, n):
        bull = low[i] > high[i - 2]
        bear = high[i] < low[i - 2]

        if bull:
            bullish_zones.append((i, high[i - 2], low[i]))
        if bear:
            bearish_zones.append((i, high[i], low[i - 2]))

        cutoff = i - max(int(lookback), 1)
        bullish_zones = [z for z in bullish_zones if z[0] >= cutoff]
        bearish_zones = [z for z in bearish_zones if z[0] >= cutoff]
        ifvg_zones = [z for z in ifvg_zones if z["created_idx"] >= cutoff]

        if bull and bearish_zones:
            lo_bull = high[i - 2]
            hi_bull = low[i]
            for (_, lo_bear, hi_bear) in bearish_zones:
                lo = max(lo_bull, lo_bear)
                hi = min(hi_bull, hi_bear)
                if lo < hi:
                    ifvg_zones.append(
                        {
                            "created_idx": i,
                            "lo": lo,
                            "hi": hi,
                            "fills": 0,
                            "side": None,
                            "invalid": False,
                        }
                    )
        if bear and bullish_zones:
            lo_bear = high[i]
            hi_bear = low[i - 2]
            for (_, lo_bull, hi_bull) in bullish_zones:
                lo = max(lo_bull, lo_bear)
                hi = min(hi_bull, hi_bear)
                if lo < hi:
                    ifvg_zones.append(
                        {
                            "created_idx": i,
                            "lo": lo,
                            "hi": hi,
                            "fills": 0,
                            "side": None,
                            "invalid": False,
                        }
                    )

        if ifvg_zones:
            price = close[i]
            prev_price = close[i - 1] if i > 0 else price
            for zone in ifvg_zones:
                if zone["invalid"] and invalidate_on_fill:
                    continue
                lo = zone["lo"]
                hi = zone["hi"]
                if (price >= lo - tolerance) and (price <= hi + tolerance):
                    mask[i] = True
                if zone["side"] is None:
                    if prev_price > hi:
                        zone["side"] = "above"
                    elif prev_price < lo:
                        zone["side"] = "below"
                width = hi - lo
                if width <= 0:
                    continue
                threshold = max(0.0, min(float(fill_threshold), 1.0))
                in_zone = not (df[high_col].iloc[i] < lo or df[low_col].iloc[i] > hi)
                if not in_zone:
                    continue
                if threshold <= 0.0:
                    filled = True
                else:
                    if zone["side"] == "above":
                        target = hi - width * threshold
                        filled = df[low_col].iloc[i] <= target
                    elif zone["side"] == "below":
                        target = lo + width * threshold
                        filled = df[high_col].iloc[i] >= target
                    else:
                        target_hi = hi - width * threshold
                        target_lo = lo + width * threshold
                        filled = (df[low_col].iloc[i] <= target_hi) or (df[high_col].iloc[i] >= target_lo)
                if filled:
                    zone["fills"] += 1
                    if zone["fills"] >= max(int(fill_count), 1):
                        zone["invalid"] = True

    return pd.Series(mask, index=df.index)


def _fib_mask(
    df: pd.DataFrame,
    *,
    lookback: int,
    levels: Iterable[float],
    tolerance: float,
    high_col: str,
    low_col: str,
    close_col: str,
) -> pd.Series:
    n = len(df)
    high = df[high_col].astype(float).to_numpy()
    low = df[low_col].astype(float).to_numpy()
    close = df[close_col].astype(float).to_numpy()
    levels = [float(lvl) for lvl in levels]
    mask = np.zeros(n, dtype=bool)
    for i in range(n):
        start = max(0, i - max(int(lookback), 1) + 1)
        swing_high = float(np.nanmax(high[start : i + 1]))
        swing_low = float(np.nanmin(low[start : i + 1]))
        if swing_high <= swing_low:
            continue
        price = close[i]
        for lvl in levels:
            retrace = swing_high - (swing_high - swing_low) * lvl
            if abs(price - retrace) <= tolerance:
                mask[i] = True
                break
    return pd.Series(mask, index=df.index)


def _atr_series(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / float(window), adjust=False).mean()


def _order_block_mask(
    df: pd.DataFrame,
    *,
    lookback: int,
    atr_window: int,
    impulse_atr_mult: float,
    use_body: bool,
    tolerance: float,
    invalidate_on_fill: bool,
    fill_count: int,
    open_col: str,
    high_col: str,
    low_col: str,
    close_col: str,
) -> pd.Series:
    open_ = df[open_col].astype(float).to_numpy()
    high = df[high_col].astype(float).to_numpy()
    low = df[low_col].astype(float).to_numpy()
    close = df[close_col].astype(float).to_numpy()
    atr = _atr_series(
        df[high_col].astype(float),
        df[low_col].astype(float),
        df[close_col].astype(float),
        atr_window,
    ).to_numpy()

    zones: list[dict] = []
    mask = np.zeros(len(df), dtype=bool)
    for i in range(1, len(df)):
        impulse = abs(close[i] - close[i - 1]) >= impulse_atr_mult * atr[i] if not np.isnan(atr[i]) else False
        if impulse:
            direction = "bull" if close[i] > close[i - 1] else "bear"
            start = max(0, i - max(int(lookback), 1))
            last_idx = None
            for j in range(i - 1, start - 1, -1):
                if direction == "bull" and close[j] < open_[j]:
                    last_idx = j
                    break
                if direction == "bear" and close[j] > open_[j]:
                    last_idx = j
                    break
            if last_idx is not None:
                if use_body:
                    lo = min(open_[last_idx], close[last_idx])
                    hi = max(open_[last_idx], close[last_idx])
                else:
                    lo = low[last_idx]
                    hi = high[last_idx]
                zones.append(
                    {"lo": lo, "hi": hi, "fills": 0, "invalid": False}
                )
        price = close[i]
        updated = []
        for zone in zones:
            if zone["invalid"] and invalidate_on_fill:
                updated.append(zone)
                continue
            if (price >= zone["lo"] - tolerance) and (price <= zone["hi"] + tolerance):
                mask[i] = True
                zone["fills"] += 1
                if zone["fills"] >= max(int(fill_count), 1):
                    zone["invalid"] = True
            updated.append(zone)
        zones = updated
    return pd.Series(mask, index=df.index)


def _breaker_block_mask(
    df: pd.DataFrame,
    *,
    sweep_lookback: int,
    retrace_level: float,
    retrace_tol: float,
    continue_bars: int,
    tolerance: float,
    high_col: str,
    low_col: str,
    close_col: str,
) -> pd.Series:
    high = df[high_col].astype(float).to_numpy()
    low = df[low_col].astype(float).to_numpy()
    close = df[close_col].astype(float).to_numpy()
    mask = np.zeros(len(df), dtype=bool)
    last_low_sweep = None
    last_high_sweep = None

    for i in range(1, len(df)):
        start = max(0, i - max(int(sweep_lookback), 1))
        recent_low = np.min(low[start:i])
        recent_high = np.max(high[start:i])
        low_sweep = low[i] < recent_low - tolerance
        high_sweep = high[i] > recent_high + tolerance
        if low_sweep:
            last_low_sweep = i
        if high_sweep:
            last_high_sweep = i

        if last_low_sweep is not None and last_high_sweep is not None:
            if last_low_sweep < last_high_sweep:
                lo = low[last_low_sweep]
                hi = high[last_high_sweep]
                retrace = lo + (hi - lo) * float(retrace_level)
                if abs(close[i] - retrace) <= float(retrace_tol) * (hi - lo):
                    end = min(len(df) - 1, i + max(int(continue_bars), 1))
                    if np.any(close[i:end + 1] > hi):
                        mask[i] = True
            else:
                hi = high[last_high_sweep]
                lo = low[last_low_sweep]
                retrace = hi - (hi - lo) * float(retrace_level)
                if abs(close[i] - retrace) <= float(retrace_tol) * (hi - lo):
                    end = min(len(df) - 1, i + max(int(continue_bars), 1))
                    if np.any(close[i:end + 1] < lo):
                        mask[i] = True
    return pd.Series(mask, index=df.index)


def _model10_mask(
    df: pd.DataFrame,
    *,
    sweep_lookback: int,
    consolidation_bars: int,
    max_range_pct: float,
    tolerance: float,
    high_col: str,
    low_col: str,
    close_col: str,
) -> pd.Series:
    high = df[high_col].astype(float).to_numpy()
    low = df[low_col].astype(float).to_numpy()
    close = df[close_col].astype(float).to_numpy()
    mask = np.zeros(len(df), dtype=bool)
    protected_idx = None
    protected_side = None

    for i in range(1, len(df)):
        start = max(0, i - max(int(sweep_lookback), 1))
        recent_low = np.min(low[start:i])
        recent_high = np.max(high[start:i])
        low_sweep = low[i] < recent_low - tolerance
        high_sweep = high[i] > recent_high + tolerance
        if low_sweep:
            protected_idx = i
            protected_side = "low"
        if high_sweep:
            protected_idx = i
            protected_side = "high"

        if protected_idx is None:
            continue
        if i - protected_idx < max(int(consolidation_bars), 1):
            continue

        cons_start = i - int(consolidation_bars) + 1
        cons_high = np.max(high[cons_start : i + 1])
        cons_low = np.min(low[cons_start : i + 1])
        mid = close[i]
        if mid == 0:
            continue
        range_pct = (cons_high - cons_low) / mid
        if range_pct > float(max_range_pct):
            continue

        if protected_side == "low":
            zone_lo = low[protected_idx]
            zone_hi = cons_high
        else:
            zone_lo = cons_low
            zone_hi = high[protected_idx]
        if zone_lo > zone_hi:
            zone_lo, zone_hi = zone_hi, zone_lo
        if (close[i] >= zone_lo - tolerance) and (close[i] <= zone_hi + tolerance):
            mask[i] = True
    return pd.Series(mask, index=df.index)


def ict_poi_filter(
    df: pd.DataFrame,
    *,
    level_types: Optional[Iterable[str]] = None,
    symbol: Optional[str] = None,
    mode: str = "in_zone",
    tolerance: float = 0.0,
    max_distance: Optional[float] = None,
    distance_unit: str = "abs",
    include_ifvg: bool = False,
    ifvg_lookback: int = 200,
    ifvg_invalidate_on_fill: bool = True,
    ifvg_fill_threshold: float = 1.0,
    ifvg_fill_count: int = 1,
    include_ob: bool = False,
    ob_lookback: int = 50,
    ob_atr_window: int = 14,
    ob_impulse_atr_mult: float = 2.0,
    ob_use_body: bool = True,
    ob_invalidate_on_fill: bool = True,
    ob_fill_count: int = 1,
    include_breaker: bool = False,
    breaker_sweep_lookback: int = 20,
    breaker_retrace_level: float = 0.5,
    breaker_retrace_tol: float = 0.05,
    breaker_continue_bars: int = 5,
    include_model10: bool = False,
    model10_sweep_lookback: int = 30,
    model10_consolidation_bars: int = 5,
    model10_max_range_pct: float = 0.01,
    include_fib: bool = False,
    fib_lookback: int = 200,
    fib_levels: Iterable[float] = (0.382, 0.5, 0.618),
    fib_tolerance: Optional[float] = None,
    allow_if_missing: bool = True,
    allow_if_empty: bool = True,
    close_col: str = "close",
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
) -> pd.Series:
    """Return True when price interacts with ICT POI (levels/IFVG/fib)."""
    if close_col not in df.columns:
        if allow_if_missing:
            return pd.Series(True, index=df.index)
        raise ValueError(f"DataFrame missing column: {close_col}")

    lvl_types = _to_level_types(level_types)
    pieces: list[pd.Series] = []

    if lvl_types:
        if lvl_repo is None or lvl_helpers is None or symbol is None:
            if not allow_if_missing:
                raise ValueError("levels repository not available or symbol missing")
        else:
            try:
                engine = lvl_repo.get_engine() if hasattr(lvl_repo, "get_engine") else None
                levels = lvl_repo.select_levels(
                    engine=engine,
                    table_fqn="marketdata.levels",
                    symbol=symbol,
                    level_types=lvl_types,
                    active_only=True,
                    start=df.index.min().isoformat(),
                    end=df.index.max().isoformat(),
                    limit=100000,
                )
            except Exception:
                levels = pd.DataFrame()
                if not allow_if_missing:
                    raise
            if isinstance(levels, pd.DataFrame) and not levels.empty:
                mode_key = str(mode).strip().lower()
                close = df[close_col].astype(float)
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
            elif not allow_if_empty:
                pieces.append(pd.Series(False, index=df.index))

    if include_ifvg:
        for col in (high_col, low_col):
            if col not in df.columns:
                if allow_if_missing:
                    return pd.Series(True, index=df.index)
                raise ValueError(f"DataFrame missing column: {col}")
        pieces.append(
            _ifvg_mask(
                df,
                lookback=ifvg_lookback,
                tolerance=float(tolerance),
                invalidate_on_fill=ifvg_invalidate_on_fill,
                fill_threshold=ifvg_fill_threshold,
                fill_count=ifvg_fill_count,
                high_col=high_col,
                low_col=low_col,
                close_col=close_col,
            )
        )

    if include_ob:
        for col in (open_col, high_col, low_col):
            if col not in df.columns:
                if allow_if_missing:
                    return pd.Series(True, index=df.index)
                raise ValueError(f"DataFrame missing column: {col}")
        pieces.append(
            _order_block_mask(
                df,
                lookback=ob_lookback,
                atr_window=ob_atr_window,
                impulse_atr_mult=ob_impulse_atr_mult,
                use_body=ob_use_body,
                tolerance=float(tolerance),
                invalidate_on_fill=ob_invalidate_on_fill,
                fill_count=ob_fill_count,
                open_col=open_col,
                high_col=high_col,
                low_col=low_col,
                close_col=close_col,
            )
        )

    if include_breaker:
        for col in (high_col, low_col):
            if col not in df.columns:
                if allow_if_missing:
                    return pd.Series(True, index=df.index)
                raise ValueError(f"DataFrame missing column: {col}")
        pieces.append(
            _breaker_block_mask(
                df,
                sweep_lookback=breaker_sweep_lookback,
                retrace_level=breaker_retrace_level,
                retrace_tol=breaker_retrace_tol,
                continue_bars=breaker_continue_bars,
                tolerance=float(tolerance),
                high_col=high_col,
                low_col=low_col,
                close_col=close_col,
            )
        )

    if include_model10:
        for col in (high_col, low_col):
            if col not in df.columns:
                if allow_if_missing:
                    return pd.Series(True, index=df.index)
                raise ValueError(f"DataFrame missing column: {col}")
        pieces.append(
            _model10_mask(
                df,
                sweep_lookback=model10_sweep_lookback,
                consolidation_bars=model10_consolidation_bars,
                max_range_pct=model10_max_range_pct,
                tolerance=float(tolerance),
                high_col=high_col,
                low_col=low_col,
                close_col=close_col,
            )
        )

    if include_fib:
        for col in (high_col, low_col):
            if col not in df.columns:
                if allow_if_missing:
                    return pd.Series(True, index=df.index)
                raise ValueError(f"DataFrame missing column: {col}")
        tol = float(fib_tolerance) if fib_tolerance is not None else float(tolerance)
        pieces.append(
            _fib_mask(
                df,
                lookback=fib_lookback,
                levels=fib_levels,
                tolerance=tol,
                high_col=high_col,
                low_col=low_col,
                close_col=close_col,
            )
        )

    if not pieces:
        return pd.Series(True, index=df.index) if allow_if_empty else pd.Series(False, index=df.index)

    combined = pieces[0]
    for piece in pieces[1:]:
        combined |= piece
    return combined.reindex(df.index).fillna(False)


__all__ = ["ict_poi_filter"]
