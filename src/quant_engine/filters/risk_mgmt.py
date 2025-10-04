"""Risk and money management oriented filters."""
from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pandas as pd


def _daily_groups(idx: pd.DatetimeIndex) -> pd.Index:
    """Return a normalized UTC date index suitable for grouping by day."""

    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError("Index must be a DatetimeIndex")

    if idx.tz is None:
        utc_idx = idx.tz_localize("UTC")
    else:
        utc_idx = idx.tz_convert("UTC")
    return utc_idx.normalize()


def daily_loss_cap_filter(
    df: pd.DataFrame,
    loss_cap: float,
    mode: Literal["pnl", "equity", "ret_notional"] = "pnl",
    pnl_col: str = "pnl",
    equity_col: str = "equity",
    ret_col: str = "ret",
    notional: Optional[float] = None,
) -> pd.Series:
    """Return ``False`` once the cumulative daily loss reaches the cap.

    Lockout is triggered when the cumulative PnL for the current UTC day is
    less than or equal to ``-loss_cap``. The filter returns ``True`` elsewhere.

    Missing inputs (columns or ``notional`` when required) result in a no-op
    filter that returns ``True`` everywhere.
    """

    if loss_cap <= 0:
        raise ValueError("loss_cap must be positive")

    try:
        groups = _daily_groups(df.index)
    except Exception:
        return pd.Series(True, index=df.index)

    try:
        if mode == "pnl" and pnl_col in df.columns:
            pnl = df[pnl_col].astype(float)
        elif mode == "equity" and equity_col in df.columns:
            pnl = df[equity_col].astype(float).diff()
        elif (
            mode == "ret_notional"
            and ret_col in df.columns
            and notional is not None
        ):
            pnl = df[ret_col].astype(float) * float(notional)
        else:
            return pd.Series(True, index=df.index)

        daily_cum = pnl.groupby(groups).cumsum()
        breached = daily_cum <= -abs(float(loss_cap))
        lock = breached.groupby(groups).cummax()
        return (~lock).fillna(True)
    except Exception:
        return pd.Series(True, index=df.index)


def daily_trades_cap_filter(
    df: pd.DataFrame,
    signal_col: str,
    max_trades_per_day: int = 3,
) -> pd.Series:
    """Allow at most ``max_trades_per_day`` True signals per UTC day.

    Missing ``signal_col`` results in a no-op returning ``True`` everywhere.
    """

    if signal_col not in df.columns:
        return pd.Series(True, index=df.index)

    if max_trades_per_day < 1:
        raise ValueError("max_trades_per_day must be >= 1")

    try:
        groups = _daily_groups(df.index)
    except Exception:
        return pd.Series(True, index=df.index)

    sig = df[signal_col].fillna(False).astype(bool)
    count = sig.groupby(groups).cumsum()
    allowed = count <= int(max_trades_per_day)
    return allowed.reindex(df.index, fill_value=True)


def cooldown_bars_filter(
    df: pd.DataFrame,
    signal_col: str,
    cooldown_bars: int = 10,
) -> pd.Series:
    """Impose a cooldown of ``cooldown_bars`` bars after a True signal.

    Missing ``signal_col`` returns ``True`` everywhere.
    """

    if signal_col not in df.columns:
        return pd.Series(True, index=df.index)

    if cooldown_bars < 0:
        raise ValueError("cooldown_bars must be >= 0")

    sig = df[signal_col].fillna(False).astype(bool)
    if cooldown_bars == 0:
        return pd.Series(True, index=df.index)

    positions = pd.Series(np.arange(len(df)), index=df.index, dtype=float)
    true_positions = positions.where(sig)
    last_true_before = true_positions.shift().ffill()
    distance = positions - last_true_before
    allowed = last_true_before.isna() | (distance > cooldown_bars)
    return allowed.fillna(True)


def atr_risk_gate_filter(
    df: pd.DataFrame,
    atr_window: int = 14,
    max_atr_pct: float = 0.02,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pd.Series:
    """Block when ATR to close ratio exceeds ``max_atr_pct``.

    Missing OHLC columns results in a no-op returning ``True`` everywhere.
    """

    required = {high_col, low_col, close_col}
    if not required.issubset(df.columns):
        return pd.Series(True, index=df.index)

    if atr_window <= 0:
        raise ValueError("atr_window must be positive")
    if max_atr_pct <= 0:
        raise ValueError("max_atr_pct must be positive")

    h = df[high_col].astype(float)
    l = df[low_col].astype(float)
    c = df[close_col].astype(float)

    prev_close = c.shift(1)
    tr_components = pd.concat(
        [(h - l).abs(), (h - prev_close).abs(), (l - prev_close).abs()], axis=1
    )
    tr = tr_components.max(axis=1)

    atr = tr.ewm(alpha=1.0 / float(atr_window), adjust=False).mean()
    atr_pct = atr / c.replace(0.0, np.nan)
    allowed = atr_pct <= float(max_atr_pct)
    return allowed.fillna(True)


def equity_dd_lockout_filter(
    df: pd.DataFrame,
    equity_col: str = "equity",
    max_dd_pct: float = 0.1,
) -> pd.Series:
    """Lock out once the equity drawdown exceeds ``max_dd_pct``.

    Missing ``equity_col`` returns ``True`` everywhere.
    """

    if equity_col not in df.columns:
        return pd.Series(True, index=df.index)

    if max_dd_pct < 0:
        raise ValueError("max_dd_pct must be non-negative")

    eq = df[equity_col].astype(float)
    peak = eq.cummax()
    drawdown = (peak - eq) / peak.replace(0.0, np.nan)
    allowed = drawdown <= float(max_dd_pct)
    return allowed.fillna(True)
