"""Gate filter using persisted market stats with conditional fallback."""
from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

import pandas as pd

from ..persistence import db
from ..stats import conditions as stats_conditions
from ..stats import events as stats_events

LOGGER = logging.getLogger(__name__)


def _lookup_metric_row(
    conn,
    *,
    symbol: str,
    timeframe: str,
    event: str,
    target: str,
    split: str,
    condition_name: Optional[str],
    condition_value: Optional[str],
) -> Optional[Mapping[str, Any]]:
    cur = conn.cursor()
    if condition_name is None:
        cur.execute(
            """
            SELECT * FROM market_stats
            WHERE symbol = ? AND timeframe = ? AND event = ? AND target = ? AND split = ?
              AND condition_name IS NULL
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (symbol, timeframe, event, target, split),
        )
    else:
        cur.execute(
            """
            SELECT * FROM market_stats
            WHERE symbol = ? AND timeframe = ? AND event = ? AND target = ? AND split = ?
              AND condition_name = ? AND condition_value = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (symbol, timeframe, event, target, split, condition_name, condition_value),
        )
    row = cur.fetchone()
    return dict(row) if row else None


def _resolve_condition_value(
    df: pd.DataFrame,
    condition_name: str,
    params: Mapping[str, Any],
) -> Optional[str]:
    if "condition_value" in params:
        value = params.get("condition_value")
        return None if value is None else str(value)
    func = getattr(stats_conditions, condition_name, None)
    if func is None:
        raise ValueError(f"Unknown condition '{condition_name}'")
    df_local = df.copy()
    if "ts" not in df_local.columns:
        df_local["ts"] = df_local.index
    series = func(df_local, **(params.get("condition_params") or {}))
    if series.empty:
        return None
    value = series.iloc[-1]
    return None if pd.isna(value) else str(value)


def _resolve_event_trigger(
    df: pd.DataFrame,
    event: str,
    event_params: Mapping[str, Any],
) -> bool:
    func = getattr(stats_events, event, None)
    if func is None:
        raise ValueError(f"Unknown event '{event}'")
    series = func(df, **event_params)
    if series.empty:
        return False
    return bool(series.iloc[-1])


def _resolve_stats_row(
    df: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str,
    event: str,
    target: str,
    split: str,
    condition_name: Optional[str],
    condition_params: Optional[Mapping[str, Any]],
    condition_value: Optional[str],
) -> Optional[Mapping[str, Any]]:
    cond_name = condition_name
    cond_value = condition_value
    if cond_name:
        cond_value = _resolve_condition_value(
            df, cond_name, {"condition_value": cond_value, "condition_params": condition_params}
        )

    with db.session() as conn:
        row = None
        if cond_name:
            row = _lookup_metric_row(
                conn,
                symbol=symbol,
                timeframe=timeframe,
                event=event,
                target=target,
                split=split,
                condition_name=cond_name,
                condition_value=cond_value,
            )
        if row is None:
            row = _lookup_metric_row(
                conn,
                symbol=symbol,
                timeframe=timeframe,
                event=event,
                target=target,
                split=split,
                condition_name=None,
                condition_value=None,
            )
    return row


def stats_gate_filter(
    df: pd.DataFrame,
    *,
    event: str,
    target: str,
    threshold: float,
    comparator: str = "gte",
    metric: str = "p_hat",
    split: str = "test",
    condition_name: Optional[str] = None,
    condition_params: Optional[Mapping[str, Any]] = None,
    condition_value: Optional[str] = None,
    min_samples: int = 300,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    allow_if_missing: bool = True,
    allow_if_insufficient: bool = True,
) -> pd.Series:
    """Return True if the DB stats meet the configured threshold."""
    if df.empty:
        return pd.Series(False, index=df.index)

    symbol_value = symbol
    if symbol_value is None and "symbol" in df.columns:
        symbol_value = str(df["symbol"].iloc[-1])
    if symbol_value is None:
        raise ValueError("stats_gate_filter requires symbol")

    tf_value = timeframe
    if tf_value is None and "timeframe" in df.columns:
        tf_value = str(df["timeframe"].iloc[-1])
    if tf_value is None:
        raise ValueError("stats_gate_filter requires timeframe")

    if not _resolve_event_trigger(df, event, event_params=condition_params or {}):
        return pd.Series(True, index=df.index)

    row = _resolve_stats_row(
        df,
        symbol=symbol_value,
        timeframe=tf_value,
        event=event,
        target=target,
        split=split,
        condition_name=condition_name,
        condition_params=condition_params,
        condition_value=condition_value,
    )

    if row is None:
        if allow_if_missing:
            return pd.Series(True, index=df.index)
        raise ValueError("No stats found for stats_gate_filter")

    n = row.get("n")
    if n is not None and int(n) < int(min_samples):
        if allow_if_insufficient:
            return pd.Series(True, index=df.index)
        raise ValueError("Stats sample size below min_samples")

    metric_value = row.get(metric)
    if metric_value is None:
        if allow_if_missing:
            return pd.Series(True, index=df.index)
        raise ValueError(f"Metric '{metric}' missing in stats row")

    metric_value = float(metric_value)
    threshold_val = float(threshold)
    comparator_key = comparator.lower()
    if comparator_key in {"gt", "greater"}:
        ok = metric_value > threshold_val
    elif comparator_key in {"lt", "less"}:
        ok = metric_value < threshold_val
    elif comparator_key in {"lte", "less_equal"}:
        ok = metric_value <= threshold_val
    else:
        ok = metric_value >= threshold_val

    return pd.Series(bool(ok), index=df.index)


def stats_gate_score(
    df: pd.DataFrame,
    *,
    event: str,
    target: str,
    metric: str = "p_hat",
    split: str = "test",
    condition_name: Optional[str] = None,
    condition_params: Optional[Mapping[str, Any]] = None,
    condition_value: Optional[str] = None,
    min_samples: int = 300,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    allow_if_missing: bool = True,
    allow_if_insufficient: bool = True,
    scale_min: float = 0.0,
    scale_max: float = 1.0,
) -> pd.Series:
    """Return a score series based on the selected metric."""
    if df.empty:
        return pd.Series(0.0, index=df.index)

    symbol_value = symbol
    if symbol_value is None and "symbol" in df.columns:
        symbol_value = str(df["symbol"].iloc[-1])
    if symbol_value is None:
        raise ValueError("stats_gate_score requires symbol")

    tf_value = timeframe
    if tf_value is None and "timeframe" in df.columns:
        tf_value = str(df["timeframe"].iloc[-1])
    if tf_value is None:
        raise ValueError("stats_gate_score requires timeframe")

    if not _resolve_event_trigger(df, event, event_params=condition_params or {}):
        return pd.Series(0.0, index=df.index)

    row = _resolve_stats_row(
        df,
        symbol=symbol_value,
        timeframe=tf_value,
        event=event,
        target=target,
        split=split,
        condition_name=condition_name,
        condition_params=condition_params,
        condition_value=condition_value,
    )

    if row is None:
        return pd.Series(0.0 if allow_if_missing else float("nan"), index=df.index)

    n = row.get("n")
    if n is not None and int(n) < int(min_samples):
        return pd.Series(0.0 if allow_if_insufficient else float("nan"), index=df.index)

    metric_value = row.get(metric)
    if metric_value is None:
        return pd.Series(0.0 if allow_if_missing else float("nan"), index=df.index)

    value = float(metric_value)
    if scale_max > scale_min:
        value = max(scale_min, min(scale_max, value))
        value = (value - scale_min) / (scale_max - scale_min)
    return pd.Series(value, index=df.index)


__all__ = ["stats_gate_filter", "stats_gate_score"]
