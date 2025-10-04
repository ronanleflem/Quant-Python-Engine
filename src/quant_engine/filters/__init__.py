"""Collection of reusable pre-trade filters."""
from __future__ import annotations

from .volatility_trend import adx_filter, atr_filter, ema_slope_filter
from .volume_profile import volume_surge_filter, vwap_side_filter, poc_distance_filter
from .structure_ict import liquidity_sweep_filter, bos_filter, mss_filter
from .time_seasonality import (
    session_time_filter,
    day_of_week_filter,
    day_of_month_filter,
    month_of_year_filter,
    intraday_time_filter,
)

__all__ = [
    "adx_filter",
    "atr_filter",
    "ema_slope_filter",
    "volume_surge_filter",
    "vwap_side_filter",
    "poc_distance_filter",
    "liquidity_sweep_filter",
    "bos_filter",
    "mss_filter",
    "session_time_filter",
    "day_of_week_filter",
    "day_of_month_filter",
    "month_of_year_filter",
    "intraday_time_filter",
    "filters_registry",
    "list_filter_types",
]

filters_registry = {
    "adx": adx_filter,
    "atr": atr_filter,
    "ema_slope": ema_slope_filter,
    "volume_surge": volume_surge_filter,
    "vwap_side": vwap_side_filter,
    "poc_distance": poc_distance_filter,
    "liquidity_sweep": liquidity_sweep_filter,
    "bos": bos_filter,
    "mss": mss_filter,
    "session_time": session_time_filter,
    "day_of_week": day_of_week_filter,
    "day_of_month": day_of_month_filter,
    "month_of_year": month_of_year_filter,
    "intraday_time": intraday_time_filter,
}


def list_filter_types() -> list[str]:
    """Return the list of available filter identifiers."""

    return sorted(filters_registry)
