"""Collection of reusable pre-trade filters."""
from __future__ import annotations

from .volatility_trend import adx_filter, atr_filter, ema_slope_filter
from .volume_profile import volume_surge_filter, vwap_side_filter, poc_distance_filter

__all__ = [
    "adx_filter",
    "atr_filter",
    "ema_slope_filter",
    "volume_surge_filter",
    "vwap_side_filter",
    "poc_distance_filter",
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
}


def list_filter_types() -> list[str]:
    """Return the list of available filter identifiers."""

    return sorted(filters_registry)
