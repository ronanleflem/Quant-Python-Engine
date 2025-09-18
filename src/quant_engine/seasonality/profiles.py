"""Helpers to work with seasonality profiles."""
from __future__ import annotations

from typing import Dict, Sequence

try:  # pragma: no cover - optional dependency
    import polars as pl
except ModuleNotFoundError:  # pragma: no cover - used when dependency missing
    pl = None  # type: ignore


def _require_polars() -> None:
    if pl is None:  # pragma: no cover - exercised when dependency missing
        raise RuntimeError("polars is required for seasonality computations")

from ..api.schemas import SeasonalityProfileSpec, SeasonalitySignalSpec


def select_active_bins(
    profiles: pl.DataFrame,
    signal_spec: SeasonalitySignalSpec,
    profile_spec: SeasonalityProfileSpec,
) -> Dict[str, Sequence]:
    """Return the active bins for each dimension according to the strategy."""

    _require_polars()
    active: Dict[str, Sequence] = {}
    if profiles.is_empty():
        return active

    for dim in signal_spec.dims:
        selected = _select_bins_for_dimension(profiles, dim, signal_spec, profile_spec)
        if selected:
            active[dim] = selected
    return active


def _select_bins_for_dimension(
    profiles: pl.DataFrame,
    dim: str,
    signal_spec: SeasonalitySignalSpec,
    profile_spec: SeasonalityProfileSpec,
) -> Sequence:
    _require_polars()
    metric_col = "p_hat" if profile_spec.measure == "direction" else "ret_mean"
    table = profiles.filter((pl.col("dim") == dim) & (~pl.col("insufficient")))
    if table.is_empty():
        return []
    table = table.filter(pl.col(metric_col).is_not_null())
    if table.is_empty():
        return []
    if signal_spec.method == "threshold":
        filtered = table.filter(pl.col(metric_col) >= signal_spec.threshold)
    else:
        filtered = table.sort(metric_col, descending=True).head(signal_spec.topk)
    if filtered.is_empty():
        return []
    return filtered.get_column("bin").to_list()


def summarise_profiles(profiles: pl.DataFrame) -> Dict[str, list[Dict[str, object]]]:
    """Return a serialisable representation of the computed profiles."""

    _require_polars()
    if profiles.is_empty():
        return {}

    serialised: Dict[str, list[Dict[str, object]]] = {}
    dims = profiles.get_column("dim").unique().to_list()
    for dim in dims:
        table = profiles.filter(pl.col("dim") == dim)
        serialised[str(dim)] = table.to_dicts()
    return serialised
