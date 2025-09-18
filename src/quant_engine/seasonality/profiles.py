"""Helpers to work with seasonality profiles."""
from __future__ import annotations

from typing import Dict, Mapping, Sequence

try:  # pragma: no cover - optional dependency
    import polars as pl
except ModuleNotFoundError:  # pragma: no cover - used when dependency missing
    pl = None  # type: ignore


def _require_polars() -> None:
    if pl is None:  # pragma: no cover - exercised when dependency missing
        raise RuntimeError("polars is required for seasonality computations")

from ..api.schemas import SeasonalityProfileSpec, SeasonalitySignalSpec


def select_active_bins(
    profiles: Mapping[str, pl.DataFrame],
    signal_spec: SeasonalitySignalSpec,
    profile_spec: SeasonalityProfileSpec,
) -> Dict[str, Sequence]:
    """Return the active bins for each dimension according to the strategy."""

    active: Dict[str, Sequence] = {}
    _require_polars()
    for dim in signal_spec.dims:
        table = profiles.get(dim)
        if table is None or table.is_empty():
            continue
        selected = _select_bins_for_dimension(table, signal_spec, profile_spec)
        if selected:
            active[dim] = selected
    return active


def _select_bins_for_dimension(
    table: pl.DataFrame,
    signal_spec: SeasonalitySignalSpec,
    profile_spec: SeasonalityProfileSpec,
) -> Sequence:
    _require_polars()
    metric_col = "p_hat" if profile_spec.measure == "direction" else "mean_ret"
    if signal_spec.method == "threshold":
        threshold = signal_spec.threshold
        if profile_spec.measure == "return":
            mask = table[metric_col] >= threshold
        else:
            mask = table[metric_col] >= threshold
        filtered = table.filter(mask)
    else:
        sorted_table = table.sort(metric_col, descending=True)
        filtered = sorted_table.head(signal_spec.topk)
    return filtered.get_column(table.columns[0]).to_list()


def summarise_profiles(profiles: Mapping[str, pl.DataFrame]) -> Dict[str, list[Dict[str, object]]]:
    """Return a serialisable representation of the computed profiles."""

    _require_polars()
    serialised: Dict[str, list[Dict[str, object]]] = {}
    for dim, table in profiles.items():
        serialised[dim] = table.to_dicts()
    return serialised
