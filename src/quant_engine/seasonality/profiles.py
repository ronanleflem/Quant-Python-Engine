"""Helpers to work with seasonality profiles."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, MutableMapping, Sequence

try:  # pragma: no cover - optional dependency
    import polars as pl
except ModuleNotFoundError:  # pragma: no cover - used when dependency missing
    pl = None  # type: ignore


def _require_polars() -> None:
    if pl is None:  # pragma: no cover - exercised when dependency missing
        raise RuntimeError("polars is required for seasonality computations")

@dataclass
class SeasonalityRules:
    """Container holding the active bins and metadata for a signal."""

    active_bins: Dict[str, set[int]] = field(default_factory=dict)
    combine: str = "and"
    metadata: MutableMapping[str, Any] = field(default_factory=dict)

    def to_serialisable(self) -> Dict[str, Any]:
        """Return a JSON-friendly representation of the rules."""

        return {
            "active_bins": {dim: sorted(bins) for dim, bins in self.active_bins.items()},
            "combine": self.combine,
            "metadata": dict(self.metadata),
        }


def _score_column(measure: str, table: pl.DataFrame) -> str | None:
    """Return the name of the score column used for ranking bins."""

    preferred = ["p_hat", "lift"] if measure == "direction" else ["ret_mean", "lift"]
    for col in preferred:
        if col in table.columns:
            return col
    return None


def _normalise_threshold(threshold: float | None, measure: str) -> float | None:
    """Normalise the threshold according to the measure semantics."""

    if threshold is None:
        return None
    if measure != "return":
        return float(threshold)
    value = float(threshold)
    if abs(value) >= 0.01:
        # Treat large magnitudes as basis points for convenience.
        return value / 10000.0
    return value


def _select_bins_for_dimension(
    table: pl.DataFrame,
    *,
    method: str,
    threshold: float | None,
    topk: int,
    score_col: str,
) -> tuple[set[int], float | None]:
    """Return the selected bins for a single dimension."""

    table = table.filter(pl.col(score_col).is_not_null())
    if table.is_empty():
        return set(), None
    if method == "threshold":
        thr = threshold if threshold is not None else float("-inf")
        filtered = table.filter(pl.col(score_col) >= thr)
        cutoff = thr
    else:
        k = max(int(topk), 0)
        if k == 0:
            return set(), None
        filtered = table.sort(score_col, descending=True).head(k)
        cutoff = (
            float(filtered.get_column(score_col).min()) if not filtered.is_empty() else None
        )
    if filtered.is_empty():
        return set(), cutoff
    bins = set(int(b) for b in filtered.get_column("bin").to_list())
    return bins, cutoff


def select_bins(
    profiles: pl.DataFrame,
    *,
    method: str = "threshold",
    threshold: float | None = 0.54,
    topk: int = 3,
    dims: Sequence[str] | None = None,
    measure: str = "direction",
    combine: str = "and",
) -> SeasonalityRules:
    """Convert profile tables into tradable rules."""

    _require_polars()
    if profiles.is_empty():
        return SeasonalityRules(metadata={"thresholds": {}, "counts": {}})

    requested_dims = list(dims) if dims is not None else []
    if not requested_dims:
        requested_dims = profiles.get_column("dim").unique().cast(pl.Utf8).to_list()

    normalised_threshold = _normalise_threshold(threshold, measure)
    active: Dict[str, set[int]] = {}
    thresholds_meta: Dict[str, float | None] = {}
    counts_meta: Dict[str, int] = {}

    for dim in requested_dims:
        table = profiles.filter((pl.col("dim") == dim) & (~pl.col("insufficient")))
        if table.is_empty():
            active[dim] = set()
            thresholds_meta[dim] = normalised_threshold
            counts_meta[dim] = 0
            continue
        score_col = _score_column(measure, table)
        if score_col is None:
            active[dim] = set()
            thresholds_meta[dim] = normalised_threshold
            counts_meta[dim] = 0
            continue
        bins, cutoff = _select_bins_for_dimension(
            table,
            method=method,
            threshold=normalised_threshold,
            topk=topk,
            score_col=score_col,
        )
        active[dim] = bins
        thresholds_meta[dim] = cutoff if cutoff is not None else normalised_threshold
        counts_meta[dim] = len(bins)

    metadata: Dict[str, Any] = {
        "thresholds": thresholds_meta,
        "counts": counts_meta,
        "method": method,
        "measure": measure,
    }
    return SeasonalityRules(active_bins=active, combine=combine, metadata=metadata)


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
