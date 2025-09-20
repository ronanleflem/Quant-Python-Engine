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

    active_bins: Dict[str, set[Any]] = field(default_factory=dict)
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
) -> tuple[set[Any], float | None]:
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
    bins = set(filtered.get_column("bin").to_list())
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
    active: Dict[str, set[Any]] = {}
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


def compare_profiles(
    df_a: "pl.DataFrame", df_b: "pl.DataFrame", dim: str
) -> tuple["pl.DataFrame", float | None]:
    """Compare seasonality lifts between two symbols for a given dimension."""

    _require_polars()

    if df_a.is_empty() or df_b.is_empty():
        return pl.DataFrame(), None

    def _prepare(table: "pl.DataFrame") -> tuple["pl.DataFrame", str] | None:
        filtered = table.filter(pl.col("dim") == dim)
        if "insufficient" in filtered.columns:
            filtered = filtered.filter(~pl.col("insufficient"))
        if filtered.is_empty():
            return None
        symbols = (
            filtered.get_column("symbol")
            .drop_nulls()
            .unique()
            .cast(pl.Utf8)
            .to_list()
        )
        label = symbols[0] if symbols else "symbol"
        aggregations: list[pl.Expr] = []
        if "lift" in filtered.columns:
            aggregations.append(pl.col("lift").mean().alias("lift"))
        if "baseline" in filtered.columns:
            aggregations.append(pl.col("baseline").mean().alias("baseline"))
        if "n" in filtered.columns:
            aggregations.append(pl.col("n").sum().alias("n"))
        if "p_hat" in filtered.columns:
            aggregations.append(pl.col("p_hat").mean().alias("p_hat"))
        if not aggregations:
            aggregations.append(pl.len().alias("count"))
        prepared = filtered.group_by("bin").agg(aggregations)
        rename_map = {
            col: col if col == "bin" else f"{col}_{label}"
            for col in prepared.columns
        }
        prepared = prepared.rename(rename_map)
        return prepared, label

    prepared_a = _prepare(df_a)
    prepared_b = _prepare(df_b)
    if prepared_a is None or prepared_b is None:
        return pl.DataFrame(), None

    table_a, label_a = prepared_a
    table_b, label_b = prepared_b
    comparison = table_a.join(table_b, on="bin", how="inner")
    if comparison.is_empty():
        return pl.DataFrame(), None

    lift_col_a = f"lift_{label_a}"
    lift_col_b = f"lift_{label_b}"
    if lift_col_a in comparison.columns and lift_col_b in comparison.columns:
        comparison = comparison.with_columns(
            (pl.col(lift_col_a) - pl.col(lift_col_b)).alias("lift_diff")
        )
        valid = comparison.filter(
            pl.col(lift_col_a).is_not_null() & pl.col(lift_col_b).is_not_null()
        )
        if valid.is_empty():
            corr: float | None = None
        else:
            corr_value = valid.select(
                pl.corr(pl.col(lift_col_a), pl.col(lift_col_b))
            ).to_series()
            corr = float(corr_value[0]) if corr_value.len() else None
    else:
        corr = None

    return comparison.sort("bin"), corr
