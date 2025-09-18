"""Computation helpers for seasonality profiles."""
from __future__ import annotations

from typing import Dict, Iterable

try:  # pragma: no cover - optional dependency
    import polars as pl
except ModuleNotFoundError:  # pragma: no cover - used when dependency missing
    pl = None  # type: ignore

from ..api.schemas import SeasonalityProfileSpec
from ..stats.estimators import freq_with_wilson

# ---------------------------------------------------------------------------
# Feature preparation


def _require_polars() -> None:
    if pl is None:  # pragma: no cover - exercised when dependency missing
        raise RuntimeError("polars is required for seasonality computations")


def prepare_features(dataset: pl.DataFrame, profile: SeasonalityProfileSpec) -> pl.DataFrame:
    """Return a dataframe with forward returns and calendar bins.

    Parameters
    ----------
    dataset:
        Input bars sorted by symbol and timestamp containing ``close`` prices.
    profile:
        Profile specification describing the return horizon and measure.
    """

    _require_polars()

    if dataset.is_empty():
        return dataset.clone()

    df = dataset.sort(["symbol", "timestamp"])
    horizon = max(int(profile.ret_horizon), 1)
    df = df.with_columns(
        pl.col("close").shift(-horizon).over("symbol").alias("close_fwd")
    )
    df = df.with_columns(
        pl.when(pl.col("close").abs() <= 1e-12)
        .then(None)
        .otherwise((pl.col("close_fwd") - pl.col("close")) / pl.col("close"))
        .alias("forward_ret")
    )
    df = df.with_columns(
        pl.col("forward_ret").gt(0).cast(pl.Int64).alias("direction")
    )
    df = df.with_columns(
        pl.col("timestamp").dt.hour().alias("hour"),
        pl.col("timestamp").dt.weekday().alias("dow"),
        pl.col("timestamp").dt.month().alias("month"),
    )
    return df.filter(pl.col("forward_ret").is_not_null())


# ---------------------------------------------------------------------------
# Aggregation helpers


def _aggregate_dimension(
    dataset: pl.DataFrame,
    dim: str,
    profile: SeasonalityProfileSpec,
) -> pl.DataFrame:
    """Aggregate forward returns for a single dimension."""

    _require_polars()

    if dataset.is_empty():
        return pl.DataFrame({dim: [], "n": [], "successes": [], "p_hat": [], "ci_low": [], "ci_high": [], "mean_ret": []})

    rows: list[Dict[str, float | int]] = []
    for group in dataset.partition_by(dim, maintain_order=False):
        bin_value = group.get_column(dim)[0]
        n = group.height
        if n < profile.min_samples_bin:
            continue
        if profile.measure == "direction":
            successes = int(group.get_column("direction").sum())
        else:
            successes = int((group.get_column("forward_ret") > 0).sum())
        mean_val = group.get_column("forward_ret").mean()
        mean_ret = float(mean_val) if mean_val is not None else 0.0
        p_hat, ci_low, ci_high = freq_with_wilson(successes, n)
        rows.append(
            {
                dim: bin_value,
                "n": n,
                "successes": successes,
                "p_hat": p_hat,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "mean_ret": mean_ret,
            }
        )
    if not rows:
        return pl.DataFrame({dim: [], "n": [], "successes": [], "p_hat": [], "ci_low": [], "ci_high": [], "mean_ret": []})
    return pl.DataFrame(rows).sort(dim)


def compute_profiles(dataset: pl.DataFrame, profile: SeasonalityProfileSpec) -> Dict[str, pl.DataFrame]:
    """Compute seasonality profiles for the requested dimensions."""

    result: Dict[str, pl.DataFrame] = {}
    if profile.by_hour:
        result["hour"] = _aggregate_dimension(dataset, "hour", profile)
    if profile.by_dow:
        result["dow"] = _aggregate_dimension(dataset, "dow", profile)
    if profile.by_month:
        result["month"] = _aggregate_dimension(dataset, "month", profile)
    return result


def iter_active_bins(profile: SeasonalityProfileSpec) -> Iterable[str]:
    """Yield the dimensions enabled by the profile specification."""

    if profile.by_hour:
        yield "hour"
    if profile.by_dow:
        yield "dow"
    if profile.by_month:
        yield "month"
