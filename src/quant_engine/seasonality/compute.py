"""Computation helpers for seasonality profiles."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

try:  # pragma: no cover - optional dependency
    import polars as pl
except ModuleNotFoundError:  # pragma: no cover - used when dependency missing
    pl = None  # type: ignore

from ..api.schemas import SeasonalityProfileSpec
from ..stats.estimators import freq_with_wilson


# ---------------------------------------------------------------------------
# Feature preparation helpers
# ---------------------------------------------------------------------------


def _require_polars() -> None:
    if pl is None:  # pragma: no cover - exercised when dependency missing
        raise RuntimeError("polars is required for seasonality computations")


def add_time_bins(df: pl.DataFrame) -> pl.DataFrame:
    """Augment the dataset with hour/day-of-week/month calendar bins."""

    _require_polars()
    if df.is_empty():
        return df.clone()
    return df.with_columns(
        pl.col("timestamp").dt.hour().alias("hour"),
        pl.col("timestamp").dt.weekday().alias("dow"),
        pl.col("timestamp").dt.month().alias("month"),
    )


def outcome_return(df: pl.DataFrame, horizon: int = 1) -> pl.DataFrame:
    """Compute forward returns over the requested horizon."""

    _require_polars()
    if df.is_empty():
        return df.clone()
    horizon = max(int(horizon), 1)
    return_col = f"return_h{horizon}"
    fwd_col = f"close_fwd_h{horizon}"
    eps = 1e-12
    df = df.with_columns(
        pl.col("close").shift(-horizon).over("symbol").alias(fwd_col)
    )
    df = df.with_columns(
        pl.when(
            pl.col("close").is_null()
            | pl.col(fwd_col).is_null()
            | (pl.col("close").abs() <= eps)
        )
        .then(None)
        .otherwise(pl.col(fwd_col) / pl.col("close") - 1.0)
        .alias(return_col)
    )
    df = df.drop(fwd_col)
    # expose the canonical forward return column expected elsewhere
    df = df.with_columns(pl.col(return_col).alias("forward_ret"))
    return df


def outcome_direction(df: pl.DataFrame, horizon: int = 1) -> pl.DataFrame:
    """Compute the up/down direction over the requested horizon."""

    _require_polars()
    if df.is_empty():
        return df.clone()
    horizon = max(int(horizon), 1)
    return_col = f"return_h{horizon}"
    direction_col = f"direction_h{horizon}"
    if return_col not in df.columns:
        df = outcome_return(df, horizon)
    df = df.with_columns(
        pl.when(pl.col(return_col).is_null())
        .then(None)
        .otherwise(pl.col(return_col) > 0)
        .alias(direction_col)
    )
    df = df.with_columns(pl.col(direction_col).cast(pl.Int64).alias("direction"))
    return df


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def _empty_profile_frame(
    df: pl.DataFrame, group_cols: Sequence[str], measure: str
) -> pl.DataFrame:
    """Return an empty dataframe with the expected schema for aggregations."""

    _require_polars()
    schema: list[tuple[str, pl.PolarsDataType]] = []
    for col in group_cols:
        dtype = df.schema.get(col, pl.Int64)
        schema.append((col, dtype))
    schema.append(("n", pl.Int64))
    if measure == "direction":
        schema.extend(
            [
                ("successes", pl.Int64),
                ("p_hat", pl.Float64),
                ("ci_low", pl.Float64),
                ("ci_high", pl.Float64),
            ]
        )
    else:
        schema.extend(
            [
                ("ret_mean", pl.Float64),
                ("ret_median", pl.Float64),
                ("ret_std", pl.Float64),
            ]
        )
    schema.extend(
        [
            ("baseline", pl.Float64),
            ("lift", pl.Float64),
            ("insufficient", pl.Boolean),
        ]
    )
    return pl.DataFrame(schema=schema)


def profile_direction(
    df: pl.DataFrame,
    group_cols: Sequence[str],
    horizon: int,
    min_samples: int,
) -> pl.DataFrame:
    """Aggregate directional hit-rates with Wilson intervals."""

    _require_polars()
    if not group_cols:
        raise ValueError("group_cols must not be empty")
    direction_col = f"direction_h{max(int(horizon), 1)}"
    if direction_col not in df.columns:
        df = outcome_direction(df, horizon)
    filtered = df.filter(pl.col(direction_col).is_not_null())
    if filtered.is_empty():
        return _empty_profile_frame(df, group_cols, "direction")

    grouped = filtered.group_by(group_cols).agg(
        pl.len().alias("n"),
        pl.col(direction_col).cast(pl.Int64).sum().alias("successes"),
    )
    if grouped.is_empty():
        return _empty_profile_frame(df, group_cols, "direction")

    baseline = filtered.group_by("symbol").agg(
        pl.col(direction_col).cast(pl.Float64).mean().alias("baseline")
    )

    grouped = grouped.join(baseline, on="symbol", how="left")
    grouped = grouped.with_columns(
        pl.struct(["successes", "n"]).map_elements(
            lambda s: freq_with_wilson(int(s["successes"]), int(s["n"]))
        ).alias("wilson")
    )
    grouped = grouped.with_columns(
        pl.col("wilson").struct.field("field_0").alias("p_hat"),
        pl.col("wilson").struct.field("field_1").alias("ci_low"),
        pl.col("wilson").struct.field("field_2").alias("ci_high"),
    ).drop("wilson")
    grouped = grouped.with_columns((pl.col("n") < min_samples).alias("insufficient"))
    grouped = grouped.with_columns(
        pl.when(pl.col("insufficient"))
        .then(None)
        .otherwise(pl.col("p_hat"))
        .alias("p_hat"),
        pl.when(pl.col("insufficient"))
        .then(None)
        .otherwise(pl.col("ci_low"))
        .alias("ci_low"),
        pl.when(pl.col("insufficient"))
        .then(None)
        .otherwise(pl.col("ci_high"))
        .alias("ci_high"),
    )
    grouped = grouped.with_columns(
        pl.when(pl.col("insufficient"))
        .then(None)
        .otherwise(pl.col("p_hat") - pl.col("baseline"))
        .alias("lift")
    )
    return grouped.sort(group_cols)


def profile_return(
    df: pl.DataFrame,
    group_cols: Sequence[str],
    horizon: int,
    min_samples: int,
) -> pl.DataFrame:
    """Aggregate forward returns by the requested grouping."""

    _require_polars()
    if not group_cols:
        raise ValueError("group_cols must not be empty")
    return_col = f"return_h{max(int(horizon), 1)}"
    if return_col not in df.columns:
        df = outcome_return(df, horizon)
    filtered = df.filter(pl.col(return_col).is_not_null())
    if filtered.is_empty():
        return _empty_profile_frame(df, group_cols, "return")

    grouped = filtered.group_by(group_cols).agg(
        pl.len().alias("n"),
        pl.col(return_col).mean().alias("ret_mean"),
        pl.col(return_col).median().alias("ret_median"),
        pl.col(return_col).std().alias("ret_std"),
    )
    if grouped.is_empty():
        return _empty_profile_frame(df, group_cols, "return")

    baseline = filtered.group_by("symbol").agg(
        pl.col(return_col).mean().alias("baseline")
    )
    grouped = grouped.join(baseline, on="symbol", how="left")
    grouped = grouped.with_columns((pl.col("n") < min_samples).alias("insufficient"))
    grouped = grouped.with_columns(
        pl.when(pl.col("insufficient"))
        .then(None)
        .otherwise(pl.col("ret_mean"))
        .alias("ret_mean"),
        pl.when(pl.col("insufficient"))
        .then(None)
        .otherwise(pl.col("ret_median"))
        .alias("ret_median"),
        pl.when(pl.col("insufficient"))
        .then(None)
        .otherwise(pl.col("ret_std"))
        .alias("ret_std"),
    )
    grouped = grouped.with_columns(
        pl.when(pl.col("insufficient"))
        .then(None)
        .otherwise(pl.col("ret_mean") - pl.col("baseline"))
        .alias("lift")
    )
    return grouped.sort(group_cols)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def prepare_features(dataset: pl.DataFrame, profile: SeasonalityProfileSpec) -> pl.DataFrame:
    """Return the feature set with forward returns and time bins."""

    _require_polars()
    if dataset.is_empty():
        return dataset.clone()

    df = dataset.sort(["symbol", "timestamp"])
    horizon = max(int(profile.ret_horizon), 1)
    df = add_time_bins(df)
    df = outcome_return(df, horizon)
    df = outcome_direction(df, horizon)
    return df.filter(pl.col(f"return_h{horizon}").is_not_null())


def _empty_profiles_table(
    df: pl.DataFrame, profile: SeasonalityProfileSpec
) -> pl.DataFrame:
    """Return an empty table with the final schema for the requested measure."""

    _require_polars()
    symbol_type = df.schema.get("symbol", pl.Utf8)
    base_schema: list[tuple[str, pl.PolarsDataType]] = [
        ("symbol", symbol_type),
        ("dim", pl.Utf8),
        ("bin", pl.Int64),
        ("n", pl.Int64),
    ]
    if profile.measure == "direction":
        base_schema.extend(
            [
                ("successes", pl.Int64),
                ("p_hat", pl.Float64),
                ("ci_low", pl.Float64),
                ("ci_high", pl.Float64),
            ]
        )
    else:
        base_schema.extend(
            [
                ("ret_mean", pl.Float64),
                ("ret_median", pl.Float64),
                ("ret_std", pl.Float64),
            ]
        )
    base_schema.extend(
        [
            ("baseline", pl.Float64),
            ("lift", pl.Float64),
            ("insufficient", pl.Boolean),
            ("timeframe", pl.Utf8),
            ("period_start", pl.Datetime),
            ("period_end", pl.Datetime),
            ("horizon", pl.Int64),
        ]
    )
    return pl.DataFrame(schema=base_schema)


def compute_profiles(
    dataset: pl.DataFrame,
    profile: SeasonalityProfileSpec,
    *,
    timeframe: str | None = None,
    period_start: datetime | None = None,
    period_end: datetime | None = None,
    artifacts_out_dir: str | None = None,
) -> pl.DataFrame:
    """Compute seasonality profiles and optionally persist them to parquet."""

    _require_polars()
    horizon = max(int(profile.ret_horizon), 1)

    if period_start is None and "timestamp" in dataset.columns and not dataset.is_empty():
        period_start = dataset.get_column("timestamp").min()
    if period_end is None and "timestamp" in dataset.columns and not dataset.is_empty():
        period_end = dataset.get_column("timestamp").max()

    dims: list[str] = []
    if profile.by_hour:
        dims.append("hour")
    if profile.by_dow:
        dims.append("dow")
    if profile.by_month:
        dims.append("month")

    tables: list[pl.DataFrame] = []
    for dim in dims:
        group_cols: list[str] = ["symbol", dim]
        if profile.measure == "direction":
            table = profile_direction(dataset, group_cols, horizon, profile.min_samples_bin)
        else:
            table = profile_return(dataset, group_cols, horizon, profile.min_samples_bin)
        if table.is_empty():
            continue
        table = table.rename({dim: "bin"})
        table = table.with_columns(pl.lit(dim).alias("dim"))
        tables.append(table)

    if tables:
        combined = pl.concat(tables, how="vertical", rechunk=True)
    else:
        combined = _empty_profiles_table(dataset, profile)

    combined = combined.with_columns(
        pl.lit(timeframe or "").alias("timeframe"),
        pl.lit(period_start, dtype=pl.Datetime).alias("period_start"),
        pl.lit(period_end, dtype=pl.Datetime).alias("period_end"),
        pl.lit(horizon, dtype=pl.Int64).alias("horizon"),
    )

    if profile.measure == "direction":
        ordered_cols = [
            "symbol",
            "timeframe",
            "dim",
            "bin",
            "n",
            "successes",
            "p_hat",
            "ci_low",
            "ci_high",
            "baseline",
            "lift",
            "insufficient",
            "period_start",
            "period_end",
            "horizon",
        ]
    else:
        ordered_cols = [
            "symbol",
            "timeframe",
            "dim",
            "bin",
            "n",
            "ret_mean",
            "ret_median",
            "ret_std",
            "baseline",
            "lift",
            "insufficient",
            "period_start",
            "period_end",
            "horizon",
        ]
    combined = combined.select(ordered_cols)

    if artifacts_out_dir:
        out_path = Path(artifacts_out_dir) / "seasonality_profiles.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        combined.write_parquet(out_path)

    return combined


def iter_active_bins(profile: SeasonalityProfileSpec) -> Iterable[str]:
    """Yield the dimensions enabled by the profile specification."""

    if profile.by_hour:
        yield "hour"
    if profile.by_dow:
        yield "dow"
    if profile.by_month:
        yield "month"
