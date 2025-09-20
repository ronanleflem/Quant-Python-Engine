"""Computation helpers for seasonality profiles."""
from __future__ import annotations

import calendar
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

try:  # pragma: no cover - optional dependency
    import polars as pl
except ModuleNotFoundError:  # pragma: no cover - used when dependency missing
    pl = None  # type: ignore

from ..api.schemas import SeasonalityProfileSpec
from ..stats.estimators import freq_with_wilson


CONDITIONAL_METRIC_NAMES = [
    "run_len_up_mean",
    "run_len_down_mean",
    "n_runs",
    "p_reversal_n",
    "p_reversal_ci_low",
    "p_reversal_ci_high",
    "p_reversal_lift",
    "p_reversal_baseline",
    "amp_mean",
    "amp_std",
    "atr_mean",
    "p_breakout_up",
    "p_breakout_down",
    "p_in_range",
]


def _conditional_metrics_schema() -> list[tuple[str, "pl.PolarsDataType"]]:
    """Return the schema for conditional seasonality metrics."""

    _require_polars()
    dtype_map: dict[str, "pl.PolarsDataType"] = {
        "run_len_up_mean": pl.Float64,
        "run_len_down_mean": pl.Float64,
        "n_runs": pl.Int64,
        "p_reversal_n": pl.Float64,
        "p_reversal_ci_low": pl.Float64,
        "p_reversal_ci_high": pl.Float64,
        "p_reversal_lift": pl.Float64,
        "p_reversal_baseline": pl.Float64,
        "amp_mean": pl.Float64,
        "amp_std": pl.Float64,
        "atr_mean": pl.Float64,
        "p_breakout_up": pl.Float64,
        "p_breakout_down": pl.Float64,
        "p_in_range": pl.Float64,
    }
    return [(name, dtype_map[name]) for name in CONDITIONAL_METRIC_NAMES]


def _ensure_metric_columns(table: "pl.DataFrame") -> "pl.DataFrame":
    """Ensure the conditional metric columns exist in the table."""

    _require_polars()
    for name, dtype in _conditional_metrics_schema():
        if name not in table.columns:
            table = table.with_columns(pl.lit(None).cast(dtype).alias(name))
    return table


# ---------------------------------------------------------------------------
# Feature preparation helpers
# ---------------------------------------------------------------------------


def _require_polars() -> None:
    if pl is None:  # pragma: no cover - exercised when dependency missing
        raise RuntimeError("polars is required for seasonality computations")


def assign_session(ts_utc: datetime | None) -> str | None:
    """Return the trading session bucket for a UTC timestamp."""

    if ts_utc is None:
        return None
    hour = ts_utc.hour
    if 0 <= hour < 7:
        return "Asia"
    if 7 <= hour < 12:
        return "Europe"
    if 12 <= hour < 16:
        return "EU_US_overlap"
    if 16 <= hour < 21:
        return "US"
    return "Other"


def _is_month_end(ts_utc: datetime | None) -> bool | None:
    if ts_utc is None:
        return None
    last_day = calendar.monthrange(ts_utc.year, ts_utc.month)[1]
    return ts_utc.day == last_day


def _is_third_friday(ts_utc: datetime | None) -> bool | None:
    """Return whether the timestamp falls on the third Friday of the month."""

    if ts_utc is None:
        return None
    if ts_utc.weekday() != calendar.FRIDAY:
        return False
    return 15 <= ts_utc.day <= 21


def add_time_bins(df: pl.DataFrame) -> pl.DataFrame:
    """Augment the dataset with hour/day-of-week/month calendar bins."""

    _require_polars()
    if df.is_empty():
        return df.clone()
    df = df.with_columns(
        pl.col("timestamp").dt.hour().alias("hour"),
        pl.col("timestamp").dt.weekday().alias("dow"),
        pl.col("timestamp").dt.month().alias("month"),
        pl.col("timestamp")
        .map_elements(assign_session, return_dtype=pl.Utf8)
        .alias("session"),
        pl.col("timestamp").dt.day().eq(1).alias("is_month_start"),
        pl.col("timestamp")
        .map_elements(_is_month_end, return_dtype=pl.Boolean)
        .alias("is_month_end"),
        pl.col("timestamp")
        .dt.hour()
        .is_in([13, 14, 20])
        .alias("is_news_hour"),
        pl.col("timestamp")
        .map_elements(_is_third_friday, return_dtype=pl.Boolean)
        .alias("is_third_friday"),
    )

    if "roll_id" in df.columns:
        df = df.with_columns(
            pl.col("roll_id")
            .ne(pl.col("roll_id").shift(1).over("symbol"))
            .fill_null(False)
            .alias("is_rollover_day")
        )

    return df


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
    schema.extend(_conditional_metrics_schema())
    schema.extend(
        [
            ("baseline", pl.Float64),
            ("lift", pl.Float64),
            ("insufficient", pl.Boolean),
        ]
    )
    return pl.DataFrame(schema=schema)


def _aggregate_conditional_metrics(
    df: pl.DataFrame,
    group_cols: Sequence[str],
    horizon: int,
) -> pl.DataFrame:
    """Compute run-length, reversal, amplitude and breakout metrics."""

    _require_polars()
    if df.is_empty():
        schema = [
            *[(col, df.schema.get(col, pl.Int64)) for col in group_cols],
            *_conditional_metrics_schema(),
        ]
        return pl.DataFrame(schema=schema)

    base = df.select(group_cols).unique()
    result = base

    # Run-length and reversal metrics rely on run starts
    if {"run_start", "run_length", "reversal_within_h"}.issubset(df.columns):
        run_starts = df.filter(pl.col("run_start"))
        if not run_starts.is_empty():
            runs = run_starts.group_by(group_cols).agg(
                pl.col("run_length_up").mean().alias("run_len_up_mean"),
                pl.col("run_length_down").mean().alias("run_len_down_mean"),
                pl.len().alias("n_runs"),
                pl.col("reversal_within_h").cast(pl.Int64).sum().alias("reversal_successes"),
            )
            baseline = run_starts.group_by("symbol").agg(
                pl.col("reversal_within_h")
                .cast(pl.Float64)
                .mean()
                .alias("p_reversal_baseline")
            )
            runs = runs.join(baseline, on="symbol", how="left")
            runs = runs.with_columns(
                pl.struct(["reversal_successes", "n_runs"]).map_elements(
                    lambda s: freq_with_wilson(int(s["reversal_successes"]), int(s["n_runs"]))
                ).alias("_rev"),
            )
            runs = runs.with_columns(
                pl.col("_rev").struct.field("field_0").alias("p_reversal_n"),
                pl.col("_rev").struct.field("field_1").alias("p_reversal_ci_low"),
                pl.col("_rev").struct.field("field_2").alias("p_reversal_ci_high"),
            ).drop("_rev")
            runs = runs.with_columns(
                pl.when(pl.col("n_runs") == 0)
                .then(None)
                .otherwise(pl.col("p_reversal_n") - pl.col("p_reversal_baseline"))
                .alias("p_reversal_lift"),
            )
            runs = runs.drop("reversal_successes")
            result = result.join(runs, on=group_cols, how="left")
    # Amplitude metrics
    if "amplitude" in df.columns:
        amp_stats = df.group_by(group_cols).agg(
            pl.col("amplitude").mean().alias("amp_mean"),
            pl.col("amplitude").std().alias("amp_std"),
        )
        result = result.join(amp_stats, on=group_cols, how="left")
    # ATR mean if available
    if "atr" in df.columns:
        atr_stats = df.group_by(group_cols).agg(
            pl.col("atr").mean().alias("atr_mean")
        )
        result = result.join(atr_stats, on=group_cols, how="left")
    # Breakout probabilities
    breakout_cols = {"breakout_up", "breakout_down", "in_range"}
    if breakout_cols.issubset(set(df.columns)):
        breakout_stats = df.group_by(group_cols).agg(
            pl.col("breakout_up").cast(pl.Float64).mean().alias("p_breakout_up"),
            pl.col("breakout_down").cast(pl.Float64).mean().alias("p_breakout_down"),
            pl.col("in_range").cast(pl.Float64).mean().alias("p_in_range"),
        )
        result = result.join(breakout_stats, on=group_cols, how="left")

    result = _ensure_metric_columns(result)
    return result


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
    extras = _aggregate_conditional_metrics(filtered, group_cols, horizon)
    grouped = grouped.join(extras, on=group_cols, how="left")
    grouped = _ensure_metric_columns(grouped)
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
    extras = _aggregate_conditional_metrics(filtered, group_cols, horizon)
    grouped = grouped.join(extras, on=group_cols, how="left")
    grouped = _ensure_metric_columns(grouped)
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

    if "direction" in df.columns:
        df = df.with_columns(
            pl.col("direction")
            .shift(1)
            .over("symbol")
            .alias("_prev_direction")
        )
        df = df.with_columns(
            pl.col("direction")
            .ne(pl.col("_prev_direction"))
            .fill_null(True)
            .alias("_run_start"),
        )
        df = df.with_columns(
            pl.col("_run_start")
            .cast(pl.Int64)
            .cumsum()
            .over("symbol")
            .alias("_run_id"),
        )
        df = df.with_columns(
            pl.len().over(["symbol", "_run_id"]).alias("run_length"),
            pl.col("_run_start").alias("run_start"),
        )
        df = df.with_columns(
            pl.when(pl.col("run_start") & (pl.col("direction") == 1))
            .then(pl.col("run_length"))
            .otherwise(None)
            .alias("run_length_up"),
            pl.when(pl.col("run_start") & (pl.col("direction") == 0))
            .then(pl.col("run_length"))
            .otherwise(None)
            .alias("run_length_down"),
            pl.when(pl.col("run_start"))
            .then(pl.col("run_length") <= horizon)
            .otherwise(None)
            .alias("reversal_within_h"),
        )
        df = df.drop("_prev_direction", "_run_start", "_run_id")

    if {"high", "low"}.issubset(df.columns):
        df = df.with_columns((pl.col("high") - pl.col("low")).alias("amplitude"))

    if {"timestamp", "high", "low"}.issubset(df.columns):
        df = df.with_columns(pl.col("timestamp").dt.date().alias("_trade_date"))
        daily = (
            df.group_by(["symbol", "_trade_date"])
            .agg(
                pl.col("high").max().alias("_day_high"),
                pl.col("low").min().alias("_day_low"),
            )
            .sort(["symbol", "_trade_date"])
            .with_columns(
                pl.col("_day_high")
                .shift(1)
                .over("symbol")
                .alias("_prev_day_high"),
                pl.col("_day_low")
                .shift(1)
                .over("symbol")
                .alias("_prev_day_low"),
            )
            .select(["symbol", "_trade_date", "_prev_day_high", "_prev_day_low"])
        )
        df = df.join(daily, on=["symbol", "_trade_date"], how="left")
        df = df.with_columns(
            pl.when(pl.col("_prev_day_high").is_not_null())
            .then(pl.col("high") >= pl.col("_prev_day_high"))
            .otherwise(None)
            .alias("breakout_up"),
            pl.when(pl.col("_prev_day_low").is_not_null())
            .then(pl.col("low") <= pl.col("_prev_day_low"))
            .otherwise(None)
            .alias("breakout_down"),
        )
        df = df.with_columns(
            pl.when(
                pl.col("_prev_day_high").is_not_null()
                & pl.col("_prev_day_low").is_not_null()
            )
            .then(~pl.col("breakout_up") & ~pl.col("breakout_down"))
            .otherwise(None)
            .alias("in_range"),
        )
        df = df.drop("_trade_date", "_prev_day_high", "_prev_day_low")

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
    base_schema.extend(_conditional_metrics_schema())
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
    if profile.by_session:
        dims.append("session")
    if profile.by_month_start:
        dims.append("is_month_start")
    if profile.by_month_end:
        dims.append("is_month_end")
    if profile.by_news_hour:
        dims.append("is_news_hour")
    if profile.by_third_friday:
        dims.append("is_third_friday")
    if profile.by_rollover_day:
        dims.append("is_rollover_day")

    tables: list[pl.DataFrame] = []
    for dim in dims:
        if dim not in dataset.columns:
            continue
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

    metric_cols = list(CONDITIONAL_METRIC_NAMES)

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
            *metric_cols,
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
            *metric_cols,
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
    if profile.by_session:
        yield "session"
    if profile.by_month_start:
        yield "is_month_start"
    if profile.by_month_end:
        yield "is_month_end"
    if profile.by_news_hour:
        yield "is_news_hour"
    if profile.by_third_friday:
        yield "is_third_friday"
    if profile.by_rollover_day:
        yield "is_rollover_day"
