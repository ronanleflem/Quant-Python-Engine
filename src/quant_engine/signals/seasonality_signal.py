"""Helpers to convert seasonality profiles into trading signals."""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence

try:  # pragma: no cover - optional dependency
    import polars as pl
except ModuleNotFoundError:  # pragma: no cover - used when dependency missing
    pl = None  # type: ignore

from ..seasonality.profiles import SeasonalityRules


def _require_polars() -> None:
    if pl is None:  # pragma: no cover - exercised when dependency missing
        raise RuntimeError("polars is required for seasonality signals")


DIMENSION_TO_COLUMN = {
    "hour": "hour",
    "dow": "dow",
    "month": "month",
    "month_of_year": "month_of_year",
    "session": "session",
    "is_month_start": "is_month_start",
    "is_month_end": "is_month_end",
    "week_in_month": "week_in_month",
    "day_in_month": "day_in_month",
    "quarter": "quarter",
    "last_1": "last_1",
    "last_2": "last_2",
    "last_3": "last_3",
    "last_4": "last_4",
    "last_5": "last_5",
    "is_january": "is_january",
    "is_february": "is_february",
    "is_march": "is_march",
    "is_april": "is_april",
    "is_may": "is_may",
    "is_june": "is_june",
    "is_july": "is_july",
    "is_august": "is_august",
    "is_september": "is_september",
    "is_october": "is_october",
    "is_november": "is_november",
    "is_december": "is_december",
}


def _combine_masks(
    masks: Sequence[pl.Expr],
    method: str,
    *,
    sum_threshold: int = 1,
) -> pl.Expr:
    """Combine the per-dimension activation masks according to ``method``."""

    _require_polars()
    if not masks:
        return pl.lit(False)
    if method == "and":
        return pl.all_horizontal(masks)
    if method == "or":
        return pl.any_horizontal(masks)
    summed = pl.sum_horizontal([expr.cast(pl.Int64) for expr in masks])
    return summed.ge(sum_threshold)


def make_seasonality_signals(dataset: pl.DataFrame, rules: SeasonalityRules) -> pl.DataFrame:
    """Return the dataset with boolean long/short signal columns."""

    _require_polars()
    df = dataset
    masks: list[pl.Expr] = []
    for dim, bins in rules.active_bins.items():
        column = DIMENSION_TO_COLUMN.get(dim)
        if column is None or not bins:
            continue
        masks.append(pl.col(column).is_in(list(bins)))
    if masks:
        sum_threshold = int(rules.metadata.get("sum_threshold", 1)) if rules.metadata else 1
        long_expr = _combine_masks(masks, rules.combine, sum_threshold=sum_threshold)
    else:
        long_expr = pl.lit(False)
    df = df.with_columns(
        long_expr.alias("long"),
        pl.lit(False).alias("short"),
    )
    return df
