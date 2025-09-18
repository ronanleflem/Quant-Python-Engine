"""Helpers to convert seasonality profiles into trading signals."""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence

try:  # pragma: no cover - optional dependency
    import polars as pl
except ModuleNotFoundError:  # pragma: no cover - used when dependency missing
    pl = None  # type: ignore


def _require_polars() -> None:
    if pl is None:  # pragma: no cover - exercised when dependency missing
        raise RuntimeError("polars is required for seasonality signals")


DIMENSION_TO_COLUMN = {
    "hour": "hour",
    "dow": "dow",
    "month": "month",
}


def _combine_masks(masks: Sequence[pl.Expr], method: str) -> pl.Expr:
    _require_polars()
    if not masks:
        return pl.lit(False)
    if method == "and":
        return pl.all_horizontal(masks)
    if method == "or":
        return pl.any_horizontal(masks)
    # sum -> at least one match
    summed = pl.sum_horizontal([expr.cast(pl.Int64) for expr in masks])
    return summed.gt(0)


def build_seasonality_signal(
    dataset: pl.DataFrame,
    active_bins: Mapping[str, Iterable],
    combine: str = "and",
) -> pl.DataFrame:
    """Return the dataset with long/short columns for the seasonality signal."""

    df = dataset
    _require_polars()
    masks: list[pl.Expr] = []
    for dim, bins in active_bins.items():
        column = DIMENSION_TO_COLUMN.get(dim)
        if column is None or not bins:
            continue
        masks.append(pl.col(column).is_in(list(bins)))
    if masks:
        long_expr = _combine_masks(masks, combine).cast(pl.Int64)
    else:
        long_expr = pl.lit(0, dtype=pl.Int64)
    df = df.with_columns(
        long_expr.alias("long"),
        pl.lit(0, dtype=pl.Int64).alias("short"),
    )
    return df
