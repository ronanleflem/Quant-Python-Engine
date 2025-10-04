"""Smoke tests for time seasonality filters."""
from __future__ import annotations

import pandas as pd

from quant_engine.filters.time_seasonality import (
    day_of_month_filter,
    day_of_week_filter,
    intraday_time_filter,
    month_of_year_filter,
    session_time_filter,
)


def _make_df(start: str, periods: int, freq: str = "min") -> pd.DataFrame:
    index = pd.date_range(start=start, periods=periods, freq=freq, tz="UTC")
    return pd.DataFrame(index=index)


def test_session_time_filter_london_window() -> None:
    df = _make_df("2025-01-01", periods=3 * 24 * 60)
    mask = session_time_filter(df, session="london")

    assert isinstance(mask, pd.Series)
    assert mask.index.equals(df.index)
    assert mask.dtype == bool

    selected_hours = df.index[mask]
    if not selected_hours.empty:
        hours = selected_hours.to_series().dt.hour
        assert ((hours >= 7) & (hours < 15)).all()

    outside_hours = df.index[~mask]
    if not outside_hours.empty:
        hours = outside_hours.to_series().dt.hour
        assert ((hours < 7) | (hours >= 15)).all()


def test_day_of_week_filter_blocks_weekend() -> None:
    df = _make_df("2025-01-01", periods=10 * 24 * 60)
    mask = day_of_week_filter(df, blocked_days=[5, 6])

    weekdays = df.index.to_series().dt.dayofweek
    assert mask.dtype == bool
    assert mask.index.equals(df.index)
    assert mask[weekdays < 5].all()
    assert (~mask)[weekdays >= 5].all()


def test_day_of_month_filter_first_three_days() -> None:
    df = _make_df("2025-01-01", periods=10 * 24 * 60)
    mask = day_of_month_filter(df, mode="first", n=3)

    days = df.index.to_series().dt.day
    assert mask.dtype == bool
    assert mask.index.equals(df.index)
    assert mask[days <= 3].all()
    assert (~mask)[days > 3].all()


def test_month_of_year_filter_only_january() -> None:
    df = _make_df("2024-12-30", periods=7 * 24, freq="h")
    mask = month_of_year_filter(df, allowed_months=[1])

    months = df.index.to_series().dt.month
    assert mask.dtype == bool
    assert mask.index.equals(df.index)
    assert mask[months == 1].all()
    assert (~mask)[months != 1].all()


def test_intraday_time_filter_window() -> None:
    df = _make_df("2025-01-01", periods=2 * 24 * 60)
    mask = intraday_time_filter(df, start="09:00", end="12:00")

    minutes_of_day = df.index.to_series().dt.hour * 60 + df.index.to_series().dt.minute
    assert mask.dtype == bool
    assert mask.index.equals(df.index)
    assert mask[(minutes_of_day >= 9 * 60) & (minutes_of_day < 12 * 60)].all()
    assert (~mask)[(minutes_of_day < 9 * 60) | (minutes_of_day >= 12 * 60)].all()
