"""Seasonality and time-based filters.

These filters operate solely on the time index of the input dataframe and
return boolean series aligned with the source index.
"""
from __future__ import annotations

from typing import Iterable, Literal, Optional

import pandas as pd

SessionName = Literal["asia", "london", "newyork"]


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    """Return a timezone-aware DatetimeIndex for computations."""

    if not isinstance(df.index, pd.DatetimeIndex):  # pragma: no cover - defensive
        raise TypeError("DataFrame index must be a DatetimeIndex")
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    return idx
def session_time_filter(
    df: pd.DataFrame,
    session: SessionName = "london",
    tz: str = "UTC",
) -> pd.Series:
    """Return ``True`` when timestamps fall within the configured session."""

    idx = _ensure_datetime_index(df).tz_convert(tz)
    hours = idx.hour
    if session == "asia":
        mask = (hours >= 23) | (hours < 7)
    elif session == "london":
        mask = (hours >= 7) & (hours < 15)
    else:  # "newyork"
        mask = (hours >= 13) & (hours < 21)
    return pd.Series(mask, index=df.index)


def day_of_week_filter(
    df: pd.DataFrame,
    allowed_days: Optional[Iterable[int]] = None,
    blocked_days: Optional[Iterable[int]] = None,
) -> pd.Series:
    """Return ``True`` for timestamps occurring on allowed weekdays."""

    idx = _ensure_datetime_index(df)
    weekdays = pd.Series(idx.dayofweek, index=df.index)
    if allowed_days is not None and blocked_days is not None:
        raise ValueError("Specify either allowed_days or blocked_days, not both")
    if allowed_days is not None:
        return weekdays.isin(list(allowed_days))
    if blocked_days is not None:
        return ~weekdays.isin(list(blocked_days))
    return pd.Series(True, index=df.index)


def day_of_month_filter(
    df: pd.DataFrame,
    mode: Literal["first", "last"] = "first",
    n: int = 1,
) -> pd.Series:
    """Return ``True`` for timestamps within the first/last ``n`` days of a month."""

    if n <= 0:
        raise ValueError("n must be a positive integer")
    idx = _ensure_datetime_index(df)
    days = pd.Series(idx.day, index=df.index)
    if mode == "first":
        return days <= n
    if mode == "last":
        month_lengths = pd.Series(idx.days_in_month, index=df.index)
        return (month_lengths - days + 1) <= n
    raise ValueError("mode must be either 'first' or 'last'")


def month_of_year_filter(
    df: pd.DataFrame,
    allowed_months: Optional[Iterable[int]] = None,
    blocked_months: Optional[Iterable[int]] = None,
) -> pd.Series:
    """Return ``True`` for timestamps occurring in allowed months."""

    idx = _ensure_datetime_index(df)
    months = pd.Series(idx.month, index=df.index)
    if allowed_months is not None and blocked_months is not None:
        raise ValueError("Specify either allowed_months or blocked_months, not both")
    if allowed_months is not None:
        return months.isin(list(allowed_months))
    if blocked_months is not None:
        return ~months.isin(list(blocked_months))
    return pd.Series(True, index=df.index)


def intraday_time_filter(
    df: pd.DataFrame,
    start: str = "09:00",
    end: str = "12:00",
    tz: str = "UTC",
) -> pd.Series:
    """Return ``True`` for timestamps within the half-open intraday window."""

    idx = _ensure_datetime_index(df).tz_convert(tz)
    times = idx.hour * 60 + idx.minute
    try:
        start_hour, start_minute = map(int, start.split(":"))
        end_hour, end_minute = map(int, end.split(":"))
    except ValueError as exc:  # pragma: no cover - defensive validation
        raise ValueError("start and end must be formatted as 'HH:MM'") from exc
    start_total = start_hour * 60 + start_minute
    end_total = end_hour * 60 + end_minute
    if end_total < start_total:
        raise ValueError("end time must be after start time within the same day")
    mask = (times >= start_total) & (times < end_total)
    return pd.Series(mask, index=df.index)
