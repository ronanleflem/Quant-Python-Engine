"""Timezone-aware trading session utilities."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timezone
from typing import Literal, Mapping, Optional
from zoneinfo import ZoneInfo

import pandas as pd

SessionName = Literal["asia", "london", "newyork"]


@dataclass(frozen=True)
class MarketSessionSpec:
    """Describe a market session in its local timezone."""

    tz: str
    start: time
    end: time


MARKET_SESSIONS: Mapping[SessionName, MarketSessionSpec] = {
    "asia": MarketSessionSpec("Asia/Tokyo", time(9, 0), time(17, 0)),
    "london": MarketSessionSpec("Europe/London", time(8, 0), time(16, 30)),
    "newyork": MarketSessionSpec("America/New_York", time(9, 30), time(16, 0)),
}


def _ensure_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _seconds_since_midnight(value: time) -> int:
    return value.hour * 3600 + value.minute * 60 + value.second


def _is_in_window(ts_utc: datetime, spec: MarketSessionSpec, tz: Optional[str] = None) -> bool:
    tz_name = tz or spec.tz
    local = ts_utc.astimezone(ZoneInfo(tz_name))
    local_seconds = local.hour * 3600 + local.minute * 60 + local.second
    start_seconds = _seconds_since_midnight(spec.start)
    end_seconds = _seconds_since_midnight(spec.end)
    if start_seconds <= end_seconds:
        return start_seconds <= local_seconds < end_seconds
    return local_seconds >= start_seconds or local_seconds < end_seconds


def assign_session_label(ts: datetime) -> str:
    """Return the trading session bucket for a timestamp (timezone-aware)."""

    ts_utc = _ensure_utc(ts)
    in_asia = _is_in_window(ts_utc, MARKET_SESSIONS["asia"])
    in_london = _is_in_window(ts_utc, MARKET_SESSIONS["london"])
    in_newyork = _is_in_window(ts_utc, MARKET_SESSIONS["newyork"])
    if in_london and in_newyork:
        return "EU_US_overlap"
    if in_london:
        return "Europe"
    if in_newyork:
        return "US"
    if in_asia:
        return "Asia"
    return "Other"


def session_time_mask(
    index: pd.DatetimeIndex,
    session: SessionName,
    tz: Optional[str] = None,
) -> pd.Series:
    """Return a boolean mask for timestamps falling within a market session."""

    if not isinstance(index, pd.DatetimeIndex):  # pragma: no cover - defensive
        raise TypeError("index must be a DatetimeIndex")
    idx = index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    spec = MARKET_SESSIONS[session]
    tz_name = tz or spec.tz
    local_idx = idx.tz_convert(tz_name)
    local_seconds = local_idx.hour * 3600 + local_idx.minute * 60 + local_idx.second
    start_seconds = _seconds_since_midnight(spec.start)
    end_seconds = _seconds_since_midnight(spec.end)
    if start_seconds <= end_seconds:
        mask = (local_seconds >= start_seconds) & (local_seconds < end_seconds)
    else:
        mask = (local_seconds >= start_seconds) | (local_seconds < end_seconds)
    return pd.Series(mask, index=index)


__all__ = ["MARKET_SESSIONS", "SessionName", "assign_session_label", "session_time_mask"]
