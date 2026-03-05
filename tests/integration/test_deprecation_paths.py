from __future__ import annotations

import pandas as pd
import pytest

from quant_engine.stats import conditions as stats_conditions
from quant_engine.stats import events as stats_events


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=6, freq="h", tz="UTC"),
            "open": [100, 101, 102, 101, 103, 104],
            "high": [101, 102, 103, 104, 105, 106],
            "low": [99, 100, 101, 100, 102, 103],
            "close": [101, 102, 101, 103, 104, 103],
            "session_id": ["asia", "asia", "london", "london", "newyork", "newyork"],
        }
    )


def test_stats_event_deprecation_warning_contains_target_version_and_date() -> None:
    df = _sample_df()

    with pytest.warns(DeprecationWarning, match=r"v0\.15\.0.*2026-04-30"):
        _ = stats_events.gap_up(df)


def test_stats_condition_deprecation_warning_contains_target_version_and_date() -> None:
    df = _sample_df()

    with pytest.warns(DeprecationWarning, match=r"v0\.15\.0.*2026-04-30"):
        _ = stats_conditions.day_of_week(df)
