from __future__ import annotations

import pandas as pd
import pytest

from quant_engine.market_intelligence import conditions as mi_conditions
from quant_engine.market_intelligence import events as mi_events
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


def test_stats_event_wrapper_matches_mi_and_warns() -> None:
    df = _sample_df()

    expected = mi_events.gap_up(df)
    with pytest.warns(DeprecationWarning):
        observed = stats_events.gap_up(df)

    pd.testing.assert_series_equal(observed, expected)


def test_stats_condition_wrapper_matches_mi_and_warns() -> None:
    df = _sample_df()

    expected = mi_conditions.day_of_week(df)
    with pytest.warns(DeprecationWarning):
        observed = stats_conditions.day_of_week(df)

    pd.testing.assert_series_equal(observed, expected)
