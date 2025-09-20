from __future__ import annotations

from datetime import datetime

import pytest

from quant_engine.api import schemas
from quant_engine.seasonality import compute
from quant_engine.seasonality.compute import CONDITIONAL_METRIC_NAMES


def test_assign_session_buckets() -> None:
    assert compute.assign_session(datetime(2025, 1, 1, 0, 0)) == "Asia"
    assert compute.assign_session(datetime(2025, 1, 1, 8, 0)) == "Europe"
    assert compute.assign_session(datetime(2025, 1, 1, 14, 0)) == "EU_US_overlap"
    assert compute.assign_session(datetime(2025, 1, 1, 18, 0)) == "US"
    assert compute.assign_session(datetime(2025, 1, 1, 23, 0)) == "Other"


def test_add_time_bins_includes_sessions_and_month_edges() -> None:
    pl = pytest.importorskip("polars")
    df = pl.DataFrame(
        {
            "timestamp": [
                datetime(2025, 1, 1, 0, 0),
                datetime(2025, 1, 15, 8, 0),
                datetime(2025, 1, 31, 14, 0),
                datetime(2025, 2, 1, 18, 0),
            ],
            "symbol": ["TEST"] * 4,
            "close": [1.0, 1.1, 1.2, 1.3],
        }
    )
    enriched = compute.add_time_bins(df)
    assert "session" in enriched.columns
    assert "is_month_start" in enriched.columns
    assert "is_month_end" in enriched.columns
    assert {"day_in_month", "week_in_month", "month_of_year", "quarter"}.issubset(
        set(enriched.columns)
    )
    sessions = enriched.get_column("session").to_list()
    assert sessions[0] == "Asia"
    assert sessions[2] == "EU_US_overlap"
    month_start = enriched.get_column("is_month_start").to_list()
    month_end = enriched.get_column("is_month_end").to_list()
    assert month_start[0] is True
    assert month_end[2] is True
    day_in_month = enriched.get_column("day_in_month").to_list()
    assert day_in_month[0] == 1
    assert day_in_month[2] == 31
    week_in_month = enriched.get_column("week_in_month").to_list()
    assert week_in_month[0] == 1
    assert week_in_month[2] == 5
    quarters = enriched.get_column("quarter").to_list()
    assert quarters[0] == 1


def test_add_time_bins_adds_special_flags() -> None:
    pl = pytest.importorskip("polars")
    df = pl.DataFrame(
        {
            "timestamp": [
                datetime(2025, 1, 16, 12, 0),
                datetime(2025, 1, 17, 13, 0),
                datetime(2025, 1, 20, 20, 0),
                datetime(2025, 1, 21, 9, 0),
            ],
            "symbol": ["TEST"] * 4,
            "close": [1.0, 1.1, 1.2, 1.3],
            "roll_id": [1, 1, 2, 2],
        }
    )
    enriched = compute.add_time_bins(df)
    assert {"is_news_hour", "is_third_friday", "is_rollover_day"}.issubset(
        set(enriched.columns)
    )
    assert set(compute.LAST_DAY_COLUMNS).issubset(set(enriched.columns))
    assert set(compute.MONTH_FLAG_COLUMNS).issubset(set(enriched.columns))
    assert enriched.get_column("is_news_hour").to_list() == [False, True, True, False]
    assert enriched.get_column("is_third_friday").to_list() == [False, True, False, False]
    assert enriched.get_column("is_rollover_day").to_list() == [False, False, True, False]
    last_flags = {
        name: enriched.get_column(name).to_list() for name in compute.LAST_DAY_COLUMNS
    }
    assert last_flags["last_1"][2] is True
    assert last_flags["last_2"][2] is False
    month_flags = {
        name: enriched.get_column(name).sum() for name in compute.MONTH_FLAG_COLUMNS
    }
    assert month_flags["is_january"] >= 1
    assert month_flags["is_february"] >= 0


def test_compute_profiles_handles_new_dimensions() -> None:
    pl = pytest.importorskip("polars")
    df = pl.DataFrame(
        {
            "timestamp": [
                datetime(2025, 1, 1, 0, 0),
                datetime(2025, 1, 15, 8, 0),
                datetime(2025, 1, 31, 14, 0),
                datetime(2025, 2, 1, 18, 0),
            ],
            "symbol": ["TEST"] * 4,
            "close": [1.0, 1.1, 1.2, 1.3],
            "roll_id": [1, 1, 2, 2],
        }
    )
    profile_spec = schemas.SeasonalityProfileSpec(
        by_hour=False,
        by_dow=False,
        by_month=True,
        by_session=True,
        by_month_start=True,
        by_month_end=True,
        by_news_hour=True,
        by_third_friday=True,
        by_rollover_day=True,
        by_week_in_month=True,
        by_day_in_month=True,
        by_month_last_days=True,
        by_quarter=True,
        by_month_flags=True,
        measure="direction",
        ret_horizon=1,
        min_samples_bin=1,
    )
    features = compute.prepare_features(df, profile_spec)
    profiles_df = compute.compute_profiles(features, profile_spec)
    dims = set(profiles_df.get_column("dim").to_list())
    assert {
        "session",
        "is_month_start",
        "is_month_end",
        "is_news_hour",
        "is_third_friday",
        "is_rollover_day",
        "week_in_month",
        "day_in_month",
        "last_1",
        "quarter",
        "is_january",
        "month",
        "month_of_year",
    }.issubset(dims)
    for col in CONDITIONAL_METRIC_NAMES:
        assert col in profiles_df.columns


def test_signal_spec_accepts_new_dims() -> None:
    spec = schemas.SeasonalitySignalSpec(
        dims=[
            "session",
            "is_month_end",
            "is_news_hour",
            "is_third_friday",
            "week_in_month",
            "quarter",
            "last_1",
            "is_may",
        ]
    )
    assert spec.dims == [
        "session",
        "is_month_end",
        "is_news_hour",
        "is_third_friday",
        "week_in_month",
        "quarter",
        "last_1",
        "is_may",
    ]
