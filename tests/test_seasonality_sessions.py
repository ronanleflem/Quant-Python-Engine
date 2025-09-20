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
    sessions = enriched.get_column("session").to_list()
    assert sessions[0] == "Asia"
    assert sessions[2] == "EU_US_overlap"
    month_start = enriched.get_column("is_month_start").to_list()
    month_end = enriched.get_column("is_month_end").to_list()
    assert month_start[0] is True
    assert month_end[2] is True


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
        }
    )
    profile_spec = schemas.SeasonalityProfileSpec(
        by_hour=False,
        by_dow=False,
        by_month=False,
        by_session=True,
        by_month_start=True,
        by_month_end=True,
        measure="direction",
        ret_horizon=1,
        min_samples_bin=1,
    )
    features = compute.prepare_features(df, profile_spec)
    profiles_df = compute.compute_profiles(features, profile_spec)
    dims = set(profiles_df.get_column("dim").to_list())
    assert {"session", "is_month_start", "is_month_end"}.issubset(dims)
    for col in CONDITIONAL_METRIC_NAMES:
        assert col in profiles_df.columns


def test_signal_spec_accepts_new_dims() -> None:
    spec = schemas.SeasonalitySignalSpec(dims=["session", "is_month_end"])
    assert spec.dims == ["session", "is_month_end"]
