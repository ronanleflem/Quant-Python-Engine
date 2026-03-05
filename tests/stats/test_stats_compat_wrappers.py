from __future__ import annotations

from collections.abc import Callable

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


@pytest.mark.parametrize(
    ("name", "call"),
    [
        ("k_consecutive", lambda mod, df: mod.k_consecutive(df, k=2, direction="up")),
        ("shock_atr", lambda mod, df: mod.shock_atr(df, mult=1.5, window=3)),
        ("breakout_hhll", lambda mod, df: mod.breakout_hhll(df, lookback=2, type="hh")),
        ("bullish_candle", lambda mod, df: mod.bullish_candle(df)),
        ("bearish_candle", lambda mod, df: mod.bearish_candle(df)),
        ("bullish_engulfing", lambda mod, df: mod.bullish_engulfing(df)),
        ("bearish_engulfing", lambda mod, df: mod.bearish_engulfing(df)),
        ("bullish_streak", lambda mod, df: mod.bullish_streak(df, k=2)),
        ("bearish_streak", lambda mod, df: mod.bearish_streak(df, k=2)),
        ("gap_up", lambda mod, df: mod.gap_up(df)),
        ("gap_down", lambda mod, df: mod.gap_down(df)),
        ("always_true", lambda mod, df: mod.always_true(df)),
    ],
)
def test_stats_event_wrapper_matches_mi_and_warns(name: str, call: Callable[..., pd.Series]) -> None:
    df = _sample_df()

    expected = call(mi_events, df)
    with pytest.warns(
        DeprecationWarning,
        match=rf"quant_engine\.stats\.events\.{name}.*v0\.15\.0.*2026-04-30.*market_intelligence\.events",
    ):
        observed = call(stats_events, df)

    pd.testing.assert_series_equal(observed, expected)


@pytest.mark.parametrize(
    ("name", "call"),
    [
        ("htf_trend", lambda mod, df: mod.htf_trend(df, tf_multiplier=2, ema_period=2)),
        ("vol_tertile", lambda mod, df: mod.vol_tertile(df, window=3)),
        ("session", lambda mod, df: mod.session(df)),
        ("hour_bin", lambda mod, df: mod.hour_bin(df)),
        ("day_of_week", lambda mod, df: mod.day_of_week(df)),
        ("month_of_year", lambda mod, df: mod.month_of_year(df)),
        ("session_from_ts", lambda mod, df: mod.session_from_ts(df)),
    ],
)
def test_stats_condition_wrapper_matches_mi_and_warns(name: str, call: Callable[..., pd.Series]) -> None:
    df = _sample_df()

    expected = call(mi_conditions, df)
    with pytest.warns(
        DeprecationWarning,
        match=rf"quant_engine\.stats\.conditions\.{name}.*v0\.15\.0.*2026-04-30.*market_intelligence\.conditions",
    ):
        observed = call(stats_conditions, df)

    pd.testing.assert_series_equal(observed, expected)
