from __future__ import annotations

import pandas as pd

from quant_engine.strategies.dca_benchmark import DcaBenchmarkStrategy


def _ohlc_business_days() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", "2024-04-30", freq="B", tz="UTC")
    close = pd.Series(range(100, 100 + len(idx)), dtype=float)
    return pd.DataFrame(
        {
            "ts": idx,
            "open": close - 0.5,
            "high": close + 0.5,
            "low": close - 1.0,
            "close": close,
            "volume": 1000,
        }
    )


def test_monthly_fixed_weekend_fallback_next_business_day() -> None:
    strat = DcaBenchmarkStrategy(
        strategy_id="bench",
        params={"variant": "monthly_fixed", "day_of_month": 14, "amount": 100.0},
    )
    sigs = strat.backtest(_ohlc_business_days(), {"symbol": "SPY"})
    jan = [s for s in sigs if s.ts_open_utc.month == 1][0]
    assert jan.ts_open_utc.day == 15
    assert jan.meta["benchmark"]["fallback"] == "next_business_day"


def test_monthly_randomized_is_deterministic_with_seed() -> None:
    params = {"variant": "monthly_randomized", "seed": 42, "amount": 100.0}
    s1 = DcaBenchmarkStrategy("a", params).backtest(_ohlc_business_days(), {"symbol": "SPY"})
    s2 = DcaBenchmarkStrategy("b", params).backtest(_ohlc_business_days(), {"symbol": "SPY"})
    assert [s.ts_open_utc for s in s1] == [s.ts_open_utc for s in s2]


def test_mid_month_respects_window_and_holiday_fallback() -> None:
    strat = DcaBenchmarkStrategy(
        strategy_id="bench",
        params={
            "variant": "mid_month",
            "start_day": 10,
            "end_day": 20,
            "day_of_month": 15,
            "holiday_dates": ["2024-01-15"],
            "amount": 100.0,
        },
    )
    sigs = strat.backtest(_ohlc_business_days(), {"symbol": "SPY"})
    jan = [s for s in sigs if s.ts_open_utc.month == 1][0]
    assert jan.ts_open_utc.day == 16


def test_turn_of_month_and_weekly_fixed_emit_expected_markers() -> None:
    ohlc = _ohlc_business_days()
    turn = DcaBenchmarkStrategy("t", {"variant": "turn_of_month", "offset": -1, "amount": 100.0})
    week = DcaBenchmarkStrategy("w", {"variant": "weekly_fixed", "weekday": 2, "amount": 100.0})
    turn_sigs = turn.backtest(ohlc, {"symbol": "SPY"})
    week_sigs = week.backtest(ohlc, {"symbol": "SPY"})
    assert len(turn_sigs) == 4
    assert len(week_sigs) >= 16
    assert all(s.meta["benchmark"]["capital_curve"] > 0 for s in turn_sigs)
