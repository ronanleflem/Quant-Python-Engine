import datetime as dt

import numpy as np
import pandas as pd

from quant_engine.filters.risk_mgmt import (
    atr_risk_gate_filter,
    cooldown_bars_filter,
    daily_loss_cap_filter,
    daily_trades_cap_filter,
    equity_dd_lockout_filter,
)


def _make_sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(123)
    idx = pd.date_range("2025-01-01", periods=2 * 24 * 60, freq="min", tz="UTC")
    n = len(idx)

    close = 100 + np.cumsum(rng.normal(0, 0.3, size=n))
    open_ = close + rng.normal(0, 0.05, size=n)
    span = 0.5 + rng.random(n)
    high = np.maximum(open_, close) + span
    low = np.minimum(open_, close) - span

    pnl = rng.normal(0, 5.0, size=n)
    first_day_mask = idx < pd.Timestamp("2025-01-02", tz="UTC")
    pnl[first_day_mask] = rng.normal(5.0, 1.5, size=first_day_mask.sum())
    loss_window = (idx >= pd.Timestamp("2025-01-02 00:00:00", tz="UTC")) & (
        idx < pd.Timestamp("2025-01-02 00:30:00", tz="UTC")
    )
    pnl[loss_window] = -50.0
    big_drop_ts = pd.Timestamp("2025-01-02 01:00:00", tz="UTC")
    pnl[idx.get_loc(big_drop_ts)] = -15000.0

    equity = 100_000 + pnl.cumsum()

    entry_signal = np.zeros(n, dtype=bool)
    for ts in (
        "2025-01-01 08:00:00",
        "2025-01-01 09:00:00",
        "2025-01-01 10:00:00",
        "2025-01-02 05:00:00",
        "2025-01-02 05:12:00",
        "2025-01-02 05:25:00",
    ):
        entry_signal[idx.get_loc(pd.Timestamp(ts, tz="UTC"))] = True

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "pnl": pnl,
            "equity": equity,
            "entry_signal": entry_signal,
        },
        index=idx,
    )

    volatile_ts = pd.Timestamp("2025-01-02 02:00:00", tz="UTC")
    df.loc[volatile_ts, "high"] = df.loc[volatile_ts, "close"] + 12.0
    df.loc[volatile_ts, "low"] = df.loc[volatile_ts, "close"] - 12.0

    return df


def test_daily_loss_cap_filter_blocks_after_threshold():
    df = _make_sample_df()
    flt = daily_loss_cap_filter(df, loss_cap=100.0, mode="pnl", pnl_col="pnl")

    first_day = df.index.normalize() == pd.Timestamp("2025-01-01", tz="UTC")
    assert flt[first_day].all()

    second_day = df.index.normalize() == pd.Timestamp("2025-01-02", tz="UTC")
    breached_idx = flt[second_day][~flt[second_day]].index[0]
    assert breached_idx >= pd.Timestamp("2025-01-02 00:00:00+00:00")
    assert not flt.loc[breached_idx]
    post_breach = flt[second_day].loc[breached_idx:]
    assert (~post_breach).all()


def test_equity_dd_lockout_filter_triggers_on_drawdown():
    df = _make_sample_df()
    flt = equity_dd_lockout_filter(df, equity_col="equity", max_dd_pct=0.1)

    breach_points = flt[~flt]
    assert not breach_points.empty
    first_breach = breach_points.index[0]
    assert first_breach >= pd.Timestamp("2025-01-02 01:00:00+00:00")
    assert not flt.loc[first_breach]
    assert flt.loc[pd.Timestamp("2025-01-01 12:00:00", tz="UTC")]


def test_atr_risk_gate_filter_has_expected_behavior():
    df = _make_sample_df()
    flt = atr_risk_gate_filter(df, atr_window=5, max_atr_pct=0.02)

    assert flt.dtype == bool
    assert flt.any()
    assert (~flt).any()


def test_daily_trades_cap_filter_limits_entries():
    df = _make_sample_df()
    flt = daily_trades_cap_filter(df, signal_col="entry_signal", max_trades_per_day=2)

    first_signals = [
        pd.Timestamp("2025-01-01 08:00:00", tz="UTC"),
        pd.Timestamp("2025-01-01 09:00:00", tz="UTC"),
        pd.Timestamp("2025-01-01 10:00:00", tz="UTC"),
    ]
    assert flt.loc[first_signals[0]]
    assert flt.loc[first_signals[1]]
    assert not flt.loc[first_signals[2]]
    assert not flt.loc[pd.Timestamp("2025-01-01 15:00:00", tz="UTC")]


def test_cooldown_bars_filter_enforces_gap():
    df = _make_sample_df()
    flt = cooldown_bars_filter(df, signal_col="entry_signal", cooldown_bars=10)

    ts = pd.Timestamp("2025-01-02 05:00:00", tz="UTC")
    assert flt.loc[ts]
    for minutes in range(1, 11):
        check_ts = ts + pd.Timedelta(minutes=minutes)
        assert not flt.loc[check_ts]
    assert flt.loc[ts + pd.Timedelta(minutes=11)]
    second_signal = ts + pd.Timedelta(minutes=12)
    assert flt.loc[second_signal]
    for minutes in range(13, 23):
        check_ts = ts + pd.Timedelta(minutes=minutes)
        assert not flt.loc[check_ts]
    assert flt.loc[ts + pd.Timedelta(minutes=23)]
    assert flt.loc[ts + pd.Timedelta(minutes=25)]
