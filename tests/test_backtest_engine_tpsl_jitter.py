import random

import pytest

from quant_engine.backtest import engine


def test_backtest_engine_applies_tpsl_jitter_on_tp_exit() -> None:
    dataset = [
        {
            "timestamp": "2020-01-01",
            "open": 100.0,
            "high": 100.5,
            "low": 99.5,
            "close": 100.2,
        },
        {
            "timestamp": "2020-01-02",
            "open": 101.0,
            "high": 103.5,
            "low": 100.6,
            "close": 102.0,
        },
        {
            "timestamp": "2020-01-03",
            "open": 105.0,
            "high": 106.0,
            "low": 104.0,
            "close": 105.5,
        },
    ]
    signals = [1, 0, 0]
    atr_values = [1.0, 1.0, 1.0]
    jitter_cfg = {"dist": "uniform", "tp_bps": 10.0, "sl_bps": 0.0, "seed": 7}

    trades, _, _ = engine.run(
        dataset,
        signals,
        atr_values,
        atr_mult=1.0,
        r_mult=1.0,
        tpsl_jitter=jitter_cfg,
    )

    assert len(trades) == 1
    trade = trades[0]
    expected_jitter = random.Random(7).uniform(-10.0, 10.0)
    expected_exit = dataset[2]["open"] * (1 + expected_jitter / 10000.0)
    assert trade["price_exit"] == pytest.approx(expected_exit, rel=1e-9)


def test_backtest_engine_tpsl_jitter_disabled_is_noop() -> None:
    dataset = [
        {
            "timestamp": "2020-01-01",
            "open": 100.0,
            "high": 100.5,
            "low": 99.5,
            "close": 100.2,
        },
        {
            "timestamp": "2020-01-02",
            "open": 101.0,
            "high": 103.5,
            "low": 100.6,
            "close": 102.0,
        },
        {
            "timestamp": "2020-01-03",
            "open": 105.0,
            "high": 106.0,
            "low": 104.0,
            "close": 105.5,
        },
    ]
    signals = [1, 0, 0]
    atr_values = [1.0, 1.0, 1.0]
    jitter_cfg = {"dist": "uniform", "tp_bps": 10.0, "sl_bps": 10.0, "seed": 7, "enabled": False}

    trades, _, _ = engine.run(
        dataset,
        signals,
        atr_values,
        atr_mult=1.0,
        r_mult=1.0,
        tpsl_jitter=jitter_cfg,
    )

    assert len(trades) == 1
    trade = trades[0]
    assert trade["price_exit"] == dataset[2]["open"]
