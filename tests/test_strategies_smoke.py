"""Smoke tests covering the high-level strategies and runner plumbing."""
from __future__ import annotations

import logging

import pandas as pd
import pytest

from quant_engine.strategies.dca_equity import DcaEquityStrategy
from quant_engine.strategies.dca_etf import DcaEtfStrategy
from quant_engine.strategies.crypto_grid import CryptoGridStrategy


def _make_ohlc(closes: list[float], start: str = "2023-01-01") -> pd.DataFrame:
    dates = pd.date_range(start=start, periods=len(closes), freq="D", tz="UTC")
    df = pd.DataFrame(
        {
            "ts": dates,
            "open": closes,
            "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes],
            "close": closes,
            "volume": [1_000_000] * len(closes),
        }
    )
    return df


def test_dca_equity_strategy_backtest_and_live() -> None:
    params = {
        "asset_class": "EQUITY",
        "grid": [
            {"dd": -10.0, "weight": 0.15},
            {"dd": -20.0, "weight": 0.20},
            {"dd": -30.0, "weight": 0.25},
            {"dd": -40.0, "weight": 0.25},
            {"dd": -50.0, "weight": 0.15},
        ],
        "tp_sl": {
            "enabled": True,
            "mode": "per_grid_max_dd",
            "rules": [
                {"max_dd_reached": -50.0, "tp_pct": 25.0, "be_pct": 10.0},
            ],
        },
    }
    strategy = DcaEquityStrategy("EQ", params)
    closes = [100, 90, 80, 70, 60, 50, 55, 60, 65, 90]
    df = _make_ohlc(closes)
    context = {"symbol": "EQ", "asset_class": "EQUITY"}
    signals = strategy.backtest(df, context)
    buys = [sig for sig in signals if sig.side == "BUY"]
    sells = [sig for sig in signals if sig.side == "SELL"]
    assert len(buys) == 5
    assert len(sells) == 1
    assert sells[0].meta["tp_mode"] == "per_grid_max_dd"
    live_context = {"symbol": "EQ", "asset_class": "EQUITY", "state": {}}
    live_signals = strategy.evaluate_live_bar(df, live_context)
    assert live_signals and live_signals[-1].side == "SELL"


def test_dca_equity_fallback_for_non_temporal_index(caplog: pytest.LogCaptureFixture) -> None:
    params = {
        "asset_class": "EQUITY",
        "grid": [
            {"dd": -10.0, "weight": 0.5},
        ],
    }
    strategy = DcaEquityStrategy("EQ", params)
    closes = [100, 95, 90, 92]
    df = pd.DataFrame(
        {
            "open": closes,
            "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes],
            "close": closes,
            "volume": [1_000] * len(closes),
        }
    )
    context = {"symbol": "EQ", "asset_class": "EQUITY"}
    with caplog.at_level(logging.WARNING, logger="quant_engine.strategies.dca_equity"):
        strategy.backtest(df, context)
    assert any("Rolling drawdown window" in record.message for record in caplog.records)


def test_dca_etf_strategy_smoke() -> None:
    params = {
        "asset_class": "ETF",
        "grid": [
            {"dd": -5.0, "weight": 0.15},
            {"dd": -10.0, "weight": 0.20},
            {"dd": -15.0, "weight": 0.25},
            {"dd": -20.0, "weight": 0.25},
            {"dd": -30.0, "weight": 0.15},
        ],
        "activation_limit": {"period_days": 60, "max_signals": 10},
        "reset_on_new_high": True,
    }
    strategy = DcaEtfStrategy("ETF", params)
    closes = [100, 94, 90, 85, 80, 70, 68, 72]
    df = _make_ohlc(closes)
    context = {"symbol": "ETF", "asset_class": "ETF"}
    signals = strategy.backtest(df, context)
    buys = [sig for sig in signals if sig.side == "BUY"]
    assert len(buys) == 5
    assert all(sig.meta["grid_config"] for sig in buys)
    live_context = {"symbol": "ETF", "asset_class": "ETF", "state": {}}
    live_signals = strategy.evaluate_live_bar(df, live_context)
    assert live_signals == []
    rolling_context = {"symbol": "ETF", "asset_class": "ETF", "state": {}}
    collected: list[str] = []
    for i in range(len(df)):
        window = df.iloc[: i + 1]
        signals_now = strategy.evaluate_live_bar(window, rolling_context)
        collected.extend(sig.side for sig in signals_now)
    assert collected.count("BUY") == 5


def test_crypto_grid_strategy_generates_buy_and_sell() -> None:
    params = {
        "asset_class": "CRYPTO",
        "grid": [
            {"dd": -30.0, "action": "increase", "intensity": "small"},
            {"dd": -45.0, "action": "increase", "intensity": "medium"},
            {"dd": -60.0, "action": "increase", "intensity": "large"},
        ],
        "tp_sl": {
            "enabled": True,
            "mode": "per_grid_max_dd",
            "rules": [
                {"max_dd_reached": -60.0, "tp_pct": 60.0, "be_pct": 20.0}
            ],
        },
    }
    strategy = CryptoGridStrategy("CRYPTO", params)
    closes = [100, 90, 70, 55, 40, 35, 45, 65]
    df = _make_ohlc(closes)
    context = {"symbol": "ETH", "asset_class": "CRYPTO"}
    signals = strategy.backtest(df, context)
    buys = [sig for sig in signals if sig.side == "BUY"]
    sells = [sig for sig in signals if sig.side == "SELL"]
    assert len(buys) == 3
    assert sells and sells[0].meta["action"] == "rebalance"
    live_context = {"symbol": "ETH", "asset_class": "CRYPTO", "state": {}}
    collected: list[str] = []
    for i in range(len(df)):
        window = df.iloc[: i + 1]
        signals_now = strategy.evaluate_live_bar(window, live_context)
        collected.extend(sig.side for sig in signals_now)
    assert collected.count("BUY") == 3
    assert collected[-1] == "SELL"
