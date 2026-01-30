"""Tests for stress test helpers (Monte Carlo + scenarios)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from quant_engine.performance.models import CompletedTrade
from quant_engine.performance.stress_tests import (
    apply_scenarios_to_returns,
    run_monte_carlo_on_trades,
)


def _make_trade(*, pnl: float, exit_offset_days: int, symbol: str = "ABC") -> CompletedTrade:
    entry_time = datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=exit_offset_days - 1)
    exit_time = datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=exit_offset_days)
    entry_price = 100.0
    exit_price = entry_price + pnl
    quantity = 1.0
    gross_pnl = pnl
    gross_pnl_pct = (pnl / entry_price) * 100.0
    return CompletedTrade(
        strategy_id="stress",
        run_id="run-1",
        symbol=symbol,
        asset_class="EQUITY",
        side="LONG",
        cycle_id=exit_offset_days,
        entry_time_utc=entry_time,
        exit_time_utc=exit_time,
        entry_price=entry_price,
        exit_price=exit_price,
        quantity=quantity,
        gross_pnl=gross_pnl,
        gross_pnl_pct=gross_pnl_pct,
        max_dd_pct=None,
        meta={},
    )


def test_run_monte_carlo_on_trades_outputs_metrics_and_distributions() -> None:
    trades = [
        _make_trade(pnl=10.0, exit_offset_days=1),
        _make_trade(pnl=-5.0, exit_offset_days=2),
        _make_trade(pnl=7.0, exit_offset_days=3),
    ]
    params = {"n_sims": 50, "seed": 7, "method": "bootstrap", "initial_capital": 1_000.0}
    result = run_monte_carlo_on_trades(trades, parameters=params)

    metrics = result["metrics"]
    assert "max_drawdown" in metrics
    assert "final_capital" in metrics
    assert "p50" in metrics["final_capital"]

    distributions = result["distributions"]
    assert len(distributions["equity_curves"]) == params["n_sims"]
    assert len(distributions["equity_curves"][0]) == len(trades) + 1

    parameters = result["parameters"]
    assert parameters["n_simulations"] == params["n_sims"]
    assert parameters["seed"] == params["seed"]


def test_run_monte_carlo_rejects_invalid_config() -> None:
    trades = [_make_trade(pnl=5.0, exit_offset_days=1)]
    with pytest.raises(ValueError):
        run_monte_carlo_on_trades(trades, parameters={"n_sims": 0})


def test_run_monte_carlo_block_method_supported() -> None:
    trades = [
        _make_trade(pnl=10.0, exit_offset_days=1),
        _make_trade(pnl=-5.0, exit_offset_days=2),
        _make_trade(pnl=7.0, exit_offset_days=3),
        _make_trade(pnl=-3.0, exit_offset_days=4),
        _make_trade(pnl=2.0, exit_offset_days=5),
    ]
    params = {"n_sims": 20, "seed": 5, "method": "block", "block_size": 2, "initial_capital": 1_000.0}
    result = run_monte_carlo_on_trades(trades, parameters=params)

    metrics = result["metrics"]
    assert "final_capital" in metrics
    assert metrics["final_capital"]["p50"] is not None


def test_run_monte_carlo_light_mode_reduces_equity_curves() -> None:
    trades = [
        _make_trade(pnl=10.0, exit_offset_days=1),
        _make_trade(pnl=-5.0, exit_offset_days=2),
        _make_trade(pnl=7.0, exit_offset_days=3),
        _make_trade(pnl=-3.0, exit_offset_days=4),
        _make_trade(pnl=2.0, exit_offset_days=5),
        _make_trade(pnl=1.0, exit_offset_days=6),
    ]
    params = {
        "n_sims": 30,
        "seed": 11,
        "method": "bootstrap",
        "initial_capital": 1_000.0,
        "output": {"mode": "light", "max_curves": 5, "curve_stride": 2},
    }
    result = run_monte_carlo_on_trades(trades, parameters=params)

    equity_curves = result["distributions"]["equity_curves"]
    assert len(equity_curves) == 5
    assert len(equity_curves[0]) < len(trades) + 1


def test_run_monte_carlo_light_strict_percentiles_and_curves() -> None:
    trades = [
        _make_trade(pnl=10.0, exit_offset_days=1),
        _make_trade(pnl=-5.0, exit_offset_days=2),
        _make_trade(pnl=7.0, exit_offset_days=3),
        _make_trade(pnl=-3.0, exit_offset_days=4),
        _make_trade(pnl=2.0, exit_offset_days=5),
        _make_trade(pnl=1.0, exit_offset_days=6),
    ]
    params = {
        "n_sims": 40,
        "seed": 13,
        "method": "bootstrap",
        "initial_capital": 1_000.0,
        "output": {"mode": "light_strict", "max_curves": 6, "curve_stride": 3},
    }
    result = run_monte_carlo_on_trades(trades, parameters=params)

    metrics = result["metrics"]
    assert "p10" in metrics["final_capital"]
    assert "p99" in metrics["final_capital"]
    assert "p5" not in metrics["final_capital"]

    distributions = result["distributions"]
    assert set(distributions.keys()) == {"equity_curves"}
    assert len(distributions["equity_curves"]) == 6


def test_apply_scenarios_to_returns_outputs_scenarios() -> None:
    returns = [10.0, -5.0, 8.0, -2.0]
    params = {
        "initial_capital": 1_000.0,
        "scenarios": [
            {"name": "crash_now", "type": "crash", "shock_pct": -0.2, "index": "start"},
            {"name": "vol_x2", "type": "volatility", "vol_multiplier": 2.0},
        ],
    }
    result = apply_scenarios_to_returns(returns, parameters=params)

    metrics = result["metrics"]["scenarios"]
    assert set(metrics.keys()) == {"crash_now", "vol_x2"}

    distributions = result["distributions"]["scenarios"]
    assert len(distributions["crash_now"]["returns"]) == len(returns)
    assert len(distributions["vol_x2"]["returns"]) == len(returns)

    assert len(result["parameters"]["scenarios"]) == 2


def test_apply_scenarios_to_returns_rejects_bad_parameters() -> None:
    with pytest.raises(ValueError):
        apply_scenarios_to_returns([1.0, 2.0], parameters="bad")
