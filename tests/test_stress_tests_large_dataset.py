"""Stress test + Monte Carlo on a large dataset sample."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from quant_engine.performance.stress_tests import (
    apply_scenarios_to_returns,
    run_monte_carlo_on_equity_curve,
    run_monte_carlo_on_returns,
    run_scenarios_on_equity_curve,
)


DATA_PATH = Path("specs/examples/data/forex/EURUSD_20250101_20250601_1min.csv")


def _load_returns(max_rows: int = 20000) -> list[float]:
    df = pd.read_csv(DATA_PATH, nrows=max_rows)
    if "Close" not in df.columns:
        raise ValueError("Expected Close column in EURUSD CSV.")
    closes = pd.to_numeric(df["Close"], errors="coerce").dropna()
    returns = closes.diff().dropna()
    return returns.astype(float).tolist()


def _build_equity_curve(returns: list[float], initial_capital: float) -> list[float]:
    equity = [float(initial_capital)]
    for value in returns:
        equity.append(equity[-1] + float(value))
    return equity


@pytest.mark.skipif(not DATA_PATH.exists(), reason="Large dataset not available.")
def test_monte_carlo_large_dataset_sample() -> None:
    returns = _load_returns()
    result = run_monte_carlo_on_returns(
        returns,
        parameters={"n_sims": 200, "seed": 123, "method": "bootstrap", "initial_capital": 10_000.0},
    )

    metrics = result["metrics"]
    assert "final_capital" in metrics
    assert metrics["final_capital"]["p50"] is not None
    assert result["parameters"]["n_simulations"] == 200


@pytest.mark.skipif(not DATA_PATH.exists(), reason="Large dataset not available.")
def test_stress_scenarios_large_dataset_sample() -> None:
    returns = _load_returns()
    params = {
        "initial_capital": 10_000.0,
        "scenarios": [
            {"name": "crash_mid", "type": "crash", "shock_pct": -0.15, "index": "mid"},
            {"name": "vol_x2", "type": "volatility", "vol_multiplier": 2.0},
        ],
    }
    result = apply_scenarios_to_returns(returns, parameters=params)

    metrics = result["metrics"]["scenarios"]
    assert set(metrics.keys()) == {"crash_mid", "vol_x2"}
    distributions = result["distributions"]["scenarios"]
    assert len(distributions["crash_mid"]["returns"]) == len(returns)


@pytest.mark.skipif(not DATA_PATH.exists(), reason="Large dataset not available.")
def test_monte_carlo_and_scenarios_on_equity_curve() -> None:
    returns = _load_returns()
    equity = _build_equity_curve(returns, initial_capital=10_000.0)

    mc_result = run_monte_carlo_on_equity_curve(
        equity,
        parameters={"n_sims": 200, "seed": 123, "method": "bootstrap"},
    )
    assert mc_result["parameters"]["n_simulations"] == 200
    assert mc_result["parameters"]["initial_capital"] == pytest.approx(10_000.0)
    assert mc_result["metrics"]["final_capital"]["p50"] is not None

    scen_result = run_scenarios_on_equity_curve(
        equity,
        parameters={
            "scenarios": [
                {"name": "crash_mid", "type": "crash", "shock_pct": -0.15, "index": "mid"},
                {"name": "vol_x2", "type": "volatility", "vol_multiplier": 2.0},
            ]
        },
    )
    assert set(scen_result["metrics"]["scenarios"].keys()) == {"crash_mid", "vol_x2"}
