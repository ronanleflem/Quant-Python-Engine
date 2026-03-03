from datetime import datetime, timedelta, timezone

import pytest

from quant_engine.backtest import metrics


def test_final_performance_normalized() -> None:
    assert metrics.final_performance_normalized(1300.0, 1000.0) == pytest.approx(0.3)


def test_twr_compounds_period_returns() -> None:
    assert metrics.twr([0.1, -0.05, 0.02]) == pytest.approx((1.1 * 0.95 * 1.02) - 1.0)


def test_xirr_converges_on_regular_cashflows() -> None:
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    value, status = metrics.xirr(
        [
            (start, -1000.0),
            (start + timedelta(days=365), 1100.0),
        ]
    )
    assert status == "ok"
    assert value == pytest.approx(0.1, rel=1e-3)


def test_xirr_invalid_without_sign_change() -> None:
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    value, status = metrics.xirr([(start, -100.0), (start + timedelta(days=1), -10.0)])
    assert value is None
    assert status == "invalid_cashflows"


def test_drawdown_and_time_under_water_on_contributed_capital() -> None:
    equity = [100.0, 150.0, 120.0, 140.0, 170.0]
    contributed = [100.0, 120.0, 120.0, 140.0, 140.0]
    assert metrics.max_drawdown_on_contributed_capital(equity, contributed) == pytest.approx(0.25)
    assert metrics.time_under_water(equity) == 2


def test_return_over_stress_ratio_uses_drawdown_denominator() -> None:
    ratio = metrics.return_over_stress_ratio(0.3, 0.15)
    assert ratio["version"] == "return_over_stress_ratio_v1"
    assert ratio["numerator"] == pytest.approx(0.3)
    assert ratio["denominator_raw"] == pytest.approx(0.15)
    assert ratio["denominator"] == pytest.approx(0.15)
    assert ratio["ratio"] == pytest.approx(2.0)


def test_return_over_stress_ratio_uses_epsilon_for_near_zero_drawdown() -> None:
    ratio = metrics.return_over_stress_ratio(0.1, 0.0, epsilon=1e-6)
    assert ratio["denominator"] == pytest.approx(1e-6)
    assert ratio["ratio"] == pytest.approx(100000.0)


def test_capital_efficiency_index_nominal() -> None:
    cei = metrics.capital_efficiency_index(1300.0, [100.0, 400.0, 1000.0])
    assert cei == pytest.approx(1.3)


def test_capital_efficiency_index_zero_or_negative_max_contributed_capital() -> None:
    assert metrics.capital_efficiency_index(1000.0, []) == pytest.approx(0.0)
    assert metrics.capital_efficiency_index(1000.0, [0.0, -10.0]) == pytest.approx(0.0)


def test_capital_efficiency_index_short_series() -> None:
    cei = metrics.capital_efficiency_index(950.0, [1000.0])
    assert cei == pytest.approx(0.95)
