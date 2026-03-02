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
