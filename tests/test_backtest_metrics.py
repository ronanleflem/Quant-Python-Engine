import pytest

from quant_engine.backtest import metrics


def test_empty_series_metrics_zero():
    result = metrics.compute([], [])
    assert result == {
        "sharpe": 0.0,
        "sortino": 0.0,
        "max_drawdown": 0.0,
        "cagr": 0.0,
        "hit_rate": 0.0,
        "avg_R": 0.0,
        "trades": 0.0,
    }


def test_constant_series_sharpe_sortino_zero():
    returns = [1.0, 1.0, 1.0]
    assert metrics.sharpe_ratio(returns) == 0.0
    assert metrics.sortino_ratio(returns) == 0.0


def test_simple_drawdown():
    equity = [0.0, 2.0, 1.0]
    assert metrics.max_drawdown(equity) == 1.0


def test_known_trades_metrics():
    trades = [
        {"pnl": -1.0, "r_multiple": -1.0},
        {"pnl": 2.0, "r_multiple": 2.0},
        {"pnl": 2.0, "r_multiple": 2.0},
    ]
    equity = [0.0, -1.0, 1.0, 3.0, 2.0, 4.0, 1.0]
    result = metrics.compute(trades, equity)

    assert result["sharpe"] == pytest.approx((3 ** 0.5) / (2 ** 0.5))
    assert result["sortino"] == pytest.approx(3.0)
    assert result["max_drawdown"] == pytest.approx(3.0)


def test_hit_rate_and_avg_r():
    trades = [
        {"pnl": 1.0, "r_multiple": 2.0},
        {"pnl": -1.0, "r_multiple": -1.0},
        {"pnl": 0.5, "r_multiple": 1.5},
    ]
    assert metrics.hit_rate(trades) == pytest.approx(2 / 3)
    assert metrics.avg_r(trades) == pytest.approx(2.5 / 3)
