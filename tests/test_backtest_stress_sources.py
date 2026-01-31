import pytest

from quant_engine.performance.backtest_builder import build_backtest_payload


def _sample_trades() -> list[dict]:
    return [
        {
            "ts_entry": "2025-01-01T00:00:00Z",
            "ts_exit": "2025-01-01T00:01:00Z",
            "price_entry": 1.0,
            "price_exit": 1.001,
            "quantity": 1.0,
            "pnl": 10.0,
            "side": "LONG",
            "r_multiple": 1.0,
        },
        {
            "ts_entry": "2025-01-01T00:02:00Z",
            "ts_exit": "2025-01-01T00:03:00Z",
            "price_entry": 1.0,
            "price_exit": 0.999,
            "quantity": 1.0,
            "pnl": -5.0,
            "side": "LONG",
            "r_multiple": -0.5,
        },
        {
            "ts_entry": "2025-01-01T00:04:00Z",
            "ts_exit": "2025-01-01T00:05:00Z",
            "price_entry": 1.0,
            "price_exit": 1.002,
            "quantity": 1.0,
            "pnl": 7.0,
            "side": "LONG",
            "r_multiple": 0.7,
        },
    ]


def _build_payload(source: str) -> dict:
    equity = [0.0, 0.0, 0.0]
    return build_backtest_payload(
        strategy_id="stress-src",
        run_id="run-src",
        asset_class="FX",
        symbol="EURUSD",
        timeframe="1m",
        trades=_sample_trades(),
        equity=equity,
        start_ts="2025-01-01T00:00:00Z",
        end_ts="2025-01-01T00:06:00Z",
        config={
            "initial_capital": 10000.0,
            "stress_tests": {
                "enabled": True,
                "monte_carlo": {
                    "source": source,
                    "n_sims": 50,
                    "seed": 1,
                    "method": "bootstrap",
                },
            },
        },
    )


def test_backtest_monte_carlo_source_trades_changes_output() -> None:
    payload_trades = _build_payload("trades")
    payload_equity = _build_payload("equity")

    mc_trades = payload_trades["stress_tests"]["monte_carlo"]
    mc_equity = payload_equity["stress_tests"]["monte_carlo"]

    trades_p50 = mc_trades["metrics"]["total_return"]["p50"]
    equity_p50 = mc_equity["metrics"]["total_return"]["p50"]

    assert equity_p50 == pytest.approx(0.0)
    assert trades_p50 != pytest.approx(0.0)
