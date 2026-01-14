from __future__ import annotations

from pathlib import Path

from quant_engine.backtest import runner as backtest_runner


SPEC_DIR = Path("specs/tests")


def test_backtest_csv_emits_trades() -> None:
    spec = backtest_runner.load_backtest_spec(SPEC_DIR / "backtest_csv_trades.json")
    result = backtest_runner.run_backtest_from_spec(spec)
    payload = result.get("payload", {})
    trades = payload.get("trades", [])
    assert trades
