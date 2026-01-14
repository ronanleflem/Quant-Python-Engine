from __future__ import annotations

from datetime import datetime
from pathlib import Path

from quant_engine.backtest import runner as backtest_runner
from quant_engine.strategies import runner as strategies_runner


SPEC_DIR = Path("specs/tests")


def _parse_iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def test_backtest_trade_fields_are_consistent() -> None:
    spec = backtest_runner.load_backtest_spec(SPEC_DIR / "backtest_csv_trades.json")
    result = backtest_runner.run_backtest_from_spec(spec)
    trades = result.get("payload", {}).get("trades", [])
    assert trades
    sample = trades[0]
    assert str(sample.get("side", "")).lower() == "long"
    pnl = sample.get("pnl")
    if pnl is None:
        pnl = sample.get("grossPnl")
    assert pnl is not None
    entry_ts = sample.get("ts_entry") or sample.get("entryTimeUtc")
    exit_ts_raw = sample.get("ts_exit") or sample.get("exitTimeUtc")
    assert entry_ts is not None
    assert exit_ts_raw is not None
    entry = _parse_iso(entry_ts)
    exit_ts = _parse_iso(exit_ts_raw)
    assert entry <= exit_ts


def test_dca_trades_have_exit_action() -> None:
    spec = strategies_runner.load_strategy_spec(SPEC_DIR / "strategy_dca_equity_csv_trades.json")
    result = strategies_runner.run_backtest_with_payload(spec)
    trades = result.get("payload", {}).get("trades", [])
    assert trades
    assert all(t.get("meta", {}).get("action") in {"take_profit", "break_even", "stop_loss"} for t in trades)
    assert all((t.get("quantity") or 0) > 0 for t in trades)
