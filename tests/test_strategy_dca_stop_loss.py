from __future__ import annotations

from pathlib import Path

from quant_engine.strategies import runner as strategies_runner


SPEC_DIR = Path("specs/tests")


def _load_spec(name: str) -> dict:
    return strategies_runner.load_strategy_spec(SPEC_DIR / name)


def _reset_cache() -> None:
    strategies_runner._OHLC_CACHE.clear()


def _assert_stop_loss(result: dict) -> None:
    trades = result.get("payload", {}).get("trades", [])
    assert trades
    assert any(t.get("meta", {}).get("action") == "stop_loss" for t in trades)


def test_strategy_dca_equity_stop_loss_intracandle(monkeypatch) -> None:
    monkeypatch.delenv("DB_DSN", raising=False)
    _reset_cache()
    spec = _load_spec("strategy_dca_equity_stop_loss.json")
    result = strategies_runner.run_backtest_with_payload(spec)
    _assert_stop_loss(result)


def test_strategy_dca_equity_stop_loss_bar_close(monkeypatch) -> None:
    monkeypatch.delenv("DB_DSN", raising=False)
    _reset_cache()
    spec = _load_spec("strategy_dca_equity_stop_loss_bar_close.json")
    result = strategies_runner.run_backtest_with_payload(spec)
    _assert_stop_loss(result)


def test_strategy_dca_equity_trailing_stop_bar_close(monkeypatch) -> None:
    monkeypatch.delenv("DB_DSN", raising=False)
    _reset_cache()
    spec = _load_spec("strategy_dca_equity_trailing_stop_bar_close.json")
    result = strategies_runner.run_backtest_with_payload(spec)
    trades = result.get("payload", {}).get("trades", [])
    counts = result.get("result", {}).get("counts", {})
    assert counts.get("SPY", 0) >= 1
    if trades:
        assert any(t.get("meta", {}).get("action") == "trailing_stop" for t in trades)
