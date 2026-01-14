from __future__ import annotations

from pathlib import Path

from quant_engine.strategies import runner as strategies_runner


SPEC_DIR = Path("specs/tests")


def _load_spec(name: str) -> dict:
    return strategies_runner.load_strategy_spec(SPEC_DIR / name)


def _assert_payload(result: dict) -> None:
    payload = result.get("payload", {})
    assert "run" in payload
    assert "trades" in payload
    run = payload["run"]
    assert run["symbol"] == "SPY"
    assert run["assetClass"] in {"EQUITY", "ETF"}


def _reset_cache() -> None:
    strategies_runner._OHLC_CACHE.clear()


def test_strategy_dca_equity_csv_basic(monkeypatch) -> None:
    monkeypatch.delenv("DB_DSN", raising=False)
    _reset_cache()
    spec = _load_spec("strategy_dca_equity_csv_basic.json")
    result = strategies_runner.run_backtest_with_payload(spec)
    _assert_payload(result)


def test_strategy_dca_equity_csv_filters(monkeypatch) -> None:
    monkeypatch.delenv("DB_DSN", raising=False)
    _reset_cache()
    spec = _load_spec("strategy_dca_equity_csv_filters.json")
    result = strategies_runner.run_backtest_with_payload(spec)
    _assert_payload(result)


def test_strategy_dca_etf_csv_basic(monkeypatch) -> None:
    monkeypatch.delenv("DB_DSN", raising=False)
    _reset_cache()
    spec = _load_spec("strategy_dca_etf_csv_basic.json")
    result = strategies_runner.run_backtest_with_payload(spec)
    _assert_payload(result)


def test_strategy_dca_equity_trades_emitted(monkeypatch) -> None:
    monkeypatch.delenv("DB_DSN", raising=False)
    _reset_cache()
    spec = _load_spec("strategy_dca_equity_csv_trades.json")
    result = strategies_runner.run_backtest_with_payload(spec)
    payload = result.get("payload", {})
    assert payload["trades"]
