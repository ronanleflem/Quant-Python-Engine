from __future__ import annotations

from pathlib import Path

from quant_engine.backtest import runner as backtest_runner


SPEC_DIR = Path("specs/tests")


def _run(spec_name: str) -> dict:
    spec = backtest_runner.load_backtest_spec(SPEC_DIR / spec_name)
    return backtest_runner.run_backtest_from_spec(spec)


def test_backtest_csv_m1_timeframe() -> None:
    result = _run("backtest_csv_basic.json")
    run = result.get("payload", {}).get("run", {})
    assert run.get("timeframe") == "1m"


def test_backtest_csv_h1_timeframe() -> None:
    result = _run("backtest_csv_h1.json")
    run = result.get("payload", {}).get("run", {})
    assert run.get("timeframe") == "1h"


def test_backtest_csv_d1_timeframe() -> None:
    result = _run("backtest_csv_d1.json")
    run = result.get("payload", {}).get("run", {})
    assert run.get("timeframe") == "1d"
