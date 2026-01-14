from __future__ import annotations

from pathlib import Path

import copy

from quant_engine.backtest import runner as backtest_runner


SPEC_DIR = Path("specs/tests")


def _run(spec_name: str) -> dict:
    spec = backtest_runner.load_backtest_spec(SPEC_DIR / spec_name)
    return backtest_runner.run_backtest_from_spec(spec)


def _run_with_timeframe(timeframe: str | None) -> dict:
    spec = backtest_runner.load_backtest_spec(SPEC_DIR / "backtest_csv_basic.json")
    spec = copy.deepcopy(spec)
    if timeframe is None:
        spec["data"].pop("timeframe", None)
    else:
        spec["data"]["timeframe"] = timeframe
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


def test_timeframe_aliases() -> None:
    result = _run_with_timeframe("M1")
    run = result.get("payload", {}).get("run", {})
    assert run.get("timeframe") == "1m"
    result = _run_with_timeframe("1min")
    run = result.get("payload", {}).get("run", {})
    assert run.get("timeframe") == "1m"


def test_timeframe_missing_defaults() -> None:
    result = _run_with_timeframe(None)
    run = result.get("payload", {}).get("run", {})
    assert run.get("timeframe") == "1m"
