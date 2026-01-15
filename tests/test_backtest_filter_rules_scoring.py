from __future__ import annotations

from pathlib import Path

from quant_engine.backtest import runner as backtest_runner


SPEC_DIR = Path("specs/tests")


def test_backtest_filter_rules_scoring_runs() -> None:
    spec = backtest_runner.load_backtest_spec(
        SPEC_DIR / "backtest_csv_filter_rules_scoring.json"
    )
    result = backtest_runner.run_backtest_from_spec(spec)
    assert result.get("payload") is not None
