from __future__ import annotations

from pathlib import Path

from quant_engine.backtest import runner as backtest_runner


SPEC_DIR = Path("specs/tests")


def test_backtest_dynamic_stop_loss_runs() -> None:
    spec = backtest_runner.load_backtest_spec(
        SPEC_DIR / "backtest_csv_dynamic_stop_loss.json"
    )
    result = backtest_runner.run_backtest_from_spec(spec)
    assert result.get("payload") is not None
