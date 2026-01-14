from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from quant_engine.api.schemas import SeasonalitySpec
from quant_engine.backtest import runner as backtest_runner
from quant_engine.seasonality import runner as seasonality_runner
from quant_engine.strategies import runner as strategies_runner


SPEC_DIR = Path("specs/tests")
PYTHONPATH = str(Path(__file__).resolve().parents[1] / "src")


def test_combo_backtest_dca_seasonality(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("DB_DSN", raising=False)

    backtest_spec = backtest_runner.load_backtest_spec(SPEC_DIR / "backtest_csv_basic.json")
    backtest_result = backtest_runner.run_backtest_from_spec(backtest_spec)
    assert backtest_result.get("payload")

    dca_spec = strategies_runner.load_strategy_spec(SPEC_DIR / "strategy_dca_equity_csv_basic.json")
    dca_result = strategies_runner.run_backtest_with_payload(dca_spec)
    assert dca_result.get("payload")

    seas_payload = json.loads((SPEC_DIR / "seasonality_basic.json").read_text())
    seas_payload["artifacts"] = {"out_dir": str(tmp_path / "combo_seasonality")}
    seas_spec = SeasonalitySpec.model_validate(seas_payload)
    seas_result = seasonality_runner.run(seas_spec)
    assert "best_metrics" in seas_result
    artifact_root = tmp_path / "combo_seasonality"
    assert artifact_root.exists()
    for path in tmp_path.rglob("*"):
        if path == tmp_path:
            continue
        assert str(path).startswith(str(artifact_root))


def test_combo_cli_runs(tmp_path: Path) -> None:
    pytest.importorskip("typer")
    env = {
        **os.environ,
        "DB_DSN": "sqlite:///:memory:",
        "PYTHONPATH": PYTHONPATH,
    }
    backtest_spec = SPEC_DIR / "backtest_csv_basic.json"
    dca_spec = SPEC_DIR / "strategy_dca_equity_csv_basic.json"
    seas_spec = json.loads((SPEC_DIR / "seasonality_basic.json").read_text())
    seas_spec["artifacts"] = {"out_dir": str(tmp_path / "cli_seasonality")}
    seas_path = tmp_path / "seasonality_cli.json"
    seas_path.write_text(json.dumps(seas_spec))

    commands = [
        [sys.executable, "-m", "quant_engine.cli.main", "backtest", "run", "--spec", str(backtest_spec)],
        [sys.executable, "-m", "quant_engine.cli.main", "strategy", "backtest", "--spec", str(dca_spec)],
        [sys.executable, "-m", "quant_engine.cli.main", "seasonality", "run", "--spec", str(seas_path)],
    ]
    for cmd in commands:
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        assert result.returncode == 0
