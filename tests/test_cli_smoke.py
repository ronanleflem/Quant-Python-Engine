import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("typer")

SPEC_PATH = Path(__file__).parent / "data" / "spec_example.json"
PYTHONPATH = str(Path(__file__).resolve().parents[1] / "src")


@pytest.mark.skipif(not SPEC_PATH.exists(), reason="spec file missing")
def test_cli_smoke() -> None:
    env = {**os.environ, "DB_DSN": "sqlite:///:memory:", "PYTHONPATH": PYTHONPATH}
    result = subprocess.run(
        [sys.executable, "-m", "quant_engine.cli.main", "run-local", "--spec", str(SPEC_PATH)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0
    assert "metrics" in result.stdout
