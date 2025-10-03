from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from quant_engine.api import app as api_app
from quant_engine.levels import repo as levels_repo


@pytest.mark.skipif(
    "QE_MARKETDATA_MYSQL_URL" not in os.environ,
    reason="Requires QE_MARKETDATA_MYSQL_URL for persistence",
)
def test_levels_phase15_end_to_end() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    build_spec = repo_root / "specs" / "levels_phase15_build.json"
    fill_spec = repo_root / "specs" / "levels_phase15_fill.json"
    env = {**os.environ, "PYTHONPATH": str(repo_root / "src")}

    build_cmd = [
        sys.executable,
        "-m",
        "quant_engine.cli.main",
        "levels",
        "build",
        "--spec",
        str(build_spec),
    ]
    build_result = subprocess.run(
        build_cmd,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert build_result.returncode == 0, build_result.stderr

    fill_cmd = [
        sys.executable,
        "-m",
        "quant_engine.cli.main",
        "levels",
        "fill",
        "--spec",
        str(fill_spec),
    ]
    fill_result = subprocess.run(
        fill_cmd,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert fill_result.returncode == 0, fill_result.stderr

    active = api_app.levels_active(symbol="EURUSD", level_types=["FVG"], limit=10)
    assert isinstance(active, list)

    engine = levels_repo.get_engine()
    table_fqn = "marketdata.levels"
    df_sessions = levels_repo.select_levels(
        engine,
        table_fqn,
        symbol="EURUSD",
        level_types=["SESSION_HIGH", "SESSION_LOW"],
        active_only=False,
        start="2025-01-01T00:00:00Z",
        end="2025-01-07T23:59:00Z",
        limit=10000,
    )
    assert isinstance(df_sessions, pd.DataFrame)
    assert not df_sessions.empty
