import json
import os
from pathlib import Path

import pandas as pd
import pytest

from quant_engine.api import schemas
from quant_engine.stats import runner


@pytest.mark.skipif(
    not os.getenv("QE_MARKETDATA_MYSQL_URL"),
    reason="QE_MARKETDATA_MYSQL_URL is required for MySQL-backed smoke test",
)
def test_stats_levels_integration_smoke(tmp_path: Path) -> None:
    spec_path = Path("specs/stats_levels_examples.json")
    if not spec_path.exists():
        pytest.skip("stats_levels_examples.json spec is missing")
    payload = json.loads(spec_path.read_text())
    payload.setdefault("artifacts", {})["out_dir"] = str(tmp_path / "stats_levels_examples")
    spec = schemas.StatsSpec(**payload)

    df_summary = runner.run_stats(spec)

    assert isinstance(df_summary, pd.DataFrame)
    assert not df_summary.empty
