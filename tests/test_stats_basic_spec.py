from __future__ import annotations

import json
from pathlib import Path

from quant_engine.api.schemas import StatsSpec
from quant_engine.stats import runner as stats_runner


SPEC_PATH = Path("specs/tests/stats_basic.json")


def test_stats_basic_from_spec(tmp_path: Path) -> None:
    payload = json.loads(SPEC_PATH.read_text())
    payload["artifacts"] = {"out_dir": str(tmp_path / "stats_basic")}
    spec = StatsSpec.model_validate(payload)
    df = stats_runner.run_stats(spec)
    assert not df.empty
    assert "lift_freq" in df.columns
