from __future__ import annotations

import json
from pathlib import Path

from quant_engine.api.schemas import SeasonalitySpec, StatsSpec
from quant_engine.seasonality import runner as seasonality_runner
from quant_engine.stats import runner as stats_runner


STATS_SPEC = Path("specs/tests/stats_seasonality_combo_stats.json")
SEAS_SPEC = Path("specs/tests/stats_seasonality_combo_seasonality.json")


def test_stats_seasonality_combo(tmp_path: Path) -> None:
    stats_payload = json.loads(STATS_SPEC.read_text())
    stats_payload["artifacts"] = {"out_dir": str(tmp_path / "combo_stats")}
    stats_spec = StatsSpec.model_validate(stats_payload)
    stats_df = stats_runner.run_stats(stats_spec)
    assert not stats_df.empty

    seas_payload = json.loads(SEAS_SPEC.read_text())
    seas_payload["artifacts"] = {"out_dir": str(tmp_path / "combo_seasonality")}
    seas_spec = SeasonalitySpec.model_validate(seas_payload)
    result = seasonality_runner.run(seas_spec)
    assert "active_bins" in result
