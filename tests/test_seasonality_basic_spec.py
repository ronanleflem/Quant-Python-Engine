from __future__ import annotations

import json
from pathlib import Path

from quant_engine.api.schemas import SeasonalitySpec
from quant_engine.seasonality import runner as seasonality_runner


SPEC_PATH = Path("specs/tests/seasonality_basic.json")
RETURNS_TOPK_SPEC_PATH = Path("specs/tests/seasonality_returns_topk.json")


def test_seasonality_basic_from_spec(tmp_path: Path) -> None:
    payload = json.loads(SPEC_PATH.read_text())
    payload["artifacts"] = {"out_dir": str(tmp_path / "seasonality_basic")}
    spec = SeasonalitySpec.model_validate(payload)
    result = seasonality_runner.run(spec)
    assert "best_metrics" in result
    assert "active_bins" in result


def test_seasonality_returns_topk_artifacts(tmp_path: Path) -> None:
    payload = json.loads(RETURNS_TOPK_SPEC_PATH.read_text())
    payload["artifacts"] = {"out_dir": str(tmp_path / "seasonality_returns")}
    spec = SeasonalitySpec.model_validate(payload)
    result = seasonality_runner.run(spec)
    artifacts = result.get("artifacts", {})
    assert artifacts.get("trades")
    assert artifacts.get("equity")
    assert Path(artifacts["trades"]).exists()
    assert Path(artifacts["equity"]).exists()
    if artifacts.get("profiles"):
        assert Path(artifacts["profiles"]).exists()
