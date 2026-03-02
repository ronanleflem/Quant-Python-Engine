import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from quant_engine.api import schemas
from quant_engine.core.spec import ArtifactsSpec
from quant_engine.optimize import runner as optimize_runner
from quant_engine.stats import runner


def test_stats_pipeline_writes_dca_robustness_artifacts(tmp_path: Path) -> None:
    start = datetime(2021, 1, 1)
    rows = []
    for i in range(50):
        ts = start + timedelta(days=i)
        close = 100 + i
        rows.append(
            {
                "timestamp": ts.isoformat(),
                "symbol": "TST",
                "open": close - 1,
                "high": close + 1,
                "low": close - 2,
                "close": close,
                "volume": 1000,
            }
        )

    dataset_path = tmp_path / "tiny.json"
    dataset_path.write_text(json.dumps(rows))

    out_dir = tmp_path / "artifacts"
    spec = schemas.StatsSpec(
        data=schemas.StatsDataSpec(
            dataset_path=str(dataset_path),
            symbols=["TST"],
            timeframe="1d",
            start="2021-01-01",
            end="2021-02-19",
        ),
        events=[schemas.StatsEventSpec(name="k_consecutive", params={"k": 2, "direction": "up"})],
        targets=[schemas.StatsTargetSpec(name="up_next_bar")],
        artifacts=ArtifactsSpec(out_dir=str(out_dir)),
    )

    out = runner.run_stats(spec)
    assert not out.empty

    robustness_json = out_dir / "dca_robustness_v1.json"
    robustness_parquet = out_dir / "dca_robustness_v1.parquet"
    assert robustness_json.exists()
    assert robustness_parquet.exists()

    payload = json.loads(robustness_json.read_text())
    assert payload["version"] == "dca-grid-process-v1"
    assert payload["seed"] == 42
    assert "percentile" in payload
    assert "dominance" in payload
    assert len(payload["stress_grid"]) == 3

    run_manifest = out_dir / "run_manifest.json"
    checksums = out_dir / "checksums.txt"
    assert run_manifest.exists()
    assert checksums.exists()

    manifest_payload = json.loads(run_manifest.read_text())
    assert manifest_payload["schema_version"] == "dca-grid-process-v1"
    assert manifest_payload["seed"] == 42
    assert isinstance(manifest_payload.get("dataset_hash"), str)
    assert len(manifest_payload["dataset_hash"]) == 64
    assert isinstance(manifest_payload.get("spec_hash"), str)
    assert len(manifest_payload["spec_hash"]) == 64
    assert isinstance(manifest_payload.get("runtime_versions"), dict)
    assert "python" in manifest_payload["runtime_versions"]

    checksum_lines = [line for line in checksums.read_text().splitlines() if line.strip()]
    assert checksum_lines
    assert any(line.endswith("run_manifest.json") for line in checksum_lines)
    assert any(line.endswith("dca_robustness_v1.json") for line in checksum_lines)
    assert any(line.endswith("best_plausible_passive_ex_ante.json") for line in checksum_lines)

    ex_ante_json = out_dir / "best_plausible_passive_ex_ante.json"
    ex_ante_parquet = out_dir / "best_plausible_passive_ex_ante.parquet"
    assert ex_ante_json.exists()
    assert ex_ante_parquet.exists()

    ex_ante_payload = json.loads(ex_ante_json.read_text())
    metadata = ex_ante_payload["metadata"]
    assert metadata["schema_version"] == "dca-grid-process-v1"
    assert metadata["seed"] == 42
    assert isinstance(metadata["dataset_hash"], str)
    assert len(metadata["dataset_hash"]) == 64
    assert metadata["config_version"] == "dca-grid-process-v1"
    assert "universe_rules_version" in metadata

    row = ex_ante_payload["rows"][0]
    assert row["percentile"] == payload["percentile"]
    assert row["perf_dominance_prob"] == payload["dominance"]["perf_dominance_prob"]
    assert row["drawdown_dominance_prob"] == payload["dominance"]["drawdown_dominance_prob"]


def test_optimize_runner_bridge_maps_to_stats_robustness() -> None:
    stats_out = pd.DataFrame(
        [
            {"p_hat": 0.55, "lift_freq": 0.10},
            {"p_hat": 0.45, "lift_freq": 0.05},
            {"p_hat": 0.60, "lift_freq": 0.12},
        ]
    )

    via_stats = runner.compute_dca_robustness_artifacts(stats_out, seed=77)
    via_optimize = optimize_runner.compute_dca_robustness_from_stats(stats_out, seed=77)

    assert via_optimize == via_stats
    assert len(via_optimize["stress_grid"]) == 3
