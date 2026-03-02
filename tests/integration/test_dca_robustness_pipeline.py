import json
from datetime import datetime, timedelta
from pathlib import Path

from quant_engine.api import schemas
from quant_engine.core.spec import ArtifactsSpec
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
