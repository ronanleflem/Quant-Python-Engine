import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

from quant_engine.api import schemas
from quant_engine.core.spec import ArtifactsSpec
from quant_engine.stats import runner


def _build_canonical_dataset(path: Path) -> None:
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
    path.write_text(json.dumps(rows))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _extract_checksum_map(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        digest, file_name = line.split(maxsplit=1)
        out[file_name.strip()] = digest.strip()
    return out


def _classify_reproducibility(
    baseline: dict,
    manifest: dict,
    robustness: dict,
    checksums: dict[str, str],
    observed_hashes: dict[str, str],
) -> tuple[str, list[str], list[str]]:
    non_conformities: list[str] = []
    minor_drift: list[str] = []

    expected_manifest = baseline["manifest"]
    for key, expected_value in expected_manifest.items():
        if manifest.get(key) != expected_value:
            non_conformities.append(f"manifest.{key}")

    for key in ("dataset_hash", "spec_hash", "commit", "runtime_versions"):
        if key not in manifest:
            minor_drift.append(f"manifest.{key}:missing")

    for artifact_name, expected_hash in baseline["artifact_sha256"].items():
        observed_hash = observed_hashes.get(artifact_name)
        if observed_hash != expected_hash:
            non_conformities.append(f"sha256.{artifact_name}")

    for artifact_name, expected_hash in baseline["checksums"].items():
        observed_checksum = checksums.get(artifact_name)
        if observed_checksum != expected_hash:
            non_conformities.append(f"checksums.{artifact_name}")

    expected_robustness = baseline["robustness"]
    tolerance = float(baseline["numeric_tolerance"]["abs"])
    for field_name, expected_value in expected_robustness.items():
        observed_value = robustness.get(field_name)
        if not isinstance(observed_value, (int, float)):
            non_conformities.append(f"robustness.{field_name}:missing_or_non_numeric")
            continue
        if abs(float(observed_value) - float(expected_value)) > tolerance:
            non_conformities.append(f"robustness.{field_name}")

    if non_conformities:
        return "NON_CONFORME", non_conformities, minor_drift
    if minor_drift:
        return "DRIFT_MINEUR", non_conformities, minor_drift
    return "REPRODUCIBLE", non_conformities, minor_drift


def test_dca_canonical_rerun_reproducibility_status(tmp_path: Path) -> None:
    baseline_dir = Path("tests/fixtures/dca_canonical_baseline")
    baseline = json.loads((baseline_dir / "baseline.json").read_text())
    baseline_checksums = _extract_checksum_map(baseline_dir / "checksums.txt")

    dataset_path = tmp_path / "canonical_dataset.json"
    _build_canonical_dataset(dataset_path)

    out_dir = tmp_path / "canonical_run"
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

    manifest_path = out_dir / "run_manifest.json"
    checksums_path = out_dir / "checksums.txt"
    robustness_path = out_dir / "dca_robustness_v1.json"

    manifest = json.loads(manifest_path.read_text())
    robustness = json.loads(robustness_path.read_text())
    checksums = _extract_checksum_map(checksums_path)
    observed_hashes = {
        "dca_robustness_v1.json": _sha256(robustness_path),
        "run_manifest.json": _sha256(manifest_path),
    }

    baseline_with_fixture_checksums = dict(baseline)
    baseline_with_fixture_checksums["checksums"] = baseline_checksums

    status, non_conformities, minor_drift = _classify_reproducibility(
        baseline=baseline_with_fixture_checksums,
        manifest=manifest,
        robustness=robustness,
        checksums=checksums,
        observed_hashes=observed_hashes,
    )

    status_payload = {
        "status": status,
        "non_conformities": non_conformities,
        "minor_drift": minor_drift,
    }
    status_path = out_dir / "canonical_repro_status.json"
    status_path.write_text(json.dumps(status_payload, indent=2, sort_keys=True))

    assert status_path.exists()
    assert json.loads(status_path.read_text())["status"] == baseline["expected_status"]
