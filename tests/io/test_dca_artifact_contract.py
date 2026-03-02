import json
from datetime import datetime, timezone

import pandas as pd

from quant_engine.io.dca_artifacts import (
    DEFAULT_UNIVERSE_RULES_VERSION,
    SCHEMA_VERSION,
    ArtifactEnvelope,
    write_dca_contract_artifacts,
)


def test_dca_artifact_contract_writes_json_and_parquet(tmp_path):
    out_dir = tmp_path / "artifacts"
    dataset = [{"timestamp": "2024-01-01T00:00:00Z", "symbol": "AAPL", "close": 100.0}]
    summary = pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "target": "up_next_bar",
                "n": 10,
                "successes": 6,
                "lift_freq": 0.12,
                "lift_bayes": 0.11,
            }
        ]
    )
    robustness = {
        "quantiles": {"p50": 0.5},
        "percentile": 0.75,
        "dominance": {"perf_dominance_prob": 0.8, "drawdown_dominance_prob": 0.7, "is_dominant": True},
    }

    written = write_dca_contract_artifacts(
        out_dir,
        generated_at=datetime.now(timezone.utc).isoformat(),
        seed=7,
        dataset_rows=dataset,
        config_version=SCHEMA_VERSION,
        universe_rules_version="asset-universe-rules-v2",
        stats_summary=summary,
        robustness=robustness,
    )

    assert set(written.keys()) == {"metrics", "distributions", "capital_curves", "rolling", "score"}
    for name, paths in written.items():
        assert out_dir.joinpath(f"{name}.json").exists()
        assert out_dir.joinpath(f"{name}.parquet").exists()
        payload = json.loads(out_dir.joinpath(f"{name}.json").read_text())
        parsed = ArtifactEnvelope.model_validate(payload)
        assert parsed.metadata.schema_version == SCHEMA_VERSION
        assert parsed.metadata.reproducibility.seed == 7
        assert parsed.metadata.universe_rules_version == "asset-universe-rules-v2"
        assert paths["json"].endswith(f"{name}.json")
        assert paths["parquet"].endswith(f"{name}.parquet")


def test_artifact_envelope_back_compat_without_universe_rules_version():
    payload = {
        "artifact": "metrics",
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "contract_version": "1.0.0",
            "generated_at": "2026-01-01T00:00:00+00:00",
            "reproducibility": {
                "seed": 42,
                "dataset_hash": "abc",
                "config_version": SCHEMA_VERSION,
            },
        },
        "rows": [],
    }

    parsed = ArtifactEnvelope.model_validate(payload)

    assert parsed.metadata.universe_rules_version == DEFAULT_UNIVERSE_RULES_VERSION
