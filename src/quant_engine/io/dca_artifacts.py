from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


SCHEMA_VERSION = "dca-grid-process-v1"
CONTRACT_VERSION = "1.0.0"


class ReproducibilityMetadata(BaseModel):
    seed: int
    dataset_hash: str
    config_version: str


class ContractMetadata(BaseModel):
    schema_version: str = SCHEMA_VERSION
    contract_version: str = CONTRACT_VERSION
    generated_at: str
    reproducibility: ReproducibilityMetadata


class ArtifactEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact: str
    metadata: ContractMetadata
    rows: List[Dict[str, Any]] = Field(default_factory=list)


def compute_dataset_hash(rows: List[Dict[str, Any]]) -> str:
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_metadata(*, generated_at: str, seed: int, dataset_hash: str, config_version: str) -> ContractMetadata:
    return ContractMetadata(
        generated_at=generated_at,
        reproducibility=ReproducibilityMetadata(seed=seed, dataset_hash=dataset_hash, config_version=config_version),
    )


def _write_json_and_parquet(out_dir: Path, envelope: ArtifactEnvelope) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{envelope.artifact}.json"
    parquet_path = out_dir / f"{envelope.artifact}.parquet"
    json_path.write_text(envelope.model_dump_json(indent=2), encoding="utf-8")

    df = pd.DataFrame(envelope.rows)
    if df.empty:
        df = pd.DataFrame([{}])
    df.to_parquet(parquet_path, index=False)
    return {"json": str(json_path), "parquet": str(parquet_path)}


def write_dca_contract_artifacts(
    out_dir: str | Path,
    *,
    generated_at: str,
    seed: int,
    dataset_rows: List[Dict[str, Any]],
    config_version: str,
    stats_summary: pd.DataFrame,
    robustness: Dict[str, Any],
) -> Dict[str, Dict[str, str]]:
    out_path = Path(out_dir)
    dataset_hash = compute_dataset_hash(dataset_rows)
    meta = build_metadata(
        generated_at=generated_at,
        seed=seed,
        dataset_hash=dataset_hash,
        config_version=config_version,
    )

    rows = stats_summary.to_dict("records")
    metrics_rows = [
        {
            "symbol": r.get("symbol"),
            "target": r.get("target"),
            "n": int(r.get("n", 0) or 0),
            "successes": int(r.get("successes", 0) or 0),
            "lift_freq": float(r.get("lift_freq", 0.0) or 0.0),
            "lift_bayes": float(r.get("lift_bayes", 0.0) or 0.0),
        }
        for r in rows
    ]
    distributions_rows = [
        {
            "quantile": k,
            "value": float(v),
        }
        for k, v in (robustness.get("quantiles") or {}).items()
    ]

    capital_curve = []
    cumulative = 0.0
    for idx, row in enumerate(metrics_rows):
        cumulative += float(row.get("lift_freq", 0.0))
        capital_curve.append({"step": idx, "equity_index": cumulative})

    rolling_rows = []
    window = 5
    for idx in range(len(metrics_rows)):
        start = max(0, idx - window + 1)
        values = [float(m.get("lift_freq", 0.0)) for m in metrics_rows[start : idx + 1]]
        rolling_rows.append({"step": idx, "window": window, "rolling_lift_freq": sum(values) / len(values) if values else 0.0})

    score_rows = [
        {
            "percentile": float(robustness.get("percentile", 0.0) or 0.0),
            "perf_dominance_prob": float((robustness.get("dominance") or {}).get("perf_dominance_prob", 0.0) or 0.0),
            "drawdown_dominance_prob": float((robustness.get("dominance") or {}).get("drawdown_dominance_prob", 0.0) or 0.0),
            "is_dominant": bool((robustness.get("dominance") or {}).get("is_dominant", False)),
        }
    ]

    artifacts = {
        "metrics": metrics_rows,
        "distributions": distributions_rows,
        "capital_curves": capital_curve,
        "rolling": rolling_rows,
        "score": score_rows,
    }

    written: Dict[str, Dict[str, str]] = {}
    for name, artifact_rows in artifacts.items():
        envelope = ArtifactEnvelope(artifact=name, metadata=meta, rows=artifact_rows)
        written[name] = _write_json_and_parquet(out_path, envelope)
    return written
