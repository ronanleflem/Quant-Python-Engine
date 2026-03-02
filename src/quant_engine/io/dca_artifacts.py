from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator


SCHEMA_VERSION = "dca-grid-process-v1"
CONTRACT_VERSION = "1.0.0"
DEFAULT_UNIVERSE_RULES_VERSION = "asset-universe-rules-v1"


class ReproducibilityMetadata(BaseModel):
    seed: int
    dataset_hash: str
    config_version: str


class ContractMetadata(BaseModel):
    schema_version: str = SCHEMA_VERSION
    contract_version: str = CONTRACT_VERSION
    universe_rules_version: str = DEFAULT_UNIVERSE_RULES_VERSION
    seed: int | None = None
    dataset_hash: str | None = None
    config_version: str | None = None
    generated_at: str
    reproducibility: ReproducibilityMetadata

    @model_validator(mode="after")
    def _populate_flat_reproducibility_fields(self) -> "ContractMetadata":
        if self.seed is None:
            self.seed = self.reproducibility.seed
        if self.dataset_hash is None:
            self.dataset_hash = self.reproducibility.dataset_hash
        if self.config_version is None:
            self.config_version = self.reproducibility.config_version
        return self


class ArtifactEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact: str
    metadata: ContractMetadata
    rows: List[Dict[str, Any]] = Field(default_factory=list)


def compute_dataset_hash(rows: List[Dict[str, Any]]) -> str:
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_metadata(
    *,
    generated_at: str,
    seed: int,
    dataset_hash: str,
    config_version: str,
    universe_rules_version: str | None = None,
) -> ContractMetadata:
    return ContractMetadata(
        universe_rules_version=str(universe_rules_version or DEFAULT_UNIVERSE_RULES_VERSION),
        seed=seed,
        dataset_hash=dataset_hash,
        config_version=config_version,
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
    universe_rules_version: str | None = None,
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
        universe_rules_version=universe_rules_version,
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

    plausibility_constraints = {
        "min_percentile": 0.50,
        "min_perf_dominance_prob": 0.60,
        "min_drawdown_dominance_prob": 0.60,
        "requires_dominance_flag": True,
    }
    best_candidate: Dict[str, Any] = {}
    if metrics_rows:
        best_candidate = max(metrics_rows, key=lambda item: float(item.get("lift_freq", 0.0) or 0.0))
    percentile_value = float(robustness.get("percentile", 0.0) or 0.0)
    dominance_payload = robustness.get("dominance") or {}
    perf_prob = float(dominance_payload.get("perf_dominance_prob", 0.0) or 0.0)
    drawdown_prob = float(dominance_payload.get("drawdown_dominance_prob", 0.0) or 0.0)
    is_dominant = bool(dominance_payload.get("is_dominant", False))
    passes_constraints = (
        bool(best_candidate)
        and percentile_value >= plausibility_constraints["min_percentile"]
        and perf_prob >= plausibility_constraints["min_perf_dominance_prob"]
        and drawdown_prob >= plausibility_constraints["min_drawdown_dominance_prob"]
        and is_dominant
    )
    best_plausible_passive_ex_ante_rows = [
        {
            "selection_method": "max_lift_freq_under_plausibility",
            "symbol": best_candidate.get("symbol"),
            "target": best_candidate.get("target"),
            "n": int(best_candidate.get("n", 0) or 0),
            "successes": int(best_candidate.get("successes", 0) or 0),
            "lift_freq": float(best_candidate.get("lift_freq", 0.0) or 0.0),
            "lift_bayes": float(best_candidate.get("lift_bayes", 0.0) or 0.0),
            "percentile": percentile_value,
            "perf_dominance_prob": perf_prob,
            "drawdown_dominance_prob": drawdown_prob,
            "is_dominant": is_dominant,
            "constraints": plausibility_constraints,
            "passes_constraints": passes_constraints,
        }
    ]

    artifacts = {
        "metrics": metrics_rows,
        "distributions": distributions_rows,
        "capital_curves": capital_curve,
        "rolling": rolling_rows,
        "score": score_rows,
        "best_plausible_passive_ex_ante": best_plausible_passive_ex_ante_rows,
    }

    written: Dict[str, Dict[str, str]] = {}
    for name, artifact_rows in artifacts.items():
        envelope = ArtifactEnvelope(artifact=name, metadata=meta, rows=artifact_rows)
        written[name] = _write_json_and_parquet(out_path, envelope)
    return written
