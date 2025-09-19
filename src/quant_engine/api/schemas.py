"""API request/response models (light-weight)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field

from ..core.spec import (
    ValidationSpec as CoreValidationSpec,
    ArtifactsSpec as CoreArtifactsSpec,
    PersistenceSpec as CorePersistenceSpec,
)


@dataclass
class SubmitResponse:
    id: str


@dataclass
class StatusResponse:
    status: str


@dataclass
class ResultResponse:
    result: Dict[str, Any] | None


class StatsDataSpec(BaseModel):
    """Dataset location and window for statistics runs."""

    dataset_path: str
    symbols: List[str]
    timeframe: str
    start: str
    end: str


class StatsEventSpec(BaseModel):
    """Specification for an event to detect in the dataset."""

    name: str
    params: Dict[str, Any] = Field(default_factory=dict)


class StatsConditionSpec(BaseModel):
    """Specification for conditioning regime."""

    name: str
    params: Dict[str, Any] = Field(default_factory=dict)


class StatsTargetSpec(BaseModel):
    """Specification for a target metric."""

    name: str
    params: Dict[str, Any] = Field(default_factory=dict)


class StatsSpec(BaseModel):
    """Top-level specification for a statistics run."""

    data: StatsDataSpec
    events: List[StatsEventSpec] = Field(default_factory=list)
    conditions: List[StatsConditionSpec] = Field(default_factory=list)
    targets: List[StatsTargetSpec] = Field(default_factory=list)
    validation: CoreValidationSpec | None = None
    artifacts: CoreArtifactsSpec | None = None
    persistence: CorePersistenceSpec | None = None


class ExecutionSpec(BaseModel):
    """Light-weight execution configuration for seasonality runs."""

    slippage_bps: float = 0.0
    commission_bps: float = 0.0


class RiskSpec(BaseModel):
    """Basic risk settings used when generating the signal."""

    max_positions: int = 1
    max_allocation: float = 1.0


class TPSSLSpec(BaseModel):
    """Simple take-profit/stop-loss configuration placeholder."""

    take_profit: float | None = None
    stop_loss: float | None = None


class ValidationSpec(BaseModel):
    """Validation configuration for seasonality workflows."""

    min_trades: int = 0
    train_months: int = 0
    test_months: int = 1
    folds: int = 1
    embargo_days: int = 0


class ArtifactsSpec(BaseModel):
    """Configuration describing where to persist generated artifacts."""

    out_dir: str | None = None


class PersistenceSpec(BaseModel):
    """Minimal database persistence settings."""

    enabled: bool = False
    spec_id: str | None = None
    dataset_id: str | None = None


class SeasonalityDataSpec(BaseModel):
    dataset_path: str
    symbols: list[str]
    timeframe: str
    start: str
    end: str


class SeasonalityProfileSpec(BaseModel):
    # quelles dimensions de saisonnalité calculer
    by_hour: bool = True
    by_dow: bool = True
    by_month: bool = False
    # mesure : 'direction' (P(close_{t+1} > close_t)) ou 'return'
    measure: Literal["direction", "return"] = "direction"
    ret_horizon: int = 1  # nb de barres à regarder
    min_samples_bin: int = 300  # n_min par bin


class SeasonalitySignalSpec(BaseModel):
    # comment transformer les profils en signal tradable
    method: Literal["threshold", "topk"] = "threshold"
    threshold: float = 0.54  # si measure=direction alors p_hat>=seuil
    topk: int = 3  # si method=topk: prendre K bins les + forts
    dims: list[Literal["hour", "dow", "month"]] = ["hour", "dow"]
    combine: Literal["and", "or", "sum"] = "and"  # combine multi-dims


class SeasonalityComputeSpec(BaseModel):
    """Configuration pour la boucle d'optimisation."""

    max_trials: int = 30
    search_space: Dict[str, Any] = Field(default_factory=dict)


class SeasonalitySpec(BaseModel):
    data: SeasonalityDataSpec
    profile: SeasonalityProfileSpec = SeasonalityProfileSpec()
    signal: SeasonalitySignalSpec = SeasonalitySignalSpec()
    compute: SeasonalityComputeSpec = SeasonalityComputeSpec()
    execution: ExecutionSpec = ExecutionSpec()
    risk: RiskSpec = RiskSpec()
    tp_sl: TPSSLSpec = TPSSLSpec()
    validation: ValidationSpec = ValidationSpec()
    artifacts: ArtifactsSpec = ArtifactsSpec()
    persistence: PersistenceSpec = PersistenceSpec()

