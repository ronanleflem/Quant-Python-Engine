"""API request/response models (light-weight)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, ConfigDict

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


class MySQLDataSpec(BaseModel):
    """MySQL data feed configuration."""

    model_config = ConfigDict(protected_namespaces=())

    connection_url: Optional[str] = None
    env_var: Optional[str] = "QE_MARKETDATA_MYSQL_URL"
    schema_: Optional[str] = Field("marketdata", alias="schema")
    table: str = "ohlcv"
    symbol_col: str = "symbol"
    ts_col: str = "ts"
    open_col: str = "open"
    high_col: str = "high"
    low_col: str = "low"
    close_col: str = "close"
    volume_col: str = "volume"
    timeframe_col: Optional[str] = "timeframe"
    extra_where: Optional[str] = None
    chunk_minutes: int = 0

    @property
    def schema(self) -> Optional[str]:
        return self.schema_


class DataInputSpec(BaseModel):
    dataset_path: Optional[str] = None
    mysql: Optional[MySQLDataSpec] = None
    symbols: List[str]
    timeframe: str
    start: str
    end: str


class StatsDataSpec(DataInputSpec):
    """Dataset location and window for statistics runs."""


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


class SeasonalityDataSpec(DataInputSpec):
    pass


class SeasonalityProfileSpec(BaseModel):
    # quelles dimensions de saisonnalité calculer
    by_hour: bool = True
    by_dow: bool = True
    by_month: bool = False
    by_session: bool = False
    by_month_start: bool = False
    by_month_end: bool = False
    by_news_hour: bool = False
    by_rollover_day: bool = False
    by_third_friday: bool = False
    by_week_in_month: bool = False
    by_day_in_month: bool = False
    by_month_last_days: bool = False
    by_quarter: bool = False
    by_month_flags: bool = False
    # mesure : 'direction' (P(close_{t+1} > close_t)) ou 'return'
    measure: Literal["direction", "return"] = "direction"
    ret_horizon: int = 1  # nb de barres à regarder
    min_samples_bin: int = 300  # n_min par bin


class SeasonalitySignalSpec(BaseModel):
    # comment transformer les profils en signal tradable
    method: Literal["threshold", "topk"] = "threshold"
    threshold: float = 0.54  # si measure=direction alors p_hat>=seuil
    topk: int = 3  # si method=topk: prendre K bins les + forts
    dims: list[
        Literal[
            "hour",
            "dow",
            "month",
            "month_of_year",
            "session",
            "is_month_start",
            "is_month_end",
            "is_news_hour",
            "is_rollover_day",
            "is_third_friday",
            "week_in_month",
            "day_in_month",
            "quarter",
            "last_1",
            "last_2",
            "last_3",
            "last_4",
            "last_5",
            "is_january",
            "is_february",
            "is_march",
            "is_april",
            "is_may",
            "is_june",
            "is_july",
            "is_august",
            "is_september",
            "is_october",
            "is_november",
            "is_december",
        ]
    ] = ["hour", "dow"]
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

