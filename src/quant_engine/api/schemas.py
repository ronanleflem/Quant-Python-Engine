"""API request/response models (light-weight)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, ConfigDict, model_validator

from ..core.spec import (
    ValidationSpec as CoreValidationSpec,
    ArtifactsSpec as CoreArtifactsSpec,
)


@dataclass
class SubmitResponse:
    id: str


@dataclass
class StatusResponse:
    status: str
    id: Optional[str] = None


@dataclass
class ResultResponse:
    result: Dict[str, Any] | None


@dataclass
class RunEnqueueResponse:
    run_id: str
    status: str
    reused: bool = False


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
    symbol_lookup_table: Optional[str] = None
    symbol_lookup_symbol_col: str = "symbol"
    symbol_lookup_id_col: str = "id"

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
    asset_class: Optional[str] = None
    currency: Optional[str] = None
    delta_base: Optional[str] = None
    delta_prefix: Optional[str] = None
    delta_exchange: Optional[str] = None
    delta_market_type: Optional[str] = None
    delta_quotes: Optional[str] = None
    delta_broker: Optional[str] = None
    delta_brokers: Optional[List[str]] = None
    delta_asset_dir: Optional[str] = None
    delta_table: Optional[str] = None
    delta_symbol: Optional[str] = None
    delta_calendar: Optional[str] = None
    delta_min_coverage: Optional[float] = None


class StatsDataSpec(DataInputSpec):
    """Dataset location and window for statistics runs."""


class StatsEventSpec(BaseModel):
    """Specification for an event to detect in the dataset."""

    name: str
    type: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)


class StatsConditionSpec(BaseModel):
    """Specification for conditioning regime."""

    name: str
    type: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)


class FilterConditionSpec(BaseModel):
    """Specification for applying a pre-trade filter."""

    type: Literal[
        "adx",
        "atr",
        "ema_slope",
        "volume_surge",
        "vwap_side",
        "poc_distance",
        "liquidity_sweep",
        "bos",
        "mss",
        "session_time",
        "day_of_week",
        "day_of_month",
        "month_of_year",
        "intraday_time",
        "k_consecutive",
        "seasonality_bin",
        "hurst_regime",
        "entropy_window",
        "daily_loss_cap",
        "daily_trades_cap",
        "cooldown_bars",
        "atr_risk_gate",
        "equity_dd_lockout",
        "benford_law",
        "cycles",
        "donchian_channels",
        "liquidity_cmf",
        "statistical_arbitrage",
        "psychologic_ulcer",
        "stationarity",
        "volatility",
        "ema_structure",
        "rsi_entry",
        "macd_entry",
        "volume_above_average",
        "fractal_analysis",
        "mean_reversion",
        "contradictory_signals",
        "linear_regression_macd_cross",
        "atr_rising",
        "market_regime",
        "trend",
        "biais_institutional",
        "stats_gate",
    ]
    params: Dict[str, Any] = Field(default_factory=dict)


class StatsTargetSpec(BaseModel):
    """Specification for a target metric."""

    name: str
    type: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)


class StatsPersistenceSpec(BaseModel):
    """Persistence settings for statistics runs."""

    model_config = ConfigDict(protected_namespaces=())

    store_stats_in_db: Optional[bool] = None
    enabled: Optional[bool] = None
    spec_id: Optional[str] = None
    dataset_id: Optional[str] = None

    @model_validator(mode="after")
    def _resolve_enabled(self) -> "StatsPersistenceSpec":
        if self.enabled is None:
            if self.store_stats_in_db is not None:
                self.enabled = bool(self.store_stats_in_db)
            else:
                self.enabled = False
        return self


class StatsSpec(BaseModel):
    """Top-level specification for a statistics run."""

    data: StatsDataSpec
    events: List[StatsEventSpec] = Field(default_factory=list)
    conditions: List[StatsConditionSpec] = Field(default_factory=list)
    targets: List[StatsTargetSpec] = Field(default_factory=list)
    validation: CoreValidationSpec | None = None
    artifacts: CoreArtifactsSpec | None = None
    persistence: StatsPersistenceSpec | None = None


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


class LiveDataSpec(BaseModel):
    """Specification of the live data feed for on-bar-close execution."""

    model_config = ConfigDict(protected_namespaces=(), populate_by_name=True)

    mysql_read_env: str = "QE_MARKETDATA_MYSQL_URL"
    schema_: str = Field("marketdata", alias="schema")
    table: str = "ohlcv"
    symbol_col: str = "symbol"
    ts_col: str = "ts"
    open_col: str = "open"
    high_col: str = "high"
    low_col: str = "low"
    close_col: str = "close"
    volume_col: str = "volume"
    symbols: List[str]
    timeframe: str
    warmup_bars: int = 300
    poll_interval_sec: int = 5
    timeframe_col: str | None = None
    scans: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def schema(self) -> str:
        return self.schema_


class LiveFilterSpec(BaseModel):
    type: str
    params: Dict[str, Any] = Field(default_factory=dict)


class LiveRuleSpec(BaseModel):
    type: str
    params: Dict[str, Any] = Field(default_factory=dict)


class LiveRiskGateSpec(BaseModel):
    type: str
    params: Dict[str, Any] = Field(default_factory=dict)


class LiveTPSLMgmtSpec(BaseModel):
    type: str
    params: Dict[str, Any] = Field(default_factory=dict)


class StrategyImplSpec(BaseModel):
    type: str
    params: Dict[str, Any] = Field(default_factory=dict)


class LiveStrategySpec(BaseModel):
    strategy_id: str
    filters: List[LiveFilterSpec] = Field(default_factory=list)
    rules: List[LiveRuleSpec] = Field(default_factory=list)
    risk_gates: List[LiveRiskGateSpec] = Field(default_factory=list)
    tp_sl_mgmt: LiveTPSLMgmtSpec | None = None
    impl: StrategyImplSpec | None = None


class LiveWriteDBSpec(BaseModel):
    model_config = ConfigDict(protected_namespaces=(), populate_by_name=True)

    mysql_write_env: str = "QE_WRITE_MYSQL_URL"
    schema_: str = Field("quant", alias="schema")
    table: str = "trades_live"

    @property
    def schema(self) -> str:
        return self.schema_


class LiveJavaEmitSpec(BaseModel):
    enabled: bool = False
    url_env: str = "QE_JAVA_LIVE_URL"
    path: str = "/live/signal"


class LiveDestinationsSpec(BaseModel):
    write_db: LiveWriteDBSpec | None = None
    emit_java: LiveJavaEmitSpec | None = None


class LiveSpec(BaseModel):
    data: LiveDataSpec
    strategy: LiveStrategySpec
    destinations: LiveDestinationsSpec | None = None


class SeasonalityDataSpec(DataInputSpec):
    pass


class SeasonalityProfileSpec(BaseModel):
    # quelles dimensions de saisonnalitÃ© calculer
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
    ret_horizon: int = 1  # nb de barres Ã  regarder
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
