"""Canonical run request contract used by the Java -> Python entrypoint."""
from __future__ import annotations

from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, TypeAdapter, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class DataRangeBlock(StrictModel):
    symbol: str
    timeframe: str
    currency: Optional[str] = None
    start_date: str = Field(validation_alias=AliasChoices("start_date", "startDate"))
    end_date: str = Field(validation_alias=AliasChoices("end_date", "endDate"))
    dataset_path: Optional[str] = None
    path: Optional[str] = None
    mysql: Optional[Dict[str, Any]] = None
    symbols: Optional[List[str]] = None


class MarketStatsDataBlock(StrictModel):
    symbol: Optional[str] = None
    timeframe: str
    start_date: Optional[str] = Field(default=None, validation_alias=AliasChoices("start_date", "startDate"))
    end_date: Optional[str] = Field(default=None, validation_alias=AliasChoices("end_date", "endDate"))
    asset_class: Optional[str] = Field(default=None, validation_alias=AliasChoices("asset_class", "assetClass"))
    currency: Optional[str] = None
    lookback: Optional[int] = None
    stats_pack: Optional[str] = Field(default=None, validation_alias=AliasChoices("stats_pack", "statsPack"))
    session: Optional[str] = None
    include_weekends: Optional[bool] = Field(
        default=None, validation_alias=AliasChoices("include_weekends", "includeWeekends")
    )
    dataset_path: Optional[str] = None
    path: Optional[str] = None
    mysql: Optional[Dict[str, Any]] = None
    symbols: Optional[List[str]] = None


class SeasonalityDataBlock(StrictModel):
    symbol: Optional[str] = None
    timeframe: str
    start_date: Optional[str] = Field(default=None, validation_alias=AliasChoices("start_date", "startDate"))
    end_date: Optional[str] = Field(default=None, validation_alias=AliasChoices("end_date", "endDate"))
    asset_class: Optional[str] = Field(default=None, validation_alias=AliasChoices("asset_class", "assetClass"))
    currency: Optional[str] = None
    window: Optional[str] = None
    start_year: Optional[int] = Field(default=None, validation_alias=AliasChoices("start_year", "startYear"))
    end_year: Optional[int] = Field(default=None, validation_alias=AliasChoices("end_year", "endYear"))
    dataset_path: Optional[str] = None
    path: Optional[str] = None
    mysql: Optional[Dict[str, Any]] = None
    symbols: Optional[List[str]] = None


class FilterSpec(StrictModel):
    id: str
    params: Dict[str, Any] = Field(default_factory=dict)


class FilterRuleSpec(StrictModel):
    id: str
    mode: Optional[str] = None
    weight: Optional[float] = None


class FilterRulesConfig(StrictModel):
    min_score: Optional[float] = Field(default=None, validation_alias=AliasChoices("min_score", "minScore"))
    min_score_pct: Optional[float] = Field(
        default=None, validation_alias=AliasChoices("min_score_pct", "minScorePct")
    )


class FiltersBlock(StrictModel):
    filters: List[FilterSpec] = Field(default_factory=list)
    rules: List[FilterRuleSpec] = Field(default_factory=list)
    rules_config: Optional[FilterRulesConfig] = Field(
        default=None, validation_alias=AliasChoices("rules_config", "rulesConfig")
    )


class BacktestSignalBlock(StrictModel):
    type: str
    fast: Optional[int] = None
    slow: Optional[int] = None
    require_crossing: Optional[bool] = Field(
        default=None, validation_alias=AliasChoices("require_crossing", "requireCrossing")
    )


class BacktestStrategyBlock(StrictModel):
    name: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)


class DcaStrategyBlock(StrictModel):
    type: str
    grid: List[str] = Field(default_factory=list)
    params: Dict[str, Any] = Field(default_factory=dict)


class DcaDataBlock(StrictModel):
    symbol: Optional[str] = None
    timeframe: str
    currency: Optional[str] = None
    start_date: str = Field(validation_alias=AliasChoices("start_date", "startDate"))
    end_date: str = Field(validation_alias=AliasChoices("end_date", "endDate"))
    dataset_path: Optional[str] = None
    path: Optional[str] = None
    mysql: Optional[Dict[str, Any]] = None
    symbols: Optional[List[str]] = None


class DcaUniverseItem(StrictModel):
    symbol: str
    asset_class: Optional[str] = Field(default=None, validation_alias=AliasChoices("asset_class", "assetClass"))
    name: Optional[str] = None
    exchange: Optional[str] = None
    currency: Optional[str] = None
    broker: Optional[str] = None
    market_type: Optional[str] = Field(default=None, validation_alias=AliasChoices("market_type", "marketType"))


class MarketLeafSpec(StrictModel):
    id: str
    params: Dict[str, Any] = Field(default_factory=dict)


class MarketStatsBlock(StrictModel):
    event: MarketLeafSpec
    condition: MarketLeafSpec
    target: MarketLeafSpec
    validation: Optional[Dict[str, Any]] = None


class SeasonalityBlock(StrictModel):
    profile: Dict[str, Any]
    signal: Dict[str, Any]
    compute: Optional[Dict[str, Any]] = None
    execution: Optional[Dict[str, Any]] = None
    risk: Optional[Dict[str, Any]] = None
    tp_sl: Optional[Dict[str, Any]] = Field(default=None, validation_alias=AliasChoices("tp_sl", "tpSl"))


class StressTestsBlock(StrictModel):
    enabled: Optional[bool] = None
    source: Optional[str] = None
    method: Optional[str] = None
    n_sims: Optional[int] = Field(default=None, validation_alias=AliasChoices("n_sims", "nSims"))
    seed: Optional[int] = None
    block_size: Optional[int] = Field(default=None, validation_alias=AliasChoices("block_size", "blockSize"))
    overlapping: Optional[bool] = None
    time_distribution: Optional[Dict[str, Any]] = Field(
        default=None, validation_alias=AliasChoices("time_distribution", "timeDistribution", "time_dist")
    )
    param_drift: Optional[Dict[str, Any]] = Field(default=None, validation_alias=AliasChoices("param_drift", "paramDrift"))
    sizing: Optional[Dict[str, Any]] = None
    output: Optional[Dict[str, Any]] = None
    scenarios: Optional[List[Dict[str, Any]] | Dict[str, Any]] = None
    multi_asset: Optional[Dict[str, Any]] = Field(default=None, validation_alias=AliasChoices("multi_asset", "multiAsset"))
    aggregation: Optional[str] = None
    weights: Optional[List[float] | str] = None
    timestamp_alignment: Optional[str] = Field(
        default=None, validation_alias=AliasChoices("timestamp_alignment", "timestampAlignment")
    )


class PerformanceBlock(StrictModel):
    initial_capital: Optional[float] = Field(
        default=None, validation_alias=AliasChoices("initial_capital", "initialCapital")
    )
    capital_per_unit: Optional[float] = Field(
        default=None, validation_alias=AliasChoices("capital_per_unit", "capitalPerUnit")
    )
    max_capital_per_trade: Optional[float] = Field(
        default=None, validation_alias=AliasChoices("max_capital_per_trade", "maxCapitalPerTrade")
    )
    risk_pct: Optional[float] = Field(default=None, validation_alias=AliasChoices("risk_pct", "riskPct"))
    risk_free_rate_pct: Optional[float] = Field(
        default=None, validation_alias=AliasChoices("risk_free_rate_pct", "riskFreeRatePct")
    )
    stress_tests: Optional[StressTestsBlock] = Field(
        default=None, validation_alias=AliasChoices("stress_tests", "stressTests")
    )


class RunRequestCommon(StrictModel):
    catalog_version: str = Field(validation_alias=AliasChoices("catalog_version", "catalogVersion"))
    request_id: Optional[str] = Field(default=None, validation_alias=AliasChoices("request_id", "requestId"))
    output: Optional[Dict[str, Any]] = None
    persistence: Optional[Dict[str, Any]] = None
    filters: Optional[FiltersBlock] = None
    performance: Optional[PerformanceBlock] = None


class BacktestRunRequest(RunRequestCommon):
    spec_type: Literal["backtest"]
    data: DataRangeBlock
    signal: BacktestSignalBlock
    strategy: Optional[BacktestStrategyBlock] = None


class DcaRunRequest(RunRequestCommon):
    spec_type: Literal["dca"]
    data: DcaDataBlock
    strategy: DcaStrategyBlock
    universe: List[DcaUniverseItem] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_symbol_or_universe(self) -> "DcaRunRequest":
        has_symbol = bool((self.data.symbol or "").strip()) if self.data is not None else False
        has_universe = bool(self.universe)
        if not has_symbol and not has_universe:
            raise ValueError("dca requires data.symbol or universe")
        return self


class MarketStatsRunRequest(RunRequestCommon):
    spec_type: Literal["market_stats"]
    data: MarketStatsDataBlock
    stats: MarketStatsBlock

    @model_validator(mode="after")
    def _validate_symbol_or_symbols(self) -> "MarketStatsRunRequest":
        has_symbol = bool((self.data.symbol or "").strip()) if self.data is not None else False
        has_symbols = bool(self.data.symbols) if self.data is not None else False
        if not has_symbol and not has_symbols:
            raise ValueError("market_stats requires data.symbol or data.symbols")
        return self


class SeasonalityRunRequest(RunRequestCommon):
    spec_type: Literal["seasonality"]
    data: SeasonalityDataBlock
    seasonality: SeasonalityBlock

    @model_validator(mode="after")
    def _validate_symbol_or_symbols(self) -> "SeasonalityRunRequest":
        has_symbol = bool((self.data.symbol or "").strip()) if self.data is not None else False
        has_symbols = bool(self.data.symbols) if self.data is not None else False
        if not has_symbol and not has_symbols:
            raise ValueError("seasonality requires data.symbol or data.symbols")
        return self


class StressTestsDataBlock(StrictModel):
    base_run_id: str = Field(
        validation_alias=AliasChoices(
            "base_run_id",
            "baseRunId",
            "source_run_id",
            "sourceRunId",
        )
    )


class StressTestsRunRequest(RunRequestCommon):
    spec_type: Literal["stress_tests"]
    data: StressTestsDataBlock
    strategy: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def _validate_stress_tests_block(self) -> "StressTestsRunRequest":
        base_run_id = (self.data.base_run_id or "").strip() if self.data is not None else ""
        if not base_run_id:
            raise ValueError("stress_tests requires data.base_run_id")
        stress_block = self.performance.stress_tests if self.performance is not None else None
        if stress_block is None:
            raise ValueError("stress_tests requires performance.stress_tests")
        return self


class OptimizationObjectiveBlock(StrictModel):
    metric: str
    direction: Literal["max", "min"]


class OptimizationBudgetBlock(StrictModel):
    max_trials: int = Field(validation_alias=AliasChoices("max_trials", "maxTrials"))
    timeout_seconds: Optional[int] = Field(
        default=None,
        validation_alias=AliasChoices("timeout_seconds", "timeoutSeconds"),
    )
    seed: Optional[int] = None

    @model_validator(mode="after")
    def _validate_budget_bounds(self) -> "OptimizationBudgetBlock":
        if int(self.max_trials) < 1:
            raise ValueError("optimization.budget.max_trials must be >= 1")
        if self.timeout_seconds is not None and int(self.timeout_seconds) < 1:
            raise ValueError("optimization.budget.timeout_seconds must be >= 1")
        if self.seed is not None and int(self.seed) < 0:
            raise ValueError("optimization.budget.seed must be >= 0")
        return self


class OptimizationBlock(StrictModel):
    base_run_id: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("base_run_id", "baseRunId"),
    )
    base_spec: Optional[Dict[str, Any]] = None
    search_space: Dict[str, Any] = Field(validation_alias=AliasChoices("search_space", "searchSpace"))
    objective: OptimizationObjectiveBlock
    budget: OptimizationBudgetBlock

    @model_validator(mode="after")
    def _validate_base_and_search_space(self) -> "OptimizationBlock":
        base_run_id = (self.base_run_id or "").strip()
        base_spec = self.base_spec if isinstance(self.base_spec, dict) else None
        if not base_run_id and not base_spec:
            raise ValueError("optimization requires optimization.base_run_id or optimization.base_spec")
        if not isinstance(self.search_space, dict) or len(self.search_space) == 0:
            raise ValueError("optimization.search_space must be a non-empty object")
        return self


class OptimizeBacktestRunRequest(RunRequestCommon):
    spec_type: Literal["optimize_backtest"]
    optimization: OptimizationBlock

    @model_validator(mode="after")
    def _validate_base_spec_type(self) -> "OptimizeBacktestRunRequest":
        base_spec = self.optimization.base_spec if self.optimization is not None else None
        if isinstance(base_spec, dict):
            base_spec_type = str(base_spec.get("spec_type") or "").strip().lower()
            if base_spec_type and base_spec_type != "backtest":
                raise ValueError("optimize_backtest requires optimization.base_spec.spec_type=backtest")
        return self


class OptimizeDcaRunRequest(RunRequestCommon):
    spec_type: Literal["optimize_dca"]
    optimization: OptimizationBlock

    @model_validator(mode="after")
    def _validate_base_spec_type(self) -> "OptimizeDcaRunRequest":
        base_spec = self.optimization.base_spec if self.optimization is not None else None
        if isinstance(base_spec, dict):
            base_spec_type = str(base_spec.get("spec_type") or "").strip().lower()
            if base_spec_type and base_spec_type != "dca":
                raise ValueError("optimize_dca requires optimization.base_spec.spec_type=dca")
        return self


RunRequestInput = Annotated[
    Union[
        BacktestRunRequest,
        DcaRunRequest,
        MarketStatsRunRequest,
        SeasonalityRunRequest,
        StressTestsRunRequest,
        OptimizeBacktestRunRequest,
        OptimizeDcaRunRequest,
    ],
    Field(discriminator="spec_type"),
]

run_request_input_adapter = TypeAdapter(RunRequestInput)


def validate_run_request_input(payload: Dict[str, Any]) -> RunRequestInput:
    """Validate and parse a canonical run request payload."""
    if "spec_type" not in payload and "specType" in payload:
        payload = dict(payload)
        payload["spec_type"] = payload.get("specType")
        payload.pop("specType", None)
    return run_request_input_adapter.validate_python(payload)
