"""Canonical run request contract used by the Java -> Python entrypoint."""
from __future__ import annotations

from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, TypeAdapter


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class DataRangeBlock(StrictModel):
    symbol: str
    timeframe: str
    start_date: str = Field(validation_alias=AliasChoices("start_date", "startDate"))
    end_date: str = Field(validation_alias=AliasChoices("end_date", "endDate"))
    dataset_path: Optional[str] = None
    path: Optional[str] = None
    mysql: Optional[Dict[str, Any]] = None
    symbols: Optional[List[str]] = None


class MarketStatsDataBlock(StrictModel):
    symbol: str
    timeframe: str
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
    symbol: str
    timeframe: str
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
    method: Optional[str] = None
    n_sims: Optional[int] = Field(default=None, validation_alias=AliasChoices("n_sims", "nSims"))
    seed: Optional[int] = None
    block_size: Optional[int] = Field(default=None, validation_alias=AliasChoices("block_size", "blockSize"))


class PerformanceBlock(StrictModel):
    initial_capital: Optional[float] = Field(
        default=None, validation_alias=AliasChoices("initial_capital", "initialCapital")
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
    data: DataRangeBlock
    strategy: DcaStrategyBlock


class MarketStatsRunRequest(RunRequestCommon):
    spec_type: Literal["market_stats"]
    data: MarketStatsDataBlock
    stats: MarketStatsBlock


class SeasonalityRunRequest(RunRequestCommon):
    spec_type: Literal["seasonality"]
    data: SeasonalityDataBlock
    seasonality: SeasonalityBlock


class StressTestsRunRequest(RunRequestCommon):
    spec_type: Literal["stress_tests"]
    data: DataRangeBlock
    strategy: Optional[Dict[str, Any]] = None


RunRequestInput = Annotated[
    Union[
        BacktestRunRequest,
        DcaRunRequest,
        MarketStatsRunRequest,
        SeasonalityRunRequest,
        StressTestsRunRequest,
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
