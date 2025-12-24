"""API contract for strategy stress tests.

This module defines a unified interface for running stress tests and returning
results that can be stored in ``StrategyRunResult.extra``. Implementations can
plug into the placeholders to provide Monte Carlo or deterministic scenario
analysis while keeping a consistent payload schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from math import ceil
import random
from statistics import mean, pstdev
from typing import Any, Dict, List, Mapping, Optional, Sequence, TypedDict, Union

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .models import CompletedTrade


TimeSeries = Union[Sequence[float], Mapping[datetime, float]]
MultiAssetReturns = Mapping[str, TimeSeries]

METRIC_NAME_MAP = {
    "maxDrawdown": "max_drawdown",
    "max_drawdown": "max_drawdown",
    "maxDrawdownPct": "max_drawdown_pct",
    "max_drawdown_pct": "max_drawdown_pct",
    "cagr": "cagr",
    "cagr_pct": "cagr",
    "ruinProbability": "ruin_probability",
    "ruin_probability": "ruin_probability",
    "time_to_recovery": "time_to_recovery_days",
    "time_to_recovery_days": "time_to_recovery_days",
}

EXPECTED_STRESS_TEST_METRICS: Dict[str, Dict[str, List[str]]] = {
    "monte_carlo": {
        "summary": ["max_drawdown", "max_drawdown_pct", "cagr", "ruin_probability", "time_to_recovery_days"],
        "level1": [
            "final_capital",
            "return_pct",
            "max_drawdown",
            "max_drawdown_pct",
            "volatility_pct",
            "sharpe",
            "sortino",
            "winrate_pct",
            "total_return",
            "average_trade",
            "win_count",
            "loss_count",
        ],
    },
    "scenarios": {
        "level1": [
            "final_capital",
            "return_pct",
            "max_drawdown",
            "max_drawdown_pct",
            "volatility_pct",
            "sharpe",
            "sortino",
            "winrate_pct",
            "total_return",
            "average_trade",
            "win_count",
            "loss_count",
        ]
    },
}


def _normalize_metric_names(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for key, value in metrics.items():
        target = METRIC_NAME_MAP.get(key, key)
        normalized[target] = value
    return normalized


def _normalize_scenario_metrics(metrics_map: Mapping[str, Any]) -> Dict[str, Any]:
    return {name: _normalize_metric_names(metrics) for name, metrics in metrics_map.items()}


class StressTestResult(TypedDict, total=False):
    """Standardized output keys for stress tests.

    Keys
    ----
    metrics:
        Aggregated figures such as percentiles, worst-case loss, or VaR.
    distributions:
        Raw distributions or arrays produced by the stress test (e.g. Monte Carlo
        returns, scenario PnL paths).
    parameters:
        Parameters actually used for the run (sample size, scenario shocks,
        random seed, horizon, etc.).
    warnings:
        Human-readable warnings for data quality or configuration issues.
    """

    metrics: Dict[str, Any]
    distributions: Dict[str, Any]
    parameters: Dict[str, Any]
    warnings: List[str]


class ScenarioDefinition(TypedDict, total=False):
    name: str
    type: str
    description: str
    shock_pct: float
    vol_multiplier: float
    drawdown_pct: float
    window: int
    index: Union[int, str]
    start_index: Optional[int]


class ScenarioConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    type: str
    description: Optional[str] = None
    shock_pct: Optional[float] = None
    vol_multiplier: Optional[float] = None
    drawdown_pct: Optional[float] = None
    window: Optional[int] = None
    index: Optional[Union[int, str]] = None
    start_index: Optional[int] = None

    @model_validator(mode="after")
    def validate_scenario_fields(self) -> "ScenarioConfig":
        scenario_type = self.type.lower()
        if scenario_type in {"crash", "gap"} and self.shock_pct is None:
            raise ValueError(f"Scenario '{self.name}' requires shock_pct for type '{self.type}'.")
        if scenario_type in {"vol", "volatility", "volatility_spike"} and self.vol_multiplier is None:
            raise ValueError(f"Scenario '{self.name}' requires vol_multiplier for type '{self.type}'.")
        if scenario_type in {"drawdown", "prolonged_drawdown"}:
            if self.drawdown_pct is None:
                raise ValueError(f"Scenario '{self.name}' requires drawdown_pct for type '{self.type}'.")
            if self.window is None:
                raise ValueError(f"Scenario '{self.name}' requires window for type '{self.type}'.")
        return self


class MonteCarloConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    n_sims: int = Field(
        1_000,
        validation_alias=AliasChoices("n_sims", "n_simulations"),
    )
    seed: Optional[int] = 42
    method: str = "bootstrap"
    block_size: int = Field(5, validation_alias=AliasChoices("block_size", "blockSize"))
    overlapping: bool = True
    initial_capital: float = 0.0
    ruin_threshold: float = 0.0
    multi_asset: bool = False

    @field_validator("n_sims", "block_size")
    @classmethod
    def validate_positive_ints(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("n_sims and block_size must be positive integers.")
        return value

    @field_validator("method")
    @classmethod
    def validate_method(cls, value: str) -> str:
        method = value.lower()
        allowed = {"bootstrap", "iid", "shuffle", "block", "block_bootstrap"}
        if method not in allowed:
            raise ValueError(f"method must be one of {sorted(allowed)}.")
        return method


class StressTestConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    monte_carlo: MonteCarloConfig
    scenarios: List[ScenarioConfig]


@dataclass(frozen=True)
class StandardTrade:
    pnl: float
    r_multiple: Optional[float]
    entry_time_utc: Optional[datetime]
    exit_time_utc: Optional[datetime]
    symbol: Optional[str] = None


def _parse_dt(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.utcfromtimestamp(float(value))
        except Exception:
            return None
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except Exception:
            try:
                from pandas import to_datetime

                return to_datetime(value, utc=True).to_pydatetime()
            except Exception:
                return None
    return None


def standardize_trades(trades: Sequence[Union[CompletedTrade, Mapping[str, Any]]]) -> List[StandardTrade]:
    """Normalize trade inputs (DCA/strategies) to a common format.

    Expected fields include pnl, r_multiple, and entry/exit timestamps.
    """

    normalized: List[StandardTrade] = []
    for tr in trades:
        if isinstance(tr, CompletedTrade):
            r_multiple = None
            if isinstance(tr.meta, dict):
                r_multiple = tr.meta.get("r_multiple")
            normalized.append(
                StandardTrade(
                    pnl=float(tr.gross_pnl),
                    r_multiple=float(r_multiple) if r_multiple is not None else None,
                    entry_time_utc=_parse_dt(tr.entry_time_utc),
                    exit_time_utc=_parse_dt(tr.exit_time_utc),
                    symbol=tr.symbol,
                )
            )
            continue

        if isinstance(tr, Mapping):
            pnl = tr.get("pnl")
            if pnl is None:
                pnl = tr.get("gross_pnl")
            r_multiple = tr.get("r_multiple")
            entry_time = tr.get("ts_entry") or tr.get("entry_time") or tr.get("entryTimeUtc")
            exit_time = tr.get("ts_exit") or tr.get("exit_time") or tr.get("exitTimeUtc")
            symbol = tr.get("symbol")
            if pnl is None:
                continue
            normalized.append(
                StandardTrade(
                    pnl=float(pnl),
                    r_multiple=float(r_multiple) if r_multiple is not None else None,
                    entry_time_utc=_parse_dt(entry_time),
                    exit_time_utc=_parse_dt(exit_time),
                    symbol=str(symbol) if symbol is not None else None,
                )
            )
    return normalized


def _percentile(values: Sequence[float], pct: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = (len(ordered) - 1) * pct
    low = int(idx)
    high = min(low + 1, len(ordered) - 1)
    frac = idx - low
    return ordered[low] + (ordered[high] - ordered[low]) * frac


def _max_drawdown(equity: Sequence[float]) -> float:
    peak = float("-inf")
    max_dd = 0.0
    for value in equity:
        peak = max(peak, value)
        max_dd = max(max_dd, peak - value)
    return max_dd


def _time_to_recovery(
    equity: Sequence[float], timestamps: Optional[Sequence[datetime]]
) -> Optional[float]:
    if not equity:
        return None
    max_duration = 0.0
    peak = equity[0]
    peak_time = timestamps[0] if timestamps else None
    drawdown_start: Optional[datetime] = None
    drawdown_start_idx: Optional[int] = None
    for idx, value in enumerate(equity[1:], start=1):
        current_time = timestamps[idx] if timestamps else None
        if value >= peak:
            if drawdown_start is not None:
                duration = (current_time - drawdown_start).total_seconds() if current_time else (idx - drawdown_start_idx)
                max_duration = max(max_duration, duration)
                drawdown_start = None
                drawdown_start_idx = None
            peak = value
            peak_time = current_time
            continue
        if drawdown_start is None:
            drawdown_start = peak_time
            drawdown_start_idx = idx
    if drawdown_start is not None:
        end_time = timestamps[-1] if timestamps else None
        duration = (end_time - drawdown_start).total_seconds() if end_time else (len(equity) - 1 - drawdown_start_idx)
        max_duration = max(max_duration, duration)
    if timestamps:
        return max_duration / 86400.0
    return float(max_duration)


def _build_timestamps(trades: Sequence[StandardTrade]) -> Optional[List[datetime]]:
    exit_times = [t.exit_time_utc for t in trades if t.exit_time_utc is not None]
    if len(exit_times) != len(trades):
        return None
    return exit_times


def _build_sampled_timestamps(
    base_start: datetime, deltas: Sequence[timedelta], sample_size: int, rng: random.Random
) -> List[datetime]:
    if not deltas:
        return [base_start + timedelta(days=i) for i in range(sample_size)]
    timestamps = [base_start]
    for _ in range(sample_size - 1):
        delta = rng.choice(deltas)
        timestamps.append(timestamps[-1] + delta)
    return timestamps


def _summary_stats(values: Sequence[float]) -> Dict[str, Optional[float]]:
    return {
        "p5": _percentile(values, 0.05),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "mean": mean(values) if values else None,
        "std": pstdev(values) if len(values) > 1 else None,
    }


def _block_start_indices(sample_size: int, block_size: int, overlapping: bool) -> List[int]:
    if block_size <= 0 or sample_size <= 0 or block_size > sample_size:
        return []
    if overlapping:
        return list(range(0, sample_size - block_size + 1))
    return list(range(0, sample_size - block_size + 1, block_size))


def _equity_timestamps_from_trades(trades: Sequence[StandardTrade]) -> Optional[List[datetime]]:
    exit_times = [t.exit_time_utc for t in trades if t.exit_time_utc is not None]
    if len(exit_times) != len(trades) or not exit_times:
        return None
    return [exit_times[0]] + exit_times


def _is_datetime_key(value: Any) -> bool:
    return _parse_dt(value) is not None


def _normalize_timeseries(returns: TimeSeries) -> tuple[List[float], Optional[List[datetime]]]:
    if isinstance(returns, Mapping):
        ordered = sorted(returns.items(), key=lambda item: _parse_dt(item[0]) or datetime.min)
        timestamps = [_parse_dt(ts) for ts, _ in ordered]
        values = [float(val) for _, val in ordered]
        return values, timestamps
    return [float(val) for val in returns], None


def _is_multi_asset_returns(returns: Any) -> bool:
    if not isinstance(returns, Mapping):
        return False
    if not returns:
        return False
    sample_key = next(iter(returns.keys()))
    if _is_datetime_key(sample_key):
        return False
    sample_value = next(iter(returns.values()))
    return isinstance(sample_value, (Mapping, Sequence)) and not isinstance(sample_value, (str, bytes))


def _normalize_multi_asset_returns(
    returns: MultiAssetReturns,
) -> tuple[Dict[str, List[float]], Dict[str, Optional[List[datetime]]]]:
    normalized: Dict[str, List[float]] = {}
    timestamps: Dict[str, Optional[List[datetime]]] = {}
    for symbol, series in returns.items():
        values, ts = _normalize_timeseries(series)
        normalized[str(symbol)] = values
        timestamps[str(symbol)] = ts
    return normalized, timestamps


def _aggregate_multi_asset_returns(
    per_asset: Mapping[str, List[float]],
    timestamps: Mapping[str, Optional[List[datetime]]],
) -> tuple[List[float], Optional[List[datetime]]]:
    if per_asset and all(timestamps.get(symbol) for symbol in per_asset):
        combined: Dict[datetime, float] = {}
        for symbol, values in per_asset.items():
            ts_list = timestamps.get(symbol) or []
            for ts, value in zip(ts_list, values):
                if ts is None:
                    continue
                combined[ts] = combined.get(ts, 0.0) + float(value)
        if combined:
            ordered = sorted(combined.items(), key=lambda item: item[0])
            return [val for _, val in ordered], [ts for ts, _ in ordered]

    max_len = max((len(values) for values in per_asset.values()), default=0)
    aggregated = [0.0] * max_len
    for values in per_asset.values():
        for idx, value in enumerate(values):
            aggregated[idx] += float(value)
    return aggregated, None


def _estimate_scale(returns: Sequence[float]) -> float:
    if returns:
        return max(mean([abs(val) for val in returns]), 1e-9)
    return 1.0


def _shock_value(shock_pct: float, returns: Sequence[float], initial_capital: float) -> float:
    shock_pct = -abs(shock_pct)
    scale = initial_capital if initial_capital else _estimate_scale(returns)
    return scale * shock_pct


def _scenario_index(returns: Sequence[float], scenario: Mapping[str, Any]) -> int:
    if not returns:
        return 0
    index = scenario.get("index", "mid")
    if isinstance(index, int):
        return max(0, min(index, len(returns) - 1))
    if str(index).lower() == "start":
        return 0
    if str(index).lower() == "end":
        return len(returns) - 1
    return len(returns) // 2


def _apply_scenario_to_returns(
    returns: Sequence[float],
    scenario: Mapping[str, Any],
    *,
    initial_capital: float,
) -> List[float]:
    adjusted = [float(val) for val in returns]
    if not adjusted:
        return adjusted

    scenario_type = str(scenario.get("type", "")).lower()
    if scenario_type in {"crash", "gap"}:
        shock_pct = float(scenario.get("shock_pct", -0.2))
        idx = _scenario_index(adjusted, scenario)
        adjusted[idx] += _shock_value(shock_pct, adjusted, initial_capital)
        return adjusted

    if scenario_type in {"vol", "volatility", "volatility_spike"}:
        multiplier = float(scenario.get("vol_multiplier", 2.0))
        avg = mean(adjusted)
        return [avg + (val - avg) * multiplier for val in adjusted]

    if scenario_type in {"drawdown", "prolonged_drawdown"}:
        window = max(int(scenario.get("window", 10)), 1)
        drawdown_pct = float(scenario.get("drawdown_pct", -0.2))
        shock_total = _shock_value(drawdown_pct, adjusted, initial_capital)
        start_idx = scenario.get("start_index")
        if start_idx is None:
            start_idx = max((len(adjusted) - window) // 2, 0)
        start_idx = max(0, min(int(start_idx), max(len(adjusted) - window, 0)))
        end_idx = min(start_idx + window, len(adjusted))
        per_step = shock_total / max(end_idx - start_idx, 1)
        for idx in range(start_idx, end_idx):
            adjusted[idx] += per_step
        return adjusted

    return adjusted


def _build_equity_curve(returns: Sequence[float], initial_capital: float) -> List[float]:
    equity = [initial_capital]
    for value in returns:
        equity.append(equity[-1] + float(value))
    return equity


def _parse_scenarios(params: Mapping[str, Any]) -> List[ScenarioConfig]:
    scenarios = params.get("scenarios")
    if scenarios is None:
        scenarios = _default_scenarios(params)
    if isinstance(scenarios, Mapping) or isinstance(scenarios, (str, bytes)):
        raise ValueError("Scenarios must be provided as a list of scenario definitions.")
    if not isinstance(scenarios, Sequence):
        raise ValueError("Scenarios must be provided as a list of scenario definitions.")
    if not scenarios:
        raise ValueError("Scenario list cannot be empty.")
    try:
        return [ScenarioConfig.model_validate(scenario) for scenario in scenarios]
    except ValidationError as exc:
        raise ValueError(f"Invalid scenario configuration: {exc}") from exc


def _parse_monte_carlo_config(parameters: Optional[Mapping[str, Any]]) -> MonteCarloConfig:
    params = parameters or {}
    if not isinstance(params, Mapping):
        raise ValueError("Monte Carlo parameters must be provided as a mapping.")
    try:
        return MonteCarloConfig.model_validate(params)
    except ValidationError as exc:
        raise ValueError(f"Invalid Monte Carlo configuration: {exc}") from exc


def _default_scenarios(parameters: Mapping[str, Any]) -> List[ScenarioDefinition]:
    return [
        {
            "name": "crash",
            "type": "crash",
            "description": "Single-period crash shock applied mid-series.",
            "shock_pct": float(parameters.get("crash_shock_pct", -0.25)),
            "index": parameters.get("crash_index", "mid"),
        },
        {
            "name": "gap",
            "type": "gap",
            "description": "Opening gap down applied at the start of the series.",
            "shock_pct": float(parameters.get("gap_shock_pct", -0.1)),
            "index": parameters.get("gap_index", "start"),
        },
        {
            "name": "vol_x2",
            "type": "volatility",
            "description": "Volatility spike with returns scaled by 2x.",
            "vol_multiplier": float(parameters.get("vol_multiplier", 2.0)),
        },
        {
            "name": "drawdown_prolonged",
            "type": "drawdown",
            "description": "Prolonged drawdown applied across a rolling window.",
            "drawdown_pct": float(parameters.get("drawdown_pct", -0.2)),
            "window": int(parameters.get("drawdown_window", 10)),
            "start_index": parameters.get("drawdown_start_index"),
        },
    ]


def apply_scenarios_to_returns(
    returns: Union[TimeSeries, MultiAssetReturns],
    *,
    parameters: Optional[Mapping[str, Any]] = None,
) -> StressTestResult:
    """Apply deterministic scenarios to a single or multi-asset returns series."""

    params = parameters or {}
    if not isinstance(params, Mapping):
        raise ValueError("Scenario parameters must be provided as a mapping.")
    scenarios = _parse_scenarios(params)
    initial_capital = float(params.get("initial_capital", 0.0))
    warnings: List[str] = []

    scenario_results: Dict[str, Any] = {}
    scenario_metrics: Dict[str, Any] = {}

    if _is_multi_asset_returns(returns):
        per_asset, timestamps = _normalize_multi_asset_returns(returns)
        if not per_asset:
            warnings.append("No returns available for scenarios.")
        for scenario in scenarios:
            scenario_data = scenario.model_dump()
            name = str(scenario_data.get("name", "scenario"))
            per_asset_results: Dict[str, Any] = {}
            adjusted_assets: Dict[str, List[float]] = {}
            for symbol, values in per_asset.items():
                adjusted = _apply_scenario_to_returns(values, scenario_data, initial_capital=initial_capital)
                adjusted_assets[symbol] = adjusted
                metrics = _normalize_metric_names(
                    _compute_level1_metrics(adjusted, _build_equity_curve(adjusted, initial_capital), initial_capital)
                )
                per_asset_results[symbol] = {
                    "returns": adjusted,
                    "timestamps": timestamps.get(symbol),
                    "metrics": metrics,
                    "parameters": scenario_data,
                }

            portfolio_returns, portfolio_ts = _aggregate_multi_asset_returns(adjusted_assets, timestamps)
            portfolio_metrics = _normalize_metric_names(
                _compute_level1_metrics(
                    portfolio_returns, _build_equity_curve(portfolio_returns, initial_capital), initial_capital
                )
            )
            scenario_results[name] = {
                "portfolio_level": {
                    "metrics": portfolio_metrics,
                    "returns": portfolio_returns,
                    "timestamps": portfolio_ts,
                    "parameters": scenario_data,
                },
                "by_symbol": per_asset_results,
            }
            scenario_metrics[name] = portfolio_metrics
    else:
        values, timestamps = _normalize_timeseries(returns)
        if not values:
            warnings.append("No returns available for scenarios.")
        for scenario in scenarios:
            scenario_data = scenario.model_dump()
            name = str(scenario_data.get("name", "scenario"))
            adjusted = _apply_scenario_to_returns(values, scenario_data, initial_capital=initial_capital)
            metrics = _normalize_metric_names(
                _compute_level1_metrics(adjusted, _build_equity_curve(adjusted, initial_capital), initial_capital)
            )
            scenario_results[name] = {
                "metrics": metrics,
                "returns": adjusted,
                "timestamps": timestamps,
                "parameters": scenario_data,
            }
            scenario_metrics[name] = metrics

    return {
        "metrics": {"scenarios": _normalize_scenario_metrics(scenario_metrics)},
        "distributions": {"scenarios": scenario_results},
        "parameters": {
            "initial_capital": initial_capital,
            "scenarios": [scenario.model_dump() for scenario in scenarios],
        },
        "warnings": warnings,
    }


def _compute_level1_metrics(
    pnl_values: Sequence[float],
    equity: Sequence[float],
    initial_capital: float,
) -> Dict[str, Optional[float]]:
    if not equity:
        return {
            "final_capital": None,
            "return_pct": None,
            "max_drawdown": None,
            "max_drawdown_pct": None,
            "volatility_pct": None,
            "sharpe": None,
            "sortino": None,
            "winrate_pct": None,
            "total_return": None,
            "average_trade": None,
        }

    returns_pct: List[float] = []
    if initial_capital:
        returns_pct = [pnl / initial_capital * 100.0 for pnl in pnl_values]

    win_count = sum(1 for r in returns_pct if r > 0)
    loss_count = sum(1 for r in returns_pct if r <= 0)
    nb_trades = len(returns_pct)
    total_return = sum(returns_pct) if returns_pct else None
    average_trade = (total_return / nb_trades) if nb_trades and total_return is not None else None

    final_capital = equity[-1]
    return_pct = ((final_capital - initial_capital) / initial_capital * 100.0) if initial_capital else None

    max_drawdown = _max_drawdown(equity)
    peak = equity[0]
    max_dd_value = 0.0
    for value in equity[1:]:
        peak = max(peak, value)
        max_dd_value = max(max_dd_value, peak - value)
    max_drawdown_pct = (max_dd_value / peak * 100.0) if peak else None

    volatility_pct = pstdev(returns_pct) if len(returns_pct) > 1 else None
    mean_ret = mean(returns_pct) if returns_pct else None
    sharpe = None
    sortino = None
    if volatility_pct and volatility_pct != 0 and mean_ret is not None:
        sharpe = mean_ret / volatility_pct * (len(returns_pct) ** 0.5)
    if returns_pct:
        downside = [r for r in returns_pct if r < 0]
        if len(downside) > 1:
            downside_std = pstdev(downside)
            if downside_std:
                sortino = mean_ret / downside_std * (len(returns_pct) ** 0.5) if mean_ret is not None else None

    winrate_pct = (win_count / nb_trades * 100.0) if nb_trades else None

    return {
        "final_capital": final_capital,
        "return_pct": return_pct,
        "max_drawdown": max_drawdown,
        "max_drawdown_pct": max_drawdown_pct,
        "volatility_pct": volatility_pct,
        "sharpe": sharpe,
        "sortino": sortino,
        "winrate_pct": winrate_pct,
        "total_return": total_return,
        "average_trade": average_trade,
        "win_count": float(win_count),
        "loss_count": float(loss_count),
    }


def _sample_iid(trades: Sequence[StandardTrade], rng: random.Random) -> List[StandardTrade]:
    indices = list(range(len(trades)))
    rng.shuffle(indices)
    return [trades[i] for i in indices]


def _sample_bootstrap(trades: Sequence[StandardTrade], rng: random.Random) -> List[StandardTrade]:
    return [rng.choice(trades) for _ in range(len(trades))]


def _sample_block(
    trades: Sequence[StandardTrade],
    rng: random.Random,
    *,
    block_size: int,
    overlapping: bool,
) -> List[StandardTrade]:
    sample_size = len(trades)
    starts = _block_start_indices(sample_size, block_size, overlapping)
    if not starts:
        return _sample_bootstrap(trades, rng)
    num_blocks = ceil(sample_size / block_size)
    indices: List[int] = []
    for _ in range(num_blocks):
        start = rng.choice(starts)
        indices.extend(range(start, min(start + block_size, sample_size)))
    indices = indices[:sample_size]
    return [trades[i] for i in indices]


def _sample_block_multi_asset(
    trades: Sequence[StandardTrade],
    rng: random.Random,
    *,
    block_size: int,
    overlapping: bool,
) -> List[StandardTrade]:
    by_symbol: Dict[str, List[StandardTrade]] = {}
    for tr in trades:
        if tr.symbol is None:
            return _sample_block(trades, rng, block_size=block_size, overlapping=overlapping)
        by_symbol.setdefault(tr.symbol, []).append(tr)

    all_exit_times = [t.exit_time_utc for t in trades if t.exit_time_utc is not None]
    if len(all_exit_times) != len(trades):
        return _sample_block(trades, rng, block_size=block_size, overlapping=overlapping)

    sorted_times = sorted(set(all_exit_times))
    starts = _block_start_indices(len(sorted_times), block_size, overlapping)
    if not starts:
        return _sample_block(trades, rng, block_size=block_size, overlapping=overlapping)

    num_blocks = ceil(len(sorted_times) / block_size)
    sampled: List[StandardTrade] = []
    ordered_by_symbol = {
        symbol: sorted(symbol_trades, key=lambda t: t.exit_time_utc)
        for symbol, symbol_trades in by_symbol.items()
    }
    for _ in range(num_blocks):
        start_idx = rng.choice(starts)
        end_idx = min(start_idx + block_size, len(sorted_times)) - 1
        start_ts = sorted_times[start_idx]
        end_ts = sorted_times[end_idx]
        for symbol, ordered in ordered_by_symbol.items():
            for tr in ordered:
                if tr.exit_time_utc is not None and start_ts <= tr.exit_time_utc <= end_ts:
                    sampled.append(tr)
    sampled.sort(key=lambda t: t.exit_time_utc or datetime.min)
    if not sampled:
        return _sample_block(trades, rng, block_size=block_size, overlapping=overlapping)
    return sampled


def _monte_carlo_bootstrap(
    trades: Sequence[StandardTrade],
    *,
    parameters: Optional[Mapping[str, Any]] = None,
) -> StressTestResult:
    config = _parse_monte_carlo_config(parameters)
    n_simulations = config.n_sims
    seed = config.seed
    initial_capital = float(config.initial_capital)
    ruin_threshold = float(config.ruin_threshold)
    method = config.method
    block_size = config.block_size
    overlapping = config.overlapping
    multi_asset = config.multi_asset
    rng = random.Random(seed)

    if not trades:
        return {
            "metrics": {},
            "distributions": {},
            "parameters": {
                "n_simulations": n_simulations,
                "seed": seed,
                "method": method,
            },
            "warnings": ["No trades available for bootstrap."],
        }

    base_timestamps = _build_timestamps(trades)
    base_start = base_timestamps[0] if base_timestamps else datetime.utcnow()
    deltas: List[timedelta] = []
    if base_timestamps:
        for prev, nxt in zip(base_timestamps[:-1], base_timestamps[1:]):
            deltas.append(nxt - prev)

    sample_size = len(trades)
    pnl_values = [t.pnl for t in trades]

    equity_curves: List[List[float]] = []
    max_drawdowns: List[float] = []
    cagrs: List[float] = []
    time_to_recovery: List[float] = []
    ruin_flags: List[bool] = []
    level1_metrics: Dict[str, List[float]] = {
        "final_capital": [],
        "return_pct": [],
        "max_drawdown": [],
        "max_drawdown_pct": [],
        "volatility_pct": [],
        "sharpe": [],
        "sortino": [],
        "winrate_pct": [],
        "total_return": [],
        "average_trade": [],
        "win_count": [],
        "loss_count": [],
    }

    base_period_days: Optional[float] = None
    if base_timestamps:
        base_period_days = (base_timestamps[-1] - base_timestamps[0]).total_seconds() / 86400.0

    ordered_trades = sorted(
        trades, key=lambda t: t.exit_time_utc if t.exit_time_utc is not None else datetime.min
    )

    for _ in range(n_simulations):
        if method in {"iid", "shuffle"}:
            sampled_trades = _sample_iid(ordered_trades, rng)
        elif method in {"block", "block_bootstrap"}:
            if multi_asset:
                sampled_trades = _sample_block_multi_asset(
                    ordered_trades, rng, block_size=block_size, overlapping=overlapping
                )
            else:
                sampled_trades = _sample_block(ordered_trades, rng, block_size=block_size, overlapping=overlapping)
        else:
            sampled_trades = _sample_bootstrap(ordered_trades, rng)

        sampled_pnl = [t.pnl for t in sampled_trades]
        equity = [initial_capital]
        for pnl in sampled_pnl:
            equity.append(equity[-1] + pnl)
        equity_curves.append(equity)

        max_drawdowns.append(_max_drawdown(equity))

        timestamps = _equity_timestamps_from_trades(sampled_trades)
        if timestamps is None and base_timestamps:
            timestamps = _build_sampled_timestamps(base_start, deltas, len(equity), rng)
        ttr = _time_to_recovery(equity, timestamps)
        if ttr is not None:
            time_to_recovery.append(float(ttr))

        ruin_flags.append(min(equity) <= ruin_threshold)

        years = None
        if base_period_days is not None and base_period_days > 0:
            years = base_period_days / 365.25
        else:
            years = sample_size / 252.0 if sample_size else None
        if years and equity[0] > 0 and equity[-1] > 0:
            cagr = (equity[-1] / equity[0]) ** (1 / years) - 1
            cagrs.append(cagr)

        level1 = _compute_level1_metrics(sampled_pnl, equity, initial_capital)
        for key, value in level1.items():
            if value is not None:
                level1_metrics[key].append(float(value))

    ruin_probability = sum(ruin_flags) / len(ruin_flags) if ruin_flags else None

    metrics = {
        "max_drawdown": _summary_stats(max_drawdowns),
        "cagr": _summary_stats(cagrs),
        "time_to_recovery_days": _summary_stats(time_to_recovery),
        "ruin_probability": ruin_probability,
    }

    for key, values in level1_metrics.items():
        metrics[key] = _summary_stats(values)

    metrics = _normalize_metric_names(metrics)
    normalized_level1 = _normalize_metric_names(level1_metrics)

    distributions = {
        "max_drawdown": max_drawdowns,
        "cagr": cagrs,
        "time_to_recovery_days": time_to_recovery,
        "equity_curves": equity_curves,
        "ruin": ruin_flags,
        "level1": normalized_level1,
    }

    return {
        "metrics": metrics,
        "distributions": distributions,
        "parameters": {
            "n_simulations": n_simulations,
            "seed": seed,
            "initial_capital": initial_capital,
            "ruin_threshold": ruin_threshold,
            "method": method,
            "block_size": block_size,
            "overlapping": overlapping,
            "multi_asset": multi_asset,
        },
    }


def run_monte_carlo_on_trades(
    trades: Sequence[CompletedTrade],
    *,
    metadata: Optional[Mapping[str, Any]] = None,
    parameters: Optional[Mapping[str, Any]] = None,
) -> StressTestResult:
    """Run Monte Carlo stress tests using the completed trades.

    Parameters
    ----------
    trades:
        Sequence of ``CompletedTrade`` describing the run.
    metadata:
        Optional contextual metadata (strategy id, asset class, etc.).
    parameters:
        Optional configuration for the Monte Carlo run (sample size, horizon,
        random seed, bootstrap method, etc.).

    Returns
    -------
    StressTestResult
        Dict compatible with ``StrategyRunResult.extra``.
    """

    normalized = standardize_trades(trades)
    result = _monte_carlo_bootstrap(normalized, parameters=parameters)
    if metadata:
        result.setdefault("parameters", {}).update({"metadata": dict(metadata)})
    return result


def run_monte_carlo_on_returns(
    returns: TimeSeries,
    *,
    metadata: Optional[Mapping[str, Any]] = None,
    parameters: Optional[Mapping[str, Any]] = None,
) -> StressTestResult:
    """Run Monte Carlo stress tests using the returns series.

    Parameters
    ----------
    returns:
        Sequence or mapping of returns (chronological order if sequence).
    metadata:
        Optional contextual metadata (strategy id, asset class, etc.).
    parameters:
        Optional configuration for the Monte Carlo run (sample size, horizon,
        random seed, bootstrap method, etc.).

    Returns
    -------
    StressTestResult
        Dict compatible with ``StrategyRunResult.extra``.
    """

    if isinstance(returns, Mapping):
        ordered = sorted(returns.items(), key=lambda item: item[0])
        trades = [
            StandardTrade(
                pnl=float(value),
                r_multiple=None,
                entry_time_utc=_parse_dt(ts),
                exit_time_utc=_parse_dt(ts),
                symbol=None,
            )
            for ts, value in ordered
        ]
    else:
        trades = [
            StandardTrade(pnl=float(value), r_multiple=None, entry_time_utc=None, exit_time_utc=None, symbol=None)
            for value in returns
        ]
    result = _monte_carlo_bootstrap(trades, parameters=parameters)
    if metadata:
        result.setdefault("parameters", {}).update({"metadata": dict(metadata)})
    return result


def run_monte_carlo_on_equity_curve(
    equity_curve: TimeSeries,
    *,
    metadata: Optional[Mapping[str, Any]] = None,
    parameters: Optional[Mapping[str, Any]] = None,
) -> StressTestResult:
    """Run Monte Carlo stress tests using the equity curve.

    Parameters
    ----------
    equity_curve:
        Sequence or mapping of equity values (chronological order if sequence).
    metadata:
        Optional contextual metadata (strategy id, asset class, etc.).
    parameters:
        Optional configuration for the Monte Carlo run (sample size, horizon,
        random seed, bootstrap method, etc.).

    Returns
    -------
    StressTestResult
        Dict compatible with ``StrategyRunResult.extra``.
    """

    if isinstance(equity_curve, Mapping):
        ordered = sorted(equity_curve.items(), key=lambda item: item[0])
        pnl_series: List[StandardTrade] = []
        prev_value: Optional[float] = None
        prev_ts: Optional[datetime] = None
        for ts, value in ordered:
            val = float(value)
            if prev_value is None:
                prev_value = val
                prev_ts = _parse_dt(ts)
                continue
            pnl_series.append(
                StandardTrade(
                    pnl=val - prev_value,
                    r_multiple=None,
                    entry_time_utc=prev_ts,
                    exit_time_utc=_parse_dt(ts),
                    symbol=None,
                )
            )
            prev_value = val
            prev_ts = _parse_dt(ts)
    else:
        pnl_series = []
        prev_value = None
        for value in equity_curve:
            val = float(value)
            if prev_value is None:
                prev_value = val
                continue
            pnl_series.append(
                StandardTrade(pnl=val - prev_value, r_multiple=None, entry_time_utc=None, exit_time_utc=None, symbol=None)
            )
            prev_value = val
    if parameters is None:
        parameters = {}
    if parameters.get("initial_capital") is None:
        if isinstance(equity_curve, Mapping):
            ordered = sorted(equity_curve.items(), key=lambda item: item[0])
            if ordered:
                parameters = dict(parameters)
                parameters["initial_capital"] = float(ordered[0][1])
        else:
            try:
                first_value = next(iter(equity_curve))
                parameters = dict(parameters)
                parameters["initial_capital"] = float(first_value)
            except StopIteration:
                pass
    result = _monte_carlo_bootstrap(pnl_series, parameters=parameters)
    if metadata:
        result.setdefault("parameters", {}).update({"metadata": dict(metadata)})
    return result


def run_scenarios_on_trades(
    trades: Sequence[CompletedTrade],
    *,
    metadata: Optional[Mapping[str, Any]] = None,
    parameters: Optional[Mapping[str, Any]] = None,
) -> StressTestResult:
    """Run deterministic scenario stress tests using the completed trades.

    Parameters
    ----------
    trades:
        Sequence of ``CompletedTrade`` describing the run.
    metadata:
        Optional contextual metadata (strategy id, asset class, etc.).
    parameters:
        Optional scenario definitions (shock size, volatility spike, etc.).

    Returns
    -------
    StressTestResult
        Dict compatible with ``StrategyRunResult.extra``.
    """
    params = parameters or {}
    normalized = standardize_trades(trades)
    if not normalized:
        result: StressTestResult = {
            "metrics": {"scenarios": {}},
            "distributions": {"scenarios": {}},
            "parameters": {"scenarios": []},
            "warnings": ["No trades available for scenarios."],
        }
        if metadata:
            result.setdefault("parameters", {}).update({"metadata": dict(metadata)})
        return result

    multi_asset = bool(params.get("multi_asset"))
    symbols = {trade.symbol for trade in normalized if trade.symbol}
    if len(symbols) > 1:
        multi_asset = True

    if multi_asset:
        returns_by_symbol: Dict[str, List[float]] = {}
        for trade in normalized:
            symbol = trade.symbol or "UNKNOWN"
            returns_by_symbol.setdefault(symbol, []).append(trade.pnl)
        result = apply_scenarios_to_returns(returns_by_symbol, parameters=params)
    else:
        ordered = sorted(
            normalized, key=lambda t: t.exit_time_utc if t.exit_time_utc is not None else datetime.min
        )
        pnl_series = [trade.pnl for trade in ordered]
        result = apply_scenarios_to_returns(pnl_series, parameters=params)

    if metadata:
        result.setdefault("parameters", {}).update({"metadata": dict(metadata)})
    return result


def run_scenarios_on_returns(
    returns: TimeSeries,
    *,
    metadata: Optional[Mapping[str, Any]] = None,
    parameters: Optional[Mapping[str, Any]] = None,
) -> StressTestResult:
    """Run deterministic scenario stress tests using the returns series.

    Parameters
    ----------
    returns:
        Sequence or mapping of returns (chronological order if sequence).
    metadata:
        Optional contextual metadata (strategy id, asset class, etc.).
    parameters:
        Optional scenario definitions (shock size, volatility spike, etc.).

    Returns
    -------
    StressTestResult
        Dict compatible with ``StrategyRunResult.extra``.
    """
    result = apply_scenarios_to_returns(returns, parameters=parameters)
    if metadata:
        result.setdefault("parameters", {}).update({"metadata": dict(metadata)})
    return result


def run_scenarios_on_equity_curve(
    equity_curve: TimeSeries,
    *,
    metadata: Optional[Mapping[str, Any]] = None,
    parameters: Optional[Mapping[str, Any]] = None,
) -> StressTestResult:
    """Run deterministic scenario stress tests using the equity curve.

    Parameters
    ----------
    equity_curve:
        Sequence or mapping of equity values (chronological order if sequence).
    metadata:
        Optional contextual metadata (strategy id, asset class, etc.).
    parameters:
        Optional scenario definitions (shock size, volatility spike, etc.).

    Returns
    -------
    StressTestResult
        Dict compatible with ``StrategyRunResult.extra``.
    """
    if isinstance(equity_curve, Mapping):
        ordered = sorted(equity_curve.items(), key=lambda item: item[0])
        pnl_series: List[float] = []
        prev_value: Optional[float] = None
        for _, value in ordered:
            val = float(value)
            if prev_value is None:
                prev_value = val
                continue
            pnl_series.append(val - prev_value)
            prev_value = val
        initial_capital = float(ordered[0][1]) if ordered else 0.0
    else:
        pnl_series = []
        prev_value = None
        values = list(equity_curve)
        initial_capital = float(values[0]) if values else 0.0
        for value in values:
            val = float(value)
            if prev_value is None:
                prev_value = val
                continue
            pnl_series.append(val - prev_value)
            prev_value = val

    params = dict(parameters or {})
    params.setdefault("initial_capital", initial_capital)
    result = apply_scenarios_to_returns(pnl_series, parameters=params)
    if metadata:
        result.setdefault("parameters", {}).update({"metadata": dict(metadata)})
    return result
