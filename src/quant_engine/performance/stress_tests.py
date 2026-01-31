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
try:  # optional: numpy for faster Monte Carlo
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    np = None  # type: ignore[assignment]


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
    model_config = ConfigDict(extra="allow")

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


def _parse_time_distribution(parameters: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(parameters, Mapping):
        return None
    cfg = parameters.get("time_distribution") or parameters.get("time_dist")
    if not isinstance(cfg, Mapping):
        return None
    if cfg.get("enabled") is False:
        return None
    mode = str(cfg.get("mode", "exit_deltas")).strip().lower()
    aliases = {
        "deltas": "exit_deltas",
        "exit_delta": "exit_deltas",
        "durations": "trade_durations",
        "trade_duration": "trade_durations",
        "sessions": "exit_times",
        "session": "exit_times",
    }
    mode = aliases.get(mode, mode)
    allowed = {"exit_deltas", "exit_times", "trade_durations"}
    if mode not in allowed:
        raise ValueError(f"time_distribution.mode must be one of {sorted(allowed)}.")
    seed = cfg.get("seed")
    return {"mode": mode, "seed": seed}


def _parse_param_drift_config(parameters: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(parameters, Mapping):
        return None
    cfg = parameters.get("param_drift")
    if not isinstance(cfg, Mapping):
        return None
    if cfg.get("enabled") is False:
        return None
    dist = str(cfg.get("dist", "normal")).strip().lower()
    if dist not in {"uniform", "normal"}:
        raise ValueError(f"param_drift.dist must be one of {sorted(['normal', 'uniform'])}.")
    mode = str(cfg.get("mode", "per_sim")).strip().lower()
    allowed = {"per_sim", "per_trade", "random_walk"}
    if mode not in allowed:
        raise ValueError(f"param_drift.mode must be one of {sorted(allowed)}.")
    return {
        "dist": dist,
        "mode": mode,
        "mu": cfg.get("mu", 1.0),
        "sigma": cfg.get("sigma", 0.05),
        "low": cfg.get("low", 0.9),
        "high": cfg.get("high", 1.1),
        "min": cfg.get("min"),
        "max": cfg.get("max"),
        "seed": cfg.get("seed"),
    }


def _draw_param_drift_multiplier(rng: random.Random, cfg: Mapping[str, Any]) -> float:
    dist = cfg["dist"]
    if dist == "uniform":
        value = rng.uniform(float(cfg.get("low", 0.9)), float(cfg.get("high", 1.1)))
    else:
        value = rng.gauss(float(cfg.get("mu", 1.0)), float(cfg.get("sigma", 0.05)))
    min_val = cfg.get("min")
    max_val = cfg.get("max")
    if min_val is not None:
        value = max(float(min_val), value)
    if max_val is not None:
        value = min(float(max_val), value)
    return float(value)


def _apply_param_drift(
    sampled_pnl: Sequence[float],
    rng: random.Random,
    cfg: Mapping[str, Any],
) -> List[float]:
    mode = cfg["mode"]
    if mode == "per_sim":
        mult = _draw_param_drift_multiplier(rng, cfg)
        return [float(pnl) * mult for pnl in sampled_pnl]
    if mode == "per_trade":
        return [float(pnl) * _draw_param_drift_multiplier(rng, cfg) for pnl in sampled_pnl]
    cumulative = 1.0
    adjusted: List[float] = []
    for pnl in sampled_pnl:
        step = _draw_param_drift_multiplier(rng, cfg)
        cumulative *= step
        min_val = cfg.get("min")
        max_val = cfg.get("max")
        if min_val is not None:
            cumulative = max(float(min_val), cumulative)
        if max_val is not None:
            cumulative = min(float(max_val), cumulative)
        adjusted.append(float(pnl) * cumulative)
    return adjusted


def _build_trade_durations(trades: Sequence[StandardTrade]) -> List[timedelta]:
    durations: List[timedelta] = []
    for trade in trades:
        if trade.entry_time_utc is None or trade.exit_time_utc is None:
            return []
        durations.append(trade.exit_time_utc - trade.entry_time_utc)
    return durations


def _build_time_distribution_timestamps(
    mode: str,
    *,
    base_start: datetime,
    base_timestamps: Sequence[datetime],
    base_deltas: Sequence[timedelta],
    trade_durations: Sequence[timedelta],
    sample_size: int,
    rng: random.Random,
) -> Optional[List[datetime]]:
    if mode == "exit_times":
        if not base_timestamps:
            return None
        sampled_exit = [rng.choice(list(base_timestamps)) for _ in range(sample_size)]
        sampled_exit.sort()
        if not sampled_exit:
            return None
        return [sampled_exit[0]] + sampled_exit
    if mode == "trade_durations":
        if not trade_durations:
            return None
        timestamps = [base_start]
        for _ in range(sample_size):
            delta = rng.choice(list(trade_durations))
            timestamps.append(timestamps[-1] + delta)
        return timestamps
    if not base_deltas:
        return None
    return _build_sampled_timestamps(base_start, base_deltas, sample_size + 1, rng)


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
    weights: Optional[Mapping[str, float]] = None,
    *,
    timestamp_alignment: str = "union",
) -> tuple[List[float], Optional[List[datetime]]]:
    if per_asset and all(timestamps.get(symbol) for symbol in per_asset):
        combined: Dict[datetime, float] = {}
        common_timestamps: Optional[set[datetime]] = None
        if timestamp_alignment == "intersection":
            for symbol in per_asset:
                symbol_ts = set(timestamps.get(symbol) or [])
                if common_timestamps is None:
                    common_timestamps = symbol_ts
                else:
                    common_timestamps &= symbol_ts
            if not common_timestamps:
                return [], []
        for symbol, values in per_asset.items():
            weight = float(weights.get(symbol, 1.0)) if weights else 1.0
            ts_list = timestamps.get(symbol) or []
            for ts, value in zip(ts_list, values):
                if ts is None:
                    continue
                if common_timestamps is not None and ts not in common_timestamps:
                    continue
                combined[ts] = combined.get(ts, 0.0) + float(value) * weight
        if combined:
            ordered = sorted(combined.items(), key=lambda item: item[0])
            return [val for _, val in ordered], [ts for ts, _ in ordered]

    max_len = max((len(values) for values in per_asset.values()), default=0)
    aggregated = [0.0] * max_len
    for symbol, values in per_asset.items():
        weight = float(weights.get(symbol, 1.0)) if weights else 1.0
        for idx, value in enumerate(values):
            aggregated[idx] += float(value) * weight
    return aggregated, None


def _normalize_weights(
    symbols: Sequence[str],
    raw_weights: Mapping[str, float],
) -> Optional[Dict[str, float]]:
    if not symbols:
        return None
    weights = {symbol: float(raw_weights.get(symbol, 0.0)) for symbol in symbols}
    total = sum(weights.values())
    if total == 0:
        return None
    return {symbol: value / total for symbol, value in weights.items()}


def _volatility_weights(
    per_asset: Mapping[str, Sequence[float]],
    *,
    weighting: str,
) -> Optional[Dict[str, float]]:
    if not per_asset:
        return None
    volatilities: Dict[str, float] = {}
    for symbol, values in per_asset.items():
        if len(values) > 1:
            vol = pstdev(values)
            if vol > 0:
                if weighting == "inverse":
                    volatilities[symbol] = 1.0 / vol
                else:
                    volatilities[symbol] = vol
    if len(volatilities) != len(per_asset):
        return None
    total = sum(volatilities.values())
    if total == 0:
        return None
    return {symbol: value / total for symbol, value in volatilities.items()}


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
    aggregation = str(params.get("aggregation", "equal_weight")).lower()
    raw_weights = params.get("weights")
    weights_source = params.get("weights_source")
    volatility_weighting = str(params.get("volatility_weighting", "inverse")).lower()
    timestamp_alignment = str(params.get("timestamp_alignment", "union")).lower()
    weights_used: Optional[Dict[str, float]] = None

    scenario_results: Dict[str, Any] = {}
    scenario_metrics: Dict[str, Any] = {}

    if _is_multi_asset_returns(returns):
        per_asset, timestamps = _normalize_multi_asset_returns(returns)
        if not per_asset:
            warnings.append("No returns available for scenarios.")
        symbols = list(per_asset.keys())
        if aggregation == "value_weighted":
            if isinstance(raw_weights, Mapping) and raw_weights:
                weights_used = _normalize_weights(symbols, raw_weights)
                if weights_used is None:
                    warnings.append("Value-weighted aggregation requires non-zero weights; falling back to equal weights.")
            else:
                warnings.append("Value-weighted aggregation requires weights; falling back to equal weights.")
        elif aggregation == "vol_weighted":
            if volatility_weighting not in {"direct", "inverse"}:
                warnings.append(
                    f"Unknown volatility_weighting '{volatility_weighting}'; falling back to inverse weighting."
                )
                volatility_weighting = "inverse"
            weights_used = _volatility_weights(per_asset, weighting=volatility_weighting)
            if weights_used is None:
                warnings.append("Volatility-weighted aggregation unavailable; falling back to equal weights.")
        elif aggregation != "equal_weight":
            warnings.append(f"Unknown aggregation '{aggregation}'; falling back to equal weights.")

        if weights_used is None and symbols:
            weights_used = {symbol: 1.0 / len(symbols) for symbol in symbols}
            aggregation = "equal_weight"

        if timestamp_alignment not in {"union", "intersection"}:
            warnings.append(
                f"Unknown timestamp_alignment '{timestamp_alignment}'; falling back to union alignment."
            )
            timestamp_alignment = "union"

        for scenario in scenarios:
            scenario_data = scenario.model_dump()
            name = str(scenario_data.get("name", "scenario"))
            per_asset_results: Dict[str, Any] = {}
            adjusted_assets: Dict[str, List[float]] = {}
            scenario_parameters = {
                **scenario_data,
                "aggregation": aggregation,
                "weights": weights_used,
                "weights_source": weights_source,
                "volatility_weighting": volatility_weighting if aggregation == "vol_weighted" else None,
                "timestamp_alignment": timestamp_alignment,
            }
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
                    "parameters": scenario_parameters,
                }

            portfolio_returns, portfolio_ts = _aggregate_multi_asset_returns(
                adjusted_assets,
                timestamps,
                weights_used,
                timestamp_alignment=timestamp_alignment,
            )
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
                    "parameters": scenario_parameters,
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
            scenario_parameters = {
                **scenario_data,
                "aggregation": None,
                "weights": None,
                "weights_source": None,
                "volatility_weighting": None,
                "timestamp_alignment": None,
            }
            scenario_results[name] = {
                "metrics": metrics,
                "returns": adjusted,
                "timestamps": timestamps,
                "parameters": scenario_parameters,
            }
            scenario_metrics[name] = metrics

    return {
        "metrics": {"scenarios": _normalize_scenario_metrics(scenario_metrics)},
        "distributions": {"scenarios": scenario_results},
        "parameters": {
            "initial_capital": initial_capital,
            "scenarios": [scenario.model_dump() for scenario in scenarios],
            "aggregation": aggregation if _is_multi_asset_returns(returns) else None,
            "weights": weights_used,
            "weights_source": weights_source if _is_multi_asset_returns(returns) else None,
            "volatility_weighting": volatility_weighting if aggregation == "vol_weighted" else None,
            "timestamp_alignment": timestamp_alignment if _is_multi_asset_returns(returns) else None,
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
    else:
        returns_pct = [float(pnl) for pnl in pnl_values]

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
    rng_sampling = random.Random(seed)
    rng_sizing = random.Random((seed or 0) + 999_983)
    time_cfg = _parse_time_distribution(parameters)
    time_seed = None if time_cfg is None else time_cfg.get("seed")
    rng_time = random.Random(time_seed if time_seed is not None else (seed or 0) + 424_242)
    param_cfg = _parse_param_drift_config(parameters)
    param_seed = None if param_cfg is None else param_cfg.get("seed")
    rng_param = random.Random(param_seed if param_seed is not None else (seed or 0) + 777_777)
    sizing_cfg = _parse_sizing_config(parameters)

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

    if (
        np is not None
        and method in {"bootstrap", "iid", "shuffle", "block", "block_bootstrap"}
        and not multi_asset
        and sizing_cfg is None
        and time_cfg is None
        and param_cfg is None
    ):
        result = _monte_carlo_bootstrap_numpy(
            trades,
            config=config,
        )
        return _apply_monte_carlo_output_mode(result, parameters)

    base_timestamps = _build_timestamps(trades)
    base_start = base_timestamps[0] if base_timestamps else datetime.utcnow()
    trade_durations = _build_trade_durations(trades)
    deltas: List[timedelta] = []
    if base_timestamps:
        for prev, nxt in zip(base_timestamps[:-1], base_timestamps[1:]):
            deltas.append(nxt - prev)
    time_warnings: List[str] = []
    if time_cfg is not None:
        mode = time_cfg["mode"]
        if mode in {"exit_deltas", "exit_times"} and not base_timestamps:
            time_warnings.append(
                "time_distribution requested but exit timestamps are missing; falling back to index-based timing."
            )
            time_cfg = None
        elif mode == "trade_durations" and not trade_durations:
            time_warnings.append(
                "time_distribution requested but trade durations are missing; falling back to index-based timing."
            )
            time_cfg = None

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
            sampled_trades = _sample_iid(ordered_trades, rng_sampling)
        elif method in {"block", "block_bootstrap"}:
            if multi_asset:
                sampled_trades = _sample_block_multi_asset(
                    ordered_trades, rng_sampling, block_size=block_size, overlapping=overlapping
                )
            else:
                sampled_trades = _sample_block(
                    ordered_trades, rng_sampling, block_size=block_size, overlapping=overlapping
                )
        else:
            sampled_trades = _sample_bootstrap(ordered_trades, rng_sampling)

        sampled_pnl = [t.pnl for t in sampled_trades]
        if param_cfg is not None:
            sampled_pnl = _apply_param_drift(sampled_pnl, rng_param, param_cfg)
        if sizing_cfg is not None:
            sampled_pnl = [pnl * _draw_size_multiplier(rng_sizing, sizing_cfg) for pnl in sampled_pnl]
        equity = [initial_capital]
        for pnl in sampled_pnl:
            equity.append(equity[-1] + pnl)
        equity_curves.append(equity)

        max_drawdowns.append(_max_drawdown(equity))

        timestamps = None
        period_days = base_period_days
        if time_cfg is not None:
            timestamps = _build_time_distribution_timestamps(
                time_cfg["mode"],
                base_start=base_start,
                base_timestamps=base_timestamps or [],
                base_deltas=deltas,
                trade_durations=trade_durations,
                sample_size=len(sampled_trades),
                rng=rng_time,
            )
        if timestamps is None:
            timestamps = _equity_timestamps_from_trades(sampled_trades)
        if timestamps is None and base_timestamps:
            timestamps = _build_sampled_timestamps(base_start, deltas, len(equity), rng_time)
        if timestamps:
            period_days = (timestamps[-1] - timestamps[0]).total_seconds() / 86400.0
        ttr = _time_to_recovery(equity, timestamps)
        if ttr is not None:
            time_to_recovery.append(float(ttr))

        ruin_flags.append(min(equity) <= ruin_threshold)

        years = None
        if period_days is not None and period_days > 0:
            years = period_days / 365.25
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

    result = {
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
    if time_cfg is not None:
        result["parameters"]["time_distribution"] = dict(time_cfg)
    if param_cfg is not None:
        result["parameters"]["param_drift"] = dict(param_cfg)
    if time_warnings:
        result["warnings"] = list(time_warnings)
    return _apply_monte_carlo_output_mode(result, parameters)


def _monte_carlo_bootstrap_numpy(
    trades: Sequence[StandardTrade],
    *,
    config: MonteCarloConfig,
) -> StressTestResult:
    n_simulations = config.n_sims
    seed = config.seed
    initial_capital = float(config.initial_capital)
    ruin_threshold = float(config.ruin_threshold)
    method = config.method
    sample_size = len(trades)

    ordered_trades = sorted(
        trades, key=lambda t: t.exit_time_utc if t.exit_time_utc is not None else datetime.min
    )
    pnl_values = np.array([t.pnl for t in ordered_trades], dtype=float)

    rng = np.random.default_rng(seed)
    if method in {"iid", "shuffle"}:
        random_matrix = rng.random((n_simulations, sample_size))
        indices = np.argsort(random_matrix, axis=1)
    elif method in {"block", "block_bootstrap"}:
        starts = _block_start_indices(sample_size, config.block_size, config.overlapping)
        if not starts:
            indices = rng.integers(0, sample_size, size=(n_simulations, sample_size))
        else:
            num_blocks = int(ceil(sample_size / max(config.block_size, 1)))
            starts_arr = rng.choice(np.array(starts, dtype=int), size=(n_simulations, num_blocks))
            offsets = np.arange(config.block_size, dtype=int)
            block_indices = starts_arr[:, :, None] + offsets[None, None, :]
            block_indices = np.clip(block_indices, 0, sample_size - 1)
            indices = block_indices.reshape(n_simulations, -1)[:, :sample_size]
    else:
        indices = rng.integers(0, sample_size, size=(n_simulations, sample_size))

    sampled_pnl = pnl_values[indices]
    sizing_cfg = _parse_sizing_config(config.model_dump())
    if sizing_cfg is not None:
        sampled_pnl = _apply_numpy_sizing(sampled_pnl, sizing_cfg, seed)
    equity = initial_capital + np.cumsum(sampled_pnl, axis=1)
    equity_curves = np.concatenate(
        [np.full((n_simulations, 1), initial_capital, dtype=float), equity], axis=1
    )

    running_max = np.maximum.accumulate(equity_curves, axis=1)
    max_drawdowns = (running_max - equity_curves).max(axis=1)
    ruin_flags = (equity_curves.min(axis=1) <= ruin_threshold).tolist()

    returns_pct = sampled_pnl / initial_capital * 100.0 if initial_capital else sampled_pnl
    win_count = (returns_pct > 0).sum(axis=1).astype(float)
    loss_count = (returns_pct <= 0).sum(axis=1).astype(float)
    total_return = returns_pct.sum(axis=1)
    average_trade = total_return / sample_size if sample_size else np.zeros(n_simulations)

    final_capital = equity_curves[:, -1]
    return_pct = (
        (final_capital - initial_capital) / initial_capital * 100.0 if initial_capital else np.zeros(n_simulations)
    )

    peak = running_max.max(axis=1)
    max_drawdown_pct = np.where(
        peak != 0,
        (max_drawdowns / peak) * 100.0,
        np.nan,
    )

    volatility_pct = np.std(returns_pct, axis=1, ddof=0) if sample_size > 1 else np.full(n_simulations, np.nan)
    mean_ret = returns_pct.mean(axis=1) if sample_size else np.zeros(n_simulations)
    sharpe = np.full(n_simulations, np.nan)
    valid = (volatility_pct != 0) & ~np.isnan(volatility_pct)
    if valid.any():
        sharpe[valid] = np.divide(
            mean_ret[valid],
            volatility_pct[valid],
            out=np.full_like(mean_ret[valid], np.nan),
            where=volatility_pct[valid] != 0,
        ) * (sample_size ** 0.5)

    sortino = np.full(n_simulations, np.nan)
    for i in range(n_simulations):
        downside = returns_pct[i][returns_pct[i] < 0]
        if downside.size > 1:
            downside_std = np.std(downside, ddof=0)
            if downside_std:
                sortino[i] = mean_ret[i] / downside_std * (sample_size ** 0.5)

    base_timestamps = _build_timestamps(ordered_trades)
    base_start = base_timestamps[0] if base_timestamps else datetime.utcnow()
    deltas: List[timedelta] = []
    if base_timestamps:
        for prev, nxt in zip(base_timestamps[:-1], base_timestamps[1:]):
            deltas.append(nxt - prev)

    base_period_days: Optional[float] = None
    if base_timestamps:
        base_period_days = (base_timestamps[-1] - base_timestamps[0]).total_seconds() / 86400.0

    time_to_recovery: List[float] = []
    cagrs: List[float] = []
    equity_curves_list = equity_curves.tolist()

    exit_times = [t.exit_time_utc for t in ordered_trades if t.exit_time_utc is not None]
    has_exit_times = len(exit_times) == sample_size

    for i in range(n_simulations):
        timestamps = None
        if has_exit_times:
            sampled_exit = [exit_times[idx] for idx in indices[i]]
            if sampled_exit and sampled_exit[0] is not None:
                timestamps = [sampled_exit[0]] + sampled_exit
        elif base_timestamps:
            timestamps = _build_sampled_timestamps(base_start, deltas, sample_size + 1, random.Random(int(seed or 0) + i))

        ttr = _time_to_recovery(equity_curves_list[i], timestamps)
        if ttr is not None:
            time_to_recovery.append(float(ttr))

        years = None
        if base_period_days is not None and base_period_days > 0:
            years = base_period_days / 365.25
        else:
            years = sample_size / 252.0 if sample_size else None
        if years and initial_capital > 0 and final_capital[i] > 0:
            cagr = (final_capital[i] / initial_capital) ** (1 / years) - 1
            cagrs.append(float(cagr))

    level1_metrics: Dict[str, List[float]] = {
        "final_capital": final_capital.tolist(),
        "return_pct": return_pct.tolist(),
        "max_drawdown": max_drawdowns.tolist(),
        "max_drawdown_pct": max_drawdown_pct.tolist(),
        "volatility_pct": volatility_pct.tolist(),
        "sharpe": sharpe.tolist(),
        "sortino": sortino.tolist(),
        "winrate_pct": (win_count / sample_size * 100.0).tolist() if sample_size else [],
        "total_return": total_return.tolist(),
        "average_trade": average_trade.tolist(),
        "win_count": win_count.tolist(),
        "loss_count": loss_count.tolist(),
    }

    ruin_probability = sum(ruin_flags) / len(ruin_flags) if ruin_flags else None

    def _filter_nan(values: Sequence[float]) -> List[float]:
        cleaned: List[float] = []
        for value in values:
            if value is None:
                continue
            if isinstance(value, float) and value != value:
                continue
            cleaned.append(float(value))
        return cleaned

    metrics = {
        "max_drawdown": _summary_stats(_filter_nan(level1_metrics["max_drawdown"])),
        "cagr": _summary_stats(_filter_nan(cagrs)),
        "time_to_recovery_days": _summary_stats(_filter_nan(time_to_recovery)),
        "ruin_probability": ruin_probability,
    }
    for key, values in level1_metrics.items():
        metrics[key] = _summary_stats(_filter_nan(values))

    metrics = _normalize_metric_names(metrics)
    normalized_level1 = _normalize_metric_names(level1_metrics)

    distributions = {
        "max_drawdown": level1_metrics["max_drawdown"],
        "cagr": cagrs,
        "time_to_recovery_days": time_to_recovery,
        "equity_curves": equity_curves_list,
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
            "block_size": config.block_size,
            "overlapping": config.overlapping,
            "multi_asset": config.multi_asset,
        },
    }


def _apply_monte_carlo_output_mode(
    result: StressTestResult,
    parameters: Optional[Mapping[str, Any]],
) -> StressTestResult:
    if not isinstance(parameters, Mapping):
        return result

    output_cfg = parameters.get("output")
    mode = None
    max_curves = None
    curve_stride = None

    if isinstance(output_cfg, Mapping):
        mode = output_cfg.get("mode")
        max_curves = output_cfg.get("max_curves")
        curve_stride = output_cfg.get("curve_stride")
    else:
        mode = parameters.get("output_mode")
        max_curves = parameters.get("max_curves")
        curve_stride = parameters.get("curve_stride")

    if not mode:
        return result
    mode_value = str(mode).lower()
    if mode_value not in {"light", "light_strict"}:
        return result

    distributions = result.get("distributions")
    if not isinstance(distributions, Mapping):
        return result

    equity_curves = distributions.get("equity_curves")
    if isinstance(equity_curves, list):
        try:
            max_curves_val = int(max_curves) if max_curves is not None else 50
        except Exception:
            max_curves_val = 50
        if max_curves_val < 0:
            max_curves_val = 0
        curves = equity_curves[:max_curves_val] if max_curves_val else []

        stride = 1
        try:
            stride = int(curve_stride) if curve_stride is not None else 1
        except Exception:
            stride = 1
        if stride < 1:
            stride = 1

        if stride > 1:
            reduced: List[List[float]] = []
            for curve in curves:
                if not isinstance(curve, list):
                    continue
                sliced = curve[::stride]
                if curve and (not sliced or sliced[-1] != curve[-1]):
                    sliced.append(curve[-1])
                reduced.append(sliced)
            curves = reduced

        distributions = dict(distributions)
        distributions["equity_curves"] = curves
        result = dict(result)
        result["distributions"] = distributions

    if mode_value == "light_strict":
        percentiles = [0.10, 0.25, 0.75, 0.90, 0.95, 0.99]

        def _filter_nan(values: Sequence[float]) -> List[float]:
            cleaned: List[float] = []
            for value in values:
                if value is None:
                    continue
                if isinstance(value, float) and value != value:
                    continue
                cleaned.append(float(value))
            return cleaned

        def _summary_stats_custom(values: Sequence[float]) -> Dict[str, Optional[float]]:
            cleaned = _filter_nan(values)
            stats: Dict[str, Optional[float]] = {}
            for pct in percentiles:
                stats[f"p{int(pct * 100)}"] = _percentile(cleaned, pct)
            stats["mean"] = mean(cleaned) if cleaned else None
            stats["std"] = pstdev(cleaned) if len(cleaned) > 1 else None
            return stats

        metrics = result.get("metrics")
        distributions = result.get("distributions")
        if isinstance(metrics, Mapping) and isinstance(distributions, Mapping):
            new_metrics = dict(metrics)
            for key in ("max_drawdown", "cagr", "time_to_recovery_days"):
                values = distributions.get(key)
                if isinstance(values, list):
                    new_metrics[key] = _summary_stats_custom(values)

            level1 = distributions.get("level1")
            if isinstance(level1, Mapping):
                for key, values in level1.items():
                    if isinstance(values, list):
                        new_metrics[key] = _summary_stats_custom(values)
            result = dict(result)
            result["metrics"] = new_metrics

            equity_curves = distributions.get("equity_curves") if isinstance(distributions, Mapping) else None
            if isinstance(equity_curves, list):
                result["distributions"] = {"equity_curves": equity_curves}

    return result


def _parse_sizing_config(parameters: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(parameters, Mapping):
        return None
    sizing = parameters.get("sizing")
    if not isinstance(sizing, Mapping):
        return None
    if sizing.get("enabled") is False:
        return None
    return dict(sizing)


def _draw_size_multiplier(rng: random.Random, sizing_cfg: Mapping[str, Any]) -> float:
    dist = str(sizing_cfg.get("dist") or "uniform").lower()
    mu = float(sizing_cfg.get("mu", 1.0))
    sigma = float(sizing_cfg.get("sigma", 0.1))
    low = float(sizing_cfg.get("low", 0.8))
    high = float(sizing_cfg.get("high", 1.2))
    if dist == "normal":
        value = rng.gauss(mu, sigma)
    elif dist == "lognormal":
        value = rng.lognormvariate(mu, sigma)
    else:
        value = rng.uniform(low, high)

    min_val = sizing_cfg.get("min")
    max_val = sizing_cfg.get("max")
    if min_val is not None:
        value = max(float(min_val), value)
    if max_val is not None:
        value = min(float(max_val), value)
    return float(value)


def _apply_numpy_sizing(sampled_pnl: "np.ndarray", sizing_cfg: Mapping[str, Any], seed: Optional[int]) -> "np.ndarray":
    dist = str(sizing_cfg.get("dist") or "uniform").lower()
    mu = float(sizing_cfg.get("mu", 1.0))
    sigma = float(sizing_cfg.get("sigma", 0.1))
    low = float(sizing_cfg.get("low", 0.8))
    high = float(sizing_cfg.get("high", 1.2))
    rng = np.random.default_rng(seed)

    if dist == "normal":
        multipliers = rng.normal(mu, sigma, size=sampled_pnl.shape)
    elif dist == "lognormal":
        multipliers = rng.lognormal(mu, sigma, size=sampled_pnl.shape)
    else:
        multipliers = rng.uniform(low, high, size=sampled_pnl.shape)

    min_val = sizing_cfg.get("min")
    max_val = sizing_cfg.get("max")
    if min_val is not None or max_val is not None:
        min_bound = float(min_val) if min_val is not None else None
        max_bound = float(max_val) if max_val is not None else None
        if min_bound is not None:
            multipliers = np.maximum(multipliers, min_bound)
        if max_bound is not None:
            multipliers = np.minimum(multipliers, max_bound)
    return sampled_pnl * multipliers


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
