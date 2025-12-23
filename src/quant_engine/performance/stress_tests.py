"""API contract for strategy stress tests.

This module defines a unified interface for running stress tests and returning
results that can be stored in ``StrategyRunResult.extra``. Implementations can
plug into the placeholders to provide Monte Carlo or deterministic scenario
analysis while keeping a consistent payload schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import random
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, TypedDict, Union

from .models import CompletedTrade


TimeSeries = Union[Sequence[float], Mapping[datetime, float]]


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


@dataclass(frozen=True)
class StandardTrade:
    pnl: float
    r_multiple: Optional[float]
    entry_time_utc: Optional[datetime]
    exit_time_utc: Optional[datetime]


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
            if pnl is None:
                continue
            normalized.append(
                StandardTrade(
                    pnl=float(pnl),
                    r_multiple=float(r_multiple) if r_multiple is not None else None,
                    entry_time_utc=_parse_dt(entry_time),
                    exit_time_utc=_parse_dt(exit_time),
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


def _monte_carlo_bootstrap(
    trades: Sequence[StandardTrade],
    *,
    parameters: Optional[Mapping[str, Any]] = None,
) -> StressTestResult:
    params = parameters or {}
    n_simulations = int(params.get("n_simulations", 1_000))
    seed = params.get("seed")
    initial_capital = float(params.get("initial_capital", 0.0))
    ruin_threshold = float(params.get("ruin_threshold", 0.0))
    rng = random.Random(seed)

    if not trades:
        return {
            "metrics": {},
            "distributions": {},
            "parameters": {
                "n_simulations": n_simulations,
                "seed": seed,
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

    base_period_days: Optional[float] = None
    if base_timestamps:
        base_period_days = (base_timestamps[-1] - base_timestamps[0]).total_seconds() / 86400.0

    for _ in range(n_simulations):
        indices = [rng.randrange(sample_size) for _ in range(sample_size)]
        sampled_pnl = [pnl_values[i] for i in indices]
        equity = [initial_capital]
        for pnl in sampled_pnl:
            equity.append(equity[-1] + pnl)
        equity_curves.append(equity)

        max_drawdowns.append(_max_drawdown(equity))

        timestamps = None
        if base_timestamps:
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

    ruin_probability = sum(ruin_flags) / len(ruin_flags) if ruin_flags else None

    metrics = {
        "max_drawdown": {
            "p5": _percentile(max_drawdowns, 0.05),
            "p50": _percentile(max_drawdowns, 0.50),
            "p95": _percentile(max_drawdowns, 0.95),
            "mean": mean(max_drawdowns) if max_drawdowns else None,
            "std": pstdev(max_drawdowns) if len(max_drawdowns) > 1 else None,
        },
        "cagr": {
            "p5": _percentile(cagrs, 0.05),
            "p50": _percentile(cagrs, 0.50),
            "p95": _percentile(cagrs, 0.95),
            "mean": mean(cagrs) if cagrs else None,
            "std": pstdev(cagrs) if len(cagrs) > 1 else None,
        },
        "time_to_recovery_days": {
            "p5": _percentile(time_to_recovery, 0.05),
            "p50": _percentile(time_to_recovery, 0.50),
            "p95": _percentile(time_to_recovery, 0.95),
            "mean": mean(time_to_recovery) if time_to_recovery else None,
            "std": pstdev(time_to_recovery) if len(time_to_recovery) > 1 else None,
        },
        "ruin_probability": ruin_probability,
    }

    distributions = {
        "max_drawdown": max_drawdowns,
        "cagr": cagrs,
        "time_to_recovery_days": time_to_recovery,
        "equity_curves": equity_curves,
        "ruin": ruin_flags,
    }

    return {
        "metrics": metrics,
        "distributions": distributions,
        "parameters": {
            "n_simulations": n_simulations,
            "seed": seed,
            "initial_capital": initial_capital,
            "ruin_threshold": ruin_threshold,
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
            )
            for ts, value in ordered
        ]
    else:
        trades = [StandardTrade(pnl=float(value), r_multiple=None, entry_time_utc=None, exit_time_utc=None) for value in returns]
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
            pnl_series.append(StandardTrade(pnl=val - prev_value, r_multiple=None, entry_time_utc=None, exit_time_utc=None))
            prev_value = val
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

    raise NotImplementedError("Scenario stress tests on trades are not implemented.")


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

    raise NotImplementedError("Scenario stress tests on returns are not implemented.")


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

    raise NotImplementedError("Scenario stress tests on equity curve are not implemented.")
