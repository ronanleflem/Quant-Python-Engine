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
    params = parameters or {}
    n_simulations = int(params.get("n_simulations", 1_000))
    seed = params.get("seed")
    initial_capital = float(params.get("initial_capital", 0.0))
    ruin_threshold = float(params.get("ruin_threshold", 0.0))
    method = str(params.get("method", "bootstrap")).lower()
    block_size = max(int(params.get("block_size", 5)), 1)
    overlapping = bool(params.get("overlapping", True))
    multi_asset = bool(params.get("multi_asset", False))
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

    distributions = {
        "max_drawdown": max_drawdowns,
        "cagr": cagrs,
        "time_to_recovery_days": time_to_recovery,
        "equity_curves": equity_curves,
        "ruin": ruin_flags,
        "level1": level1_metrics,
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
