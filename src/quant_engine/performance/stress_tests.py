"""API contract for strategy stress tests.

This module defines a unified interface for running stress tests and returning
results that can be stored in ``StrategyRunResult.extra``. Implementations can
plug into the placeholders to provide Monte Carlo or deterministic scenario
analysis while keeping a consistent payload schema.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence, TypedDict, Union

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

    raise NotImplementedError("Monte Carlo stress tests on trades are not implemented.")


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

    raise NotImplementedError("Monte Carlo stress tests on returns are not implemented.")


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

    raise NotImplementedError("Monte Carlo stress tests on equity curve are not implemented.")


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
