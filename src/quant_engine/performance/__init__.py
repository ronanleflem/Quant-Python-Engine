from .models import CompletedTrade, StrategyRunResult, to_backend_payload
from .dca_builder import build_backend_payload_for_java, build_dca_performance_from_signals
from .backtest_builder import build_backtest_payload, build_backtest_performance

try:  # optional dependency: pydantic
    from .stress_tests import (
        StressTestResult,
        apply_scenarios_to_returns,
        run_monte_carlo_on_equity_curve,
        run_monte_carlo_on_returns,
        run_monte_carlo_on_trades,
        run_scenarios_on_equity_curve,
        run_scenarios_on_returns,
        run_scenarios_on_trades,
    )
except Exception:  # pragma: no cover - optional dependency missing
    StressTestResult = None  # type: ignore[assignment]
    apply_scenarios_to_returns = None  # type: ignore[assignment]
    run_monte_carlo_on_equity_curve = None  # type: ignore[assignment]
    run_monte_carlo_on_returns = None  # type: ignore[assignment]
    run_monte_carlo_on_trades = None  # type: ignore[assignment]
    run_scenarios_on_equity_curve = None  # type: ignore[assignment]
    run_scenarios_on_returns = None  # type: ignore[assignment]
    run_scenarios_on_trades = None  # type: ignore[assignment]

__all__ = [
    "CompletedTrade",
    "StrategyRunResult",
    "to_backend_payload",
    "build_dca_performance_from_signals",
    "build_backend_payload_for_java",
    "build_backtest_payload",
    "build_backtest_performance",
    "StressTestResult",
    "apply_scenarios_to_returns",
    "run_monte_carlo_on_equity_curve",
    "run_monte_carlo_on_returns",
    "run_monte_carlo_on_trades",
    "run_scenarios_on_equity_curve",
    "run_scenarios_on_returns",
    "run_scenarios_on_trades",
]
