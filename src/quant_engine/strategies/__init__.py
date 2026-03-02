"""Strategy registry and factories for high-level strategies."""
from __future__ import annotations

from typing import Any, Dict, Type

from .base import Strategy, StrategySignal
from .crypto_grid import CryptoGridStrategy
from .dca_equity import DcaEquityStrategy
from .dca_etf import DcaEtfStrategy
from .dca_benchmark import DcaBenchmarkStrategy

STRATEGY_REGISTRY: Dict[str, Type[Strategy]] = {
    "dca_equity": DcaEquityStrategy,
    "dca_etf": DcaEtfStrategy,
    "dca_benchmark": DcaBenchmarkStrategy,
    "crypto_grid": CryptoGridStrategy,
}


def create_strategy(strategy_type: str, strategy_id: str, params: Dict[str, Any]) -> Strategy:
    """Instantiate a strategy implementation from its registry key."""

    key = strategy_type.lower()
    cls = STRATEGY_REGISTRY.get(key)
    if cls is None:
        available = ", ".join(sorted(STRATEGY_REGISTRY))
        raise ValueError(f"Unknown strategy type '{strategy_type}'. Available: {available}")
    return cls(strategy_id=strategy_id, params=params or {})


__all__ = [
    "create_strategy",
    "Strategy",
    "StrategySignal",
    "STRATEGY_REGISTRY",
    "CryptoGridStrategy",
    "DcaEquityStrategy",
    "DcaEtfStrategy",
    "DcaBenchmarkStrategy",
]
