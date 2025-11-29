"""Base protocol and signal representation for high-level strategies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Protocol

import pandas as pd


@dataclass
class StrategySignal:
    """Normalized representation of a strategy decision."""

    strategy_id: str
    symbol: str
    asset_class: str  # "EQUITY","ETF","CRYPTO"
    side: str  # "BUY","SELL","ROTATE"
    ts_open_utc: pd.Timestamp
    qty: float
    meta: Dict[str, Any]


class Strategy(Protocol):
    """Abstract interface implemented by every high level strategy."""

    def backtest(self, ohlc: pd.DataFrame, context: Dict[str, Any]) -> List[StrategySignal]:
        """Return the list of signals (entries/exits/rotations) for the whole period."""

    def evaluate_live_bar(
        self, ohlc: pd.DataFrame, context: Dict[str, Any]
    ) -> List[StrategySignal]:
        """
        Called on the last closed bar (on-bar-close) to generate zero or more signals.

        ``ohlc`` contains historical bars including the latest closed bar sorted by
        ascending UTC index. ``context`` may hold the open positions, cycle state,
        configuration overrides, etc.
        """


__all__ = ["Strategy", "StrategySignal"]
