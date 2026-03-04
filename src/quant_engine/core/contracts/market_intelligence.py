"""Contract for market-intelligence adapters used by strategies and screening flows."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Protocol, runtime_checkable

if TYPE_CHECKING:
    import pandas as pd

    DataFrame = pd.DataFrame
else:
    DataFrame = Any


@runtime_checkable
class MarketIntelligenceService(Protocol):
    """Provide a normalized market snapshot from OHLCV data.

    Inputs:
        symbol: Canonical instrument identifier (e.g. ``"BTC-USD"``).
        ohlcv: Time-indexed OHLCV dataframe sorted in ascending timestamp order.

    Output:
        A mapping with analytics ready to consume by strategies, filters, or risk checks.
    """

    def build_snapshot(self, symbol: str, ohlcv: DataFrame) -> Mapping[str, Any]:
        """Compute and return a market-intelligence snapshot for ``symbol``."""
