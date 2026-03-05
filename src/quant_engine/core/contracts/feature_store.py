"""Contract for feature-store components used by market-intelligence services."""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class FeatureStore(Protocol):
    """Store and retrieve feature payloads identified by deterministic keys.

    Inputs:
        feature_set: Feature family identifier (e.g. ``"market_intelligence"``).
        symbol: Canonical instrument identifier (e.g. ``"BTC-USD"``).
        timeframe: Candle timeframe identifier (e.g. ``"1h"``).
        version: Feature contract version (e.g. ``"1.0.0"``).
        payload: Any serializable or in-memory object.

    Output:
        Implementations define payload persistence and retrieval semantics.
    """

    def get(self, feature_set: str, symbol: str, timeframe: str, version: str) -> Any:
        """Return stored payload for the key or raise ``KeyError`` when missing."""

    def put(
        self,
        feature_set: str,
        symbol: str,
        timeframe: str,
        version: str,
        payload: Any,
        *,
        overwrite: bool = False,
    ) -> None:
        """Persist payload for the key."""

    def exists(self, feature_set: str, symbol: str, timeframe: str, version: str) -> bool:
        """Return whether a key is present in the store."""
