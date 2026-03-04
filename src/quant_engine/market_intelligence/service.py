"""Market Intelligence service v1 with transitional legacy adapters and caching."""

from __future__ import annotations

from typing import Any

import pandas as pd

from quant_engine.market_intelligence.adapters.legacy_filters_adapter import LegacyFiltersAdapter
from quant_engine.market_intelligence.adapters.legacy_stats_adapter import LegacyStatsAdapter
from quant_engine.market_intelligence.feature_store_memory import InMemoryFeatureStore
from quant_engine.market_intelligence.pipeline import compute_features, label_regimes, liquidity_flags


class MarketIntelligenceServiceV1:
    """Compute market intelligence snapshots with read/write-through cache semantics."""

    def __init__(
        self,
        *,
        feature_store: InMemoryFeatureStore | Any | None = None,
        stats_adapter: LegacyStatsAdapter | None = None,
        filters_adapter: LegacyFiltersAdapter | None = None,
        feature_set: str = "market_intelligence",
        feature_version: str = "1.0.0",
        timeframe: str = "1h",
    ) -> None:
        self._feature_store = feature_store
        self._stats_adapter = stats_adapter or LegacyStatsAdapter()
        self._filters_adapter = filters_adapter or LegacyFiltersAdapter()
        self._feature_set = feature_set
        self._feature_version = feature_version
        self._timeframe = timeframe

    def compute_features(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        return compute_features(ohlcv)

    def label_regimes(self, features: pd.DataFrame) -> pd.DataFrame:
        return label_regimes(features)

    def liquidity_flags(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        return liquidity_flags(ohlcv)

    def _cache_get(self, symbol: str) -> dict[str, Any] | None:
        if self._feature_store is None:
            return None
        try:
            return self._feature_store.get(self._feature_set, symbol, self._timeframe, self._feature_version)
        except KeyError:
            return None

    def _cache_put(self, symbol: str, snapshot: dict[str, Any]) -> None:
        if self._feature_store is None:
            return
        overwrite = self._feature_store.exists(self._feature_set, symbol, self._timeframe, self._feature_version)
        self._feature_store.put(
            self._feature_set,
            symbol,
            self._timeframe,
            self._feature_version,
            snapshot,
            overwrite=overwrite,
        )

    def build_snapshot(self, symbol: str, ohlcv: pd.DataFrame) -> dict[str, Any]:
        """Read-through cache, then compute and write-through on miss."""
        cached = self._cache_get(symbol)
        if cached is not None:
            return cached

        features = self.compute_features(ohlcv)
        regimes = self.label_regimes(features)
        liquidity = self.liquidity_flags(ohlcv)

        legacy_stats = self._stats_adapter.run(spec={"symbol": symbol, "ohlcv": ohlcv})
        legacy_filters = self._filters_adapter.run(ohlcv, rules=[], symbol=symbol)

        snapshot: dict[str, Any] = {
            "symbol": symbol,
            "timeframe": self._timeframe,
            "feature_version": self._feature_version,
            "features": features,
            "regimes": regimes,
            "liquidity": liquidity,
            "legacy_stats": legacy_stats,
            "legacy_filters": legacy_filters,
        }
        self._cache_put(symbol, snapshot)
        return snapshot


__all__ = ["MarketIntelligenceServiceV1"]
