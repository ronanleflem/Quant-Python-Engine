from __future__ import annotations

import pandas as pd

from quant_engine.core.contracts.market_intelligence import MarketIntelligenceService
from quant_engine.market_intelligence.service import MarketIntelligenceServiceV1


class _DummyStatsAdapter:
    def run(self, spec):
        return pd.DataFrame(
            {
                "symbol": [spec["symbol"]],
                "event": ["dummy"],
                "p_hat": [0.5],
            },
            index=pd.to_datetime(["2026-01-01T00:00:00Z"], utc=True),
        )


class _DummyFiltersAdapter:
    def run(self, ohlcv, rules, **_kwargs):
        return pd.DataFrame(
            {
                "hard_mask": [True] * len(ohlcv),
                "score": [1.0] * len(ohlcv),
                "score_pct": [1.0] * len(ohlcv),
                "final_mask": [True] * len(ohlcv),
            },
            index=ohlcv.index,
        )


def _sample_ohlcv() -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=8, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "open": [100, 101, 102, 103, 102, 101, 100, 99],
            "high": [101, 102, 103, 104, 103, 102, 101, 100],
            "low": [99, 100, 101, 102, 101, 100, 99, 98],
            "close": [100.4, 101.3, 102.2, 102.7, 101.6, 100.5, 99.4, 99.1],
            "volume": [1200, 1300, 1400, 1250, 1100, 1050, 1000, 980],
        },
        index=idx,
    )


def test_service_v1_satisfies_core_market_intelligence_service_protocol() -> None:
    service = MarketIntelligenceServiceV1()
    assert isinstance(service, MarketIntelligenceService)


def test_service_v1_snapshot_shape_matches_contract_surface() -> None:
    service = MarketIntelligenceServiceV1(
        stats_adapter=_DummyStatsAdapter(),
        filters_adapter=_DummyFiltersAdapter(),
    )
    snapshot = service.build_snapshot("BTC-USD", _sample_ohlcv())

    assert {"symbol", "timeframe", "feature_version", "features", "regimes", "liquidity"}.issubset(snapshot)
    assert snapshot["symbol"] == "BTC-USD"
    assert snapshot["timeframe"] == "1h"
    assert snapshot["feature_version"] == "1.0.0"

    assert snapshot["features"].index.tz is not None
    assert str(snapshot["features"].index.tz) == "UTC"
    assert snapshot["regimes"].index.equals(snapshot["features"].index)
    assert snapshot["liquidity"].index.equals(snapshot["features"].index)
