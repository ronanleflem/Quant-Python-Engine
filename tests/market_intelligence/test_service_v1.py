from __future__ import annotations

import pandas as pd

from quant_engine.market_intelligence.feature_store_memory import InMemoryFeatureStore
from quant_engine.market_intelligence.service import MarketIntelligenceServiceV1


class _DummyStatsAdapter:
    def __init__(self) -> None:
        self.calls = 0

    def run(self, spec):
        self.calls += 1
        return pd.DataFrame(
            {
                "symbol": [spec["symbol"]],
                "event": ["dummy"],
                "p_hat": [0.5],
            },
            index=pd.to_datetime(["2026-01-01T00:00:00Z"], utc=True),
        )


class _DummyFiltersAdapter:
    def __init__(self) -> None:
        self.calls = 0

    def run(self, ohlcv, rules, **_kwargs):
        self.calls += 1
        return pd.DataFrame(
            {
                "hard_mask": [True] * len(ohlcv),
                "score": [1.0] * len(ohlcv),
                "score_pct": [1.0] * len(ohlcv),
                "final_mask": [True] * len(ohlcv),
            },
            index=ohlcv.index.tz_localize("UTC") if ohlcv.index.tz is None else ohlcv.index,
        )


def _fixed_ohlcv() -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=12, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "open": [100, 101, 102, 103, 102, 101, 100, 99, 99.5, 100, 101, 102],
            "high": [101, 102, 103, 104, 103, 102, 101, 100, 100, 101, 102, 103],
            "low": [99, 100, 101, 102, 101, 100, 99, 98, 99, 99.5, 100, 101],
            "close": [100.5, 101.5, 102.5, 102.8, 101.7, 100.3, 99.4, 99.2, 99.7, 100.4, 101.3, 102.2],
            "volume": [1000, 1100, 1200, 1300, 900, 800, 700, 650, 680, 720, 760, 800],
        },
        index=idx,
    )


def test_service_v1_build_snapshot_is_deterministic_with_fixed_dataset() -> None:
    ohlcv = _fixed_ohlcv()
    service = MarketIntelligenceServiceV1(
        stats_adapter=_DummyStatsAdapter(),
        filters_adapter=_DummyFiltersAdapter(),
    )

    first = service.build_snapshot("BTC-USD", ohlcv)
    second = service.build_snapshot("BTC-USD", ohlcv)

    pd.testing.assert_frame_equal(first["features"], second["features"])
    pd.testing.assert_frame_equal(first["regimes"], second["regimes"])
    pd.testing.assert_frame_equal(first["liquidity"], second["liquidity"])


def test_service_v1_uses_read_through_write_through_cache() -> None:
    ohlcv = _fixed_ohlcv()
    stats = _DummyStatsAdapter()
    filters = _DummyFiltersAdapter()
    store = InMemoryFeatureStore()

    service = MarketIntelligenceServiceV1(
        feature_store=store,
        stats_adapter=stats,
        filters_adapter=filters,
    )

    cold = service.build_snapshot("ETH-USD", ohlcv)
    warm = service.build_snapshot("ETH-USD", ohlcv)

    assert stats.calls == 1
    assert filters.calls == 1
    assert store.exists("market_intelligence", "ETH-USD", "1h", "1.0.0")

    pd.testing.assert_frame_equal(cold["features"], warm["features"])
    pd.testing.assert_frame_equal(cold["legacy_stats"], warm["legacy_stats"])


def test_service_v1_methods_delegate_pipeline_shapes() -> None:
    service = MarketIntelligenceServiceV1(
        stats_adapter=_DummyStatsAdapter(),
        filters_adapter=_DummyFiltersAdapter(),
    )
    ohlcv = _fixed_ohlcv()

    features = service.compute_features(ohlcv)
    regimes = service.label_regimes(features)
    flags = service.liquidity_flags(ohlcv)

    assert list(features.columns) == ["feat_return_1", "feat_volatility_5", "feat_corr_close_volume_5"]
    assert list(regimes.columns) == ["label_regime"]
    assert list(flags.columns) == ["liq_low_volume", "liq_wide_spread", "liq_illiquid"]
    assert len(features) == len(ohlcv)
    assert len(regimes) == len(ohlcv)
    assert len(flags) == len(ohlcv)
