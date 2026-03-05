from __future__ import annotations

import pandas as pd

from quant_engine.market_intelligence.pipeline import compute_features, label_regimes, liquidity_flags

EXPECTED_FEATURE_COLUMNS = (
    "feat_return_1",
    "feat_volatility_5",
    "feat_corr_close_volume_5",
)
EXPECTED_REGIME_COLUMNS = ("label_regime",)
EXPECTED_LIQUIDITY_COLUMNS = ("liq_low_volume", "liq_wide_spread", "liq_illiquid")


def _sample_ohlcv() -> pd.DataFrame:
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


def test_market_intelligence_column_snapshot_is_stable() -> None:
    ohlcv = _sample_ohlcv()

    features = compute_features(ohlcv)
    regimes = label_regimes(features)
    liquidity = liquidity_flags(ohlcv)

    assert tuple(features.columns) == EXPECTED_FEATURE_COLUMNS
    assert tuple(regimes.columns) == EXPECTED_REGIME_COLUMNS
    assert tuple(liquidity.columns) == EXPECTED_LIQUIDITY_COLUMNS
