from __future__ import annotations

import pandas as pd
import pytest

from quant_engine.market_intelligence.feature_store_parquet import ParquetFeatureStore


def test_put_get_roundtrip_with_period_filter(tmp_path) -> None:
    store = ParquetFeatureStore(tmp_path)
    key = ("volatility", "btc-usd", "1H", "1.0.0")

    payload = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="D"),
            "feat_vol": [0.1, 0.2, 0.3, 0.4, 0.5],
            "feature_version": ["1.0.0"] * 5,
        }
    )

    path = store.put(*key, payload=payload)
    assert path.exists()
    assert str(path).endswith("volatility/symbol=BTC-USD/timeframe=1h/version=1.0.0/data.parquet")

    loaded = store.get(*key)
    pd.testing.assert_frame_equal(loaded.reset_index(drop=True), payload.reset_index(drop=True))

    filtered = store.get(*key, start="2024-01-02", end="2024-01-04")
    assert filtered["timestamp"].tolist() == list(pd.date_range("2024-01-02", periods=3, freq="D"))
    assert filtered["feat_vol"].tolist() == [0.2, 0.3, 0.4]


def test_get_raises_on_feature_version_mismatch(tmp_path) -> None:
    store = ParquetFeatureStore(tmp_path)
    key = ("momentum", "ETH-USD", "4h", "2.0.0")

    payload = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-03-01", periods=2, freq="h"),
            "feat_mom": [1.0, 1.1],
            "feature_version": ["9.9.9", "9.9.9"],
        }
    )
    path = store._build_path(*key)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload.to_parquet(path, index=False)

    with pytest.raises(ValueError, match="Stored feature_version does not match requested feature_version"):
        store.get(*key)
