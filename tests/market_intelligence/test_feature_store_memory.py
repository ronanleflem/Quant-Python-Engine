from __future__ import annotations

import pytest

from quant_engine.market_intelligence.feature_store_memory import InMemoryFeatureStore


@pytest.fixture
def store() -> InMemoryFeatureStore:
    return InMemoryFeatureStore()


def test_put_get_exists_delete_roundtrip(store: InMemoryFeatureStore) -> None:
    key = ("volatility", "BTC-USD", "1h", "1.0.0")
    payload = {"feat_vol": [0.1, 0.2, 0.3]}

    assert store.exists(*key) is False

    store.put(*key, payload=payload)

    assert store.exists(*key) is True
    assert store.get(*key) == payload
    assert store.delete(*key) is True
    assert store.exists(*key) is False
    assert store.delete(*key) is False


def test_put_rejects_existing_key_without_explicit_overwrite(store: InMemoryFeatureStore) -> None:
    key = ("momentum", "ETH-USD", "4h", "2.0.0")
    store.put(*key, payload={"feat_momentum": [1.0]})

    with pytest.raises(KeyError):
        store.put(*key, payload={"feat_momentum": [2.0]})


def test_put_allows_explicit_overwrite(store: InMemoryFeatureStore) -> None:
    key = ("trend", "SOL-USD", "15m", "3.1.0")
    store.put(*key, payload={"feat_trend": [10]})

    store.put(*key, payload={"feat_trend": [20]}, overwrite=True)

    assert store.get(*key) == {"feat_trend": [20]}


def test_sequential_writes_keep_last_value_per_key(store: InMemoryFeatureStore) -> None:
    key = ("liquidity", "ADA-USD", "1d", "1.0.1")

    for value in range(10):
        payload = {"feat_liquidity": [float(value)]}
        if store.exists(*key):
            store.put(*key, payload=payload, overwrite=True)
        else:
            store.put(*key, payload=payload)

    assert store.get(*key) == {"feat_liquidity": [9.0]}
