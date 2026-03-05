from __future__ import annotations

from typing import Any, Mapping, Sequence

from quant_engine.core.contracts import FeatureStore, MarketIntelligenceService, StrategyContract


class FakeMarketIntelligence:
    def build_snapshot(self, symbol: str, ohlcv: Any) -> Mapping[str, Any]:
        return {"symbol": symbol, "rows": len(ohlcv)}


class FakeFeatureStore:
    def __init__(self) -> None:
        self._store: dict[tuple[str, str, str, str], Any] = {}

    def get(self, feature_set: str, symbol: str, timeframe: str, version: str) -> Any:
        return self._store[(feature_set, symbol, timeframe, version)]

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
        key = (feature_set, symbol, timeframe, version)
        if key in self._store and not overwrite:
            raise KeyError(key)
        self._store[key] = payload

    def exists(self, feature_set: str, symbol: str, timeframe: str, version: str) -> bool:
        return (feature_set, symbol, timeframe, version) in self._store


class FakeStrategy:
    def evaluate(self, ohlcv: Any, context: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
        return [{"signal": "BUY", "rows": len(ohlcv), "context": dict(context)}]


def test_contracts_are_importable() -> None:
    assert MarketIntelligenceService is not None
    assert FeatureStore is not None
    assert StrategyContract is not None


def test_fake_implementations_conform_to_contracts() -> None:
    mi = FakeMarketIntelligence()
    fs = FakeFeatureStore()
    st = FakeStrategy()

    assert isinstance(mi, MarketIntelligenceService)
    assert isinstance(fs, FeatureStore)
    assert isinstance(st, StrategyContract)

    snapshot = mi.build_snapshot("BTC-USD", [{"close": 1.0}, {"close": 2.0}])
    assert snapshot["symbol"] == "BTC-USD"
    assert snapshot["rows"] == 2

    assert fs.exists("market_intelligence", "BTC-USD", "1h", "1.0.0") is False
    fs.put("market_intelligence", "BTC-USD", "1h", "1.0.0", {"features": [1.0, 2.0]})
    assert fs.exists("market_intelligence", "BTC-USD", "1h", "1.0.0") is True
    payload = fs.get("market_intelligence", "BTC-USD", "1h", "1.0.0")
    assert payload["features"] == [1.0, 2.0]

    signals = st.evaluate([{"close": 1.0}], {"mode": "backtest"})
    assert signals[0]["signal"] == "BUY"
