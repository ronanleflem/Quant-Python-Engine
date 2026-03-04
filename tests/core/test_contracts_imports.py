from __future__ import annotations

from typing import Any, Mapping, Sequence

from quant_engine.core.contracts import FeatureStore, MarketIntelligenceService, StrategyContract


class FakeMarketIntelligence:
    def build_snapshot(self, symbol: str, ohlcv: Any) -> Mapping[str, Any]:
        return {"symbol": symbol, "rows": len(ohlcv)}


class FakeFeatureStore:
    def get_or_compute(self, name: str, dataset: Sequence[Mapping[str, Any]], params: Mapping[str, Any], compute_fn):
        return list(compute_fn(dataset, params))


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

    values = fs.get_or_compute(
        "ema_2",
        [{"close": 1.0}, {"close": 2.0}],
        {"period": 2},
        lambda dataset, _: [float(row["close"]) for row in dataset],
    )
    assert values == [1.0, 2.0]

    signals = st.evaluate([{"close": 1.0}], {"mode": "backtest"})
    assert signals[0]["signal"] == "BUY"
