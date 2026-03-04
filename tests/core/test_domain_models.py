from dataclasses import FrozenInstanceError
from datetime import datetime

import pytest

from quant_engine.core.domain import Candle, Portfolio, Position, Trade


def test_domain_models_creation() -> None:
    ts = datetime(2024, 1, 1, 12, 0)

    candle = Candle(ts, "BTC-USD", 100.0, 110.0, 95.0, 105.0, 1000.0)
    trade = Trade("t-1", "BTC-USD", "buy", 2.0, 101.0, ts)
    position = Position("BTC-USD", 2.0, 101.0)
    portfolio = Portfolio(5000.0, 5200.0)

    assert candle.symbol == "BTC-USD"
    assert trade.side == "buy"
    assert position.average_price == 101.0
    assert portfolio.equity == 5200.0


def test_domain_models_are_immutable() -> None:
    candle = Candle(datetime(2024, 1, 1), "BTC-USD", 1.0, 2.0, 0.5, 1.5, 10.0)

    with pytest.raises(FrozenInstanceError):
        candle.close = 2.5  # type: ignore[misc]


def test_domain_models_require_mandatory_fields() -> None:
    with pytest.raises(TypeError):
        Candle(datetime(2024, 1, 1), "BTC-USD", 1.0, 2.0, 0.5, 1.5)  # type: ignore[call-arg]

    with pytest.raises(TypeError):
        Trade("t-1", "BTC-USD", "buy", 1.0, 100.0)  # type: ignore[call-arg]


def test_domain_models_equality() -> None:
    ts = datetime(2024, 1, 1)
    left = Position("BTC-USD", 2.0, 100.0)
    right = Position("BTC-USD", 2.0, 100.0)
    portfolio = Portfolio(1000.0, 1200.0)

    assert left == right
    assert left != Position("ETH-USD", 2.0, 100.0)
    assert portfolio == Portfolio(1000.0, 1200.0)
