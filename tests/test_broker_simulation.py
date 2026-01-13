from pytest import approx

from quant_engine.execution.broker import PortfolioBroker


def test_broker_fills_with_fees_and_slippage() -> None:
    broker = PortfolioBroker(initial_cash=2000.0, commission_rate=0.001, slippage_bps=10)

    buy_order = broker.place_order("AAPL", "buy", quantity=10, price=100.0)

    assert buy_order.status == "filled"
    assert buy_order.fill_price == approx(100.1)
    assert broker.positions["AAPL"].quantity == 10
    assert broker.positions["AAPL"].average_price == approx(100.1)

    sell_order = broker.place_order("AAPL", "sell", quantity=5, price=110.0)

    assert sell_order.status == "filled"
    assert sell_order.fill_price == approx(109.89)
    assert broker.positions["AAPL"].quantity == 5
    assert len(broker.trades) == 1
    trade = broker.trades[0]
    assert trade.entry_price == approx(100.1)
    assert trade.exit_price == approx(109.89)
    assert trade.quantity == 5


def test_broker_rejects_invalid_order() -> None:
    broker = PortfolioBroker()

    order = broker.place_order("AAPL", "buy", quantity=0, price=100.0)

    assert order.status == "rejected"
    assert order.reason == "quantity and price must be positive"
