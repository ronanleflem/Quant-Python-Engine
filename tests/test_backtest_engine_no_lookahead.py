from quant_engine.backtest import engine


def test_backtest_engine_uses_next_open_for_entry_and_exit():
    dataset = [
        {
            "timestamp": "2020-01-01",
            "open": 100.0,
            "high": 105.0,
            "low": 95.0,
            "close": 102.0,
        },
        {
            "timestamp": "2020-01-02",
            "open": 110.0,
            "high": 112.0,
            "low": 108.0,
            "close": 111.0,
        },
        {
            "timestamp": "2020-01-03",
            "open": 120.0,
            "high": 125.0,
            "low": 115.0,
            "close": 122.0,
        },
    ]
    signals = [1, 0, 0]
    atr_values = [1.0, 1.0, 1.0]

    trades, _, _ = engine.run(dataset, signals, atr_values, atr_mult=1.0, r_mult=1.0)

    assert len(trades) == 1
    trade = trades[0]
    assert trade["price_entry"] == dataset[1]["open"]
    assert trade["price_exit"] == dataset[2]["open"]
