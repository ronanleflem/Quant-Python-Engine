from quant_engine.core import dataset, spec
from quant_engine.signals.ema_cross import EmaCross
from quant_engine.core.features import atr
from quant_engine.backtest import engine


def test_backtest_runs():
    sp = spec.load_spec("tests/data/spec_example.json")
    data = dataset.load_dataset(sp.data)
    signal = EmaCross(sp.strategy.filters.ema_fast, sp.strategy.filters.ema_slow)
    sigs = signal.generate(data)
    atr_vals = atr.compute(data)
    trades, equity, summary = engine.run(
        data, sigs, atr_vals, sp.strategy.tpsl.atr_k, 1
    )
    assert len(trades) >= 0
    assert len(equity) == len(data)
    assert "sharpe" in summary
