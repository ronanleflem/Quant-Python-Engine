from quant_engine.core import dataset
from quant_engine.core.features import ema, vwap, atr
from quant_engine.core.spec import load_spec


def test_ema_vwap():
    sp = load_spec("tests/data/spec_example.json")
    data = dataset.load_dataset(sp.data)
    ema_vals = ema.compute(data, {"period": 2})
    vw_vals = vwap.compute(data)
    atr_vals = atr.compute(data, {"period": 2})
    assert len(ema_vals) == len(data)
    assert len(vw_vals) == len(data)
    assert len(atr_vals) == len(data)
    assert ema_vals[0] == data[0]["close"]
