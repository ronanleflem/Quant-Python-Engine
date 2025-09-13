from quant_engine.core import spec


def test_load_spec():
    sp = spec.load_spec("tests/data/spec_example.json")
    assert sp.data.symbols == ["ABC"]
    assert sp.strategy.filters.ema_fast == 2
    assert sp.strategy.tpsl.atr_k == 1.5
    assert sp.strategy.validation.folds == 1
