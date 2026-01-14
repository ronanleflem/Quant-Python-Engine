from quant_engine.backtest import runner


def test_apply_filter_mask_require_crossing_true():
    signal = [0, 1, 1, 1, 0, 1]
    mask = [False, False, True, False, False, False]

    gated = runner._apply_filter_mask(signal, mask, True)

    assert gated == [0, 0, 1, 0, 0, 0]


def test_apply_filter_mask_require_crossing_false():
    signal = [0, 1, 1, 1, 0, 1]
    mask = [False, False, True, False, False, False]

    gated = runner._apply_filter_mask(signal, mask, False)

    assert gated == [0, 0, 1, 1, 0, 0]


def test_resolve_require_crossing_prefers_signal_params():
    spec = {
        "signal": {"params": {"require_crossing": "false"}},
        "strategy": {"params": {"require_crossing": True}},
    }

    assert runner._resolve_require_crossing(spec) is False


def test_apply_filter_mask_gated_on_first_signal():
    signal = [0, 1, 1, 0, 1, 1, 1]
    mask = [False, True, False, False, False, True, False]

    gated = runner._apply_filter_mask(signal, mask, False)

    assert gated == [0, 1, 1, 0, 0, 1, 1]
