from __future__ import annotations

from quant_engine.performance.stress_tests import run_scenarios_on_returns


def test_regime_labels_modulate_crash_and_volatility_and_trace_payload() -> None:
    returns = [10.0, -4.0, 6.0, -2.0, 3.0]
    params = {
        "initial_capital": 1000.0,
        "regime_labels": ["bull", "regime_shift", "vol_spike", "bull", "bull"],
        "regime_shift_shock_multiplier": 2.0,
        "vol_spike_multiplier": 1.75,
        "scenarios": [
            {"name": "crash_mid", "type": "crash", "shock_pct": -0.1, "index": 1},
            {"name": "vol_event", "type": "volatility", "vol_multiplier": 2.0},
        ],
    }

    result = run_scenarios_on_returns(returns, parameters=params)

    crash = result["distributions"]["scenarios"]["crash_mid"]
    vol = result["distributions"]["scenarios"]["vol_event"]

    crash_context = crash["parameters"]["applied_regime_context"]
    assert crash_context["regime_label"] == "regime_shift"
    assert crash_context["shock_multiplier_applied"] == 2.0

    vol_context = vol["parameters"]["applied_regime_context"]
    assert vol_context["regime_label"] == "vol_spike"
    assert vol_context["vol_multiplier_applied"] == 1.75

    assert result["parameters"]["regime_labels"] == params["regime_labels"]
    assert result["parameters"]["regime_shift_shock_multiplier"] == 2.0
    assert result["parameters"]["vol_spike_multiplier"] == 1.75
