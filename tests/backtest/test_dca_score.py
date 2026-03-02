import pytest

from quant_engine.backtest.metrics import dca_composite_score


def test_dca_composite_score_is_bounded_and_decomposable() -> None:
    payload = dca_composite_score(
        final_performance_normalized_value=0.30,
        xirr_value=0.12,
        max_drawdown_on_contributed_capital_value=0.15,
        underperformance_duration_windows=1,
        underperformance_severity_pct_points=5.0,
        xirr_status="ok",
    )
    assert 0.0 <= payload["score"] <= 1.0
    assert payload["edge"] in {"weak", "medium", "strong"}
    assert set(payload["components"].keys()) == {"performance", "irr", "drawdown", "robustness"}
    assert payload["score"] == pytest.approx(sum(payload["contributions"].values()))


def test_dca_composite_score_local_monotonicity() -> None:
    base = dca_composite_score(
        final_performance_normalized_value=0.10,
        xirr_value=0.08,
        max_drawdown_on_contributed_capital_value=0.20,
        underperformance_duration_windows=3,
        underperformance_severity_pct_points=15.0,
        xirr_status="ok",
    )
    better_perf = dca_composite_score(
        final_performance_normalized_value=0.25,
        xirr_value=0.08,
        max_drawdown_on_contributed_capital_value=0.20,
        underperformance_duration_windows=3,
        underperformance_severity_pct_points=15.0,
        xirr_status="ok",
    )
    better_drawdown = dca_composite_score(
        final_performance_normalized_value=0.10,
        xirr_value=0.08,
        max_drawdown_on_contributed_capital_value=0.10,
        underperformance_duration_windows=3,
        underperformance_severity_pct_points=15.0,
        xirr_status="ok",
    )
    assert better_perf["score"] >= base["score"]
    assert better_drawdown["score"] >= base["score"]


def test_dca_composite_score_custom_weights_are_normalized() -> None:
    payload = dca_composite_score(
        final_performance_normalized_value=0.10,
        xirr_value=0.08,
        max_drawdown_on_contributed_capital_value=0.20,
        weights={"performance": 2.0, "irr": 1.0, "drawdown": 1.0, "robustness": 0.0},
    )
    assert payload["weights"]["performance"] == pytest.approx(0.5)
    assert payload["weights"]["irr"] == pytest.approx(0.25)
    assert payload["weights"]["drawdown"] == pytest.approx(0.25)
    assert payload["weights"]["robustness"] == pytest.approx(0.0)
