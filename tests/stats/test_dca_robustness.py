import pytest

from quant_engine.stats import estimators


def test_monte_carlo_calendar_distribution_is_seed_deterministic() -> None:
    values = [0.1, 0.2, 0.3, 0.4]
    a = estimators.monte_carlo_calendar_distribution(values, n_runs=20, sample_size=3, seed=123)
    b = estimators.monte_carlo_calendar_distribution(values, n_runs=20, sample_size=3, seed=123)
    c = estimators.monte_carlo_calendar_distribution(values, n_runs=20, sample_size=3, seed=124)
    assert a == b
    assert a != c


def test_percentile_rank_consistency() -> None:
    dist = [1.0, 2.0, 3.0, 4.0]
    assert estimators.percentile_rank(3.0, dist) == pytest.approx(75.0)
    assert estimators.percentile_rank(0.0, dist) == pytest.approx(0.0)


def test_stochastic_dominance_simplified_flags() -> None:
    dom = estimators.stochastic_dominance_simplified(
        observed_perf=0.9,
        observed_drawdown=0.1,
        passive_perf_distribution=[0.1, 0.2, 0.4],
        passive_drawdown_distribution=[0.2, 0.3, 0.5],
        threshold=0.6,
    )
    assert dom["perf_dominance_prob"] == pytest.approx(1.0)
    assert dom["drawdown_dominance_prob"] == pytest.approx(1.0)
    assert dom["is_dominant"] is True
