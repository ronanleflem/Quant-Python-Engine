import random
import pytest

from quant_engine.performance import stress_tests


def _expected_total_with_per_sim(returns, seed_sampling, mult):
    rng_sampling = random.Random(seed_sampling)
    indices = list(range(len(returns)))
    rng_sampling.shuffle(indices)
    shuffled = [returns[i] for i in indices]
    return sum(shuffled) * mult


def test_param_drift_per_sim_applies_single_multiplier() -> None:
    returns = [10.0, -4.0, 6.0]
    params = {
        "n_sims": 1,
        "seed": 11,
        "method": "iid",
        "initial_capital": 100.0,
        "param_drift": {
            "mode": "per_sim",
            "dist": "uniform",
            "low": 0.5,
            "high": 0.5,
            "seed": 7,
        },
    }

    result = stress_tests.run_monte_carlo_on_returns(returns, parameters=params)
    total = result["distributions"]["level1"]["total_return"][0]
    expected = _expected_total_with_per_sim(returns, seed_sampling=11, mult=0.5)
    assert total == pytest.approx(expected)


def test_param_drift_random_walk_compounds() -> None:
    returns = [5.0, 5.0, 5.0]
    params = {
        "n_sims": 1,
        "seed": 3,
        "method": "iid",
        "initial_capital": 0.0,
        "param_drift": {
            "mode": "random_walk",
            "dist": "uniform",
            "low": 1.1,
            "high": 1.1,
            "seed": 9,
        },
    }

    result = stress_tests.run_monte_carlo_on_returns(returns, parameters=params)
    total = result["distributions"]["level1"]["total_return"][0]
    expected = 5.0 * 1.1 + 5.0 * (1.1 ** 2) + 5.0 * (1.1 ** 3)
    assert total == pytest.approx(expected)
