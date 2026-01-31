import random
import pytest

from quant_engine.performance import stress_tests


def test_monte_carlo_time_distribution_exit_times_affects_ttr() -> None:
    trades = [
        {
            "pnl": 10.0,
            "ts_entry": "2020-01-01T00:00:00",
            "ts_exit": "2020-01-01T01:00:00",
        },
        {
            "pnl": -30.0,
            "ts_entry": "2020-01-02T00:00:00",
            "ts_exit": "2020-01-10T00:00:00",
        },
        {
            "pnl": 40.0,
            "ts_entry": "2020-01-15T00:00:00",
            "ts_exit": "2020-02-01T00:00:00",
        },
    ]
    params = {
        "n_sims": 1,
        "seed": 11,
        "method": "iid",
        "initial_capital": 100.0,
        "time_distribution": {"mode": "exit_times", "seed": 5},
    }

    result = stress_tests.run_monte_carlo_on_trades(trades, parameters=params)
    ttr = result["metrics"]["time_to_recovery_days"]["p50"]

    normalized = stress_tests.standardize_trades(trades)
    ordered = sorted(
        normalized,
        key=lambda t: t.exit_time_utc if t.exit_time_utc is not None else stress_tests.datetime.min,
    )
    rng_sampling = random.Random(11)
    sampled_trades = stress_tests._sample_iid(ordered, rng_sampling)
    sampled_pnl = [t.pnl for t in sampled_trades]
    equity = [100.0]
    for pnl in sampled_pnl:
        equity.append(equity[-1] + pnl)

    base_exit = [t.exit_time_utc for t in ordered]
    rng_time = random.Random(5)
    sampled_exit = [rng_time.choice(base_exit) for _ in range(len(sampled_trades))]
    sampled_exit.sort()
    timestamps = [sampled_exit[0]] + sampled_exit
    expected = stress_tests._time_to_recovery(equity, timestamps)

    assert ttr == pytest.approx(expected)
