import pandas as pd
import pytest

from quant_engine.stats import estimators


def test_freq_with_wilson_zero_case():
    assert estimators.freq_with_wilson(0, 0) == (0.0, 0.0, 0.0)


def test_benjamini_hochberg_fixed_set():
    pvals = [0.01, 0.04, 0.03, 0.002]
    qvals = estimators.benjamini_hochberg(pvals)
    assert qvals == [0.02, 0.04, 0.04, 0.008]


def test_beta_hdi_bounds():
    low, high = estimators.beta_hdi(2.5, 3.5)
    assert 0.0 <= low <= 1.0
    assert 0.0 <= high <= 1.0
    assert low < high


def test_freq_with_wilson_large_n_stability():
    p, low, high = estimators.freq_with_wilson(500_000, 1_000_000)
    assert 0.0 <= low <= p <= high <= 1.0


def test_binary_wilson_and_bayes():
    outcomes = pd.Series([1, 0, 1, 1, 0])
    summary = estimators.aggregate_binary(outcomes)

    assert summary["n"] == 5
    assert summary["successes"] == 3
    assert summary["p_hat"] == pytest.approx(0.6)
    assert 0.0 <= summary["ci_low"] <= summary["p_hat"] <= summary["ci_high"] <= 1.0

    bayes = estimators.aggregate_binary_bayes(summary["successes"], summary["n"])
    assert bayes["alpha_post"] == pytest.approx(3.5)
    assert bayes["beta_post"] == pytest.approx(2.5)
    assert bayes["p_mean"] == pytest.approx(3.5 / 6.0)
    assert bayes["p_map"] == pytest.approx(0.625)
    assert 0.0 <= bayes["hdi_low"] < bayes["hdi_high"] <= 1.0


def test_binary_zero_stability():
    outcomes = pd.Series([], dtype=float)
    summary = estimators.aggregate_binary(outcomes)
    assert summary == {
        "n": 0,
        "successes": 0,
        "p_hat": 0.0,
        "ci_low": 0.0,
        "ci_high": 0.0,
    }

    bayes = estimators.aggregate_binary_bayes(0, 0)
    assert bayes["alpha_post"] == pytest.approx(0.5)
    assert bayes["beta_post"] == pytest.approx(0.5)
    assert bayes["p_mean"] == pytest.approx(0.5)
    assert bayes["p_map"] == pytest.approx(0.5)
    assert 0.0 <= bayes["hdi_low"] < bayes["hdi_high"] <= 1.0
