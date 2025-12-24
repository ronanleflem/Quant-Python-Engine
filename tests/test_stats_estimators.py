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
