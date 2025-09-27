"""Estimator helpers for statistics runs."""
from __future__ import annotations

from math import sqrt
from statistics import NormalDist
from typing import Dict, Tuple

import math
import numpy as np
import pandas as pd


def phat(successes: int, trials: int) -> float:
    """Return the empirical success probability."""

    return successes / trials if trials else 0.0


def freq_with_wilson(successes: int, n: int, alpha: float = 0.05) -> Tuple[float, float, float]:
    """Return empirical frequency and Wilson score interval."""

    if n == 0:
        return 0.0, 0.0, 0.0
    z = NormalDist().inv_cdf(1 - alpha / 2)
    p = successes / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    ci_low = (center - margin) / denom
    ci_high = (center + margin) / denom
    return p, ci_low, ci_high


def aggregate_binary(outcomes: pd.Series) -> Dict[str, float]:
    """Aggregate binary outcomes with Wilson interval."""

    n = int(outcomes.count())
    successes = int(outcomes.sum()) if n else 0
    p, ci_low, ci_high = freq_with_wilson(successes, n)
    return {
        "n": n,
        "successes": successes,
        "p_hat": p,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def baseline(outcomes: pd.Series) -> float:
    """Return the global empirical success probability."""

    n = int(outcomes.count())
    successes = int(outcomes.sum()) if n else 0
    return phat(successes, n)


def p_value_binomial_onesided_normal(
    successes: int,
    n: int,
    p0: float,
    direction: str = "greater",
) -> float:
    """
    Approximate one-sided p-value for a binomial test using the normal
    approximation with continuity correction.

    Parameters
    ----------
    successes: int
        Number of observed successes.
    n: int
        Number of trials.
    p0: float
        Null hypothesis probability of success.
    direction: str
        'greater' for H1: p > p0, 'less' for H1: p < p0.
    """

    if n == 0:
        return 1.0
    mean = n * p0
    var = n * p0 * (1 - p0)
    if var == 0:
        if direction == "greater":
            return 0.0 if successes > mean else 1.0
        return 0.0 if successes < mean else 1.0
    sd = math.sqrt(var)
    if direction == "greater":
        z = (successes - mean - 0.5) / sd
        return 1 - NormalDist().cdf(z)
    else:
        z = (successes - mean + 0.5) / sd
        return NormalDist().cdf(z)


def benjamini_hochberg(pvals: list[float], alpha: float = 0.05) -> list[float]:
    """Return FDR-adjusted p-values using the Benjamini–Hochberg procedure."""

    m = len(pvals)
    if m == 0:
        return []
    idx = sorted(range(m), key=lambda i: pvals[i])
    sorted_p = [pvals[i] for i in idx]
    qvals = [0.0] * m
    prev = 1.0
    for rank, p in reversed(list(enumerate(sorted_p, start=1))):
        q = p * m / rank
        if q > prev:
            q = prev
        prev = q
        qvals[idx[rank - 1]] = min(q, 1.0)
    return qvals


# Bayesian estimators


def beta_binomial_posterior(
    successes: int,
    n: int,
    alpha_prior: float = 0.5,
    beta_prior: float = 0.5,
) -> Tuple[float, float]:
    """Return posterior alpha and beta parameters for a Beta prior."""

    failures = n - successes
    alpha_post = successes + alpha_prior
    beta_post = failures + beta_prior
    return alpha_post, beta_post


def posterior_mean(alpha_post: float, beta_post: float) -> float:
    """Return the posterior mean of the success probability."""

    return alpha_post / (alpha_post + beta_post)


def posterior_map(alpha_post: float, beta_post: float) -> float:
    """Return the posterior MAP estimate of the success probability."""

    if alpha_post > 1 and beta_post > 1:
        return (alpha_post - 1) / (alpha_post + beta_post - 2)
    return posterior_mean(alpha_post, beta_post)


def beta_hdi(
    alpha_post: float,
    beta_post: float,
    cred_mass: float = 0.95,
    num_points: int = 2000,
) -> Tuple[float, float]:
    """Approximate the highest density interval for a Beta distribution."""

    eps = 1e-8
    xs = np.linspace(eps, 1 - eps, num_points)
    log_pdf = (alpha_post - 1) * np.log(xs) + (beta_post - 1) * np.log(1 - xs)
    log_pdf -= np.max(log_pdf)
    pdf = np.exp(log_pdf)
    norm = np.trapz(pdf, xs)
    if norm == 0:
        return 0.0, 1.0
    pdf /= norm
    dx = xs[1] - xs[0]
    idx = np.argsort(pdf)[::-1]
    cumsum = np.cumsum(pdf[idx] * dx)
    k = np.searchsorted(cumsum, cred_mass)
    threshold = pdf[idx[k]]
    mask = pdf >= threshold
    low = float(xs[mask][0])
    high = float(xs[mask][-1])
    return low, high


def aggregate_binary_bayes(
    successes: int,
    n: int,
    cred_mass: float = 0.95,
    alpha_prior: float = 0.5,
    beta_prior: float = 0.5,
) -> Dict[str, float]:
    """Aggregate binary outcomes using a Beta-Binomial model."""

    alpha_post, beta_post = beta_binomial_posterior(
        successes, n, alpha_prior, beta_prior
    )
    p_mean = posterior_mean(alpha_post, beta_post)
    p_map = posterior_map(alpha_post, beta_post)
    hdi_low, hdi_high = beta_hdi(alpha_post, beta_post, cred_mass)
    return {
        "n": n,
        "successes": successes,
        "alpha_post": alpha_post,
        "beta_post": beta_post,
        "p_mean": p_mean,
        "p_map": p_map,
        "hdi_low": hdi_low,
        "hdi_high": hdi_high,
    }


__all__ = [
    "freq_with_wilson",
    "aggregate_binary",
    "baseline",
    "p_value_binomial_onesided_normal",
    "benjamini_hochberg",
    "beta_binomial_posterior",
    "posterior_mean",
    "posterior_map",
    "beta_hdi",
    "aggregate_binary_bayes",
]

