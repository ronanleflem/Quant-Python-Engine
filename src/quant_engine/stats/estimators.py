"""Estimator helpers for statistics runs."""
from __future__ import annotations

from math import sqrt
from statistics import NormalDist
from typing import Dict, Tuple

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
    "beta_binomial_posterior",
    "posterior_mean",
    "posterior_map",
    "beta_hdi",
    "aggregate_binary_bayes",
]

