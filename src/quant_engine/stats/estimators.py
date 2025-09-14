"""Estimator helpers for statistics runs."""
from __future__ import annotations

from math import sqrt
from statistics import NormalDist
from typing import Dict, Tuple

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


__all__ = ["freq_with_wilson", "aggregate_binary", "baseline"]

