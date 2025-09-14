"""Estimator helpers for statistics runs."""
from __future__ import annotations

from typing import Tuple


def phat(successes: int, trials: int) -> float:
    """Return the empirical success probability."""

    return successes / trials if trials else 0.0


def wilson_ci(successes: int, trials: int, z: float = 1.96) -> Tuple[float, float]:
    """Return a Wilson score confidence interval (placeholder)."""

    return (0.0, 0.0)


def baseline(trials: int) -> float:
    """Return a baseline probability estimate."""

    return 0.0

