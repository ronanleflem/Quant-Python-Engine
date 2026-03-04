"""Contract for feature-store components used to cache and retrieve engineered features."""
from __future__ import annotations

from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable


@runtime_checkable
class FeatureStore(Protocol):
    """Store and retrieve computed feature vectors.

    Inputs:
        name: Feature name (e.g. ``"ema_20"``).
        dataset: Source rows used to compute feature values.
        params: Deterministic feature parameters.
        compute_fn: Callable used when values are not already available.

    Output:
        Ordered sequence of numeric feature values aligned with ``dataset``.
    """

    def get_or_compute(
        self,
        name: str,
        dataset: Sequence[Mapping[str, Any]],
        params: Mapping[str, Any],
        compute_fn: Callable[[Sequence[Mapping[str, Any]], Mapping[str, Any]], Sequence[float]],
    ) -> Sequence[float]:
        """Return cached values or compute and persist them."""
