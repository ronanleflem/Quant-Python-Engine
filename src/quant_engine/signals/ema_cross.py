"""Simple EMA cross over signal."""
from __future__ import annotations

from typing import List, Dict, Any

from ..core.features import ema
from .base import Signal


class EmaCross(Signal):
    def __init__(self, fast: int, slow: int) -> None:
        self.fast = fast
        self.slow = slow

    def generate(self, dataset: List[Dict[str, Any]]) -> List[int]:
        fast_vals = ema.compute(dataset, {"period": self.fast})
        slow_vals = ema.compute(dataset, {"period": self.slow})
        return [1 if f > s else 0 for f, s in zip(fast_vals, slow_vals)]

