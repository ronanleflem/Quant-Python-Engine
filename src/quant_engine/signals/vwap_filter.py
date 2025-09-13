"""Filter trades based on proximity to VWAP using ATR as scale."""
from __future__ import annotations

from typing import List, Dict, Any

from ..core.features import vwap
from ..tpsl import rules
from .base import Signal


class VwapFilter(Signal):
    def __init__(self, atr_mult: float) -> None:
        self.atr_mult = atr_mult

    def generate(self, dataset: List[Dict[str, Any]]) -> List[int]:
        vw = vwap.compute(dataset)
        atr = rules.atr(dataset)
        out: List[int] = []
        for row, v, a in zip(dataset, vw, atr):
            diff = abs(row["close"] - v)
            out.append(1 if diff <= self.atr_mult * a else 0)
        return out

