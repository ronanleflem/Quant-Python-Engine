"""Exponential moving average calculations."""
from __future__ import annotations

from typing import List, Dict, Any


def compute(dataset: List[Dict[str, Any]], params: Dict[str, Any]) -> List[float]:
    period = int(params["period"])
    prices = [row["close"] for row in dataset]
    if not prices:
        return []
    alpha = 2 / (period + 1)
    ema_values = [prices[0]]
    for price in prices[1:]:
        ema_values.append(alpha * price + (1 - alpha) * ema_values[-1])
    return ema_values


def compute_many(dataset: List[Dict[str, Any]], periods: List[int]) -> Dict[int, List[float]]:
    return {p: compute(dataset, {"period": p}) for p in periods}

