"""VWAP anchored by session."""
from __future__ import annotations

from typing import List, Dict, Any


def compute(dataset: List[Dict[str, Any]], params: Dict[str, Any] | None = None) -> List[float]:
    """Compute session anchored VWAP values."""
    vwap_values: List[float] = []
    current_session = None
    pv_cum = 0.0
    vol_cum = 0.0
    for row in dataset:
        session = row["session"]
        price = row["close"]
        vol = row["volume"]
        if session != current_session:
            current_session = session
            pv_cum = 0.0
            vol_cum = 0.0
        pv_cum += price * vol
        vol_cum += vol
        vwap_values.append(pv_cum / vol_cum if vol_cum else price)
    return vwap_values

