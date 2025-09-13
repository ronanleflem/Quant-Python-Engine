"""Average True Range feature computation."""
from __future__ import annotations

from typing import List, Dict, Any


def compute(dataset: List[Dict[str, Any]], params: Dict[str, int] | None = None) -> List[float]:
    """Compute a simple ATR indicator.

    Parameters
    ----------
    dataset:
        Sequence of OHLCV rows.
    params:
        Optional ``{"period": n}`` configuration. Defaults to 14.
    """
    period = 14
    if params and "period" in params:
        period = int(params["period"])
    trs: List[float] = []
    prev_close = None
    for row in dataset:
        high = row["high"]
        low = row["low"]
        if prev_close is None:
            tr = high - low
        else:
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
        prev_close = row["close"]
    atr: List[float] = []
    window: List[float] = []
    for tr in trs:
        window.append(tr)
        if len(window) > period:
            window.pop(0)
        atr.append(sum(window) / len(window))
    return atr
