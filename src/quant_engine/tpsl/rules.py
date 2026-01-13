"""Modular stop-loss and take-profit helpers."""
from __future__ import annotations

from math import isfinite
from typing import List, Optional, Tuple


class StopInitializer:
    @staticmethod
    def fixed_atr(
        atr_values: List[float],
        atr_mult: float,
        side: int,
        idx: int,
        entry_price: float,
    ) -> Optional[Tuple[float, float]]:
        """Return stop price and distance based on ATR.

        Parameters
        ----------
        atr_values: pre-computed ATR series.
        atr_mult: multiplier applied to ATR.
        side: +1 for long, -1 for short.
        idx: index of the entry bar in ``atr_values``.
        entry_price: execution price of the trade.
        """
        if idx < 0 or idx >= len(atr_values):
            return None
        atr = atr_values[idx]
        if atr is None:
            return None
        try:
            atr_value = float(atr)
        except (TypeError, ValueError):
            return None
        if not isfinite(atr_value) or atr_value <= 0:
            return None
        if not isfinite(atr_mult) or atr_mult <= 0:
            return None
        dist = atr_value * atr_mult
        if not isfinite(dist) or dist <= 0:
            return None
        if side > 0:
            stop = entry_price - dist
        else:
            stop = entry_price + dist
        return stop, dist


class TakeProfit:
    @staticmethod
    def r_multiple(entry_price: float, stop_price: float, r_mult: float, side: int) -> float:
        """Return take profit price based on ``r_mult`` of the stop distance."""
        dist = abs(entry_price - stop_price)
        if side > 0:
            return entry_price + r_mult * dist
        return entry_price - r_mult * dist
