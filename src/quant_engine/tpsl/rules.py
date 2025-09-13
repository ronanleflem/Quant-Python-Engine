"""Modular stop-loss and take-profit helpers."""
from __future__ import annotations

from typing import List


class StopInitializer:
    @staticmethod
    def fixed_atr(atr_values: List[float], atr_mult: float, side: int, idx: int, entry_price: float) -> tuple[float, float]:
        """Return stop price and distance based on ATR.

        Parameters
        ----------
        atr_values: pre-computed ATR series.
        atr_mult: multiplier applied to ATR.
        side: +1 for long, -1 for short.
        idx: index of the entry bar in ``atr_values``.
        entry_price: execution price of the trade.
        """
        dist = atr_values[idx] * atr_mult
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
