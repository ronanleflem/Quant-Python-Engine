"""Minimal broker simulation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Trade:
    entry_price: float
    exit_price: float

    @property
    def pnl(self) -> float:
        return self.exit_price - self.entry_price


class PortfolioBroker:
    """Very small broker managing a single position."""

    def __init__(self) -> None:
        self.position = 0
        self.entry_price = 0.0
        self.trades: List[Trade] = []

    def buy(self, price: float) -> None:
        if self.position == 0:
            self.position = 1
            self.entry_price = price

    def sell(self, price: float) -> None:
        if self.position == 1:
            self.position = 0
            self.trades.append(Trade(entry_price=self.entry_price, exit_price=price))

