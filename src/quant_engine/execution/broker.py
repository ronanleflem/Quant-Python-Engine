"""Simulation broker with basic fills, fees, and slippage."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Trade:
    symbol: str
    quantity: float
    entry_price: float
    exit_price: float
    fees: float

    @property
    def pnl(self) -> float:
        direction = 1.0 if self.quantity >= 0 else -1.0
        gross = (self.exit_price - self.entry_price) * abs(self.quantity) * direction
        return gross - self.fees


@dataclass
class Position:
    quantity: float = 0.0
    average_price: float = 0.0


@dataclass
class Order:
    symbol: str
    side: str
    quantity: float
    price: float
    status: str = "submitted"
    fill_price: Optional[float] = None
    fees: float = 0.0
    reason: Optional[str] = None


class PortfolioBroker:
    """Simple broker managing multiple positions with fills."""

    def __init__(
        self,
        initial_cash: float = 0.0,
        commission_rate: float = 0.0,
        slippage_bps: float = 0.0,
        default_symbol: str = "DEFAULT",
    ) -> None:
        self.cash = initial_cash
        self.commission_rate = commission_rate
        self.slippage_bps = slippage_bps
        self.default_symbol = default_symbol
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.orders: List[Order] = []

    def buy(self, price: float, quantity: float = 1.0, symbol: Optional[str] = None) -> None:
        symbol = symbol or self.default_symbol
        self.place_order(symbol=symbol, side="buy", quantity=quantity, price=price)

    def sell(self, price: float, quantity: float = 1.0, symbol: Optional[str] = None) -> None:
        symbol = symbol or self.default_symbol
        self.place_order(symbol=symbol, side="sell", quantity=quantity, price=price)

    def place_order(self, symbol: str, side: str, quantity: float, price: float) -> Order:
        order = Order(symbol=symbol, side=side, quantity=quantity, price=price)
        if quantity <= 0 or price <= 0:
            order.status = "rejected"
            order.reason = "quantity and price must be positive"
            self.orders.append(order)
            return order
        if side not in {"buy", "sell"}:
            order.status = "rejected"
            order.reason = "side must be buy or sell"
            self.orders.append(order)
            return order
        fill_price = self._apply_slippage(side, price)
        fees = abs(quantity) * fill_price * self.commission_rate
        order.fill_price = fill_price
        order.fees = fees
        order.status = "filled"
        self._apply_fill(symbol=symbol, side=side, quantity=quantity, price=fill_price, fees=fees)
        self.orders.append(order)
        return order

    def _apply_slippage(self, side: str, price: float) -> float:
        if self.slippage_bps <= 0:
            return price
        adjustment = self.slippage_bps / 10_000
        if side == "buy":
            return price * (1 + adjustment)
        return price * (1 - adjustment)

    def _apply_fill(self, symbol: str, side: str, quantity: float, price: float, fees: float) -> None:
        signed_qty = quantity if side == "buy" else -quantity
        position = self.positions.get(symbol, Position())
        new_qty = position.quantity + signed_qty
        self.cash -= signed_qty * price
        self.cash -= fees
        if position.quantity == 0 or (position.quantity > 0 and signed_qty > 0) or (
            position.quantity < 0 and signed_qty < 0
        ):
            position.average_price = self._weighted_avg(
                position.average_price,
                position.quantity,
                price,
                signed_qty,
            )
            position.quantity = new_qty
            self.positions[symbol] = position
            return
        closing_qty = min(abs(position.quantity), abs(signed_qty))
        realized_fees = fees * (closing_qty / abs(signed_qty)) if signed_qty != 0 else 0.0
        trade_qty = closing_qty if position.quantity > 0 else -closing_qty
        self.trades.append(
            Trade(
                symbol=symbol,
                quantity=trade_qty,
                entry_price=position.average_price,
                exit_price=price,
                fees=realized_fees,
            )
        )
        remaining_qty = new_qty
        if remaining_qty == 0:
            position.quantity = 0.0
            position.average_price = 0.0
        elif (position.quantity > 0 and remaining_qty < 0) or (position.quantity < 0 and remaining_qty > 0):
            position.quantity = remaining_qty
            position.average_price = price
        else:
            position.quantity = remaining_qty
        self.positions[symbol] = position

    @staticmethod
    def _weighted_avg(avg_price: float, current_qty: float, price: float, delta_qty: float) -> float:
        total_qty = current_qty + delta_qty
        if total_qty == 0:
            return 0.0
        total_cost = avg_price * current_qty + price * delta_qty
        return total_cost / total_qty

    def total_equity(self, market_prices: Dict[str, float]) -> float:
        equity = self.cash
        for symbol, position in self.positions.items():
            price = market_prices.get(symbol)
            if price is None:
                continue
            equity += position.quantity * price
        return equity
