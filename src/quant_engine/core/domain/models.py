from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Candle:
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float

    def to_dict(self) -> dict[str, object]:
        return {
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }


@dataclass(frozen=True)
class Trade:
    trade_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    timestamp: datetime

    def to_dict(self) -> dict[str, object]:
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "price": self.price,
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True)
class Position:
    symbol: str
    quantity: float
    average_price: float

    def to_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "average_price": self.average_price,
        }


@dataclass(frozen=True)
class Portfolio:
    cash: float
    equity: float

    def to_dict(self) -> dict[str, object]:
        return {
            "cash": self.cash,
            "equity": self.equity,
        }
