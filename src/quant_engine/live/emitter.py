"""Signal emission helpers for the live runner."""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib import request, error

from sqlalchemy import Engine, text

LOGGER = logging.getLogger(__name__)


def ensure_trades_live_table(engine: Engine, fqn: str = "quant.trades_live") -> None:
    """Create the ``quant.trades_live`` table if it does not exist."""

    ddl = (
        "CREATE TABLE IF NOT EXISTS {table} ("
        " id BIGINT PRIMARY KEY AUTO_INCREMENT,"
        " strategy_id VARCHAR(64) NOT NULL,"
        " symbol VARCHAR(32) NOT NULL,"
        " timeframe VARCHAR(8) NOT NULL,"
        " ts_open DATETIME(6) NOT NULL,"
        " side ENUM('LONG','SHORT') NOT NULL,"
        " entry_price DOUBLE NOT NULL,"
        " sl DOUBLE NULL,"
        " tp DOUBLE NULL,"
        " expected_rr DOUBLE NULL,"
        " signal_payload JSON NULL,"
        " uniq_hash CHAR(64) NOT NULL,"
        " created_at TIMESTAMP(6) DEFAULT CURRENT_TIMESTAMP(6),"
        " UNIQUE KEY ux_trades_live_uniq (uniq_hash),"
        " KEY idx_trades_live_sym_tf_ts (symbol, timeframe, ts_open),"
        " KEY idx_trades_live_strategy (strategy_id)"
        ")"
    ).format(table=fqn)
    alt_table = fqn.replace(".", "_")
    with engine.begin() as conn:
        try:
            conn.exec_driver_sql(ddl)
        except Exception:  # pragma: no cover - fallback for non-MySQL engines
            conn.exec_driver_sql(ddl.replace(fqn, alt_table))


@dataclass
class TradePayload:
    strategy_id: str
    symbol: str
    timeframe: str
    ts_open: str
    side: str
    entry_price: float
    sl: Optional[float] = None
    tp: Optional[float] = None
    expected_rr: Optional[float] = None
    signal_payload: Optional[Dict[str, Any]] = None
    uniq_hash: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        data = {
            "strategy_id": self.strategy_id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "ts_open": self.ts_open,
            "side": self.side,
            "entry_price": float(self.entry_price),
            "sl": self.sl,
            "tp": self.tp,
            "expected_rr": self.expected_rr,
            "signal_payload": self.signal_payload,
        }
        if self.uniq_hash:
            data["uniq_hash"] = self.uniq_hash
        return data


def _round_for_hash(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{float(value):.10f}"


def compute_trade_hash(payload: TradePayload) -> str:
    """Compute the deterministic uniq_hash for a trade."""

    parts = [
        payload.strategy_id,
        payload.symbol,
        payload.timeframe,
        payload.ts_open,
        payload.side,
        _round_for_hash(payload.entry_price),
        _round_for_hash(payload.sl),
        _round_for_hash(payload.tp),
    ]
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return digest


def build_trade(
    strategy_id: str,
    symbol: str,
    timeframe: str,
    ts_open: str,
    side: str,
    entry_price: float,
    sl: Optional[float],
    tp: Optional[float],
    expected_rr: Optional[float],
    payload: Optional[Dict[str, Any]] = None,
) -> TradePayload:
    """Construct a trade payload enriched with ``uniq_hash``."""

    trade = TradePayload(
        strategy_id=strategy_id,
        symbol=symbol,
        timeframe=timeframe,
        ts_open=ts_open,
        side=side.upper(),
        entry_price=float(entry_price),
        sl=float(sl) if sl is not None else None,
        tp=float(tp) if tp is not None else None,
        expected_rr=float(expected_rr) if expected_rr is not None else None,
        signal_payload=payload,
    )
    trade.uniq_hash = compute_trade_hash(trade)
    return trade


def emit_to_java(
    signal_dict: Dict[str, Any],
    url_env: str = "QE_JAVA_LIVE_URL",
    path: str = "/live/signal",
    timeout: float = 3.0,
) -> bool:
    """POST the signal to the downstream Java service."""

    base = os.environ.get(url_env)
    if not base:
        LOGGER.debug("Java emission skipped - env %s not set", url_env)
        return False
    url = base.rstrip("/") + path
    data = json.dumps(signal_dict).encode()
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                LOGGER.warning("Java emit failed: HTTP %s %s", resp.status, resp.reason)
                return False
            payload = json.loads(resp.read().decode() or "{}")
    except error.URLError as exc:
        LOGGER.warning("Java emit error: %s", exc)
        return False
    ok = bool(payload.get("ok", True))
    if not ok:
        LOGGER.warning("Java emit returned non-ok payload: %s", payload)
    return ok


def write_trade(engine: Engine, table_fqn: str, trade: TradePayload) -> bool:
    """Persist the trade payload into ``quant.trades_live`` with idempotence."""

    data = trade.as_dict()
    if "uniq_hash" not in data or not data["uniq_hash"]:
        data["uniq_hash"] = compute_trade_hash(trade)
    insert_cols = (
        "strategy_id, symbol, timeframe, ts_open, side, entry_price, sl, tp,"
        " expected_rr, signal_payload, uniq_hash"
    )
    values = (
        ":strategy_id, :symbol, :timeframe, :ts_open, :side, :entry_price, :sl, :tp,"
        " :expected_rr, :signal_payload, :uniq_hash"
    )
    mysql_sql = text(
        f"INSERT INTO {table_fqn} ({insert_cols}) VALUES ({values}) "
        "ON DUPLICATE KEY UPDATE signal_payload = VALUES(signal_payload)"
    )
    sqlite_table = table_fqn.replace(".", "_")
    sqlite_sql = text(
        f"INSERT OR IGNORE INTO {sqlite_table} ({insert_cols}) VALUES ({values})"
    )
    with engine.begin() as conn:
        try:
            conn.execute(mysql_sql, data)
            return True
        except Exception:
            conn.execute(sqlite_sql, data)
            return True


__all__ = [
    "TradePayload",
    "build_trade",
    "emit_to_java",
    "ensure_trades_live_table",
    "write_trade",
]
