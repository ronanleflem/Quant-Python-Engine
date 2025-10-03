"""Persistence helpers for the ``marketdata.levels`` table."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from hashlib import sha256
from typing import Iterable, List, Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from .schemas import LevelRecord


def _resolve_mysql_url() -> str:
    url = os.environ.get("QE_LEVELS_MYSQL_URL") or os.environ.get("QE_MARKETDATA_MYSQL_URL")
    if not url:
        raise RuntimeError(
            "Missing MySQL connection URL. Define QE_LEVELS_MYSQL_URL or QE_MARKETDATA_MYSQL_URL."
        )
    return url


def get_engine(url: str | None = None) -> Engine:
    """Create a SQLAlchemy engine using the provided or resolved URL."""

    return create_engine(url or _resolve_mysql_url())


def ensure_table(engine: Engine, table_fqn: str) -> None:
    """Create the levels table if it does not already exist."""

    ddl = f"""
    CREATE TABLE IF NOT EXISTS {table_fqn} (
      id BIGINT AUTO_INCREMENT PRIMARY KEY,
      symbol VARCHAR(64) NOT NULL,
      timeframe VARCHAR(16) NOT NULL,
      level_type VARCHAR(16) NOT NULL,
      price DOUBLE NULL,
      price_lo DOUBLE NULL,
      price_hi DOUBLE NULL,
      anchor_ts DATETIME(6) NOT NULL,
      valid_from_ts DATETIME(6) NULL,
      valid_to_ts DATETIME(6) NULL,
      params_hash VARCHAR(64) NULL,
      source VARCHAR(64) NOT NULL,
      uniq_hash CHAR(64) NOT NULL,
      created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
      updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
      UNIQUE KEY uq_levels_hash (uniq_hash),
      INDEX idx_lvl_sym_type_anchor (symbol, level_type, anchor_ts),
      INDEX idx_lvl_sym_valid (symbol, valid_from_ts, valid_to_ts)
    ) ENGINE=InnoDB;
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))


def _normalise_ts(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def _ensure_datetime(value: datetime | pd.Timestamp | None) -> datetime | None:
    if value is None or pd.isna(value):  # type: ignore[arg-type]
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    return value


def _build_row(record: LevelRecord) -> dict:
    uniq_payload = f"{record.symbol}|{record.timeframe}|{record.level_type}|{record.anchor_ts.isoformat()}|"
    uniq_payload += "|".join(
        str(x if x is not None else "")
        for x in (record.price, record.price_lo, record.price_hi, record.params_hash)
    )
    uniq_hash = sha256(uniq_payload.encode()).hexdigest()
    return {
        "symbol": record.symbol,
        "timeframe": record.timeframe,
        "level_type": record.level_type,
        "price": record.price,
        "price_lo": record.price_lo,
        "price_hi": record.price_hi,
        "anchor_ts": _normalise_ts(record.anchor_ts),
        "valid_from_ts": _normalise_ts(record.valid_from_ts),
        "valid_to_ts": _normalise_ts(record.valid_to_ts),
        "params_hash": record.params_hash,
        "source": record.source,
        "uniq_hash": uniq_hash,
    }


def _chunk(records: List[dict], size: int) -> Iterable[List[dict]]:
    for i in range(0, len(records), size):
        yield records[i : i + size]


def upsert_levels(engine: Engine, table_fqn: str, records: List[LevelRecord]) -> dict:
    """Insert or update the provided level records."""

    if not records:
        return {"inserted": 0, "updated": 0}

    rows = [_build_row(record) for record in records]
    inserted = 0
    updated = 0
    insert_sql = text(
        f"""
        INSERT INTO {table_fqn}
            (symbol, timeframe, level_type, price, price_lo, price_hi,
             anchor_ts, valid_from_ts, valid_to_ts, params_hash, source, uniq_hash)
        VALUES
            (:symbol, :timeframe, :level_type, :price, :price_lo, :price_hi,
             :anchor_ts, :valid_from_ts, :valid_to_ts, :params_hash, :source, :uniq_hash)
        ON DUPLICATE KEY UPDATE
             timeframe = VALUES(timeframe),
             price = VALUES(price),
             price_lo = VALUES(price_lo),
             price_hi = VALUES(price_hi),
             valid_from_ts = VALUES(valid_from_ts),
             valid_to_ts = VALUES(valid_to_ts),
             params_hash = VALUES(params_hash),
             source = VALUES(source)
        """
    )
    with engine.begin() as conn:
        for chunk in _chunk(rows, 1000):
            placeholders = ",".join(f":h{i}" for i in range(len(chunk)))
            sel_sql = text(
                f"SELECT uniq_hash FROM {table_fqn} WHERE uniq_hash IN ({placeholders})"
            )
            params = {f"h{i}": row["uniq_hash"] for i, row in enumerate(chunk)}
            existing_rows = conn.execute(sel_sql, params).fetchall()
            existing = {
                (row._mapping.get("uniq_hash") if hasattr(row, "_mapping") else row[0])
                for row in existing_rows
            }
            conn.execute(insert_sql, chunk)
            inserted += len(chunk) - len(existing)
            updated += len(existing)
    return {"inserted": inserted, "updated": updated}


def _compute_row_hash(row: dict) -> str:
    symbol = row.get("symbol")
    timeframe = row.get("timeframe")
    level_type = row.get("level_type")
    anchor_ts = _ensure_datetime(row.get("anchor_ts"))
    if anchor_ts is None:
        raise ValueError("anchor_ts is required to compute uniq_hash")
    anchor_norm = _normalise_ts(anchor_ts)
    payload = f"{symbol}|{timeframe}|{level_type}|{anchor_norm.isoformat()}|"
    payload += "|".join(
        str(x if x is not None else "")
        for x in (row.get("price"), row.get("price_lo"), row.get("price_hi"), row.get("params_hash"))
    )
    return sha256(payload.encode()).hexdigest()


def _parse_optional_ts(value: Optional[str | datetime | pd.Timestamp]) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, str):
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    elif isinstance(value, pd.Timestamp):
        dt = value.to_pydatetime()
    else:
        dt = value
    return _normalise_ts(dt)


def select_levels(
    engine: Engine,
    table_fqn: str,
    symbol: str,
    level_types: List[str],
    active_only: bool,
    start: Optional[str | datetime] = None,
    end: Optional[str | datetime] = None,
    limit: int = 10000,
) -> pd.DataFrame:
    """Return levels filtered by symbol, type and validity flags."""

    clauses = ["symbol = :symbol"]
    params: dict = {"symbol": symbol, "limit": limit}
    if level_types:
        placeholders = ",".join(f":lt{i}" for i in range(len(level_types)))
        clauses.append(f"level_type IN ({placeholders})")
        params.update({f"lt{i}": lvl for i, lvl in enumerate(level_types)})
    if active_only:
        clauses.append("valid_to_ts IS NULL")
    start_norm = _parse_optional_ts(start)
    if start_norm is not None:
        clauses.append("anchor_ts >= :start")
        params["start"] = start_norm
    end_norm = _parse_optional_ts(end)
    if end_norm is not None:
        clauses.append("anchor_ts <= :end")
        params["end"] = end_norm
    query = (
        f"SELECT symbol, timeframe, level_type, price, price_lo, price_hi, anchor_ts, "
        f"valid_from_ts, valid_to_ts, params_hash FROM {table_fqn} "
        "WHERE " + " AND ".join(clauses) + " ORDER BY anchor_ts DESC LIMIT :limit"
    )
    with engine.connect() as conn:
        rows = conn.execute(text(query), params).fetchall()
    records: List[dict] = []
    for row in rows:
        if hasattr(row, "_mapping"):
            records.append(dict(row._mapping))
        elif hasattr(row, "keys"):
            records.append(dict(zip(row.keys(), row)))
        else:  # pragma: no cover - defensive fallback
            records.append(dict(row))
    return pd.DataFrame(records)


def upsert_valid_to_ts(engine: Engine, table_fqn: str, df_updates: pd.DataFrame) -> int:
    """Update ``valid_to_ts`` for the specified rows and return affected count."""

    if df_updates.empty:
        return 0
    df = df_updates.dropna(subset=["valid_to_ts"])
    if df.empty:
        return 0
    updates: List[dict] = []
    for _, row in df.iterrows():
        payload = row.to_dict()
        try:
            uniq_hash = _compute_row_hash(payload)
        except ValueError:
            continue
        valid_to_ts = _ensure_datetime(payload.get("valid_to_ts"))
        if valid_to_ts is None:
            continue
        updates.append({
            "uniq_hash": uniq_hash,
            "valid_to_ts": _normalise_ts(valid_to_ts),
        })
    if not updates:
        return 0
    update_sql = text(
        f"UPDATE {table_fqn} SET valid_to_ts = :valid_to_ts "
        "WHERE uniq_hash = :uniq_hash"
    )
    affected = 0
    with engine.begin() as conn:
        for chunk in _chunk(updates, 500):
            result = conn.execute(update_sql, chunk)
            affected += max(result.rowcount or 0, 0)
    return affected


__all__ = [
    "get_engine",
    "ensure_table",
    "upsert_levels",
    "select_levels",
    "upsert_valid_to_ts",
]
