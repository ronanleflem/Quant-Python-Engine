"""Smoke tests for levels DB constraints and helper views."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import List

import pytest
from sqlalchemy import text

from quant_engine.levels.repo import (
    ensure_table,
    get_engine,
    upsert_levels,
    _build_row,
)
from quant_engine.levels.schemas import LevelRecord


TABLE_FQN = "marketdata.levels"


@pytest.mark.skipif(
    "QE_MARKETDATA_MYSQL_URL" not in os.environ,
    reason="Requires QE_MARKETDATA_MYSQL_URL for persistence",
)
def test_duplicate_upsert_is_idempotent() -> None:
    engine = get_engine()
    ensure_table(engine, TABLE_FQN)

    symbol = "ZZZ_P4_SMOKE"
    base_anchor = datetime(2024, 1, 1, 12, tzinfo=timezone.utc)
    valid_from = datetime(2024, 1, 1, 0, tzinfo=timezone.utc)

    records: List[LevelRecord] = [
        LevelRecord(
            symbol=symbol,
            timeframe="H1",
            level_type="PDH",
            price=1.2345,
            price_lo=None,
            price_hi=None,
            anchor_ts=base_anchor,
            valid_from_ts=valid_from,
            valid_to_ts=None,
            params_hash="hash-point",
            source="smoke-test",
        ),
        LevelRecord(
            symbol=symbol,
            timeframe="H1",
            level_type="FVG",
            price=None,
            price_lo=1.2001,
            price_hi=1.2101,
            anchor_ts=base_anchor,
            valid_from_ts=None,
            valid_to_ts=None,
            params_hash="hash-zone",
            source="smoke-test",
        ),
    ]
    rows = [_build_row(record) for record in records]
    uniq_hashes = [row["uniq_hash"] for row in rows]

    try:
        first = upsert_levels(engine, TABLE_FQN, records)
        assert first == {"inserted": len(records), "updated": 0}

        second = upsert_levels(engine, TABLE_FQN, records)
        assert second == {"inserted": 0, "updated": len(records)}

        with engine.connect() as conn:
            params = {"symbol": symbol}
            base_points = conn.execute(
                text(
                    f"""
                    SELECT uniq_hash FROM {TABLE_FQN}
                    WHERE symbol = :symbol AND valid_to_ts IS NULL AND price IS NOT NULL
                    """
                ),
                params,
            ).fetchall()
            view_points = conn.execute(
                text(
                    """
                    SELECT uniq_hash FROM marketdata.view_levels_active_points
                    WHERE symbol = :symbol
                    """
                ),
                params,
            ).fetchall()

            base_zones = conn.execute(
                text(
                    f"""
                    SELECT uniq_hash FROM {TABLE_FQN}
                    WHERE symbol = :symbol AND valid_to_ts IS NULL AND price_lo IS NOT NULL AND price_hi IS NOT NULL
                    """
                ),
                params,
            ).fetchall()
            view_zones = conn.execute(
                text(
                    """
                    SELECT uniq_hash FROM marketdata.view_levels_active_zones
                    WHERE symbol = :symbol
                    """
                ),
                params,
            ).fetchall()

        points_from_base = {row[0] for row in base_points}
        points_from_view = {row[0] for row in view_points}
        zones_from_base = {row[0] for row in base_zones}
        zones_from_view = {row[0] for row in view_zones}

        assert points_from_view == points_from_base
        assert zones_from_view == zones_from_base
        assert set(uniq_hashes) == points_from_view.union(zones_from_view)
    finally:
        cleanup_sql = text(f"DELETE FROM {TABLE_FQN} WHERE symbol = :symbol")
        with engine.begin() as conn:
            conn.execute(cleanup_sql, {"symbol": symbol})
