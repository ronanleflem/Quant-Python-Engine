"""Repository for market statistics persistence."""

from __future__ import annotations

from typing import Iterable, Dict, Any, List

import sqlite3


class MarketStatsRepository:
    """Operations for the ``market_stats`` table."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def bulk_upsert(self, stats_rows: List[Dict[str, Any]]) -> None:
        cur = self.conn.cursor()
        rows = []
        for r in stats_rows:
            rows.append(
                (
                    r.get("symbol"),
                    r.get("timeframe"),
                    r.get("event"),
                    r.get("condition_name"),
                    r.get("condition_value"),
                    r.get("target"),
                    r.get("split"),
                    r.get("n"),
                    r.get("successes"),
                    r.get("p_hat"),
                    r.get("ci_low"),
                    r.get("ci_high"),
                    r.get("lift"),
                    r.get("start"),
                    r.get("end"),
                    r.get("spec_id"),
                    r.get("dataset_id"),
                )
            )
        cur.executemany(
            """
            INSERT INTO market_stats (
                symbol, timeframe, event, condition_name, condition_value,
                target, split, n, successes, p_hat, ci_low, ci_high, lift,
                start, end, spec_id, dataset_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(
                symbol, timeframe, event, condition_name, condition_value,
                target, split, start, end, spec_id
            ) DO UPDATE SET
                n=excluded.n,
                successes=excluded.successes,
                p_hat=excluded.p_hat,
                ci_low=excluded.ci_low,
                ci_high=excluded.ci_high,
                lift=excluded.lift,
                dataset_id=excluded.dataset_id
            """,
            rows,
        )


__all__ = ["MarketStatsRepository"]

