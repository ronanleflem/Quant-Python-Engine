"""Repository helpers for statistics and seasonality persistence."""

from __future__ import annotations

import json
from typing import Dict, Any, List, Optional, Sequence

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


class SeasonalityProfilesRepository:
    """Operations for the ``seasonality_profiles`` table."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def bulk_upsert(self, profiles_rows: Sequence[Dict[str, Any]]) -> None:
        if not profiles_rows:
            return
        cur = self.conn.cursor()
        rows: List[tuple] = []
        for r in profiles_rows:
            metrics = r.get("metrics") or {}
            metrics_json = json.dumps(metrics)
            rows.append(
                (
                    r.get("symbol"),
                    r.get("timeframe"),
                    r.get("dim"),
                    r.get("bin"),
                    r.get("measure"),
                    r.get("score"),
                    r.get("n"),
                    r.get("baseline"),
                    r.get("lift"),
                    metrics_json,
                    r.get("start"),
                    r.get("end"),
                    r.get("spec_id"),
                    r.get("dataset_id"),
                )
            )
        cur.executemany(
            """
            INSERT INTO seasonality_profiles (
                symbol, timeframe, dim, bin, measure, score, n, baseline, lift,
                metrics, start, end, spec_id, dataset_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, timeframe, dim, bin, measure, start, end, spec_id, dataset_id)
            DO UPDATE SET
                score = excluded.score,
                n = excluded.n,
                baseline = excluded.baseline,
                lift = excluded.lift,
                metrics = excluded.metrics
            """,
            rows,
        )


class SeasonalityRunsRepository:
    """Operations for the ``seasonality_runs`` table."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def create(
        self,
        run_id: str,
        *,
        spec_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
        out_dir: Optional[str] = None,
        status: str = "running",
    ) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO seasonality_runs (run_id, spec_id, dataset_id, out_dir, status)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                spec_id = excluded.spec_id,
                dataset_id = excluded.dataset_id,
                out_dir = excluded.out_dir,
                status = excluded.status
            """,
            (run_id, spec_id, dataset_id, out_dir, status),
        )

    def finish(
        self,
        run_id: str,
        status: str,
        best_summary: Optional[Dict[str, Any]] = None,
    ) -> None:
        cur = self.conn.cursor()
        payload = json.dumps(best_summary) if best_summary is not None else None
        cur.execute(
            """
            UPDATE seasonality_runs
            SET status = ?, best_summary = ?
            WHERE run_id = ?
            """,
            (status, payload, run_id),
        )


__all__ = [
    "MarketStatsRepository",
    "SeasonalityProfilesRepository",
    "SeasonalityRunsRepository",
]

