"""Repository helpers for the SQLite persistence layer."""

from __future__ import annotations

import json
from typing import Dict, Iterable, List

import sqlite3


class RunsRepository:
    """Persistence operations for ``experiment_runs``."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def create_or_running(
        self,
        run_id: str,
        spec_id: str,
        dataset_id: str,
        objective: str,
        out_dir: str,
    ) -> None:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT status FROM experiment_runs WHERE run_id = ?", (run_id,)
        )
        row = cur.fetchone()
        if row:
            if row["status"] != "RUNNING":
                cur.execute(
                    "UPDATE experiment_runs SET status = 'RUNNING' WHERE run_id = ?",
                    (run_id,),
                )
        else:
            cur.execute(
                """
                INSERT INTO experiment_runs
                    (run_id, spec_id, dataset_id, status, objective, out_dir)
                VALUES (?, ?, ?, 'RUNNING', ?, ?)
                """,
                (run_id, spec_id, dataset_id, objective, out_dir),
            )

    def finish(self, run_id: str, status: str) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            UPDATE experiment_runs
            SET status = ?, finished_at = CURRENT_TIMESTAMP
            WHERE run_id = ?
            """,
            (status, run_id),
        )


class MetricsRepository:
    """Operations for the ``run_metrics`` table."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def bulk_upsert_metrics(
        self, run_id: str, metrics_map: Dict[str, float], fold: int | None = None
    ) -> None:
        cur = self.conn.cursor()
        rows = [
            (run_id, fold, name, value)
            for name, value in metrics_map.items()
        ]
        cur.executemany(
            """
            INSERT INTO run_metrics (run_id, fold, metric_name, metric_value)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(run_id, fold, metric_name)
            DO UPDATE SET metric_value = excluded.metric_value
            """,
            rows,
        )


class TrialsRepository:
    """Operations for the ``trials`` table."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def bulk_insert_trials(self, run_id: str, trials_list: Iterable[Dict]) -> None:
        cur = self.conn.cursor()
        rows: List[tuple] = []
        for t in trials_list:
            rows.append(
                (
                    run_id,
                    t["trial_number"],
                    json.dumps(t.get("params", {})),
                    t.get("objective_value"),
                    t.get("status"),
                    t.get("n_trades"),
                    t.get("max_dd"),
                    t.get("sharpe"),
                    t.get("sortino"),
                    t.get("cagr"),
                    t.get("hit_rate"),
                    t.get("avg_r"),
                )
            )
        cur.executemany(
            """
            INSERT INTO trials (
                run_id, trial_number, params_json, objective_value, status,
                n_trades, max_dd, sharpe, sortino, cagr, hit_rate, avg_r
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, trial_number) DO NOTHING
            """,
            rows,
        )


__all__ = [
    "RunsRepository",
    "MetricsRepository",
    "TrialsRepository",
]

