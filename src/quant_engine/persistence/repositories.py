"""Repository helpers for the SQLite/MySQL persistence layer."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Mapping

import sqlite3


def extract_dca_run_metrics(extra: Mapping[str, Any]) -> Dict[str, float]:
    """Flatten DCA run ``extra`` payload to SQL metric rows.

    Keeps the contract stable for API consumers by persisting precomputed
    Python metrics directly (no recomputation in downstream services).
    """

    metrics_map: Dict[str, float] = {}
    for key in (
        "final_performance_normalized",
        "capital_efficiency_index",
        "twr",
        "xirr",
        "max_drawdown_on_contributed_capital",
        "time_under_water",
    ):
        value = extra.get(key)
        if isinstance(value, (int, float)):
            metrics_map[key] = float(value)

    ros_ratio = extra.get("return_over_stress_ratio") if isinstance(extra.get("return_over_stress_ratio"), dict) else {}
    for key in ("ratio", "numerator", "denominator", "denominator_raw", "epsilon"):
        value = ros_ratio.get(key)
        if isinstance(value, (int, float)):
            metrics_map[f"return_over_stress_ratio_{key}"] = float(value)

    xirr_status = extra.get("xirr_status")
    if isinstance(xirr_status, str):
        metrics_map["xirr_converged"] = 1.0 if xirr_status == "ok" else 0.0

    composite = extra.get("dca_composite_score") if isinstance(extra.get("dca_composite_score"), dict) else {}
    score_value = composite.get("score")
    if isinstance(score_value, (int, float)):
        metrics_map["dca_score"] = float(score_value)
    edge_value = composite.get("edge")
    if isinstance(edge_value, str):
        edge = edge_value.strip().lower()
        edge_map = {"weak": 1.0, "medium": 2.0, "strong": 3.0}
        if edge in edge_map:
            metrics_map["dca_edge_level"] = edge_map[edge]
    components = composite.get("components") if isinstance(composite.get("components"), dict) else {}
    for name, value in components.items():
        if isinstance(value, (int, float)):
            metrics_map[f"dca_score_component_{name}"] = float(value)

    return metrics_map


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
    "extract_dca_run_metrics",
]
