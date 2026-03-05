from __future__ import annotations

from typing import Any, Dict, List

from ...persistence import db
from ...stats.estimators import freq_with_wilson


def list_stats(
    symbol: str | None = None,
    timeframe: str | None = None,
    event: str | None = None,
    condition_name: str | None = None,
    target: str | None = None,
    split: str | None = None,
    min_n: int | None = None,
    significant_only: bool = False,
    method: str = "freq",
    alpha: float = 0.05,
    page: int = 1,
    page_size: int = 50,
) -> List[Dict[str, Any]]:
    with db.session() as conn:
        query = "SELECT * FROM market_stats"
        params: List[Any] = []
        clauses: List[str] = []
        if symbol:
            clauses.append("symbol = ?")
            params.append(symbol)
        if timeframe:
            clauses.append("timeframe = ?")
            params.append(timeframe)
        if event:
            clauses.append("event = ?")
            params.append(event)
        if condition_name:
            clauses.append("condition_name = ?")
            params.append(condition_name)
        if target:
            clauses.append("target = ?")
            params.append(target)
        if split:
            clauses.append("split = ?")
            params.append(split)
        if min_n is not None:
            clauses.append("n >= ?")
            params.append(min_n)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        rows = conn.execute(query, params).fetchall()

    out = [dict(r) for r in rows]
    for row in out:
        if "lift_freq" not in row and "lift" in row:
            row["lift_freq"] = row.get("lift")
        if "lift_bayes" not in row:
            row["lift_bayes"] = row.get("lift_freq")

    if significant_only:
        out = [
            row
            for row in out
            if row.get("significant") or (row.get("q_value") is not None and row["q_value"] <= alpha)
        ]

    key = "lift_bayes" if method == "bayes" else "lift_freq"
    out.sort(key=lambda row: row.get(key, 0), reverse=True)

    start = (page - 1) * page_size
    end = start + page_size
    return out[start:end]


def stats_summary(
    symbol: str | None = None,
    timeframe: str | None = None,
    event: str | None = None,
) -> List[Dict[str, Any]]:
    with db.session() as conn:
        query = (
            "SELECT condition_name, condition_value, target, SUM(n) as n, "
            "SUM(successes) as successes FROM market_stats"
        )
        params: List[Any] = []
        clauses: List[str] = []
        if symbol:
            clauses.append("symbol = ?")
            params.append(symbol)
        if timeframe:
            clauses.append("timeframe = ?")
            params.append(timeframe)
        if event:
            clauses.append("event = ?")
            params.append(event)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " GROUP BY condition_name, condition_value, target"
        rows = conn.execute(query, params).fetchall()

    out: List[Dict[str, Any]] = []
    for row in rows:
        n = int(row["n"])
        successes = int(row["successes"])
        p_hat, ci_low, ci_high = freq_with_wilson(successes, n)
        out.append(
            {
                "condition_name": row["condition_name"],
                "condition_value": row["condition_value"],
                "target": row["target"],
                "n": n,
                "successes": successes,
                "p_hat": p_hat,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    return out


def stats_heatmap(
    symbol: str,
    timeframe: str,
    event: str,
    target: str,
    condition_name: str,
) -> List[Dict[str, Any]]:
    base_query = (
        "SELECT condition_value as bin, p_hat, ci_low, ci_high, n, lift "
        "FROM market_stats WHERE symbol = ? AND timeframe = ? AND event = ? "
        "AND target = ? AND condition_name = ?"
    )
    params = [symbol, timeframe, event, target, condition_name]
    with db.session() as conn:
        rows = conn.execute(base_query + " AND split = 'test'", params).fetchall()
        if not rows:
            rows = conn.execute(base_query, params).fetchall()

    out = [dict(r) for r in rows]

    def sort_key(row: Dict[str, Any]):
        try:
            return float(row["bin"])
        except (TypeError, ValueError):
            return row["bin"]

    out.sort(key=sort_key)
    return out


def stats_top(
    symbol: str,
    timeframe: str,
    k: int = 10,
    method: str = "freq",
    significant_only: bool = False,
) -> List[Dict[str, Any]]:
    base_query = "SELECT * FROM market_stats WHERE symbol = ? AND timeframe = ?"
    params = [symbol, timeframe]
    with db.session() as conn:
        rows = conn.execute(base_query + " AND split = 'test'", params).fetchall()
        if not rows:
            rows = conn.execute(base_query, params).fetchall()

    data = [dict(row) for row in rows]
    for row in data:
        if "lift_freq" not in row and "lift" in row:
            row["lift_freq"] = row.get("lift")
        if "lift_bayes" not in row:
            row["lift_bayes"] = row.get("lift_freq")

    if significant_only:
        data = [
            row
            for row in data
            if row.get("significant") or (row.get("q_value") is not None and row["q_value"] <= 0.05)
        ]

    key = "lift_bayes" if method == "bayes" else "lift_freq"
    rows_sorted = sorted(data, key=lambda row: abs(row.get(key, 0)), reverse=True)[:k]
    return rows_sorted
