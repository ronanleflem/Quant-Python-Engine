"""Tiny in-memory API resembling the FastAPI specification.

The implementation exposes three functions ``submit``, ``status`` and
``result`` that mimic the behaviour of REST endpoints.  They can be called
synchronously which keeps the tests lightweight while preserving the public
contract of the original design.
"""
from __future__ import annotations

import json
from typing import Dict, Any, List, Sequence

from ..core.spec import Spec
from ..optimize.runner import run as run_optimisation
from ..io import ids
from ..persistence import db
from ..stats import runner as stats_runner
from ..stats.estimators import freq_with_wilson
from ..seasonality import runner as seasonality_runner
from ..seasonality.optimize import run_optimization as seasonality_run_optimization
from . import schemas

_jobs: Dict[str, Dict[str, Any]] = {}
_last_stats: Dict[str, Any] | None = None


def submit(spec: Spec) -> schemas.SubmitResponse:
    job_id = ids.generate_id()
    result = run_optimisation(spec)
    _jobs[job_id] = {"status": "completed", "result": result}
    return schemas.SubmitResponse(id=job_id)


def status(job_id: str) -> schemas.StatusResponse:
    job = _jobs.get(job_id)
    if not job:
        return schemas.StatusResponse(status="unknown")
    return schemas.StatusResponse(status=job["status"])


def result(job_id: str) -> schemas.ResultResponse:
    job = _jobs.get(job_id)
    if not job:
        return schemas.ResultResponse(result=None)
    return schemas.ResultResponse(result=job.get("result"))


# ---------------------------------------------------------------------------
# Statistics endpoints (synchronous MVP)


def stats_run(spec: schemas.StatsSpec) -> schemas.StatusResponse:
    """Execute a statistics run synchronously and store the result."""

    global _last_stats
    _last_stats = stats_runner.run_stats(spec)
    return schemas.StatusResponse(status="completed")


def stats_result() -> schemas.ResultResponse:
    """Return the last statistics result if available."""

    return schemas.ResultResponse(result=_last_stats)


# ---------------------------------------------------------------------------
# Seasonality endpoints


def seasonality_run(spec: schemas.SeasonalitySpec) -> schemas.ResultResponse:
    """Execute a seasonality run synchronously and return its summary."""

    result = seasonality_runner.run(spec)
    return schemas.ResultResponse(result=result)


def seasonality_optimize(spec: schemas.SeasonalitySpec) -> schemas.ResultResponse:
    """Launch the seasonality optimisation loop and return its outcome."""

    result = seasonality_run_optimization(spec)
    return schemas.ResultResponse(result=result)


def list_seasonality_profiles(
    symbol: str | None = None,
    timeframe: str | None = None,
    dim: str | None = None,
    measure: str | None = None,
    spec_id: str | None = None,
    dataset_id: str | None = None,
    metrics: Sequence[str] | None = None,
    page: int = 1,
    page_size: int = 50,
) -> List[Dict[str, Any]]:
    """Return paginated seasonality profiles from persistence."""

    offset = (page - 1) * page_size
    with db.session() as conn:
        query = (
            "SELECT id, symbol, timeframe, dim, bin, measure, score, n, baseline, lift, "
            "metrics, start, end, spec_id, dataset_id, created_at FROM seasonality_profiles"
        )
        params: List[Any] = []
        clauses: List[str] = []
        if symbol:
            clauses.append("symbol = ?")
            params.append(symbol)
        if timeframe:
            clauses.append("timeframe = ?")
            params.append(timeframe)
        if dim:
            clauses.append("dim = ?")
            params.append(dim)
        if measure:
            clauses.append("measure = ?")
            params.append(measure)
        if spec_id:
            clauses.append("spec_id = ?")
            params.append(spec_id)
        if dataset_id:
            clauses.append("dataset_id = ?")
            params.append(dataset_id)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([page_size, offset])
        rows = conn.execute(query, params).fetchall()
        metrics_filter = {m.strip() for m in (metrics or []) if m.strip()}
        results: List[Dict[str, Any]] = []
        for row in rows:
            payload = dict(row)
            metrics_raw = payload.get("metrics")
            metrics_map: Dict[str, Any]
            if metrics_raw in (None, ""):
                metrics_map = {}
            elif isinstance(metrics_raw, str):
                try:
                    metrics_map = json.loads(metrics_raw)
                except json.JSONDecodeError:
                    metrics_map = {}
            else:
                metrics_map = dict(metrics_raw)
            if metrics_filter:
                include = True
                for metric_name in metrics_filter:
                    if metrics_map.get(metric_name) is None:
                        include = False
                        break
                if not include:
                    continue
            payload["metrics"] = metrics_map
            results.append(payload)
        return results


def _decode_best_summary(value: Any) -> Any:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:  # pragma: no cover - defensive
            return value
    return value


def list_seasonality_runs(
    status: str | None = None,
    spec_id: str | None = None,
    dataset_id: str | None = None,
    page: int = 1,
    page_size: int = 50,
) -> List[Dict[str, Any]]:
    """Return paginated seasonality runs."""

    offset = (page - 1) * page_size
    with db.session() as conn:
        query = (
            "SELECT run_id, spec_id, dataset_id, out_dir, status, best_summary, created_at "
            "FROM seasonality_runs"
        )
        params: List[Any] = []
        clauses: List[str] = []
        if status:
            clauses.append("status = ?")
            params.append(status)
        if spec_id:
            clauses.append("spec_id = ?")
            params.append(spec_id)
        if dataset_id:
            clauses.append("dataset_id = ?")
            params.append(dataset_id)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([page_size, offset])
        rows = conn.execute(query, params).fetchall()
        out: List[Dict[str, Any]] = []
        for row in rows:
            payload = dict(row)
            payload["best_summary"] = _decode_best_summary(payload.get("best_summary"))
            out.append(payload)
        return out


def get_seasonality_run(run_id: str) -> Dict[str, Any] | None:
    """Return a single seasonality run if available."""

    with db.session() as conn:
        row = conn.execute(
            "SELECT run_id, spec_id, dataset_id, out_dir, status, best_summary, created_at "
            "FROM seasonality_runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if not row:
            return None
        payload = dict(row)
        payload["best_summary"] = _decode_best_summary(payload.get("best_summary"))
        return payload


# ---------------------------------------------------------------------------
# Read-only endpoints backed by the SQLite persistence layer


def list_runs(
    status: str | None = None,
    symbol: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    page: int = 1,
    page_size: int = 50,
) -> List[Dict[str, Any]]:
    """Return paginated runs."""

    offset = (page - 1) * page_size
    with db.session() as conn:
        query = "SELECT run_id, status, objective, out_dir, started_at, finished_at FROM experiment_runs"
        params: List[Any] = []
        clauses: List[str] = []
        if status:
            clauses.append("status = ?")
            params.append(status)
        if date_from:
            clauses.append("started_at >= ?")
            params.append(date_from)
        if date_to:
            clauses.append("started_at <= ?")
            params.append(date_to)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY started_at DESC LIMIT ? OFFSET ?"
        params.extend([page_size, offset])
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]


def get_run(run_id: str) -> Dict[str, Any] | None:
    """Return a single run with aggregated metrics."""

    with db.session() as conn:
        cur = conn.execute(
            "SELECT * FROM experiment_runs WHERE run_id = ?", (run_id,)
        )
        run = cur.fetchone()
        if not run:
            return None
        aggregated = {
            r["metric_name"]: r["metric_value"]
            for r in conn.execute(
                "SELECT metric_name, metric_value FROM run_metrics WHERE run_id = ? AND fold IS NULL",
                (run_id,),
            )
        }
        folds: Dict[int, Dict[str, float]] = {}
        for r in conn.execute(
            "SELECT fold, metric_name, metric_value FROM run_metrics WHERE run_id = ? AND fold IS NOT NULL",
            (run_id,),
        ):
            folds.setdefault(r["fold"], {})[r["metric_name"]] = r["metric_value"]
        return {"run": dict(run), "metrics": {"aggregated": aggregated, "folds": folds}}


def get_run_trials(
    run_id: str,
    order_by: str = "objective_value.desc",
    page: int = 1,
    page_size: int = 50,
) -> List[Dict[str, Any]]:
    """Return leaderboard of trials for a run."""

    offset = (page - 1) * page_size
    field, _, direction = order_by.partition(".")
    direction = "DESC" if direction.lower() == "desc" else "ASC"
    allowed_fields = {
        "objective_value",
        "n_trades",
        "max_dd",
        "sharpe",
        "sortino",
        "cagr",
        "hit_rate",
        "avg_r",
    }
    if field not in allowed_fields:
        field = "objective_value"
    with db.session() as conn:
        query = (
            f"SELECT trial_number, params_json, objective_value, status, n_trades, max_dd, "
            f"sharpe, sortino, cagr, hit_rate, avg_r FROM trials WHERE run_id = ? "
            f"ORDER BY {field} {direction} LIMIT ? OFFSET ?"
        )
        rows = conn.execute(query, (run_id, page_size, offset)).fetchall()
        return [dict(row) for row in rows]


def get_run_metrics(run_id: str) -> Dict[str, Any]:
    """Return metrics for a run (aggregated and per fold)."""

    with db.session() as conn:
        aggregated = {
            r["metric_name"]: r["metric_value"]
            for r in conn.execute(
                "SELECT metric_name, metric_value FROM run_metrics WHERE run_id = ? AND fold IS NULL",
                (run_id,),
            )
        }
        folds: Dict[int, Dict[str, float]] = {}
        for r in conn.execute(
            "SELECT fold, metric_name, metric_value FROM run_metrics WHERE run_id = ? AND fold IS NOT NULL",
            (run_id,),
        ):
            folds.setdefault(r["fold"], {})[r["metric_name"]] = r["metric_value"]
    return {"aggregated": aggregated, "folds": folds}


# ---------------------------------------------------------------------------
# Market statistics read endpoints


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
    """Return statistics rows filtered and ordered according to parameters."""

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

    for r in out:
        if "lift_freq" not in r and "lift" in r:
            r["lift_freq"] = r.get("lift")
        if "lift_bayes" not in r:
            r["lift_bayes"] = r.get("lift_freq")

    if significant_only:
        out = [
            r
            for r in out
            if r.get("significant")
            or (r.get("q_value") is not None and r["q_value"] <= alpha)
        ]

    key = "lift_bayes" if method == "bayes" else "lift_freq"
    out.sort(key=lambda r: r.get(key, 0), reverse=True)

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
        for r in rows:
            n = int(r["n"])
            successes = int(r["successes"])
            p_hat, ci_low, ci_high = freq_with_wilson(successes, n)
            out.append(
                {
                    "condition_name": r["condition_name"],
                    "condition_value": r["condition_value"],
                    "target": r["target"],
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
    """Return heatmap-style bins for a condition."""

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

    def sort_key(r: Dict[str, Any]):
        try:
            return float(r["bin"])
        except (TypeError, ValueError):
            return r["bin"]

    out.sort(key=sort_key)
    return out


def stats_top(
    symbol: str,
    timeframe: str,
    k: int = 10,
    method: str = "freq",
    significant_only: bool = False,
) -> List[Dict[str, Any]]:
    """Return top-k patterns ordered by lift for the chosen method."""

    base_query = "SELECT * FROM market_stats WHERE symbol = ? AND timeframe = ?"
    params = [symbol, timeframe]
    with db.session() as conn:
        rows = conn.execute(base_query + " AND split = 'test'", params).fetchall()
        if not rows:
            rows = conn.execute(base_query, params).fetchall()

    data = [dict(r) for r in rows]

    for r in data:
        if "lift_freq" not in r and "lift" in r:
            r["lift_freq"] = r.get("lift")
        if "lift_bayes" not in r:
            r["lift_bayes"] = r.get("lift_freq")

    if significant_only:
        data = [
            r
            for r in data
            if r.get("significant") or (r.get("q_value") is not None and r["q_value"] <= 0.05)
        ]

    key = "lift_bayes" if method == "bayes" else "lift_freq"
    rows_sorted = sorted(data, key=lambda r: abs(r.get(key, 0)), reverse=True)[:k]
    return rows_sorted


