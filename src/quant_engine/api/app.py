"""In-memory orchestration helpers and their FastAPI wrappers.

The synchronous helpers keep the test suite light-weight while the
FastAPI application exposes the same capabilities over HTTP for local
development.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from fastapi import FastAPI, HTTPException, Query
from sqlalchemy import create_engine, text

from ..core import spec as spec_module
from ..core.spec import Spec
from ..levels.runner import run_levels_build
from ..levels.schemas import LevelsBuildSpec
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

    if _last_stats is None:
        return schemas.ResultResponse(result=None)

    try:
        import pandas as pd  # type: ignore
    except Exception:  # pragma: no cover - pandas is an install dependency
        df = _last_stats
    else:
        df = _last_stats.where(pd.notna(_last_stats), None)
    payload = {
        "columns": list(df.columns),
        "rows": df.to_dict(orient="records"),
    }
    return schemas.ResultResponse(result=payload)


# ---------------------------------------------------------------------------
# Levels helpers


LEVELS_TABLE = "marketdata.levels"


def _resolve_levels_engine():
    url = os.environ.get("QE_LEVELS_MYSQL_URL") or os.environ.get("QE_MARKETDATA_MYSQL_URL")
    if not url:
        raise HTTPException(status_code=500, detail="MySQL URL not configured for levels module")
    return create_engine(url)


def _parse_iso_ts(value: str | None, field: str) -> datetime | None:
    if value in (None, ""):
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:  # pragma: no cover - defensive validation
        raise HTTPException(status_code=400, detail=f"Invalid datetime for {field}") from exc
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def _format_ts(value: datetime | None) -> str | None:
    if value is None:
        return None
    dt = value
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def _serialise_row(row) -> Dict[str, Any]:
    if hasattr(row, "_mapping"):
        data = dict(row._mapping)
    elif isinstance(row, dict):
        data = dict(row)
    else:  # pragma: no cover - legacy tuples from DB-API
        keys = getattr(row, "keys", None)
        if callable(keys):
            data = dict(zip(keys(), row))
        else:
            data = dict(row)
    for key in ("anchor_ts", "valid_from_ts", "valid_to_ts"):
        if key in data:
            data[key] = _format_ts(data.get(key))
    return data


def _fetch_levels(
    symbol: str,
    level_type: str | None = None,
    start: str | None = None,
    end: str | None = None,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    engine = _resolve_levels_engine()
    clauses = ["symbol = :symbol"]
    params: Dict[str, Any] = {"symbol": symbol, "limit": limit}
    if level_type:
        clauses.append("level_type = :level_type")
        params["level_type"] = level_type
    start_dt = _parse_iso_ts(start, "from")
    if start_dt is not None:
        clauses.append("anchor_ts >= :start")
        params["start"] = start_dt
    end_dt = _parse_iso_ts(end, "to")
    if end_dt is not None:
        clauses.append("anchor_ts <= :end")
        params["end"] = end_dt
    query = (
        f"SELECT symbol, timeframe, level_type, price, price_lo, price_hi, anchor_ts, "
        f"valid_from_ts, valid_to_ts, params_hash, source FROM {LEVELS_TABLE} "
        "WHERE " + " AND ".join(clauses) + " ORDER BY anchor_ts DESC LIMIT :limit"
    )
    with engine.connect() as conn:
        rows = conn.execute(text(query), params).fetchall()
    return [_serialise_row(row) for row in rows]


def _distance_to_price(price: float, payload: Dict[str, Any]) -> float:
    price_val = payload.get("price")
    if price_val is not None:
        return abs(float(price_val) - price)
    lo = payload.get("price_lo")
    hi = payload.get("price_hi")
    if lo is None or hi is None:
        return float("inf")
    lo_f = float(lo)
    hi_f = float(hi)
    if lo_f <= price <= hi_f:
        return 0.0
    if price < lo_f:
        return lo_f - price
    return price - hi_f


def _nearest_levels(
    symbol: str,
    price: float,
    level_type: str | None = None,
    window: float | None = None,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    fetch_limit = max(limit * 5, 200)
    engine = _resolve_levels_engine()
    clauses = ["symbol = :symbol"]
    params: Dict[str, Any] = {"symbol": symbol, "limit": fetch_limit}
    if level_type:
        clauses.append("level_type = :level_type")
        params["level_type"] = level_type
    if window is not None and window > 0:
        params["price_lo"] = price - window
        params["price_hi"] = price + window
        clauses.append(
            "((price IS NOT NULL AND price BETWEEN :price_lo AND :price_hi) OR "
            "(price_lo IS NOT NULL AND price_hi IS NOT NULL AND price_hi >= :price_lo AND price_lo <= :price_hi))"
        )
    query = (
        f"SELECT symbol, timeframe, level_type, price, price_lo, price_hi, anchor_ts, "
        f"valid_from_ts, valid_to_ts, params_hash, source FROM {LEVELS_TABLE} "
        "WHERE " + " AND ".join(clauses) + " ORDER BY anchor_ts DESC LIMIT :limit"
    )
    with engine.connect() as conn:
        rows = conn.execute(text(query), params).fetchall()
    payloads = [_serialise_row(row) for row in rows]
    for payload in payloads:
        distance = _distance_to_price(price, payload)
        payload["distance"] = distance
    filtered = [p for p in payloads if p.get("distance", float("inf")) != float("inf")]
    filtered.sort(key=lambda p: p.get("distance", float("inf")))
    return filtered[:limit]


def levels_build(spec: LevelsBuildSpec) -> Dict[str, Any]:
    """Execute a levels build request synchronously."""

    return run_levels_build(spec)


def levels_list(
    symbol: str,
    level_type: str | None = None,
    start: str | None = None,
    end: str | None = None,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    """Return persisted levels filtered by the provided criteria."""

    return _fetch_levels(symbol=symbol, level_type=level_type, start=start, end=end, limit=limit)


def levels_nearest(
    symbol: str,
    price: float,
    level_type: str | None = None,
    window: float | None = None,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """Return the nearest levels around a target price."""

    return _nearest_levels(symbol=symbol, price=price, level_type=level_type, window=window, limit=limit)


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




fastapi_app = FastAPI(title="Quant Engine API", version="0.1.0")


@fastapi_app.post('/submit', response_model=schemas.SubmitResponse)
def submit_endpoint(payload: Dict[str, Any]) -> schemas.SubmitResponse:
    """HTTP endpoint wrapping :func:`submit`."""

    try:
        spec_obj = spec_module.spec_from_dict(payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid spec: {exc}") from exc
    return submit(spec_obj)


@fastapi_app.get('/status/{job_id}', response_model=schemas.StatusResponse)
def status_endpoint(job_id: str) -> schemas.StatusResponse:
    """Return the status for a submitted job."""

    return status(job_id)


@fastapi_app.get('/result/{job_id}', response_model=schemas.ResultResponse)
def result_endpoint(job_id: str) -> schemas.ResultResponse:
    """Return the optimisation result for a job if available."""

    return result(job_id)


@fastapi_app.post('/stats/run', response_model=schemas.StatusResponse)
def stats_run_endpoint(spec: schemas.StatsSpec) -> schemas.StatusResponse:
    """Kick off a statistics computation synchronously."""

    return stats_run(spec)


@fastapi_app.get('/stats/result', response_model=schemas.ResultResponse)
def stats_result_endpoint() -> schemas.ResultResponse:
    """Return the result of the last statistics computation."""

    return stats_result()


@fastapi_app.post('/levels/build', response_model=Dict[str, Any])
def levels_build_endpoint(spec: LevelsBuildSpec) -> Dict[str, Any]:
    """Trigger a synchronous levels build run."""

    return levels_build(spec)


@fastapi_app.get('/levels', response_model=List[Dict[str, Any]])
def levels_list_endpoint(
    symbol: str,
    level_type: Optional[str] = None,
    from_: Optional[str] = Query(None, alias="from"),
    to: Optional[str] = Query(None, alias="to"),
    limit: int = Query(200, ge=1, le=1000),
) -> List[Dict[str, Any]]:
    """Return persisted levels."""

    return levels_list(symbol=symbol, level_type=level_type, start=from_, end=to, limit=limit)


@fastapi_app.get('/levels/nearest', response_model=List[Dict[str, Any]])
def levels_nearest_endpoint(
    symbol: str,
    price: float = Query(..., description="Reference price used to rank levels"),
    level_type: Optional[str] = None,
    window: Optional[float] = Query(None, description="Optional +/- window to pre-filter levels"),
    limit: int = Query(20, ge=1, le=200),
) -> List[Dict[str, Any]]:
    """Return levels closest to the requested price."""

    return levels_nearest(symbol=symbol, price=price, level_type=level_type, window=window, limit=limit)


@fastapi_app.get('/stats', response_model=List[Dict[str, Any]])
def stats_list_endpoint(
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    event: Optional[str] = None,
    condition_name: Optional[str] = None,
    target: Optional[str] = None,
    split: Optional[str] = None,
    min_n: Optional[int] = Query(None, ge=0),
    significant_only: bool = False,
    method: str = Query('freq', pattern='^(freq|bayes)$'),
    alpha: float = Query(0.05, ge=0.0, le=1.0),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
) -> List[Dict[str, Any]]:
    """List persisted statistics with optional filters."""

    return list_stats(
        symbol=symbol,
        timeframe=timeframe,
        event=event,
        condition_name=condition_name,
        target=target,
        split=split,
        min_n=min_n,
        significant_only=significant_only,
        method=method,
        alpha=alpha,
        page=page,
        page_size=page_size,
    )


@fastapi_app.get('/stats/summary', response_model=List[Dict[str, Any]])
def stats_summary_endpoint(
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    event: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return aggregated statistics summary."""

    return stats_summary(symbol=symbol, timeframe=timeframe, event=event)


@fastapi_app.get('/stats/heatmap', response_model=List[Dict[str, Any]])
def stats_heatmap_endpoint(
    symbol: str,
    timeframe: str,
    event: str,
    target: str,
    condition_name: str,
) -> List[Dict[str, Any]]:
    """Return heatmap-style bins for a condition."""

    return stats_heatmap(symbol, timeframe, event, target, condition_name)


@fastapi_app.get('/stats/top', response_model=List[Dict[str, Any]])
def stats_top_endpoint(
    symbol: str,
    timeframe: str,
    k: int = Query(10, ge=1, le=100),
    method: str = Query('freq', pattern='^(freq|bayes)$'),
    significant_only: bool = False,
) -> List[Dict[str, Any]]:
    """Return top patterns ordered by lift."""

    return stats_top(symbol, timeframe, k=k, method=method, significant_only=significant_only)


@fastapi_app.post('/seasonality/run', response_model=schemas.ResultResponse)
def seasonality_run_endpoint(spec: schemas.SeasonalitySpec) -> schemas.ResultResponse:
    """Execute a seasonality run synchronously."""

    return seasonality_run(spec)


@fastapi_app.post('/seasonality/optimize', response_model=schemas.ResultResponse)
def seasonality_optimize_endpoint(spec: schemas.SeasonalitySpec) -> schemas.ResultResponse:
    """Execute the seasonality optimisation loop."""

    return seasonality_optimize(spec)


@fastapi_app.get('/seasonality/profiles', response_model=List[Dict[str, Any]])
def seasonality_profiles_endpoint(
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    dim: Optional[str] = None,
    measure: Optional[str] = None,
    spec_id: Optional[str] = None,
    dataset_id: Optional[str] = None,
    metrics: Optional[List[str]] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
) -> List[Dict[str, Any]]:
    """List persisted seasonality profiles."""

    return list_seasonality_profiles(
        symbol=symbol,
        timeframe=timeframe,
        dim=dim,
        measure=measure,
        spec_id=spec_id,
        dataset_id=dataset_id,
        metrics=metrics,
        page=page,
        page_size=page_size,
    )


@fastapi_app.get('/seasonality/runs', response_model=List[Dict[str, Any]])
def seasonality_runs_endpoint(
    status: Optional[str] = None,
    spec_id: Optional[str] = None,
    dataset_id: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
) -> List[Dict[str, Any]]:
    """List persisted seasonality runs."""

    return list_seasonality_runs(
        status=status,
        spec_id=spec_id,
        dataset_id=dataset_id,
        page=page,
        page_size=page_size,
    )


@fastapi_app.get('/seasonality/runs/{run_id}', response_model=Dict[str, Any])
def seasonality_run_detail_endpoint(run_id: str) -> Dict[str, Any]:
    """Return details for a specific seasonality run."""

    payload = get_seasonality_run(run_id)
    if payload is None:
        raise HTTPException(status_code=404, detail='Run not found')
    return payload


@fastapi_app.get('/runs', response_model=List[Dict[str, Any]])
def runs_list_endpoint(
    status: Optional[str] = None,
    symbol: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
) -> List[Dict[str, Any]]:
    """List optimisation runs."""

    return list_runs(
        status=status,
        symbol=symbol,
        date_from=date_from,
        date_to=date_to,
        page=page,
        page_size=page_size,
    )


@fastapi_app.get('/runs/{run_id}', response_model=Dict[str, Any])
def run_detail_endpoint(run_id: str) -> Dict[str, Any]:
    """Return a single run and aggregated metrics."""

    payload = get_run(run_id)
    if payload is None:
        raise HTTPException(status_code=404, detail='Run not found')
    return payload


@fastapi_app.get('/runs/{run_id}/trials', response_model=List[Dict[str, Any]])
def run_trials_endpoint(
    run_id: str,
    order_by: str = 'objective_value.desc',
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
) -> List[Dict[str, Any]]:
    """Return leaderboard of trials for a run."""

    return get_run_trials(run_id, order_by=order_by, page=page, page_size=page_size)


@fastapi_app.get('/runs/{run_id}/metrics', response_model=Dict[str, Any])
def run_metrics_endpoint(run_id: str) -> Dict[str, Any]:
    """Return metrics for a run."""

    return get_run_metrics(run_id)


app = fastapi_app
