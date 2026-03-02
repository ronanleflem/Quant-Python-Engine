"""In-memory orchestration helpers and their FastAPI wrappers.

The synchronous helpers keep the test suite light-weight while the
FastAPI application exposes the same capabilities over HTTP for local
development.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time
import threading
import re
import logging
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from sqlalchemy import create_engine, text

from ..core import spec as spec_module
from ..backtest import runner as backtest_runner
from ..core.spec import Spec
from ..levels.runner import run_levels_build, run_levels_fill
from ..levels.repo import select_levels as repo_select_levels
from ..levels.schemas import LevelsBuildSpec
from ..optimize.runner import run as run_optimisation
from ..optimize import variants as optimize_variants
from ..io import ids
from ..io.dca_artifacts import SCHEMA_VERSION
from ..persistence import db, RunsRepository, MetricsRepository, TrialsRepository
from ..stats import runner as stats_runner
from ..stats import conditions as stats_conditions
from ..stats.estimators import freq_with_wilson
from ..seasonality import runner as seasonality_runner
from ..seasonality.optimize import run_optimization as seasonality_run_optimization
from ..strategies import runner as strategies_runner
from ..performance import stress_tests as stress_tests_runner
from ..filters import list_filter_types
from . import schemas
from .run_request_input import validate_run_request_input
from .validation_errors import (
    ApiValidationException,
    normalize_fastapi_errors,
    normalize_pydantic_errors,
    single_validation_error,
)
from .metrics import METRICS

logger = logging.getLogger(__name__)

JOB_TYPE_OPTIMIZATION = "optimization"
JOB_TYPE_STATS = "stats"
JOB_TYPE_LEVELS_BUILD = "levels_build"
JOB_TYPE_LEVELS_FILL = "levels_fill"
JOB_TYPE_SEASONALITY_RUN = "seasonality_run"
JOB_TYPE_SEASONALITY_OPTIMIZE = "seasonality_optimize"
JOB_TYPE_CANONICAL_RUN = "canonical_run"
CANONICAL_CAPABILITIES_CATALOG_VERSION = "2026-02-02"

JOB_STATUS_PENDING = "pending"
JOB_STATUS_RUNNING = "running"
JOB_STATUS_COMPLETED = "completed"
JOB_STATUS_FAILED = "failed"
JOB_STATUS_QUEUED = "QUEUED"
JOB_STATUS_RUNNING_CANONICAL = "RUNNING"
JOB_STATUS_SUCCEEDED = "SUCCEEDED"
JOB_STATUS_FAILED_CANONICAL = "FAILED"
JOB_STATUS_CANCELED = "CANCELED"

JOB_STATUSES = {
    JOB_STATUS_PENDING,
    JOB_STATUS_RUNNING,
    JOB_STATUS_COMPLETED,
    JOB_STATUS_FAILED,
    JOB_STATUS_QUEUED,
    JOB_STATUS_RUNNING_CANONICAL,
    JOB_STATUS_SUCCEEDED,
    JOB_STATUS_FAILED_CANONICAL,
    JOB_STATUS_CANCELED,
}
JOB_RESULT_WITH_ID = {
    JOB_TYPE_STATS,
    JOB_TYPE_LEVELS_BUILD,
    JOB_TYPE_LEVELS_FILL,
    JOB_TYPE_SEASONALITY_RUN,
    JOB_TYPE_SEASONALITY_OPTIMIZE,
    JOB_TYPE_CANONICAL_RUN,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalize_payload(payload: Any) -> Any:
    if is_dataclass(payload):
        return asdict(payload)
    return payload


def _serialize_payload(payload: Any) -> str | None:
    if payload is None:
        return None
    try:
        return json.dumps(_normalize_payload(payload), default=str)
    except TypeError:
        return json.dumps(str(payload))


def _decode_json(payload: Any) -> Any:
    if payload in (None, ""):
        return None
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return payload
    return payload


def _is_true_flag(value: Any) -> bool:
    """Normalize DB flags to boolean (supports MySQL BIT values as bytes)."""
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, (bytes, bytearray)):
        return any(b != 0 for b in value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "f", "no", "n", "off", ""}:
            return False
    return bool(value)


def _init_job(
    job_id: str,
    job_type: str,
    payload: Any | None,
    *,
    status: str = JOB_STATUS_PENDING,
    max_attempts: int | None = None,
    timeout_seconds: int | None = None,
) -> None:
    with db.session() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO api_jobs(
                job_id,
                job_type,
                status,
                payload_json,
                attempts,
                max_attempts,
                timeout_seconds,
                progress_json,
                cancel_requested,
                canceled_at,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                job_type,
                status,
                _serialize_payload(payload),
                0,
                max_attempts,
                timeout_seconds,
                None,
                0,
                None,
                _utc_now(),
                _utc_now(),
            ),
        )


def _update_job_status(job_id: str, status: str, *, error: str | None = None) -> None:
    if status not in JOB_STATUSES:
        raise ValueError(f"Unknown job status: {status}")
    now = _utc_now()
    normalized = status.lower()
    started_at = now if normalized == "running" else None
    finished_at = now if normalized in {"completed", "failed", "canceled"} else None
    if status in {JOB_STATUS_SUCCEEDED, JOB_STATUS_FAILED_CANONICAL, JOB_STATUS_CANCELED}:
        finished_at = now
    canceled_at = now if normalized == "canceled" or status == JOB_STATUS_CANCELED else None
    with db.session() as conn:
        conn.execute(
            """
            UPDATE api_jobs
            SET status = ?,
                error_message = COALESCE(?, error_message),
                started_at = COALESCE(?, started_at),
                finished_at = COALESCE(?, finished_at),
                canceled_at = COALESCE(?, canceled_at),
                updated_at = ?
            WHERE job_id = ?
            """,
            (status, error, started_at, finished_at, canceled_at, now, job_id),
        )


def _update_job_result(job_id: str, result: Any, *, status: str = JOB_STATUS_COMPLETED) -> None:
    if status not in JOB_STATUSES:
        raise ValueError(f"Unknown job status: {status}")
    with db.session() as conn:
        conn.execute(
            """
            UPDATE api_jobs
            SET status = ?,
                result_json = ?,
                finished_at = ?,
                updated_at = ?
            WHERE job_id = ?
            """,
            (status, _serialize_payload(result), _utc_now(), _utc_now(), job_id),
        )


def _get_job(job_id: str) -> Dict[str, Any] | None:
    with db.session() as conn:
        row = conn.execute(
            """
            SELECT job_id, job_type, status, payload_json, result_json, error_message,
                   attempts, max_attempts, timeout_seconds, progress_json,
                   cancel_requested, canceled_at,
                   created_at, started_at, finished_at, updated_at
            FROM api_jobs
            WHERE job_id = ?
            """,
            (job_id,),
        ).fetchone()
        if not row:
            return None
        payload = dict(row)
        payload["payload"] = _decode_json(payload.pop("payload_json", None))
        payload["result"] = _decode_json(payload.pop("result_json", None))
        payload["progress"] = _decode_json(payload.pop("progress_json", None))
        return payload


def _latest_job(job_type: str) -> Dict[str, Any] | None:
    with db.session() as conn:
        row = conn.execute(
            """
            SELECT job_id, job_type, status, payload_json, result_json, error_message,
                   attempts, max_attempts, timeout_seconds, progress_json,
                   cancel_requested, canceled_at,
                   created_at, started_at, finished_at, updated_at
            FROM api_jobs
            WHERE job_type = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (job_type,),
        ).fetchone()
        if not row:
            return None
        payload = dict(row)
        payload["payload"] = _decode_json(payload.pop("payload_json", None))
        payload["result"] = _decode_json(payload.pop("result_json", None))
        payload["progress"] = _decode_json(payload.pop("progress_json", None))
        return payload


def _enqueue_job(
    job_type: str,
    payload: Any | None,
    *,
    status: str = JOB_STATUS_PENDING,
    max_attempts: int | None = None,
    timeout_seconds: int | None = None,
) -> str:
    job_id = ids.generate_id()
    _init_job(
        job_id,
        job_type,
        payload=payload,
        status=status,
        max_attempts=max_attempts,
        timeout_seconds=timeout_seconds,
    )
    return job_id


def _build_job_result(job_type: str, job_id: str, result: Any) -> Any:
    if job_type == JOB_TYPE_STATS:
        payload: Dict[str, Any] = _stats_payload_from_df(result)
        payload["job_id"] = job_id
        return payload
    if job_type in JOB_RESULT_WITH_ID:
        if isinstance(result, dict):
            payload = dict(result)
        else:
            payload = {"result": result}
        payload["job_id"] = job_id
        return payload
    return result


def _job_error_payload(code: str, message: str, *, details: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    error: Dict[str, Any] = {"code": code, "message": message}
    if details:
        error["details"] = details
    return {"error": error}


def _canonical_backtest_unsupported_details(request: Dict[str, Any]) -> List[Dict[str, str]]:
    details: List[Dict[str, str]] = []

    signal_block = request.get("signal")
    if isinstance(signal_block, dict):
        signal_type = str(signal_block.get("type") or "").strip().lower()
        if signal_type and signal_type != "ema_cross":
            details.append({"field": "signal", "reason": "accepted_but_not_wired"})

    strategy_block = request.get("strategy")
    if isinstance(strategy_block, dict):
        if strategy_block.get("name"):
            details.append({"field": "strategy.name", "reason": "accepted_but_not_wired"})
        params = strategy_block.get("params")
        if isinstance(params, dict):
            _tp_sl_internal, tp_sl_supported = _canonical_backtest_tp_sl_to_internal(params)
            if not tp_sl_supported:
                details.append({"field": "strategy.params.tp_sl", "reason": "accepted_but_not_wired"})

    return details


def _canonical_backtest_tp_sl_to_internal(params: Dict[str, Any]) -> tuple[Any, bool]:
    tp_sl_raw = params.get("tp_sl", params.get("tpSl"))
    if tp_sl_raw is None:
        return None, True
    if not isinstance(tp_sl_raw, dict):
        return None, False
    supported_keys = {
        "atr_window",
        "atr_period",
        "atr_k",
        "r_mult",
        "slippage_bps",
        "fee_bps",
        "dynamic_sl",
        "jitter",
    }
    if not any(key in tp_sl_raw for key in supported_keys):
        return None, False
    return dict(tp_sl_raw), True


def _coerce_stress_weights(raw: Any) -> List[float] | None:
    if isinstance(raw, list):
        parsed: List[float] = []
        for item in raw:
            try:
                parsed.append(float(item))
            except Exception:
                return None
        return parsed if parsed else None
    if isinstance(raw, str):
        chunks = [chunk.strip() for chunk in raw.split(",") if chunk.strip()]
        if not chunks:
            return None
        parsed: List[float] = []
        for chunk in chunks:
            try:
                parsed.append(float(chunk))
            except Exception:
                return None
        return parsed if parsed else None
    return None


def _canonical_stress_tests_to_internal(stress_block: Dict[str, Any]) -> Dict[str, Any]:
    monte_carlo: Dict[str, Any] = {}
    for key in ("enabled", "source", "method", "seed", "overlapping", "aggregation"):
        if stress_block.get(key) is not None:
            monte_carlo[key] = stress_block.get(key)

    n_sims = stress_block.get("n_sims", stress_block.get("nSims"))
    if n_sims is not None:
        monte_carlo["n_sims"] = n_sims

    block_size = stress_block.get("block_size", stress_block.get("blockSize"))
    if block_size is not None:
        monte_carlo["block_size"] = block_size

    for src, dst in (
        ("time_distribution", "time_distribution"),
        ("timeDistribution", "time_distribution"),
        ("time_dist", "time_distribution"),
        ("param_drift", "param_drift"),
        ("paramDrift", "param_drift"),
        ("sizing", "sizing"),
        ("output", "output"),
    ):
        value = stress_block.get(src)
        if isinstance(value, dict):
            monte_carlo[dst] = dict(value)

    timestamp_alignment = stress_block.get("timestamp_alignment", stress_block.get("timestampAlignment"))
    if isinstance(timestamp_alignment, str) and timestamp_alignment.strip():
        monte_carlo["timestamp_alignment"] = timestamp_alignment.strip()

    weights = _coerce_stress_weights(stress_block.get("weights"))
    if weights:
        monte_carlo["weights"] = weights

    multi_asset = stress_block.get("multi_asset", stress_block.get("multiAsset"))
    if isinstance(multi_asset, dict):
        multi_copy = dict(multi_asset)
        if isinstance(multi_copy.get("timestampAlignment"), str) and not multi_copy.get("timestamp_alignment"):
            multi_copy["timestamp_alignment"] = multi_copy.get("timestampAlignment")
        if "weights" in multi_copy:
            coerced = _coerce_stress_weights(multi_copy.get("weights"))
            if coerced:
                multi_copy["weights"] = coerced
        monte_carlo["multi_asset"] = multi_copy
        if multi_copy.get("aggregation") is not None and monte_carlo.get("aggregation") is None:
            monte_carlo["aggregation"] = multi_copy.get("aggregation")
        if multi_copy.get("timestamp_alignment") is not None and monte_carlo.get("timestamp_alignment") is None:
            monte_carlo["timestamp_alignment"] = multi_copy.get("timestamp_alignment")
        if multi_copy.get("weights") is not None and monte_carlo.get("weights") is None:
            monte_carlo["weights"] = multi_copy.get("weights")

    scenarios = stress_block.get("scenarios")
    out: Dict[str, Any] = {"monte_carlo": monte_carlo}
    if isinstance(scenarios, list):
        out["scenarios"] = scenarios
    elif isinstance(scenarios, dict):
        out["scenarios"] = scenarios
    return out


def _canonical_performance_to_internal(performance_block: Dict[str, Any]) -> Dict[str, Any]:
    performance_spec: Dict[str, Any] = {}
    for key in (
        "initial_capital",
        "capital_per_unit",
        "max_capital_per_trade",
        "risk_pct",
    ):
        if performance_block.get(key) is not None:
            performance_spec[key] = performance_block.get(key)

    risk_free_rate_pct = performance_block.get("risk_free_rate_pct")
    if risk_free_rate_pct is not None:
        performance_spec["risk_free_rate_pct"] = risk_free_rate_pct
        # backtest_builder currently consumes risk_free_pct.
        performance_spec["risk_free_pct"] = risk_free_rate_pct

    stress_block = performance_block.get("stress_tests")
    if isinstance(stress_block, dict):
        performance_spec["stress_tests"] = _canonical_stress_tests_to_internal(stress_block)
    return performance_spec


def _extract_trades_from_canonical_result(result_payload: Any) -> List[Dict[str, Any]]:
    if not isinstance(result_payload, dict):
        return []

    candidates: List[Dict[str, Any]] = [result_payload]
    payload_block = result_payload.get("payload")
    if isinstance(payload_block, dict):
        candidates.append(payload_block)

    result_block = result_payload.get("result")
    if isinstance(result_block, dict):
        candidates.append(result_block)
        nested_payload = result_block.get("payload")
        if isinstance(nested_payload, dict):
            candidates.append(nested_payload)

    for candidate in candidates:
        for key in ("trades", "completed_trades"):
            value = candidate.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _normalize_stress_source_trades(trades: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for trade in trades:
        pnl = trade.get("pnl")
        if pnl is None:
            pnl = trade.get("gross_pnl")
        if pnl is None:
            pnl = trade.get("grossPnl")
        if pnl is None:
            pnl = trade.get("profit_or_loss")
        if pnl is None:
            continue

        entry_time = (
            trade.get("entry_time")
            or trade.get("entryTimeUtc")
            or trade.get("entry_timestamp")
            or trade.get("ts_entry")
        )
        exit_time = (
            trade.get("exit_time")
            or trade.get("exitTimeUtc")
            or trade.get("exit_timestamp")
            or trade.get("ts_exit")
        )

        r_multiple = trade.get("r_multiple")
        if r_multiple is None:
            meta = trade.get("meta")
            if isinstance(meta, dict):
                r_multiple = meta.get("r_multiple")
            meta_json = trade.get("meta_json")
            if r_multiple is None and isinstance(meta_json, str):
                try:
                    decoded = json.loads(meta_json)
                    if isinstance(decoded, dict):
                        r_multiple = decoded.get("r_multiple")
                except Exception:
                    pass

        normalized.append(
            {
                "symbol": trade.get("symbol"),
                "pnl": pnl,
                "r_multiple": r_multiple,
                "entry_time": entry_time,
                "exit_time": exit_time,
            }
        )
    return normalized


def _load_stress_source_trades(base_run_id: str) -> List[Dict[str, Any]]:
    trades_rows: List[Dict[str, Any]] = []

    try:
        with db.session() as conn:
            rows = conn.execute(
                """
                SELECT symbol, entry_timestamp, exit_timestamp, profit_or_loss, pnl_pct, meta_json
                FROM trades_completed
                WHERE run_id = ?
                ORDER BY exit_timestamp, entry_timestamp
                """,
                (base_run_id,),
            ).fetchall()
            for row in rows:
                trades_rows.append(dict(row))
    except Exception:
        trades_rows = []

    normalized = _normalize_stress_source_trades(trades_rows)
    if normalized:
        return normalized

    try:
        with db.session() as conn:
            row = conn.execute(
                """
                SELECT result_json
                FROM api_jobs
                WHERE job_id = ?
                LIMIT 1
                """,
                (base_run_id,),
            ).fetchone()
    except Exception:
        row = None

    if row is None:
        return []

    raw_result = _decode_json(dict(row).get("result_json"))
    extracted = _extract_trades_from_canonical_result(raw_result)
    return _normalize_stress_source_trades(extracted)


def _load_canonical_run_request(base_run_id: str) -> Dict[str, Any]:
    with db.session() as conn:
        row = conn.execute(
            """
            SELECT status, payload_json
            FROM api_jobs
            WHERE job_id = ?
            LIMIT 1
            """,
            (base_run_id,),
        ).fetchone()
    if row is None:
        raise ValueError(f"Base run '{base_run_id}' not found")
    payload = _decode_json(dict(row).get("payload_json"))
    request = payload.get("request") if isinstance(payload, dict) else None
    if not isinstance(request, dict):
        raise ValueError(f"Base run '{base_run_id}' does not contain a canonical request payload")
    status = str(dict(row).get("status") or "")
    if status != JOB_STATUS_SUCCEEDED:
        raise ValueError(f"Base run '{base_run_id}' is not SUCCEEDED (status={status})")
    return request


def _normalize_optimization_search_space(raw_space: Mapping[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for key, raw_cfg in raw_space.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("optimization.search_space keys must be non-empty strings")
        path = key.strip()
        if isinstance(raw_cfg, list):
            if not raw_cfg:
                raise ValueError(f"optimization.search_space.{path} must not be an empty list")
            normalized[path] = list(raw_cfg)
            continue
        if not isinstance(raw_cfg, dict):
            raise ValueError(f"optimization.search_space.{path} must be a list or object")
        if isinstance(raw_cfg.get("values"), list):
            values = list(raw_cfg.get("values") or [])
            if not values:
                raise ValueError(f"optimization.search_space.{path}.values must not be empty")
            normalized[path] = {"values": values}
            continue
        if isinstance(raw_cfg.get("domain"), list):
            values = list(raw_cfg.get("domain") or [])
            if not values:
                raise ValueError(f"optimization.search_space.{path}.domain must not be empty")
            normalized[path] = {"values": values}
            continue
        if "min" in raw_cfg and "max" in raw_cfg:
            try:
                min_v = float(raw_cfg.get("min"))
                max_v = float(raw_cfg.get("max"))
            except Exception as exc:
                raise ValueError(f"optimization.search_space.{path} min/max must be numeric") from exc
            if max_v < min_v:
                raise ValueError(f"optimization.search_space.{path} must satisfy max >= min")
            step = raw_cfg.get("step")
            if step is None:
                raw_type = str(raw_cfg.get("type") or "").strip().lower()
                if raw_type in {"int", "integer"} or (float(min_v).is_integer() and float(max_v).is_integer()):
                    step = 1
                else:
                    span = max_v - min_v
                    step = span / 10.0 if span > 0 else 1.0
            try:
                step_v = float(step)
            except Exception as exc:
                raise ValueError(f"optimization.search_space.{path}.step must be numeric") from exc
            if step_v <= 0:
                raise ValueError(f"optimization.search_space.{path}.step must be > 0")
            if str(raw_cfg.get("type") or "").strip().lower() in {"int", "integer"}:
                normalized[path] = {"min": int(round(min_v)), "max": int(round(max_v)), "step": int(round(step_v))}
            else:
                normalized[path] = {"min": min_v, "max": max_v, "step": step_v}
            continue
        raise ValueError(
            f"optimization.search_space.{path} must provide one of: list, values[], domain[], or min/max(/step)"
        )
    return normalized


def _canonical_optimization_objective(metric: str, direction: str) -> Any:
    metric_clean = str(metric or "").strip()
    if not metric_clean:
        raise ValueError("optimization.objective.metric must be a non-empty string")
    direction_clean = str(direction or "").strip().lower()
    if direction_clean not in {"max", "min"}:
        raise ValueError("optimization.objective.direction must be 'max' or 'min'")
    if direction_clean == "max":
        return metric_clean
    return {"weights": {metric_clean: -1.0}}


def _canonical_optimization_trials(result: Mapping[str, Any]) -> List[Dict[str, Any]]:
    trials_path = result.get("trials_path")
    if not isinstance(trials_path, str) or not trials_path.strip():
        return []
    try:
        with open(trials_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    normalized: List[Dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        raw_score = item.get("objective")
        score = None
        status = "FAILED"
        if isinstance(raw_score, (int, float)) and float(raw_score) != float("-inf"):
            score = float(raw_score)
            status = "SUCCEEDED"
        trial_id = item.get("trial_id")
        if not isinstance(trial_id, int):
            trial_id = len(normalized) + 1
        normalized.append(
            {
                "trial_id": int(trial_id),
                "score": score,
                "status": status,
                "params": item.get("params") if isinstance(item.get("params"), dict) else {},
            }
        )
    normalized.sort(key=lambda it: int(it.get("trial_id") or 0))
    return normalized


def _canonical_optimization_from_request(request: Dict[str, Any]) -> Dict[str, Any]:
    spec_type = str(request.get("spec_type") or "").strip().lower()
    if spec_type not in {"optimize_backtest", "optimize_dca"}:
        raise ValueError("Unsupported optimization spec_type")

    optimization = request.get("optimization")
    if not isinstance(optimization, dict):
        raise ValueError("Missing optimization block")

    target_spec_type = "backtest" if spec_type == "optimize_backtest" else "dca"
    base_run_id = str(optimization.get("base_run_id") or "").strip()
    base_spec = optimization.get("base_spec")
    base_request = base_spec if isinstance(base_spec, dict) else None
    if base_request is None:
        if not base_run_id:
            raise ValueError("optimization requires optimization.base_run_id or optimization.base_spec")
        base_request = _load_canonical_run_request(base_run_id)

    base_spec_type = str(base_request.get("spec_type") or "").strip().lower()
    if base_spec_type != target_spec_type:
        raise ValueError(
            f"{spec_type} requires base spec_type={target_spec_type} (got '{base_spec_type or 'unknown'}')"
        )

    if target_spec_type == "backtest":
        runner_spec = _canonical_backtest_to_spec(base_request)
        runner_fn = optimize_variants.run_backtest_optimization
    else:
        runner_spec = _canonical_dca_to_strategy_spec(base_request)
        runner_fn = optimize_variants.run_strategy_optimization

    search_space = optimization.get("search_space")
    if not isinstance(search_space, dict) or not search_space:
        raise ValueError("optimization.search_space must be a non-empty object")
    objective_block = optimization.get("objective")
    budget_block = optimization.get("budget")
    if not isinstance(objective_block, dict) or not isinstance(budget_block, dict):
        raise ValueError("optimization.objective and optimization.budget are required")

    objective_metric = str(objective_block.get("metric") or "").strip()
    objective_direction = str(objective_block.get("direction") or "max").strip().lower()
    objective_internal = _canonical_optimization_objective(objective_metric, objective_direction)

    try:
        max_trials = int(budget_block.get("max_trials"))
    except Exception as exc:
        raise ValueError("optimization.budget.max_trials must be an integer >= 1") from exc
    if max_trials < 1:
        raise ValueError("optimization.budget.max_trials must be >= 1")
    seed = budget_block.get("seed")
    if seed is not None:
        try:
            seed = int(seed)
        except Exception as exc:
            raise ValueError("optimization.budget.seed must be an integer >= 0") from exc
        if seed < 0:
            raise ValueError("optimization.budget.seed must be >= 0")

    runner_spec = dict(runner_spec)
    runner_spec["optimization"] = {
        "method": "random",
        "max_trials": max_trials,
        "seed": seed,
        "search_space": _normalize_optimization_search_space(search_space),
        "objective": objective_internal,
    }

    raw_result = runner_fn(runner_spec)
    trials = _canonical_optimization_trials(raw_result if isinstance(raw_result, dict) else {})
    succeeded = sum(1 for trial in trials if trial.get("status") == "SUCCEEDED")
    failed = len(trials) - succeeded
    raw_best = raw_result.get("best") if isinstance(raw_result, dict) else None
    best: Dict[str, Any] | None = None
    if isinstance(raw_best, dict):
        raw_score = raw_best.get("objective")
        score = float(raw_score) if isinstance(raw_score, (int, float)) and float(raw_score) != float("-inf") else None
        best = {
            "score": score,
            "params": raw_best.get("params") if isinstance(raw_best.get("params"), dict) else {},
            "run_id": base_run_id or None,
        }

    return {
        "accepted": True,
        "spec_type": request.get("spec_type"),
        "result": {
            "objective": {"metric": objective_metric, "direction": objective_direction},
            "best": best,
            "trials": trials,
            "summary": {
                "total_trials": len(trials),
                "succeeded_trials": succeeded,
                "failed_trials": failed,
            },
            "source": {
                "base_run_id": base_run_id or None,
                "base_spec_type": base_spec_type,
            },
            "artifacts": {
                "trials_path": raw_result.get("trials_path") if isinstance(raw_result, dict) else None,
                "summary_path": raw_result.get("summary") if isinstance(raw_result, dict) else None,
            },
        },
    }


def _canonical_backtest_to_spec(request: Dict[str, Any]) -> Dict[str, Any]:
    data_block = request.get("data") or {}
    signal_block = request.get("signal") or {}
    strategy_block = request.get("strategy")

    mapped_data: Dict[str, Any] = {
        "symbol": data_block.get("symbol"),
        "timeframe": data_block.get("timeframe"),
        "start": data_block.get("start_date"),
        "end": data_block.get("end_date"),
    }
    if isinstance(data_block.get("currency"), str) and data_block.get("currency").strip():
        currency = data_block.get("currency").strip().upper()
        mapped_data["currency"] = currency
        mapped_data["delta_quotes"] = currency
    data_path = data_block.get("path") or data_block.get("dataset_path")
    if isinstance(data_path, str) and data_path.strip():
        mapped_data["source"] = "csv"
        mapped_data["path"] = data_path
    if isinstance(data_block.get("mysql"), dict):
        mapped_data["mysql"] = data_block.get("mysql")
    elif "path" not in mapped_data:
        # Canonical backtest auto mode: enable strategy source resolution chain
        # (Delta -> MySQL -> Java) when no explicit source is provided.
        mapped_data["mysql_env"] = "QE_MARKETDATA_MYSQL_URL"

    signal_params: Dict[str, Any] = {}
    for key in ("fast", "slow", "require_crossing"):
        if signal_block.get(key) is not None:
            signal_params[key] = signal_block.get(key)

    mapped: Dict[str, Any] = {
        "data": mapped_data,
        "signal": {
            "type": signal_block.get("type"),
            "params": signal_params,
        },
    }

    if isinstance(strategy_block, dict):
        params = strategy_block.get("params")
        if isinstance(params, dict):
            asset_class = params.get("asset_class")
            if isinstance(asset_class, str) and asset_class.strip():
                mapped["strategy"] = {"asset_class": asset_class.strip().upper()}
            tp_sl_internal, tp_sl_supported = _canonical_backtest_tp_sl_to_internal(params)
            if tp_sl_supported and tp_sl_internal is not None:
                mapped["tpsl"] = tp_sl_internal

    filters_block = request.get("filters")
    if isinstance(filters_block, dict):
        if isinstance(filters_block.get("filters"), list):
            mapped["filters"] = [
                {"type": item.get("id"), "params": item.get("params", {})}
                for item in filters_block.get("filters", [])
                if isinstance(item, dict)
            ]
        if isinstance(filters_block.get("rules"), list):
            mapped["filter_rules"] = [
                {
                    "type": item.get("id"),
                    **({"mode": item.get("mode")} if "mode" in item else {}),
                    **({"weight": item.get("weight")} if "weight" in item else {}),
                    **({"enabled": item.get("enabled")} if "enabled" in item else {}),
                    "params": item.get("params", {}),
                }
                for item in filters_block.get("rules", [])
                if isinstance(item, dict)
            ]
        if isinstance(filters_block.get("rules_config"), dict):
            mapped["filter_rules_config"] = dict(filters_block.get("rules_config", {}))

    performance_block = request.get("performance")
    if isinstance(performance_block, dict):
        performance_spec = _canonical_performance_to_internal(performance_block)
        if performance_spec:
            mapped["performance"] = performance_spec

    if isinstance(request.get("output"), dict):
        mapped["output"] = request.get("output")
    if isinstance(request.get("persistence"), dict):
        mapped["persistence"] = request.get("persistence")

    return mapped


def _canonical_dca_unsupported_details(request: Dict[str, Any]) -> List[Dict[str, str]]:
    details: List[Dict[str, str]] = []
    strategy_block = request.get("strategy")
    if not isinstance(strategy_block, dict):
        return details

    params = strategy_block.get("params")
    if not isinstance(params, dict):
        return details

    _grid_internal, grid_supported = _canonical_dca_grid_to_internal(strategy_block, params)
    if not grid_supported:
        details.append({"field": "strategy.grid", "reason": "accepted_but_not_wired"})

    _tp_sl_internal, tp_sl_supported = _canonical_dca_tp_sl_to_internal(params)
    if not tp_sl_supported:
        details.append({"field": "strategy.params.tp_sl", "reason": "accepted_but_not_wired"})

    execution_mode = params.get("execution_mode")
    if isinstance(execution_mode, str) and execution_mode.strip().lower() not in {"bar_close", "intracandle"}:
        details.append({"field": "strategy.params.execution_mode", "reason": "accepted_but_not_wired"})

    _drawdown_reference_internal, drawdown_reference_supported = _canonical_dca_drawdown_reference_to_internal(params)
    if not drawdown_reference_supported:
        details.append({"field": "strategy.params.drawdown_reference", "reason": "accepted_but_not_wired"})

    return details


def _canonical_dca_grid_to_internal(strategy_block: Dict[str, Any], params: Dict[str, Any]) -> tuple[Any, bool]:
    params_grid = params.get("grid")
    if isinstance(params_grid, list) and params_grid:
        return params_grid, True

    raw_grid = strategy_block.get("grid")
    if raw_grid in (None, []):
        return params_grid, True
    if not isinstance(raw_grid, list) or not raw_grid:
        return None, False

    preset = str(raw_grid[0]).strip().lower()
    presets: Dict[str, List[Dict[str, float]]] = {
        "grid_balanced": [
            {"dd": -5.0, "weight": 1.0},
            {"dd": -10.0, "weight": 1.0},
            {"dd": -15.0, "weight": 1.0},
        ],
    }
    if preset not in presets:
        return None, False
    return presets[preset], True


def _canonical_dca_drawdown_reference_to_internal(params: Dict[str, Any]) -> tuple[Any, bool]:
    raw = params.get("drawdown_reference")
    if raw is None:
        return None, True
    if not isinstance(raw, str):
        return raw, True
    token = raw.strip().lower()
    aliases = {
        "rolling_high": "90D",
    }
    return aliases.get(token, raw), True


def _canonical_dca_tp_sl_to_internal(params: Dict[str, Any]) -> tuple[Any, bool]:
    tp_sl_raw = params.get("tp_sl", params.get("tpSl"))
    if tp_sl_raw is None:
        return None, True
    if isinstance(tp_sl_raw, dict):
        # Already internal-compatible shape used by the strategy runtime.
        if "rules" in tp_sl_raw or "sl_dd" in tp_sl_raw:
            trailing_cfg = tp_sl_raw.get("trailing")
            if trailing_cfg is not None and not _canonical_dca_trailing_supported(trailing_cfg):
                return None, False
            return tp_sl_raw, True

        # Canonical explicit shape: tp/sl/break_even[/trailing].
        if not (isinstance(tp_sl_raw.get("tp"), dict) and isinstance(tp_sl_raw.get("sl"), dict)):
            return None, False
        tp_cfg = tp_sl_raw.get("tp", {})
        sl_cfg = tp_sl_raw.get("sl", {})
        be_cfg = tp_sl_raw.get("break_even", {})
        trailing_cfg = tp_sl_raw.get("trailing")
        if str(tp_cfg.get("type", "")).strip().lower() != "percent":
            return None, False
        if str(sl_cfg.get("type", "")).strip().lower() != "percent":
            return None, False
        try:
            tp_value = float(tp_cfg.get("value"))
            sl_value = float(sl_cfg.get("value"))
        except Exception:
            return None, False
        if tp_value <= 0 or sl_value <= 0:
            return None, False
        be_value = None
        if isinstance(be_cfg, dict) and be_cfg.get("enabled") and be_cfg.get("trigger_pct") is not None:
            try:
                be_value = float(be_cfg.get("trigger_pct"))
            except Exception:
                return None, False

        trailing_internal = None
        if trailing_cfg is not None:
            trailing_internal, trailing_supported = _canonical_dca_trailing_to_internal(trailing_cfg)
            if not trailing_supported:
                return None, False

        converted = {
            "enabled": bool(tp_sl_raw.get("enabled", True)),
            "mode": "per_grid_max_dd",
            "sl_dd": -abs(sl_value),
            "rules": [{"max_dd_reached": 0.0, "tp_pct": tp_value, "be_pct": be_value}],
        }
        if trailing_internal is not None:
            converted["trailing"] = trailing_internal
        return converted, True

    if isinstance(tp_sl_raw, str):
        token = tp_sl_raw.strip().lower()
        import re

        match = re.fullmatch(r"tp_(\d+(?:\.\d+)?)_sl_(\d+(?:\.\d+)?)", token)
        if not match:
            return None, False
        tp_value = float(match.group(1))
        sl_value = float(match.group(2))
        if tp_value <= 0 or sl_value <= 0:
            return None, False
        return {
            "enabled": True,
            "mode": "per_grid_max_dd",
            "sl_dd": -abs(sl_value),
            "rules": [{"max_dd_reached": 0.0, "tp_pct": tp_value, "be_pct": None}],
        }, True

    return None, False


def _canonical_dca_trailing_supported(trailing_cfg: Any) -> bool:
    _internal, supported = _canonical_dca_trailing_to_internal(trailing_cfg)
    return supported


def _canonical_dca_trailing_to_internal(trailing_cfg: Any) -> tuple[Any, bool]:
    if trailing_cfg is None:
        return None, True
    if not isinstance(trailing_cfg, dict):
        return None, False
    if trailing_cfg.get("enabled", True) is False:
        return {"enabled": False}, True
    trailing_type = str(trailing_cfg.get("type", "percent")).strip().lower()
    if trailing_type != "percent":
        return None, False
    try:
        value = float(trailing_cfg.get("value"))
    except Exception:
        return None, False
    if value <= 0:
        return None, False
    trigger_raw = trailing_cfg.get("trigger_pct", trailing_cfg.get("triggerPct", value))
    try:
        trigger_pct = float(trigger_raw)
    except Exception:
        return None, False
    if trigger_pct < 0:
        return None, False
    return {
        "enabled": True,
        "type": "percent",
        "value": value,
        "trigger_pct": trigger_pct,
    }, True


def _canonical_dca_to_strategy_spec(request: Dict[str, Any]) -> Dict[str, Any]:
    data_block = request.get("data") or {}
    strategy_block = request.get("strategy") or {}
    params = dict(strategy_block.get("params") or {})

    data_spec: Dict[str, Any] = {
        "timeframe": data_block.get("timeframe"),
        "start": data_block.get("start_date"),
        "end": data_block.get("end_date"),
    }
    data_currency = None
    if isinstance(data_block.get("currency"), str) and data_block.get("currency").strip():
        data_currency = data_block.get("currency").strip().upper()
        data_spec["currency"] = data_currency
        data_spec["delta_quotes"] = data_currency
    data_path = data_block.get("path") or data_block.get("dataset_path")
    if isinstance(data_path, str) and data_path.strip():
        data_spec["source"] = "csv"
        data_spec["path"] = data_path
    if isinstance(data_block.get("mysql"), dict):
        data_spec["mysql"] = data_block.get("mysql")

    strategy_type = str(strategy_block.get("type") or "").strip()
    asset_class = str(params.get("asset_class") or ("ETF" if strategy_type == "dca_etf" else "EQUITY")).upper()
    params.setdefault("asset_class", asset_class)
    grid_internal, grid_supported = _canonical_dca_grid_to_internal(strategy_block, params)
    if grid_supported and grid_internal is not None:
        params["grid"] = grid_internal
    tp_sl_internal, tp_sl_supported = _canonical_dca_tp_sl_to_internal(params)
    if tp_sl_supported and tp_sl_internal is not None:
        params["tp_sl"] = tp_sl_internal
    drawdown_reference_internal, drawdown_reference_supported = _canonical_dca_drawdown_reference_to_internal(params)
    if drawdown_reference_supported and drawdown_reference_internal is not None:
        params["drawdown_reference"] = drawdown_reference_internal
    params.pop("tpSl", None)
    strategy_id = f"CANONICAL_{strategy_type.upper()}" if strategy_type else "CANONICAL_DCA"

    universe_block = request.get("universe")
    universe_items: List[Dict[str, Any]] = []
    if isinstance(universe_block, list) and universe_block:
        for item in universe_block:
            if not isinstance(item, dict):
                continue
            symbol = str(item.get("symbol") or "").strip()
            if not symbol:
                continue
            universe_asset = str(item.get("asset_class") or item.get("assetClass") or asset_class).upper()
            universe_item = {
                "symbol": symbol,
                "asset_class": universe_asset,
            }
            for key in ("name", "exchange", "currency", "broker", "market_type", "marketType"):
                if item.get(key) is not None:
                    target_key = "market_type" if key == "marketType" else key
                    value = item.get(key)
                    if target_key == "currency" and isinstance(value, str):
                        value = value.strip().upper()
                    universe_item[target_key] = value
            if "currency" not in universe_item and data_currency:
                universe_item["currency"] = data_currency
            universe_items.append(universe_item)
    elif data_block.get("symbol"):
        # Compatibility path kept for current clients.
        _log_event(
            {
                "event": "canonical_dca_deprecation_warning",
                "field": "data.symbol",
                "message": "data.symbol fallback is deprecated; prefer universe[] for canonical dca runs",
                "target_version": "2026-06",
                "request_id": request.get("request_id"),
            }
        )
        fallback_item: Dict[str, Any] = {"symbol": data_block.get("symbol"), "asset_class": asset_class}
        if data_currency:
            fallback_item["currency"] = data_currency
        universe_items = [fallback_item]

    if not universe_items:
        raise ValueError("dca requires data.symbol or universe")

    spec: Dict[str, Any] = {
        "strategy": {
            "strategy_id": strategy_id,
            "type": strategy_type,
            "params": params,
        },
        "data": data_spec,
        "universe": universe_items,
    }

    filters_block = request.get("filters")
    if isinstance(filters_block, dict):
        if isinstance(filters_block.get("filters"), list):
            mapped_filters: List[Dict[str, Any]] = []
            for item in filters_block.get("filters", []):
                if not isinstance(item, dict):
                    continue
                mapped_filters.append(
                    {
                        "type": item.get("id"),
                        "params": item.get("params", {}),
                    }
                )
            spec["filters"] = mapped_filters
        if isinstance(filters_block.get("rules"), list):
            mapped_rules: List[Dict[str, Any]] = []
            for item in filters_block.get("rules", []):
                if not isinstance(item, dict):
                    continue
                mapped_rule: Dict[str, Any] = {
                    "type": item.get("id"),
                    "params": item.get("params", {}),
                }
                if "mode" in item:
                    mapped_rule["mode"] = item.get("mode")
                if "weight" in item:
                    mapped_rule["weight"] = item.get("weight")
                if "enabled" in item:
                    mapped_rule["enabled"] = item.get("enabled")
                mapped_rules.append(mapped_rule)
            spec["filter_rules"] = mapped_rules
        if isinstance(filters_block.get("rules_config"), dict):
            spec["filter_rules_config"] = filters_block.get("rules_config", {})

    performance_block = request.get("performance")
    if isinstance(performance_block, dict):
        performance_spec = _canonical_performance_to_internal(performance_block)
        if performance_spec:
            spec["performance"] = performance_spec

    return spec


def _timeframe_to_timedelta(timeframe: str) -> Optional[Any]:
    raw = str(timeframe or "").strip().lower()
    match = re.fullmatch(r"(\d+)\s*([mhdw])", raw)
    if not match:
        return None
    amount = int(match.group(1))
    unit = match.group(2)
    if amount <= 0:
        return None
    import pandas as pd  # type: ignore

    if unit == "m":
        return pd.Timedelta(minutes=amount)
    if unit == "h":
        return pd.Timedelta(hours=amount)
    if unit == "d":
        return pd.Timedelta(days=amount)
    if unit == "w":
        return pd.Timedelta(weeks=amount)
    return None


def _resolve_canonical_symbols(data_block: Dict[str, Any], *, spec_type: str) -> List[str]:
    symbols_raw = data_block.get("symbols")
    if isinstance(symbols_raw, list):
        resolved = [str(item).strip() for item in symbols_raw if str(item).strip()]
        if resolved:
            return resolved
    symbol = str(data_block.get("symbol") or "").strip()
    if symbol:
        return [symbol]
    raise ValueError(f"{spec_type} requires data.symbol or data.symbols")


def _canonical_market_stats_to_spec(request: Dict[str, Any]) -> Dict[str, Any]:
    import pandas as pd  # type: ignore

    data_block = request.get("data") or {}
    stats_block = request.get("stats") or {}

    timeframe = str(data_block.get("timeframe") or "").strip()
    symbols = _resolve_canonical_symbols(data_block, spec_type="market_stats")
    if not timeframe:
        raise ValueError("market_stats requires data.timeframe")

    lookback_raw = data_block.get("lookback")
    try:
        lookback = int(lookback_raw) if lookback_raw is not None else 500
    except Exception:
        lookback = 500
    lookback = max(1, lookback)

    start_raw = data_block.get("start_date")
    end_raw = data_block.get("end_date")
    if isinstance(start_raw, str) and start_raw.strip() and isinstance(end_raw, str) and end_raw.strip():
        start_ts = pd.Timestamp(start_raw)
        end_ts = pd.Timestamp(end_raw)
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize("UTC")
        else:
            start_ts = start_ts.tz_convert("UTC")
        if end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize("UTC")
        else:
            end_ts = end_ts.tz_convert("UTC")
    else:
        end_ts = pd.Timestamp.now(tz="UTC")
        step = _timeframe_to_timedelta(timeframe) or pd.Timedelta(hours=1)
        start_ts = end_ts - (step * lookback)

    mapped_data: Dict[str, Any] = {
        "symbols": symbols,
        "timeframe": timeframe,
        "start": start_ts.isoformat(),
        "end": end_ts.isoformat(),
    }
    if isinstance(data_block.get("asset_class"), str) and data_block.get("asset_class").strip():
        mapped_data["asset_class"] = data_block.get("asset_class").strip().upper()
    if isinstance(data_block.get("currency"), str) and data_block.get("currency").strip():
        currency = data_block.get("currency").strip().upper()
        mapped_data["currency"] = currency
        mapped_data["delta_quotes"] = currency
    for key in (
        "delta_base",
        "delta_prefix",
        "delta_exchange",
        "delta_market_type",
        "delta_quotes",
        "delta_broker",
        "delta_brokers",
        "delta_asset_dir",
        "delta_table",
        "delta_symbol",
        "delta_calendar",
        "delta_min_coverage",
    ):
        if data_block.get(key) is not None:
            mapped_data[key] = data_block.get(key)
    data_path = data_block.get("path") or data_block.get("dataset_path")
    if isinstance(data_path, str) and data_path.strip():
        mapped_data["dataset_path"] = data_path.strip()
    if isinstance(data_block.get("mysql"), dict):
        mapped_data["mysql"] = data_block.get("mysql")

    def _leaf_to_item(leaf: Any) -> Dict[str, Any]:
        if not isinstance(leaf, dict):
            return {}
        return {
            "name": leaf.get("id"),
            "params": leaf.get("params", {}) if isinstance(leaf.get("params"), dict) else {},
        }

    mapped: Dict[str, Any] = {
        "data": mapped_data,
        "events": [_leaf_to_item(stats_block.get("event"))],
        "conditions": [_leaf_to_item(stats_block.get("condition"))],
        "targets": [_leaf_to_item(stats_block.get("target"))],
    }

    validation_block = stats_block.get("validation")
    if isinstance(validation_block, dict):
        mapped["validation"] = {
            "train_months": validation_block.get("train_months"),
            "test_months": validation_block.get("test_months"),
            "folds": validation_block.get("folds"),
            "embargo_days": validation_block.get("embargo_days"),
        }

    output_block = request.get("output")
    if isinstance(output_block, dict):
        out_dir = output_block.get("out_dir")
        if isinstance(out_dir, str) and out_dir.strip():
            mapped["artifacts"] = {"out_dir": out_dir.strip()}

    persistence_block = request.get("persistence")
    if isinstance(persistence_block, dict):
        mapped["persistence"] = {
            "enabled": bool(persistence_block.get("enabled", False)),
            "spec_id": persistence_block.get("spec_id"),
            "dataset_id": persistence_block.get("dataset_id"),
        }

    performance_block = request.get("performance")
    if isinstance(performance_block, dict):
        mapped["performance"] = dict(performance_block)

    return mapped


def _canonical_seasonality_method(method: Any) -> str:
    raw = str(method or "").strip().lower()
    if raw == "topk":
        return "topk"
    if raw in {"threshold", "zscore", "percentile"}:
        return "threshold"
    return "threshold"


def _canonical_seasonality_combine(combine: Any) -> str:
    raw = str(combine or "").strip().lower()
    if raw in {"and", "or", "sum"}:
        return raw
    if raw == "vote":
        return "or"
    if raw in {"mean", "weighted"}:
        return "sum"
    return "and"


def _canonical_seasonality_measure(measure: Any) -> str:
    raw = str(measure or "").strip().lower()
    if raw in {"direction", "hit_rate"}:
        return "direction"
    if raw in {"return", "avg_return", "median_return"}:
        return "return"
    return "direction"


def _canonical_seasonality_profile_flags(profile_id: Any) -> Dict[str, bool]:
    raw = str(profile_id or "").strip().lower()
    flags = {
        "by_hour": False,
        "by_dow": False,
        "by_month": False,
        "by_session": False,
        "by_month_start": False,
        "by_month_end": False,
    }
    mapping = {
        "by_hour": "by_hour",
        "by_dow": "by_dow",
        "by_month": "by_month",
        "by_session": "by_session",
        "by_month_start": "by_month_start",
        "by_month_end": "by_month_end",
    }
    key = mapping.get(raw)
    if key:
        flags[key] = True
    else:
        flags["by_hour"] = True
    return flags


def _canonical_seasonality_to_spec(request: Dict[str, Any]) -> Dict[str, Any]:
    import pandas as pd  # type: ignore

    data_block = request.get("data") or {}
    seasonality_block = request.get("seasonality") or {}
    profile_block = seasonality_block.get("profile") if isinstance(seasonality_block, dict) else {}
    signal_block = seasonality_block.get("signal") if isinstance(seasonality_block, dict) else {}
    compute_block = seasonality_block.get("compute") if isinstance(seasonality_block, dict) else {}

    timeframe = str(data_block.get("timeframe") or "").strip()
    symbols = _resolve_canonical_symbols(data_block, spec_type="seasonality")
    if not timeframe:
        raise ValueError("seasonality requires data.timeframe")

    start_raw = data_block.get("start_date")
    end_raw = data_block.get("end_date")
    if isinstance(start_raw, str) and start_raw.strip() and isinstance(end_raw, str) and end_raw.strip():
        start_ts = pd.Timestamp(start_raw)
        end_ts = pd.Timestamp(end_raw)
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize("UTC")
        else:
            start_ts = start_ts.tz_convert("UTC")
        if end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize("UTC")
        else:
            end_ts = end_ts.tz_convert("UTC")
    else:
        current_year = datetime.now(timezone.utc).year
        try:
            start_year = int(data_block.get("start_year")) if data_block.get("start_year") is not None else current_year - 5
        except Exception:
            start_year = current_year - 5
        try:
            end_year = int(data_block.get("end_year")) if data_block.get("end_year") is not None else current_year
        except Exception:
            end_year = current_year
        if end_year < start_year:
            start_year, end_year = end_year, start_year
        start_ts = pd.Timestamp(f"{start_year:04d}-01-01T00:00:00Z")
        end_ts = pd.Timestamp(f"{end_year:04d}-12-31T23:59:59Z")

    mapped_data: Dict[str, Any] = {
        "symbols": symbols,
        "timeframe": timeframe,
        "start": start_ts.isoformat(),
        "end": end_ts.isoformat(),
    }
    if isinstance(data_block.get("asset_class"), str) and data_block.get("asset_class").strip():
        mapped_data["asset_class"] = data_block.get("asset_class").strip().upper()
    if isinstance(data_block.get("currency"), str) and data_block.get("currency").strip():
        currency = data_block.get("currency").strip().upper()
        mapped_data["currency"] = currency
        mapped_data["delta_quotes"] = currency
    for key in (
        "delta_base",
        "delta_prefix",
        "delta_exchange",
        "delta_market_type",
        "delta_quotes",
        "delta_broker",
        "delta_brokers",
        "delta_asset_dir",
        "delta_table",
        "delta_symbol",
        "delta_calendar",
        "delta_min_coverage",
    ):
        if data_block.get(key) is not None:
            mapped_data[key] = data_block.get(key)
    data_path = data_block.get("path") or data_block.get("dataset_path")
    if isinstance(data_path, str) and data_path.strip():
        mapped_data["dataset_path"] = data_path.strip()
    if isinstance(data_block.get("mysql"), dict):
        mapped_data["mysql"] = data_block.get("mysql")

    profile_id = profile_block.get("id") if isinstance(profile_block, dict) else None
    profile_flags = _canonical_seasonality_profile_flags(profile_id)
    profile: Dict[str, Any] = {
        **profile_flags,
        "measure": _canonical_seasonality_measure(profile_block.get("measure") if isinstance(profile_block, dict) else None),
        "ret_horizon": (profile_block.get("ret_horizon") if isinstance(profile_block, dict) else None) or 1,
        "min_samples_bin": (profile_block.get("min_samples_bin") if isinstance(profile_block, dict) else None) or 300,
    }

    signal: Dict[str, Any] = {
        "method": _canonical_seasonality_method(signal_block.get("method") if isinstance(signal_block, dict) else None),
        "threshold": (signal_block.get("threshold") if isinstance(signal_block, dict) else None) or 0.54,
        "topk": (signal_block.get("topk") if isinstance(signal_block, dict) else None) or 3,
        "dims": (signal_block.get("dims") if isinstance(signal_block, dict) and isinstance(signal_block.get("dims"), list) else ["hour", "dow"]),
        "combine": _canonical_seasonality_combine(signal_block.get("combine") if isinstance(signal_block, dict) else None),
    }

    compute: Dict[str, Any] = {
        "max_trials": (compute_block.get("max_trials") if isinstance(compute_block, dict) else None) or 30,
        "search_space": (compute_block.get("search_space") if isinstance(compute_block, dict) and isinstance(compute_block.get("search_space"), dict) else {}),
    }

    mapped: Dict[str, Any] = {
        "data": mapped_data,
        "profile": profile,
        "signal": signal,
        "compute": compute,
    }

    output_block = request.get("output")
    if isinstance(output_block, dict):
        out_dir = output_block.get("out_dir")
        if isinstance(out_dir, str) and out_dir.strip():
            mapped["artifacts"] = {"out_dir": out_dir.strip()}

    persistence_block = request.get("persistence")
    if isinstance(persistence_block, dict):
        mapped["persistence"] = {
            "enabled": bool(persistence_block.get("enabled", False)),
            "spec_id": persistence_block.get("spec_id"),
            "dataset_id": persistence_block.get("dataset_id"),
        }

    performance_block = request.get("performance")
    if isinstance(performance_block, dict):
        mapped["performance"] = dict(performance_block)

    return mapped


def _update_job_error_result(job_id: str, payload: Dict[str, Any], *, status: str) -> None:
    if status not in JOB_STATUSES:
        raise ValueError(f"Unknown job status: {status}")
    now = _utc_now()
    with db.session() as conn:
        conn.execute(
            """
            UPDATE api_jobs
            SET status = ?,
                result_json = ?,
                finished_at = ?,
                updated_at = ?
            WHERE job_id = ?
            """,
            (status, _serialize_payload(payload), now, now, job_id),
        )


def _requeue_job(job_id: str, *, error: str | None = None, queued_status: str = JOB_STATUS_QUEUED) -> None:
    now = _utc_now()
    with db.session() as conn:
        conn.execute(
            """
            UPDATE api_jobs
            SET status = ?,
                error_message = COALESCE(?, error_message),
                started_at = NULL,
                finished_at = NULL,
                updated_at = ?
            WHERE job_id = ?
            """,
            (queued_status, error, now, job_id),
        )


def _mark_canceled(job_id: str, *, message: str = "Run canceled") -> None:
    _update_job_status(job_id, JOB_STATUS_CANCELED, error=message)
    _update_job_error_result(job_id, _job_error_payload("canceled", message), status=JOB_STATUS_CANCELED)


def request_job_cancel(job_id: str, *, message: str = "Run canceled") -> bool:
    """Request cancellation of a job. Returns True when the job is canceled immediately."""
    with db.session() as conn:
        row = conn.execute(
            "SELECT status FROM api_jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        if not row:
            return False
        status = str(row["status"])
        if status in {JOB_STATUS_QUEUED, JOB_STATUS_PENDING}:
            conn.execute(
                """
                UPDATE api_jobs
                SET status = ?, cancel_requested = 1, canceled_at = ?, finished_at = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (JOB_STATUS_CANCELED, _utc_now(), _utc_now(), _utc_now(), job_id),
            )
            return True
        conn.execute(
            "UPDATE api_jobs SET cancel_requested = 1, updated_at = ? WHERE job_id = ?",
            (_utc_now(), job_id),
        )
    return False


def update_job_progress(job_id: str, progress: Dict[str, Any]) -> None:
    with db.session() as conn:
        conn.execute(
            """
            UPDATE api_jobs
            SET progress_json = ?, updated_at = ?
            WHERE job_id = ?
            """,
            (_serialize_payload(progress), _utc_now(), job_id),
        )


def _run_job_payload(job_type: str, payload: Any | None) -> Any:
    if job_type == JOB_TYPE_OPTIMIZATION:
        spec_payload = payload["spec"] if isinstance(payload, dict) and "spec" in payload else payload
        spec_obj = spec_module.spec_from_dict(spec_payload)
        return run_optimisation(spec_obj)
    if job_type == JOB_TYPE_STATS:
        spec_obj = schemas.StatsSpec.model_validate(payload or {})
        return stats_runner.run_stats(spec_obj)
    if job_type == JOB_TYPE_LEVELS_BUILD:
        spec_obj = LevelsBuildSpec.model_validate(payload or {})
        return run_levels_build(spec_obj)
    if job_type == JOB_TYPE_LEVELS_FILL:
        spec_obj = LevelsBuildSpec.model_validate(payload or {})
        return run_levels_fill(spec_obj)
    if job_type == JOB_TYPE_SEASONALITY_RUN:
        spec_obj = schemas.SeasonalitySpec.model_validate(payload or {})
        return seasonality_runner.run(spec_obj)
    if job_type == JOB_TYPE_SEASONALITY_OPTIMIZE:
        spec_obj = schemas.SeasonalitySpec.model_validate(payload or {})
        return seasonality_run_optimization(spec_obj)
    if job_type == JOB_TYPE_CANONICAL_RUN:
        # Canonical /runs requests are queued first; execution wiring is added incrementally.
        if isinstance(payload, dict):
            request = payload.get("request", {})
            if isinstance(request, dict):
                _log_event(
                    {
                        "event": "runs_worker_payload",
                        "spec_type": request.get("spec_type"),
                        "request_id": request.get("request_id"),
                        "payload": _truncate_json_payload(request),
                    }
                )
                spec_type = str(request.get("spec_type") or "").strip().lower()
                if spec_type == "backtest":
                    details = _canonical_backtest_unsupported_details(request)
                    if details:
                        return _job_error_payload(
                            "not_implemented_feature",
                            "Feature not implemented for canonical backtest run",
                            details=details,
                        )
                    spec = _canonical_backtest_to_spec(request)
                    result = backtest_runner.run_backtest_from_spec(spec)
                    return {
                        "accepted": True,
                        "spec_type": request.get("spec_type"),
                        "result": result,
                    }
                if spec_type == "dca":
                    details = _canonical_dca_unsupported_details(request)
                    if details:
                        return _job_error_payload(
                            "not_implemented_feature",
                            "Feature not implemented for canonical dca run",
                            details=details,
                        )
                    spec = _canonical_dca_to_strategy_spec(request)
                    strategy_result = strategies_runner.run_backtest_with_payload(spec)
                    return {
                        "accepted": True,
                        "spec_type": request.get("spec_type"),
                        "result": strategy_result.get("result"),
                        "payload": strategy_result.get("payload"),
                    }
                if spec_type == "market_stats":
                    mapped_spec = _canonical_market_stats_to_spec(request)
                    spec_obj = schemas.StatsSpec.model_validate(mapped_spec)
                    stats_df = stats_runner.run_stats(spec_obj)
                    return {
                        "accepted": True,
                        "spec_type": request.get("spec_type"),
                        "result": _stats_payload_from_df(stats_df),
                    }
                if spec_type == "seasonality":
                    mapped_spec = _canonical_seasonality_to_spec(request)
                    spec_obj = schemas.SeasonalitySpec.model_validate(mapped_spec)
                    seasonality_result = seasonality_runner.run(spec_obj)
                    return {
                        "accepted": True,
                        "spec_type": request.get("spec_type"),
                        "result": seasonality_result,
                    }
                if spec_type == "stress_tests":
                    data_block = request.get("data") if isinstance(request.get("data"), dict) else {}
                    base_run_id = str(data_block.get("base_run_id") or "").strip()
                    if not base_run_id:
                        return _job_error_payload(
                            "validation_error",
                            "Missing required field: data.base_run_id",
                        )

                    performance_block = request.get("performance") if isinstance(request.get("performance"), dict) else {}
                    stress_block = performance_block.get("stress_tests")
                    if not isinstance(stress_block, dict):
                        return _job_error_payload(
                            "validation_error",
                            "Missing required block: performance.stress_tests",
                        )

                    source_trades = _load_stress_source_trades(base_run_id)
                    if not source_trades:
                        return _job_error_payload(
                            "execution_error",
                            f"No completed trades found for base run '{base_run_id}'",
                        )

                    stress_internal = _canonical_stress_tests_to_internal(stress_block)
                    monte_carlo_cfg = stress_internal.get("monte_carlo", {})
                    scenario_cfg = stress_internal.get("scenarios")

                    stress_result: Dict[str, Any] = {}
                    metadata = {"base_run_id": base_run_id}

                    if not isinstance(monte_carlo_cfg, dict):
                        monte_carlo_cfg = {}
                    if monte_carlo_cfg.get("enabled") is not False:
                        stress_result["monte_carlo"] = stress_tests_runner.run_monte_carlo_on_trades(
                            source_trades,
                            metadata=metadata,
                            parameters=monte_carlo_cfg,
                        )

                    if scenario_cfg is not None:
                        scenario_params: Dict[str, Any] = {}
                        if isinstance(scenario_cfg, list):
                            scenario_params["scenarios"] = scenario_cfg
                        elif isinstance(scenario_cfg, dict):
                            if isinstance(scenario_cfg.get("scenarios"), list):
                                scenario_params.update(scenario_cfg)
                            else:
                                scenario_params["scenarios"] = [scenario_cfg]

                        for key in ("multi_asset", "aggregation", "weights", "timestamp_alignment"):
                            if key in monte_carlo_cfg and key not in scenario_params:
                                scenario_params[key] = monte_carlo_cfg.get(key)

                        if scenario_params.get("scenarios"):
                            stress_result["scenarios"] = stress_tests_runner.run_scenarios_on_trades(
                                source_trades,
                                metadata=metadata,
                                parameters=scenario_params,
                            )

                    if not stress_result:
                        return _job_error_payload(
                            "validation_error",
                            "No stress test mode enabled (monte_carlo/scenarios)",
                        )

                    return {
                        "accepted": True,
                        "spec_type": request.get("spec_type"),
                        "base_run_id": base_run_id,
                        "result": {
                            "source": {"base_run_id": base_run_id, "trades_count": len(source_trades)},
                            "stress_tests": stress_result,
                        },
                    }
                if spec_type in {"optimize_backtest", "optimize_dca"}:
                    try:
                        return _canonical_optimization_from_request(request)
                    except ValueError as exc:
                        return _job_error_payload("validation_error", str(exc))
                return {"accepted": True, "spec_type": request.get("spec_type")}
        return {"accepted": True}
    raise ValueError(f"Unknown job type: {job_type}")


def _execute_job(
    job_id: str,
    job_type: str,
    payload: Any | None,
    *,
    set_running: bool = True,
    running_status: str = JOB_STATUS_RUNNING,
    success_status: str = JOB_STATUS_COMPLETED,
    failure_status: str = JOB_STATUS_FAILED,
) -> Any:
    if set_running:
        _update_job_status(job_id, running_status)
    try:
        result = _run_job_payload(job_type, payload)
        if job_type == JOB_TYPE_OPTIMIZATION:
            _persist_optimization_result(job_id, payload, result)
        if job_type == JOB_TYPE_CANONICAL_RUN:
            _persist_canonical_run_metrics(job_id, payload, result)
    except Exception as exc:  # pragma: no cover - defensive
        _update_job_status(job_id, failure_status, error=str(exc))
        raise
    payload_out = _build_job_result(job_type, job_id, result)
    _update_job_result(job_id, payload_out, status=success_status)
    return payload_out


def _first_numeric_metric(metrics: Dict[str, Any]) -> float | None:
    for value in metrics.values():
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _persist_canonical_run_metrics(job_id: str, payload: Any | None, result: Any) -> None:
    if not isinstance(payload, dict) or not isinstance(result, dict):
        return
    request = payload.get("request") if isinstance(payload.get("request"), dict) else {}
    spec_type = str(request.get("spec_type") or "").strip().lower()
    if spec_type != "dca" or result.get("accepted") is not True:
        return

    run_payload = result.get("payload") if isinstance(result.get("payload"), dict) else {}
    run = run_payload.get("run") if isinstance(run_payload.get("run"), dict) else {}
    extra = run.get("extra") if isinstance(run.get("extra"), dict) else {}

    metrics_map: Dict[str, float] = {}
    for key in (
        "final_performance_normalized",
        "twr",
        "xirr",
        "max_drawdown_on_contributed_capital",
        "time_under_water",
    ):
        value = extra.get(key)
        if isinstance(value, (int, float)):
            metrics_map[key] = float(value)

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

    if not metrics_map:
        return

    with db.session() as conn:
        MetricsRepository(conn).bulk_upsert_metrics(job_id, metrics_map, fold=None)


def _persist_optimization_result(job_id: str, payload: Any | None, result: Any) -> None:
    if not isinstance(payload, dict):
        return
    spec_payload = payload.get("spec") if isinstance(payload.get("spec"), dict) else {}
    strategy_payload = spec_payload.get("strategy") if isinstance(spec_payload.get("strategy"), dict) else {}
    data_payload = spec_payload.get("data") if isinstance(spec_payload.get("data"), dict) else {}
    objective = strategy_payload.get("objective") if isinstance(strategy_payload.get("objective"), str) else ""
    spec_id = (
        str(strategy_payload.get("strategy_id") or spec_payload.get("spec_id") or "")
        if isinstance(strategy_payload, dict)
        else ""
    )
    dataset_id = str(
        spec_payload.get("dataset_id")
        or data_payload.get("dataset_id")
        or data_payload.get("dataset_path")
        or ""
    )
    out_dir = ""
    if isinstance(result, dict):
        trials_path = result.get("trials_path")
        if isinstance(trials_path, str) and trials_path.strip():
            out_dir = os.path.dirname(trials_path) or ""
    with db.session() as conn:
        runs_repo = RunsRepository(conn)
        metrics_repo = MetricsRepository(conn)
        trials_repo = TrialsRepository(conn)
        runs_repo.create_or_running(
            run_id=job_id,
            spec_id=spec_id,
            dataset_id=dataset_id,
            objective=str(objective or ""),
            out_dir=out_dir,
        )

        if isinstance(result, dict):
            best = result.get("best")
            if isinstance(best, dict):
                best_metrics = best.get("metrics")
                if isinstance(best_metrics, dict) and best_metrics:
                    metrics_repo.bulk_upsert_metrics(job_id, best_metrics, fold=None)

            trials = result.get("trials")
            mapped_trials: List[Dict[str, Any]] = []
            if isinstance(trials, list):
                for index, trial in enumerate(trials, start=1):
                    if not isinstance(trial, dict):
                        continue
                    params = trial.get("params") if isinstance(trial.get("params"), dict) else {}
                    metrics = trial.get("metrics") if isinstance(trial.get("metrics"), dict) else {}
                    objective_value = None
                    if objective and objective in metrics and isinstance(metrics.get(objective), (int, float)):
                        objective_value = float(metrics.get(objective))
                    elif metrics:
                        objective_value = _first_numeric_metric(metrics)
                    mapped_trials.append(
                        {
                            "trial_number": int(trial.get("trial_number") or index),
                            "params": params,
                            "objective_value": objective_value,
                            "status": trial.get("status") or "COMPLETE",
                            "n_trades": trial.get("n_trades") or metrics.get("n_trades") or metrics.get("trades"),
                            "max_dd": trial.get("max_dd") or metrics.get("max_dd"),
                            "sharpe": trial.get("sharpe") or metrics.get("sharpe"),
                            "sortino": trial.get("sortino") or metrics.get("sortino"),
                            "cagr": trial.get("cagr") or metrics.get("cagr"),
                            "hit_rate": trial.get("hit_rate") or metrics.get("hit_rate"),
                            "avg_r": trial.get("avg_r") or metrics.get("avg_r"),
                        }
                    )
            if mapped_trials:
                trials_repo.bulk_insert_trials(job_id, mapped_trials)
                logger.info("Optimization persistence upserted trials | run_id=%s rows=%s", job_id, len(mapped_trials))

        runs_repo.finish(job_id, "COMPLETED")


def _run_job(job_type: str, payload: Any | None) -> tuple[str, Any]:
    job_id = _enqueue_job(job_type, payload=payload)
    result = _execute_job(job_id, job_type, payload, set_running=True)
    return job_id, result


def _claim_next_job(
    job_type: str | None = None,
    *,
    statuses: Sequence[str] | None = None,
    running_status: str = JOB_STATUS_RUNNING,
) -> Dict[str, Any] | None:
    if statuses is None:
        statuses = (JOB_STATUS_PENDING,)
    with db.session() as conn:
        conn.execute("BEGIN IMMEDIATE")
        placeholders = ", ".join(["?"] * len(statuses))
        query = f"SELECT job_id FROM api_jobs WHERE status IN ({placeholders})"
        params: List[Any] = []
        params.extend(list(statuses))
        if job_type:
            query += " AND job_type = ?"
            params.append(job_type)
        query += " AND (cancel_requested IS NULL OR cancel_requested = 0)"
        query += " ORDER BY created_at ASC LIMIT 1"
        row = conn.execute(query, params).fetchone()
        if not row:
            return None
        job_id = row["job_id"]
        now = _utc_now()
        update_placeholders = ", ".join(["?"] * len(statuses))
        conn.execute(
            f"""
            UPDATE api_jobs
            SET status = ?,
                started_at = ?,
                updated_at = ?,
                attempts = COALESCE(attempts, 0) + 1
            WHERE job_id = ? AND status IN ({update_placeholders})
            """,
            (running_status, now, now, job_id, *statuses),
        )
    job = _get_job(job_id)
    if job is None or job.get("status") != running_status:
        return None
    return job


def run_next_job(
    job_type: str | None = None,
    *,
    statuses: Sequence[str] | None = None,
    running_status: str = JOB_STATUS_RUNNING,
    success_status: str = JOB_STATUS_COMPLETED,
    failure_status: str = JOB_STATUS_FAILED,
) -> Dict[str, Any] | None:
    """Claim and execute the next queued job (used by async workers)."""

    job = _claim_next_job(job_type, statuses=statuses, running_status=running_status)
    if not job:
        return None
    return _execute_job(
        job_id=job["job_id"],
        job_type=job["job_type"],
        payload=job.get("payload"),
        set_running=False,
        running_status=running_status,
        success_status=success_status,
        failure_status=failure_status,
    )


def submit(spec: Spec) -> schemas.SubmitResponse:
    job_id, _ = _run_job(JOB_TYPE_OPTIMIZATION, payload={"spec": asdict(spec)})
    return schemas.SubmitResponse(id=job_id)


def submit_async(spec: Spec) -> schemas.StatusResponse:
    job_id = _enqueue_job(JOB_TYPE_OPTIMIZATION, payload={"spec": asdict(spec)})
    return schemas.StatusResponse(status="pending", id=job_id)


def status(job_id: str) -> schemas.StatusResponse:
    job = _get_job(job_id)
    if not job:
        return schemas.StatusResponse(status="unknown")
    return schemas.StatusResponse(status=job["status"])


def result(job_id: str) -> schemas.ResultResponse:
    job = _get_job(job_id)
    if not job:
        return schemas.ResultResponse(result=None)
    return schemas.ResultResponse(result=job.get("result"))


def enqueue_run_request(payload: Dict[str, Any]) -> schemas.RunEnqueueResponse:
    """Validate and enqueue a canonical run request."""

    try:
        parsed = validate_run_request_input(payload)
    except ValidationError as exc:
        raise ApiValidationException(normalize_pydantic_errors(exc)) from exc

    canonical_payload = parsed.model_dump(mode="json")
    _validate_canonical_market_stats_params(canonical_payload)
    request_id = _normalize_request_id(canonical_payload.get("request_id"))
    reused = False
    if request_id is not None:
        existing = _get_job(request_id)
        if existing and existing.get("job_type") == JOB_TYPE_CANONICAL_RUN:
            reused = True
            status = _external_job_status(str(existing.get("status", "pending")))
            return schemas.RunEnqueueResponse(run_id=request_id, status=status, reused=True)
        if existing and existing.get("job_type") != JOB_TYPE_CANONICAL_RUN:
            raise ApiValidationException(
                single_validation_error("request_id", "conflict", "request_id already exists")
            )
    else:
        request_id = ids.generate_id()

    max_attempts, timeout_seconds = _canonical_job_defaults()
    _init_job(
        request_id,
        JOB_TYPE_CANONICAL_RUN,
        payload={"request": canonical_payload},
        status=JOB_STATUS_QUEUED,
        max_attempts=max_attempts,
        timeout_seconds=timeout_seconds,
    )
    status = JOB_STATUS_QUEUED
    return schemas.RunEnqueueResponse(run_id=request_id, status=status, reused=reused)


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            if "." in text:
                maybe_float = float(text)
                if maybe_float.is_integer():
                    return int(maybe_float)
                return None
            return int(text)
        except Exception:
            return None
    return None


def _validate_canonical_market_stats_params(canonical_payload: Dict[str, Any]) -> None:
    if str(canonical_payload.get("spec_type") or "").strip().lower() != "market_stats":
        return

    stats_block = canonical_payload.get("stats")
    if not isinstance(stats_block, dict):
        return

    errors: List[Dict[str, str]] = []

    def _require_positive_int(params: Dict[str, Any], field_prefix: str, name: str) -> None:
        value = params.get(name)
        field = f"{field_prefix}.{name}"
        if value in (None, ""):
            errors.append({"field": field, "code": "missing", "message": "Field required"})
            return
        coerced = _coerce_int(value)
        if coerced is None:
            errors.append({"field": field, "code": "int_parsing", "message": "Input should be a valid integer"})
            return
        if coerced < 1:
            errors.append({"field": field, "code": "greater_than_equal", "message": "Input should be >= 1"})

    def _require_direction(params: Dict[str, Any], field_prefix: str, name: str = "direction") -> None:
        value = params.get(name)
        field = f"{field_prefix}.{name}"
        if value in (None, ""):
            errors.append({"field": field, "code": "missing", "message": "Field required"})
            return
        token = str(value).strip().lower()
        if token not in {"up", "down"}:
            errors.append({"field": field, "code": "literal_error", "message": "Input should be 'up' or 'down'"})

    event = stats_block.get("event")
    if isinstance(event, dict):
        event_id = str(event.get("id") or "").strip().lower()
        event_params = event.get("params") if isinstance(event.get("params"), dict) else {}
        if event_id == "k_consecutive":
            base = "market_stats.stats.event.params"
            _require_positive_int(event_params, base, "k")
            _require_direction(event_params, base)

    condition = stats_block.get("condition")
    if isinstance(condition, dict):
        condition_id = str(condition.get("id") or "").strip().lower()
        condition_params = condition.get("params") if isinstance(condition.get("params"), dict) else {}
        if condition_id == "htf_trend":
            base = "market_stats.stats.condition.params"
            _require_positive_int(condition_params, base, "tf_multiplier")
            _require_positive_int(condition_params, base, "ema_period")

    target = stats_block.get("target")
    if isinstance(target, dict):
        target_id = str(target.get("id") or "").strip().lower()
        target_params = target.get("params") if isinstance(target.get("params"), dict) else {}
        if target_id == "continuation_n":
            base = "market_stats.stats.target.params"
            _require_positive_int(target_params, base, "n")
            _require_direction(target_params, base)
        elif target_id == "time_to_reversal":
            base = "market_stats.stats.target.params"
            _require_positive_int(target_params, base, "max_horizon")

    if errors:
        raise ApiValidationException(errors)


def _normalize_request_id(request_id: Any) -> str | None:
    if request_id is None:
        return None
    value = str(request_id).strip()
    return value or None


def _external_job_status(status: str) -> str:
    normalized = str(status or "").strip()
    if not normalized:
        return JOB_STATUS_QUEUED
    legacy_map = {
        JOB_STATUS_PENDING: JOB_STATUS_QUEUED,
        JOB_STATUS_RUNNING: JOB_STATUS_RUNNING_CANONICAL,
        JOB_STATUS_COMPLETED: JOB_STATUS_SUCCEEDED,
        JOB_STATUS_FAILED: JOB_STATUS_FAILED_CANONICAL,
    }
    if normalized in legacy_map:
        return legacy_map[normalized]
    upper = normalized.upper()
    if upper in {
        JOB_STATUS_QUEUED,
        JOB_STATUS_RUNNING_CANONICAL,
        JOB_STATUS_SUCCEEDED,
        JOB_STATUS_FAILED_CANONICAL,
        JOB_STATUS_CANCELED,
    }:
        return upper
    return normalized


def _json_error(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"code": code, "message": message})


def _is_terminal_status(status: str) -> bool:
    return status in {JOB_STATUS_SUCCEEDED, JOB_STATUS_FAILED_CANONICAL, JOB_STATUS_CANCELED}


def _canonical_run_payload(job: Dict[str, Any]) -> Dict[str, Any]:
    status = _external_job_status(job.get("status", ""))
    payload: Dict[str, Any] = {
        "run_id": job.get("job_id"),
        "request_id": job.get("job_id"),
        "requestId": job.get("job_id"),
        "status": status,
        "job_type": job.get("job_type"),
        "created_at": job.get("created_at"),
        "updated_at": job.get("updated_at"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "attempts": job.get("attempts"),
        "max_attempts": job.get("max_attempts"),
        "timeout_seconds": job.get("timeout_seconds"),
        "progress": job.get("progress"),
    }
    if job.get("error_message"):
        payload["error"] = {"code": "execution_error", "message": job.get("error_message")}
    return payload


def _db_ready() -> tuple[bool, str | None]:
    try:
        with db.session() as conn:
            conn.execute("SELECT 1")
        return True, None
    except Exception as exc:  # pragma: no cover - defensive
        return False, str(exc)


def _get_correlation_id(request: Request) -> str | None:
    return (
        request.headers.get("x-correlation-id")
        or request.headers.get("x-correlationid")
        or request.headers.get("x-request-id")
    )


def _log_event(event: Dict[str, Any]) -> None:
    try:
        print(json.dumps(event, separators=(",", ":")), flush=True)
    except Exception:
        pass


def _truncate_json_payload(value: Any, *, max_len: int = 4000) -> str:
    try:
        payload = json.dumps(value, separators=(",", ":"), default=str)
    except Exception:
        payload = str(value)
    if len(payload) > max_len:
        return payload[: max_len - 3] + "..."
    return payload


def _canonical_job_defaults() -> tuple[int | None, int | None]:
    max_attempts_raw = os.getenv("QE_CANONICAL_MAX_ATTEMPTS", "").strip()
    timeout_raw = os.getenv("QE_CANONICAL_TIMEOUT_SECONDS", "").strip()
    max_attempts = int(max_attempts_raw) if max_attempts_raw.isdigit() else 3
    timeout_seconds = int(timeout_raw) if timeout_raw.isdigit() else None
    return max_attempts, timeout_seconds


def _canonical_runs_capabilities(spec_type: str) -> Dict[str, Any]:
    normalized = str(spec_type or "").strip().lower()
    if normalized == "market_stats":
        return {
            "spec_type": "market_stats",
            "catalog_version": CANONICAL_CAPABILITIES_CATALOG_VERSION,
            "fields": {
                "supported": [
                    "catalog_version",
                    "request_id",
                    "data.symbol",
                    "data.symbols",
                    "data.timeframe",
                    "data.dataset_path",
                    "data.path",
                    "data.mysql",
                    "data.lookback",
                    "data.stats_pack",
                    "data.session",
                    "data.include_weekends",
                    "data.asset_class",
                    "data.currency",
                    "stats.event",
                    "stats.condition",
                    "stats.target",
                    "stats.validation",
                    "output",
                    "persistence",
                ],
                "accepted_but_not_wired": [
                    "data.lookback",
                    "data.stats_pack",
                    "data.session",
                    "data.include_weekends",
                    "data.asset_class",
                    "data.currency",
                ],
            },
            "runtime_rules": {
                "execution_status": "partially_wired",
                "failure_mode": "runtime executes stats runner; accepted_but_not_wired fields are validated but ignored",
                "data_source_requirements": "data.path|data.dataset_path or data.mysql is required at runtime",
                "symbol_resolution": "data.symbols has priority over data.symbol",
            },
        }

    if normalized == "seasonality":
        return {
            "spec_type": "seasonality",
            "catalog_version": CANONICAL_CAPABILITIES_CATALOG_VERSION,
            "fields": {
                "supported": [
                    "catalog_version",
                    "request_id",
                    "data.symbol",
                    "data.symbols",
                    "data.asset_class",
                    "data.currency",
                    "data.timeframe",
                    "data.start_date",
                    "data.end_date",
                    "data.window",
                    "data.start_year",
                    "data.end_year",
                    "data.dataset_path",
                    "data.path",
                    "data.mysql",
                    "seasonality.profile",
                    "seasonality.signal",
                    "seasonality.compute",
                    "seasonality.execution",
                    "seasonality.risk",
                    "seasonality.tp_sl",
                    "output",
                    "persistence",
                ],
                "accepted_but_not_wired": [
                    "data.asset_class",
                    "data.currency",
                    "data.window",
                    "seasonality.execution",
                    "seasonality.risk",
                    "seasonality.tp_sl",
                ],
            },
            "runtime_rules": {
                "execution_status": "partially_wired",
                "failure_mode": "runtime executes seasonality runner; accepted_but_not_wired fields are validated but ignored",
                "data_source_requirements": "data.path|data.dataset_path or data.mysql is required at runtime",
                "symbol_resolution": "data.symbols has priority over data.symbol",
            },
        }

    if normalized == "backtest":
        return {
            "spec_type": "backtest",
            "catalog_version": CANONICAL_CAPABILITIES_CATALOG_VERSION,
            "fields": {
                "supported": [
                    "catalog_version",
                    "request_id",
                    "data.symbol",
                    "data.currency",
                    "data.timeframe",
                    "data.start_date",
                    "data.end_date",
                    "data.dataset_path",
                    "data.path",
                    "data.mysql",
                    "signal",
                    "filters.filters",
                    "filters.rules",
                    "filters.rules_config",
                    "strategy.name",
                    "strategy.params.tp_sl",
                    "performance.initial_capital",
                    "performance.capital_per_unit",
                    "performance.max_capital_per_trade",
                    "performance.risk_pct",
                    "performance.risk_free_rate_pct",
                    "performance.stress_tests",
                    "performance.stress_tests.enabled",
                    "performance.stress_tests.source",
                    "performance.stress_tests.method",
                    "performance.stress_tests.n_sims",
                    "performance.stress_tests.seed",
                    "performance.stress_tests.block_size",
                    "performance.stress_tests.overlapping",
                    "performance.stress_tests.time_distribution",
                    "performance.stress_tests.param_drift",
                    "performance.stress_tests.sizing",
                    "performance.stress_tests.output",
                    "performance.stress_tests.scenarios",
                    "performance.stress_tests.multi_asset",
                    "performance.stress_tests.aggregation",
                    "performance.stress_tests.weights",
                    "performance.stress_tests.timestamp_alignment",
                    "output",
                    "persistence",
                ],
                "accepted_but_not_wired": [
                    "strategy.name",
                ],
            },
            "presets": {
                "supported": {
                    "signal.type": ["ema_cross"],
                },
                "not_supported": {
                    "signal.type": ["* (except ema_cross)"],
                },
            },
            "runtime_rules": {
                "execution_status": "partially_wired",
                "failure_mode": "worker returns not_implemented_feature for accepted_but_not_wired blocks",
                "details_source": "_canonical_backtest_unsupported_details",
                "data_source_resolution": {
                    "mode": "auto_when_no_explicit_source",
                    "order": ["delta", "mysql", "java"],
                    "explicit_source_priority": ["data.path|data.dataset_path", "data.mysql"],
                },
            },
        }

    if normalized == "stress_tests":
        return {
            "spec_type": "stress_tests",
            "catalog_version": CANONICAL_CAPABILITIES_CATALOG_VERSION,
            "fields": {
                "supported": [
                    "catalog_version",
                    "request_id",
                    "data.base_run_id",
                    "performance.stress_tests",
                    "performance.stress_tests.enabled",
                    "performance.stress_tests.source",
                    "performance.stress_tests.method",
                    "performance.stress_tests.n_sims",
                    "performance.stress_tests.seed",
                    "performance.stress_tests.block_size",
                    "performance.stress_tests.overlapping",
                    "performance.stress_tests.time_distribution",
                    "performance.stress_tests.param_drift",
                    "performance.stress_tests.sizing",
                    "performance.stress_tests.output",
                    "performance.stress_tests.scenarios",
                    "performance.stress_tests.multi_asset",
                    "performance.stress_tests.aggregation",
                    "performance.stress_tests.weights",
                    "performance.stress_tests.timestamp_alignment",
                    "output",
                    "persistence",
                ],
                "accepted_but_not_wired": [
                    "output",
                    "persistence",
                ],
            },
            "runtime_rules": {
                "execution_status": "wired",
                "source_run_requirements": "data.base_run_id must reference an existing canonical DCA/backtest run with completed trades",
                "trade_source_priority": ["trades_completed table", "api_jobs.result.payload.trades"],
            },
        }

    if normalized in {"optimize_backtest", "optimize_dca"}:
        target_spec_type = "backtest" if normalized == "optimize_backtest" else "dca"
        return {
            "spec_type": normalized,
            "catalog_version": CANONICAL_CAPABILITIES_CATALOG_VERSION,
            "fields": {
                "supported": [
                    "catalog_version",
                    "request_id",
                    "optimization",
                    "optimization.base_run_id",
                    "optimization.base_spec",
                    "optimization.search_space",
                    "optimization.objective.metric",
                    "optimization.objective.direction",
                    "optimization.budget.max_trials",
                    "optimization.budget.timeout_seconds",
                    "optimization.budget.seed",
                    "output",
                    "persistence",
                ],
                "accepted_but_not_wired": [
                    "output",
                    "persistence",
                ],
            },
            "runtime_rules": {
                "execution_status": "wired",
                "failure_mode": "returns validation_error/execution_error when base reference or search space cannot be resolved",
                "target_spec_type": target_spec_type,
                "base_reference_rules": "provide optimization.base_run_id or optimization.base_spec",
                "search_space_rules": "optimization.search_space must be a non-empty object",
                "budget_rules": "optimization.budget.max_trials must be >= 1",
            },
        }

    if normalized != "dca":
        raise ApiValidationException(
            single_validation_error("spec_type", "unsupported_spec_type", "Unsupported spec_type for capabilities")
        )

    return {
        "spec_type": "dca",
        "catalog_version": CANONICAL_CAPABILITIES_CATALOG_VERSION,
        "fields": {
                "supported": [
                    "catalog_version",
                    "request_id",
                    "data.symbol",
                    "data.currency",
                    "data.timeframe",
                "data.start_date",
                "data.end_date",
                "data.dataset_path",
                "data.path",
                "data.mysql",
                "universe",
                "strategy.type",
                "strategy.params.asset_class",
                "strategy.params.grid",
                "strategy.params.execution_mode",
                "strategy.params.drawdown_reference",
                "strategy.params.tp_sl",
                "filters.filters",
                "filters.rules",
                "filters.rules_config",
                "performance.initial_capital",
                "performance.capital_per_unit",
                "performance.max_capital_per_trade",
                "performance.risk_pct",
                "performance.risk_free_rate_pct",
                "performance.stress_tests",
                "performance.stress_tests.enabled",
                "performance.stress_tests.source",
                "performance.stress_tests.method",
                "performance.stress_tests.n_sims",
                "performance.stress_tests.seed",
                "performance.stress_tests.block_size",
                "performance.stress_tests.overlapping",
                "performance.stress_tests.time_distribution",
                "performance.stress_tests.param_drift",
                "performance.stress_tests.sizing",
                "performance.stress_tests.output",
                "performance.stress_tests.scenarios",
                "performance.stress_tests.multi_asset",
                "performance.stress_tests.aggregation",
                "performance.stress_tests.weights",
                "performance.stress_tests.timestamp_alignment",
                ],
            "accepted_but_not_wired": [
                "output",
                "persistence",
            ],
        },
        "presets": {
            "supported": {
                "strategy.grid": ["grid_balanced"],
                "strategy.params.tp_sl": ["tp_X_sl_Y", "tp_sl.trailing(percent)"],
            },
            "not_supported": {
                "strategy.grid": ["grid_conservative", "grid_aggressive"],
            },
        },
        "filters": {
            "supported_ids": sorted(list_filter_types()),
            "rules_modes": ["hard", "soft"],
            "rules_weights": {"min": 0.0},
        },
        "runtime_rules": {
            "multi_symbol": {
                "execution_scope": "per_symbol",
                "aggregation": "result.counts keyed by symbol; payload trades aggregated across symbols",
                "filter_application": "filters and filter_rules are evaluated independently per symbol on each symbol OHLC",
                "missing_data_behavior": "run fails fast if any universe symbol cannot load OHLC",
            }
        },
        "legacy_dca": {
            "entrypoint": "qe strategy backtest --spec <strategy_spec.json>",
            "fields": {
                "supported_in_legacy_runner": [
                    "strategy.strategy_id",
                    "strategy.type",
                    "strategy.params.asset_class",
                    "strategy.params.execution_mode",
                    "strategy.params.drawdown_reference",
                    "strategy.params.grid",
                    "strategy.params.require_crossing",
                    "strategy.params.tp_sl",
                    "strategy.params.log_drawdown_summary",
                    "data.source",
                    "data.path",
                    "data.start",
                    "data.end",
                    "data.timeframe",
                    "data.dataset_path",
                    "data.mysql",
                    "data.delta_base",
                    "data.delta_prefix",
                    "data.delta_exchange",
                    "data.delta_market_type",
                    "data.delta_quotes",
                    "data.delta_min_coverage",
                    "universe",
                    "filters",
                    "filter_rules",
                    "filter_rules_config",
                    "screening",
                    "optimization.screening",
                    "optimization.cache_features",
                    "performance.initial_capital",
                    "performance.capital_per_unit",
                    "performance.stress_tests",
                    "output.path",
                    "output.format",
                ],
                "supported": [
                    "strategy.strategy_id",
                    "strategy.type",
                    "strategy.params.asset_class",
                    "strategy.params.execution_mode",
                    "strategy.params.drawdown_reference",
                    "strategy.params.grid",
                    "strategy.params.require_crossing",
                    "strategy.params.tp_sl",
                    "strategy.params.log_drawdown_summary",
                    "data.source",
                    "data.path",
                    "data.start",
                    "data.end",
                    "data.timeframe",
                    "data.dataset_path",
                    "data.mysql",
                    "data.delta_base",
                    "data.delta_prefix",
                    "data.delta_exchange",
                    "data.delta_market_type",
                    "data.delta_quotes",
                    "data.delta_min_coverage",
                    "universe",
                    "filters",
                    "filter_rules",
                    "filter_rules_config",
                    "screening",
                    "optimization.screening",
                    "optimization.cache_features",
                    "performance.initial_capital",
                    "performance.capital_per_unit",
                    "performance.stress_tests",
                    "output.path",
                    "output.format",
                ],
                "canonical_passthrough_supported": [
                    "strategy.type",
                    "strategy.params.asset_class",
                    "strategy.params.grid",
                    "strategy.params.execution_mode",
                    "strategy.params.drawdown_reference",
                    "strategy.params.tp_sl",
                    "filters",
                    "performance.initial_capital",
                ],
                "not_in_canonical": [
                    "strategy.strategy_id",
                    "data.source",
                    "data.start",
                    "data.end",
                    "universe",
                    "filter_rules",
                    "filter_rules_config",
                    "screening",
                    "optimization.screening",
                    "optimization.cache_features",
                    "performance.capital_per_unit",
                    "output.path",
                    "output.format",
                ],
            },
            "notes": [
                "Legacy DCA supports richer strategy specs than canonical /runs.",
                "Fields listed under legacy support are not automatically accepted as-is by canonical /runs.",
                "Use canonical_passthrough_supported to know what can be forwarded without additional Python canonical wiring.",
                "Use canonical /runs fields for production launcher payloads; use legacy section to plan incremental parity.",
            ],
        },
        "resolution": {
            "dca_symbol_source_priority": ["universe", "data.symbol"],
        },
        "deprecations": {
            "data.symbol": {
                "status": "deprecated",
                "recommended_replacement": "universe[]",
                "warning_event": "canonical_dca_deprecation_warning",
                "target_version": "2026-06",
            }
        },
    }


# ---------------------------------------------------------------------------
# Statistics endpoints (synchronous MVP)


def stats_run(spec: schemas.StatsSpec) -> schemas.StatusResponse:
    """Execute a statistics run synchronously and store the result."""

    job_id, _ = _run_job(JOB_TYPE_STATS, payload=spec.model_dump(mode="json"))
    return schemas.StatusResponse(status="completed", id=job_id)


def stats_run_async(spec: schemas.StatsSpec) -> schemas.StatusResponse:
    """Queue a statistics run to be processed by an async worker."""

    job_id = _enqueue_job(JOB_TYPE_STATS, payload=spec.model_dump(mode="json"))
    return schemas.StatusResponse(status="pending", id=job_id)


def _stats_payload_from_df(df: Any) -> Dict[str, Any]:
    try:
        import pandas as pd  # type: ignore
    except Exception:  # pragma: no cover - pandas is an install dependency
        payload_df = df
    else:
        payload_df = df.where(pd.notna(df), None)
    return {
        "columns": list(payload_df.columns),
        "rows": payload_df.to_dict(orient="records"),
    }


def stats_result() -> schemas.ResultResponse:
    """Return the last statistics result if available."""

    job = _latest_job(JOB_TYPE_STATS)
    if not job or not job.get("result"):
        return schemas.ResultResponse(result=None)
    return schemas.ResultResponse(result=job.get("result"))


def stats_condition_types() -> List[str]:
    """Return the list of supported condition factory names."""

    return stats_conditions.list_condition_types()


def filters_list() -> List[str]:
    """Return the list of supported filter identifiers."""

    return list_filter_types()


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

    _, payload = _run_job(JOB_TYPE_LEVELS_BUILD, payload=spec.model_dump(mode="json"))
    return payload


def levels_fill(spec: LevelsBuildSpec) -> Dict[str, Any]:
    """Refresh fills for GAP and FVG levels."""

    _, payload = _run_job(JOB_TYPE_LEVELS_FILL, payload=spec.model_dump(mode="json"))
    return payload


def levels_build_async(spec: LevelsBuildSpec) -> schemas.StatusResponse:
    """Queue a levels build request for async processing."""

    job_id = _enqueue_job(JOB_TYPE_LEVELS_BUILD, payload=spec.model_dump(mode="json"))
    return schemas.StatusResponse(status="pending", id=job_id)


def levels_fill_async(spec: LevelsBuildSpec) -> schemas.StatusResponse:
    """Queue a levels fill request for async processing."""

    job_id = _enqueue_job(JOB_TYPE_LEVELS_FILL, payload=spec.model_dump(mode="json"))
    return schemas.StatusResponse(status="pending", id=job_id)


def levels_list(
    symbol: str,
    level_type: str | None = None,
    start: str | None = None,
    end: str | None = None,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    """Return persisted levels filtered by the provided criteria."""

    return _fetch_levels(symbol=symbol, level_type=level_type, start=start, end=end, limit=limit)


def levels_search(
    symbol: str,
    level_types: Sequence[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    """Return levels filtered by type list using the repository helper."""

    engine = _resolve_levels_engine()
    df = repo_select_levels(
        engine,
        LEVELS_TABLE,
        symbol=symbol,
        level_types=list(level_types or []),
        active_only=False,
        start=start,
        end=end,
        limit=limit,
    )
    if df.empty:
        return []
    rows = df.sort_values("anchor_ts", ascending=False).to_dict(orient="records")
    return [_serialise_row(row) for row in rows]


def levels_active(
    symbol: str,
    level_types: Optional[List[str]] | None = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    """Return currently active levels filtered by type."""

    engine = _resolve_levels_engine()
    df = repo_select_levels(
        engine,
        LEVELS_TABLE,
        symbol=symbol,
        level_types=level_types or [],
        active_only=True,
        start=start,
        end=end,
        limit=limit,
    )
    if df.empty:
        return []
    rows = df.to_dict(orient="records")
    return [_serialise_row(row) for row in rows]


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

    _, payload = _run_job(JOB_TYPE_SEASONALITY_RUN, payload=spec.model_dump(mode="json"))
    return schemas.ResultResponse(result=payload)


def seasonality_optimize(spec: schemas.SeasonalitySpec) -> schemas.ResultResponse:
    """Launch the seasonality optimisation loop and return its outcome."""

    _, payload = _run_job(JOB_TYPE_SEASONALITY_OPTIMIZE, payload=spec.model_dump(mode="json"))
    return schemas.ResultResponse(result=payload)


def seasonality_run_async(spec: schemas.SeasonalitySpec) -> schemas.StatusResponse:
    """Queue a seasonality run for async processing."""

    job_id = _enqueue_job(JOB_TYPE_SEASONALITY_RUN, payload=spec.model_dump(mode="json"))
    return schemas.StatusResponse(status="pending", id=job_id)


def seasonality_optimize_async(spec: schemas.SeasonalitySpec) -> schemas.StatusResponse:
    """Queue a seasonality optimisation for async processing."""

    job_id = _enqueue_job(JOB_TYPE_SEASONALITY_OPTIMIZE, payload=spec.model_dump(mode="json"))
    return schemas.StatusResponse(status="pending", id=job_id)


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


@fastapi_app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.monotonic()
    correlation_id = _get_correlation_id(request)
    response: Response
    try:
        response = await call_next(request)
    except Exception as exc:  # pragma: no cover - defensive
        duration_ms = (time.monotonic() - start) * 1000.0
        METRICS.record(
            endpoint=request.url.path,
            method=request.method,
            status_code=500,
            duration_ms=duration_ms,
        )
        event_payload = {
            "event": "http_request",
            "path": request.url.path,
            "method": request.method,
            "status": 500,
            "duration_ms": round(duration_ms, 2),
            "correlation_id": correlation_id,
            "error": str(exc),
        }
        threading.Thread(target=_log_event, args=(event_payload,), daemon=True).start()
        raise
    duration_ms = (time.monotonic() - start) * 1000.0
    METRICS.record(
        endpoint=request.url.path,
        method=request.method,
        status_code=response.status_code,
        duration_ms=duration_ms,
    )
    request_id = getattr(request.state, "request_id", None)
    if correlation_id:
        response.headers["X-Correlation-Id"] = correlation_id
    if request_id:
        response.headers["X-Request-Id"] = request_id
    event_payload = {
        "event": "http_request",
        "path": request.url.path,
        "method": request.method,
        "status": response.status_code,
        "duration_ms": round(duration_ms, 2),
        "correlation_id": correlation_id,
        "request_id": request_id,
    }
    threading.Thread(target=_log_event, args=(event_payload,), daemon=True).start()
    return response


@fastapi_app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    errors = normalize_fastapi_errors(exc.errors())
    try:
        raw_body = await request.body()
        body_text = raw_body.decode("utf-8", errors="replace")
        if len(body_text) > 4000:
            body_text = body_text[:4000] + "...(truncated)"
    except Exception:
        body_text = None
    event_payload = {
        "event": "validation_error",
        "path": str(request.url.path),
        "method": request.method,
        "status": 422,
        "correlation_id": _get_correlation_id(request),
        "errors": errors,
        "body": body_text,
    }
    threading.Thread(target=_log_event, args=(event_payload,), daemon=True).start()
    return JSONResponse(status_code=422, content={"errors": errors})


@fastapi_app.exception_handler(ApiValidationException)
async def api_validation_exception_handler(
    request: Request, exc: ApiValidationException
) -> JSONResponse:
    try:
        raw_body = await request.body()
        body_text = raw_body.decode("utf-8", errors="replace")
        if len(body_text) > 4000:
            body_text = body_text[:4000] + "...(truncated)"
    except Exception:
        body_text = None
    event_payload = {
        "event": "validation_error",
        "path": str(request.url.path),
        "method": request.method,
        "status": 422,
        "correlation_id": _get_correlation_id(request),
        "errors": exc.errors,
        "body": body_text,
    }
    threading.Thread(target=_log_event, args=(event_payload,), daemon=True).start()
    return JSONResponse(status_code=422, content={"errors": exc.errors})


@fastapi_app.get('/healthz', response_model=Dict[str, Any])
def healthz_endpoint() -> Dict[str, Any]:
    """Liveness probe."""

    return {"status": "ok", "ts": _utc_now()}


@fastapi_app.get('/readyz', response_model=Dict[str, Any])
def readyz_endpoint() -> Dict[str, Any]:
    """Readiness probe (checks DB connectivity)."""

    ok, error = _db_ready()
    if not ok:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "code": "db_unavailable", "message": error},
        )
    return {"status": "ready", "ts": _utc_now()}


@fastapi_app.get('/metrics', response_model=Dict[str, Any])
def metrics_endpoint() -> Dict[str, Any]:
    """Return in-memory metrics snapshot."""

    return METRICS.snapshot()


@fastapi_app.post('/submit', response_model=schemas.SubmitResponse)
def submit_endpoint(payload: Dict[str, Any]) -> schemas.SubmitResponse:
    """HTTP endpoint wrapping :func:`submit`."""

    try:
        spec_obj = spec_module.spec_from_dict(payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise ApiValidationException(
            single_validation_error("spec", "invalid_spec", f"Invalid spec: {exc}")
        ) from exc
    except ValidationError as exc:
        raise ApiValidationException(normalize_pydantic_errors(exc)) from exc
    return submit(spec_obj)


@fastapi_app.post('/submit/async', response_model=schemas.StatusResponse)
def submit_async_endpoint(payload: Dict[str, Any]) -> schemas.StatusResponse:
    """Queue an optimisation run and return a pending job id."""

    try:
        spec_obj = spec_module.spec_from_dict(payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise ApiValidationException(
            single_validation_error("spec", "invalid_spec", f"Invalid spec: {exc}")
        ) from exc
    except ValidationError as exc:
        raise ApiValidationException(normalize_pydantic_errors(exc)) from exc
    return submit_async(spec_obj)


@fastapi_app.post('/runs', response_model=schemas.RunEnqueueResponse)
def runs_submit_endpoint(payload: Dict[str, Any], request: Request) -> schemas.RunEnqueueResponse:
    """Validate and enqueue a canonical run request."""

    _log_event(
        {
            "event": "runs_submit_payload",
            "path": "/runs",
            "method": "POST",
            "payload": _truncate_json_payload(payload),
            "payload_keys": sorted(list(payload.keys())),
        }
    )
    response = enqueue_run_request(payload)
    request.state.request_id = response.run_id
    return response


@fastapi_app.post('/runs/{run_id}/cancel', response_model=schemas.StatusResponse)
def runs_cancel_endpoint(run_id: str, request: Request) -> schemas.StatusResponse:
    """Request cancellation for a canonical run."""

    client_host = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")
    _log_event(
        {
            "event": "runs_cancel_requested",
            "path": "/runs/{run_id}/cancel",
            "method": "POST",
            "run_id": run_id,
            "client_host": client_host,
            "user_agent": user_agent,
        }
    )

    job = _get_job(run_id)
    if not job:
        return _json_error(404, "not_found", "Run not found")
    status = _external_job_status(job.get("status", ""))
    if status in {JOB_STATUS_SUCCEEDED, JOB_STATUS_FAILED_CANONICAL}:
        return _json_error(409, "already_finished", "Run already finished")
    request_job_cancel(run_id)
    job = _get_job(run_id) or job
    status = _external_job_status(job.get("status", ""))
    request.state.request_id = run_id
    _log_event(
        {
            "event": "runs_cancel_applied",
            "path": "/runs/{run_id}/cancel",
            "method": "POST",
            "run_id": run_id,
            "status": status,
            "client_host": client_host,
        }
    )
    return schemas.StatusResponse(status=status, id=run_id)


@fastapi_app.get('/status/{job_id}', response_model=schemas.StatusResponse)
def status_endpoint(job_id: str) -> schemas.StatusResponse:
    """Return the status for a submitted job."""

    return status(job_id)


@fastapi_app.get('/result/{job_id}', response_model=schemas.ResultResponse)
def result_endpoint(job_id: str) -> schemas.ResultResponse:
    """Return the optimisation result for a job if available."""

    return result(job_id)


@fastapi_app.get('/stats/conditions', response_model=List[str])
def stats_conditions_endpoint() -> List[str]:
    """Return the list of supported condition factories."""

    return stats_condition_types()


@fastapi_app.get('/filters/list', response_model=List[str])
def filters_list_endpoint() -> List[str]:
    """Return the list of available filters."""

    return filters_list()


@fastapi_app.post('/stats/run', response_model=schemas.StatusResponse)
def stats_run_endpoint(spec: schemas.StatsSpec) -> schemas.StatusResponse:
    """Kick off a statistics computation synchronously."""

    return stats_run(spec)


@fastapi_app.post('/stats/run/async', response_model=schemas.StatusResponse)
def stats_run_async_endpoint(spec: schemas.StatsSpec) -> schemas.StatusResponse:
    """Queue a statistics computation for async processing."""

    return stats_run_async(spec)


@fastapi_app.get('/stats/result', response_model=schemas.ResultResponse)
def stats_result_endpoint() -> schemas.ResultResponse:
    """Return the result of the last statistics computation."""

    return stats_result()


@fastapi_app.post('/levels/build', response_model=Dict[str, Any])
def levels_build_endpoint(spec: LevelsBuildSpec) -> Dict[str, Any]:
    """Trigger a synchronous levels build run."""

    return levels_build(spec)


@fastapi_app.post('/levels/build/async', response_model=schemas.StatusResponse)
def levels_build_async_endpoint(spec: LevelsBuildSpec) -> schemas.StatusResponse:
    """Queue a levels build run for async processing."""

    return levels_build_async(spec)


@fastapi_app.post('/levels/fill', response_model=Dict[str, Any])
def levels_fill_endpoint(spec: LevelsBuildSpec) -> Dict[str, Any]:
    """Refresh fills for active FVG/GAP levels."""

    return levels_fill(spec)


@fastapi_app.post('/levels/fill/async', response_model=schemas.StatusResponse)
def levels_fill_async_endpoint(spec: LevelsBuildSpec) -> schemas.StatusResponse:
    """Queue a levels fill run for async processing."""

    return levels_fill_async(spec)


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


@fastapi_app.get('/levels/active', response_model=List[Dict[str, Any]])
def levels_active_endpoint(
    symbol: str,
    level_type: Optional[List[str]] = Query(None),
    from_: Optional[str] = Query(None, alias="from"),
    to: Optional[str] = Query(None, alias="to"),
    limit: int = Query(200, ge=1, le=10000),
) -> List[Dict[str, Any]]:
    """Return currently active levels."""

    return levels_active(symbol=symbol, level_types=level_type, start=from_, end=to, limit=limit)


@fastapi_app.get('/levels/search', response_model=List[Dict[str, Any]])
def levels_search_endpoint(
    symbol: str,
    type_: Optional[str] = Query(None, alias="type"),
    from_: Optional[str] = Query(None, alias="from"),
    to: Optional[str] = Query(None, alias="to"),
    limit: int = Query(200, ge=1, le=10000),
) -> List[Dict[str, Any]]:
    """Search persisted levels by a comma separated type filter."""

    level_types: List[str] | None = None
    if type_:
        level_types = [t.strip().upper() for t in type_.split(",") if t.strip()]
    return levels_search(symbol=symbol, level_types=level_types, start=from_, end=to, limit=limit)


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


@fastapi_app.post('/seasonality/run/async', response_model=schemas.StatusResponse)
def seasonality_run_async_endpoint(spec: schemas.SeasonalitySpec) -> schemas.StatusResponse:
    """Queue a seasonality run for async processing."""

    return seasonality_run_async(spec)


@fastapi_app.post('/seasonality/optimize', response_model=schemas.ResultResponse)
def seasonality_optimize_endpoint(spec: schemas.SeasonalitySpec) -> schemas.ResultResponse:
    """Execute the seasonality optimisation loop."""

    return seasonality_optimize(spec)


@fastapi_app.post('/seasonality/optimize/async', response_model=schemas.StatusResponse)
def seasonality_optimize_async_endpoint(spec: schemas.SeasonalitySpec) -> schemas.StatusResponse:
    """Queue a seasonality optimisation for async processing."""

    return seasonality_optimize_async(spec)


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


@fastapi_app.get('/runs/capabilities', response_model=Dict[str, Any])
def runs_capabilities_endpoint(spec_type: str = Query(..., description="Canonical spec type")) -> Dict[str, Any]:
    """Return canonical runtime capabilities for a given spec_type."""

    return _canonical_runs_capabilities(spec_type)


@fastapi_app.get('/runs/{run_id}', response_model=Dict[str, Any])
def run_detail_endpoint(run_id: str) -> Dict[str, Any]:
    """Return a single run and aggregated metrics."""

    job = _get_job(run_id)
    if job and job.get("job_type") == JOB_TYPE_CANONICAL_RUN:
        return _canonical_run_payload(job)
    payload = get_run(run_id)
    if payload is None:
        raise HTTPException(status_code=404, detail='Run not found')
    return payload




def get_run_artifacts(run_id: str) -> Dict[str, Any]:
    """Return artifact listing for a canonical run output directory."""

    job = _get_job(run_id)
    if not job or job.get("job_type") != JOB_TYPE_CANONICAL_RUN:
        raise HTTPException(status_code=404, detail='Run not found')

    payload = job.get("payload") if isinstance(job.get("payload"), dict) else {}
    request = payload.get("request") if isinstance(payload.get("request"), dict) else {}
    output = request.get("output") if isinstance(request.get("output"), dict) else {}
    out_dir_raw = output.get("out_dir")
    out_dir = Path(str(out_dir_raw)).expanduser() if isinstance(out_dir_raw, str) and out_dir_raw.strip() else None

    files: List[Dict[str, Any]] = []
    if out_dir and out_dir.exists() and out_dir.is_dir():
        for item in sorted(out_dir.iterdir()):
            if item.is_file():
                files.append({"name": item.name, "path": str(item), "size_bytes": item.stat().st_size})

    return {
        "run_id": run_id,
        "schema_version": SCHEMA_VERSION,
        "out_dir": str(out_dir) if out_dir is not None else None,
        "files": files,
    }

@fastapi_app.get('/runs/{run_id}/result', response_model=Dict[str, Any])
def run_result_endpoint(run_id: str) -> Dict[str, Any]:
    """Return the result for a canonical run."""

    job = _get_job(run_id)
    if not job or job.get("job_type") != JOB_TYPE_CANONICAL_RUN:
        return _json_error(404, "not_found", "Run not found")
    status = _external_job_status(job.get("status", ""))
    payload: Dict[str, Any] = {
        "run_id": job.get("job_id"),
        "request_id": job.get("job_id"),
        "requestId": job.get("job_id"),
        "status": status,
    }
    if not _is_terminal_status(status):
        payload["message"] = "Result not available yet"
        return payload
    result = job.get("result")
    if result is not None:
        payload["result"] = result
    if job.get("error_message"):
        payload["error"] = {"code": "execution_error", "message": job.get("error_message")}
    if isinstance(result, dict) and "error" in result:
        payload["error"] = result.get("error")
    return payload


@fastapi_app.get('/runs/{run_id}/artifacts', response_model=Dict[str, Any])
def run_artifacts_endpoint(run_id: str) -> Dict[str, Any]:
    """Return canonical run artifact listing."""

    return get_run_artifacts(run_id)


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
