"""Optimization utilities for backtest and strategy specs."""
from __future__ import annotations

import json
import logging
import math
import random
import shutil
import os
import hashlib
import sys
from pathlib import Path
from copy import deepcopy
from itertools import product
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from ..io import artifacts

LOGGER = logging.getLogger(__name__)
from ..backtest import runner as backtest_runner
from ..strategies import runner as strategy_runner


def _iter_values(config: Any) -> List[Any]:
    if isinstance(config, list):
        return list(config)
    if isinstance(config, Mapping):
        if "values" in config:
            return list(config["values"])
        if all(k in config for k in ("min", "max", "step")):
            vmin = float(config["min"])
            vmax = float(config["max"])
            step = float(config["step"])
            if step <= 0:
                raise ValueError("step must be > 0")
            decimals = max(0, -int(math.floor(math.log10(step))) if step < 1 else 0)
            values: List[float] = []
            current = vmin
            while current <= vmax + 1e-9:
                values.append(round(current, decimals))
                current += step
            if all(float(v).is_integer() for v in values):
                return [int(v) for v in values]
            return values
    raise ValueError("search_space entries must be list or {values|min/max/step}")


def _parse_path(path: str) -> List[Any]:
    tokens: List[Any] = []
    buff = ""
    i = 0
    while i < len(path):
        ch = path[i]
        if ch == ".":
            if buff:
                tokens.append(buff)
                buff = ""
            i += 1
            continue
        if ch == "[":
            if buff:
                tokens.append(buff)
                buff = ""
            end = path.index("]", i)
            idx = path[i + 1 : end]
            tokens.append(int(idx))
            i = end + 1
            continue
        buff += ch
        i += 1
    if buff:
        tokens.append(buff)
    return tokens


def _set_path(obj: Any, path: str, value: Any) -> None:
    tokens = _parse_path(path)
    ref = obj
    for token in tokens[:-1]:
        if isinstance(token, int):
            if not isinstance(ref, list) or token >= len(ref):
                raise ValueError(f"Invalid path '{path}'")
            ref = ref[token]
        else:
            if token not in ref:
                ref[token] = {}
            ref = ref[token]
    last = tokens[-1]
    if isinstance(last, int):
        if not isinstance(ref, list) or last >= len(ref):
            raise ValueError(f"Invalid path '{path}'")
        ref[last] = value
    else:
        ref[last] = value


def _expand_search_space(space: Mapping[str, Any]) -> Tuple[List[str], List[List[Any]]]:
    keys: List[str] = []
    values: List[List[Any]] = []
    for key, cfg in space.items():
        keys.append(key)
        values.append(_iter_values(cfg))
    return keys, values


def _refine_search_space(
    space: Mapping[str, Any],
    top_trials: List[Mapping[str, Any]],
    *,
    shrink_pct: float,
    top_k: int,
    freeze_keys: Optional[List[str]] = None,
    freeze_prefixes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    refined: Dict[str, Any] = {}
    if top_k <= 0 or not top_trials:
        return dict(space)
    shrink_pct = max(0.0, min(float(shrink_pct), 1.0))
    freeze_keys = freeze_keys or []
    freeze_prefixes = freeze_prefixes or []
    for key, cfg in space.items():
        if key in freeze_keys or any(str(key).startswith(prefix) for prefix in freeze_prefixes):
            refined[key] = cfg
            continue
        values = [t.get("params", {}).get(key) for t in top_trials]
        values = [v for v in values if isinstance(v, (int, float))]
        if isinstance(cfg, Mapping) and all(k in cfg for k in ("min", "max", "step")) and values:
            vmin = float(cfg["min"])
            vmax = float(cfg["max"])
            span = vmax - vmin
            center = sorted(values)[len(values) // 2]
            half = span * shrink_pct / 2.0
            new_min = max(vmin, center - half)
            new_max = min(vmax, center + half)
            refined[key] = {"min": new_min, "max": new_max, "step": cfg["step"]}
            continue
        if isinstance(cfg, list):
            kept = [v for v in cfg if v in values]
            refined[key] = kept if kept else list(cfg)
            continue
        refined[key] = cfg
    return refined


def _numeric_bounds(values: List[Any]) -> Optional[Tuple[float, float]]:
    nums: List[float] = []
    for val in values:
        if isinstance(val, (int, float)):
            nums.append(float(val))
    if not nums:
        return None
    return min(nums), max(nums)


def _objective_value(run: Mapping[str, Any], objective: Any) -> float:
    mapping = {
        "sharpe": "sharpe",
        "sortino": "sortino",
        "return_pct": "returnPct",
        "max_drawdown_pct": "maxDrawdownPct",
        "total_return": "totalReturn",
        "winrate_pct": "winratePct",
    }
    if isinstance(objective, Mapping):
        weights = objective.get("weights") or {}
        if not isinstance(weights, Mapping):
            raise ValueError("objective.weights must be a mapping")
        penalties = objective.get("penalties") or {}
        if penalties and not isinstance(penalties, Mapping):
            raise ValueError("objective.penalties must be a mapping")
        total = 0.0
        used = 0
        for metric, weight in weights.items():
            key = mapping.get(str(metric), str(metric))
            val = run.get(key)
            try:
                val_f = float(val)
            except Exception:
                val_f = 0.0
            try:
                w_f = float(weight)
            except Exception:
                w_f = 0.0
            total += w_f * val_f
            used += 1
        for metric, cfg in penalties.items():
            if not isinstance(cfg, Mapping):
                continue
            key = mapping.get(str(metric), str(metric))
            val = run.get(key)
            try:
                val_f = float(val)
            except Exception:
                continue
            threshold = cfg.get("threshold")
            if threshold is None:
                continue
            try:
                thr_f = float(threshold)
            except Exception:
                continue
            direction = str(cfg.get("direction", "above")).lower()
            power = cfg.get("power", 1.0)
            weight = cfg.get("weight", -1.0)
            try:
                power_f = float(power)
            except Exception:
                power_f = 1.0
            try:
                weight_f = float(weight)
            except Exception:
                weight_f = -1.0
            if direction == "below":
                diff = max(0.0, thr_f - val_f)
            else:
                diff = max(0.0, val_f - thr_f)
            if diff > 0:
                total += weight_f * (diff ** power_f)
        if used == 0:
            return float("-inf")
        return total

    key = mapping.get(str(objective), str(objective))
    val = run.get(key)
    try:
        return float(val)
    except Exception:
        return float("-inf")


def _aggregate_scores(values: List[float], mode: str) -> float:
    if not values:
        return float("-inf")
    clean = [v for v in values if v is not None]
    if not clean:
        return float("-inf")
    mode = str(mode or "mean").lower()
    if mode == "min":
        return min(clean)
    if mode == "max":
        return max(clean)
    if mode == "median":
        sorted_vals = sorted(clean)
        mid = len(sorted_vals) // 2
        if len(sorted_vals) % 2 == 0:
            return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0
        return sorted_vals[mid]
    return sum(clean) / len(clean)


def _sensitivity_from_trials(
    trials: List[Mapping[str, Any]],
    *,
    top_k: int = 20,
) -> Dict[str, Any]:
    scored = [t for t in trials if isinstance(t.get("objective"), (int, float))]
    scored = sorted(scored, key=lambda t: t.get("objective", float("-inf")), reverse=True)
    top = scored[: max(1, top_k)]
    if not top:
        return {}
    params_list = [t.get("params") or {} for t in top]
    objectives = [float(t.get("objective", 0.0)) for t in top]
    mean_obj = sum(objectives) / len(objectives)
    var_obj = sum((o - mean_obj) ** 2 for o in objectives) / len(objectives)
    std_obj = math.sqrt(var_obj) if var_obj > 0 else 0.0
    param_stats: Dict[str, Dict[str, Any]] = {}
    for params in params_list:
        for key, val in params.items():
            stats = param_stats.setdefault(key, {"values": [], "unique": set()})
            stats["values"].append(val)
            stats["unique"].add(val)
    correlations: Dict[str, Optional[float]] = {}
    stable: List[str] = []
    for key, stats in param_stats.items():
        values = stats["values"]
        numeric_vals: List[float] = []
        numeric_obj: List[float] = []
        for v, obj in zip(values, objectives):
            if isinstance(v, (int, float)):
                numeric_vals.append(float(v))
                numeric_obj.append(float(obj))
        unique_count = len(stats["unique"])
        stable_ratio = unique_count / max(1, len(values))
        if stable_ratio <= 0.2:
            stable.append(key)
        if numeric_vals and std_obj > 0:
            mean_val = sum(numeric_vals) / len(numeric_vals)
            var_val = sum((v - mean_val) ** 2 for v in numeric_vals) / len(numeric_vals)
            std_val = math.sqrt(var_val) if var_val > 0 else 0.0
            if std_val > 0:
                cov = sum((v - mean_val) * (o - mean_obj) for v, o in zip(numeric_vals, numeric_obj)) / len(numeric_vals)
                correlations[key] = cov / (std_val * std_obj)
            else:
                correlations[key] = None
        else:
            correlations[key] = None
    sorted_corr = sorted(
        ((k, v) for k, v in correlations.items() if isinstance(v, (int, float))),
        key=lambda item: abs(item[1]),
        reverse=True,
    )
    return {
        "top_k": top_k,
        "objective_mean": mean_obj,
        "objective_std": std_obj,
        "correlations": {k: v for k, v in sorted_corr},
        "stable_params": stable,
    }


def _screening_pass(window_scores: List[float], cfg: Mapping[str, Any]) -> bool:
    if not window_scores:
        return True
    min_objective = cfg.get("min_window_objective")
    min_windows_passed = cfg.get("min_windows_passed")
    max_windows_failed = cfg.get("max_windows_failed")
    if min_objective is None and min_windows_passed is None and max_windows_failed is None:
        return True
    try:
        threshold = float(min_objective) if min_objective is not None else None
    except Exception:
        threshold = None
    passed = 0
    failed = 0
    for score in window_scores:
        if threshold is None:
            passed += 1
            continue
        if score >= threshold:
            passed += 1
        else:
            failed += 1
    if min_windows_passed is not None:
        try:
            if passed < int(min_windows_passed):
                return False
        except Exception:
            pass
    if max_windows_failed is not None:
        try:
            if failed > int(max_windows_failed):
                return False
        except Exception:
            pass
    if threshold is not None and passed == 0:
        return False
    return True


def _count_trials(values: List[List[Any]], method: str, max_trials: Optional[int]) -> int:
    if method == "random":
        if max_trials is None:
            raise ValueError("max_trials must be provided for random search")
        return int(max_trials)
    total = 1
    for vals in values:
        total *= max(len(vals), 1)
    return total


def _trial_specs(base_spec: Mapping[str, Any], keys: List[str], values: List[List[Any]], method: str, max_trials: Optional[int], seed: Optional[int]) -> Iterable[Dict[str, Any]]:
    if method == "random":
        rng = random.Random(seed)
        num = max_trials or 0
        if num <= 0:
            raise ValueError("max_trials must be provided for random search")
        for _ in range(num):
            spec = deepcopy(base_spec)
            for key, vals in zip(keys, values):
                _set_path(spec, key, rng.choice(vals))
            yield spec
        return
    for combo in product(*values):
        spec = deepcopy(base_spec)
        for key, val in zip(keys, combo):
            _set_path(spec, key, val)
        yield spec


def _optimization_config(spec: Mapping[str, Any]) -> Dict[str, Any]:
    cfg = spec.get("optimization") or {}
    if not isinstance(cfg, Mapping):
        raise ValueError("optimization must be a mapping")
    return dict(cfg)


def _git_head_sha(root: Path) -> Optional[str]:
    head = root / ".git" / "HEAD"
    if not head.exists():
        return None
    content = head.read_text().strip()
    if content.startswith("ref:"):
        ref = content.split(":", 1)[1].strip()
        ref_path = root / ".git" / ref
        if ref_path.exists():
            return ref_path.read_text().strip()
        return None
    return content or None


def _dataset_id(spec: Mapping[str, Any]) -> str:
    data = spec.get("data") or {}
    if not isinstance(data, Mapping):
        return "unknown"
    for key in ("dataset_path", "path"):
        if key in data:
            return f"file:{data.get(key)}"
    if data.get("delta_base") or data.get("delta_prefix") or data.get("delta_exchange"):
        parts = [
            "delta",
            str(data.get("delta_base") or ""),
            str(data.get("delta_prefix") or ""),
            str(data.get("delta_exchange") or ""),
        ]
        return "delta:" + "|".join(p for p in parts if p)
    if data.get("mysql_env") or data.get("table") or data.get("schema"):
        return "mysql:" + str(data.get("schema") or "") + "." + str(data.get("table") or "")
    return "unknown"


def _json_hash(payload: Mapping[str, Any]) -> str:
    try:
        dumped = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    except Exception:
        dumped = str(payload).encode("utf-8")
    return hashlib.sha256(dumped).hexdigest()


def _data_hash(spec: Mapping[str, Any]) -> str:
    data = spec.get("data") or {}
    if not isinstance(data, Mapping):
        return _json_hash({"data": "unknown"})
    return _json_hash(data)


def _collect_lib_versions() -> Dict[str, Any]:
    versions: Dict[str, Any] = {"python": sys.version.split()[0]}
    for lib in ("pandas", "numpy", "requests", "sqlalchemy", "deltalake"):
        try:
            module = __import__(lib)
            versions[lib] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[lib] = None
    return versions


def _trial_metadata(spec: Mapping[str, Any], cfg: Mapping[str, Any]) -> Dict[str, Any]:
    data = spec.get("data") or {}
    strategy = spec.get("strategy") or {}
    data_hash = _data_hash(spec)
    config_hash = _json_hash(spec)
    meta = {
        "seed": cfg.get("seed"),
        "dataset_id": _dataset_id(spec),
        "timeframe": data.get("timeframe"),
        "start": data.get("start"),
        "end": data.get("end"),
        "strategy_id": strategy.get("strategy_id"),
        "data_hash": data_hash,
        "config_hash": config_hash,
        "lib_versions": _collect_lib_versions(),
    }
    root = Path.cwd()
    code_version = _git_head_sha(root)
    if code_version:
        meta["code_version"] = code_version
    return meta


def _compact_payload(
    payload: Mapping[str, Any],
    *,
    equity_limit: Optional[int] = None,
    trade_limit: Optional[int] = None,
    mode: str = "full",
) -> Dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    compact: Dict[str, Any] = dict(payload)
    trades = payload.get("trades")
    trade_stats = _trade_stats(trades) if isinstance(trades, list) else None
    if isinstance(trades, list) and trade_limit:
        try:
            limit = int(trade_limit)
        except Exception:
            limit = 0
        if limit > 0 and len(trades) > limit:
            compact["trades_sample"] = trades[:limit]
            compact["trades"] = {"count": len(trades), "sampled": limit}
    if trade_stats:
        compact["trades_stats"] = trade_stats
    equity = payload.get("equity")
    if isinstance(equity, list) and equity_limit:
        try:
            limit = int(equity_limit)
        except Exception:
            limit = 0
        if limit > 0 and len(equity) > limit:
            step = max(1, len(equity) // limit)
            compact["equity_sample"] = equity[::step]
            compact["equity"] = {"count": len(equity), "sampled": len(compact["equity_sample"]), "step": step}
    if mode in {"stats", "summary"}:
        compact.pop("trades", None)
        compact.pop("trades_sample", None)
        compact.pop("equity", None)
        compact.pop("equity_sample", None)
    return compact


def _trade_stats(trades: List[Mapping[str, Any]]) -> Dict[str, Any]:
    values: List[float] = []
    durations: List[float] = []
    pnl_by_day: Dict[str, float] = {}
    mae_values: List[float] = []
    mfe_values: List[float] = []
    for trade in trades:
        for key in ("pnl_pct", "grossPnlPct", "gross_pnl_pct", "r_multiple"):
            if key in trade:
                try:
                    values.append(float(trade[key]))
                    break
                except Exception:
                    continue
        entry = trade.get("ts_entry") or trade.get("entryTimeUtc")
        exit_ts = trade.get("ts_exit") or trade.get("exitTimeUtc")
        if entry and exit_ts:
            try:
                start = pd.to_datetime(entry, utc=True)
                end = pd.to_datetime(exit_ts, utc=True)
                durations.append(float((end - start).total_seconds()))
            except Exception:
                pass
        for key in ("mae", "mae_pct", "max_adverse_excursion"):
            if key in trade:
                try:
                    mae_values.append(float(trade[key]))
                    break
                except Exception:
                    continue
        for key in ("mfe", "mfe_pct", "max_favorable_excursion"):
            if key in trade:
                try:
                    mfe_values.append(float(trade[key]))
                    break
                except Exception:
                    continue
        if exit_ts:
            try:
                exit_day = pd.to_datetime(exit_ts, utc=True).date().isoformat()
                pnl_key = None
                for key in ("pnl_pct", "grossPnlPct", "gross_pnl_pct"):
                    if key in trade:
                        pnl_key = key
                        break
                if pnl_key:
                    pnl_by_day[exit_day] = pnl_by_day.get(exit_day, 0.0) + float(trade[pnl_key])
            except Exception:
                pass
    stats: Dict[str, Any] = {"count": len(trades)}
    if values:
        values_sorted = sorted(values)
        n = len(values_sorted)
        wins = sum(1 for v in values if v > 0)
        losses = sum(1 for v in values if v <= 0)
        win_vals = [v for v in values if v > 0]
        loss_vals = [abs(v) for v in values if v < 0]
        rr = None
        if win_vals and loss_vals:
            rr = (sum(win_vals) / len(win_vals)) / (sum(loss_vals) / len(loss_vals))
        stats.update(
            {
                "min": values_sorted[0],
                "max": values_sorted[-1],
                "mean": sum(values_sorted) / n,
                "p10": values_sorted[int(0.10 * (n - 1))],
                "p25": values_sorted[int(0.25 * (n - 1))],
                "p50": values_sorted[int(0.50 * (n - 1))],
                "p75": values_sorted[int(0.75 * (n - 1))],
                "p90": values_sorted[int(0.90 * (n - 1))],
                "wins": wins,
                "losses": losses,
                "winrate_pct": (wins / n * 100.0) if n else None,
                "avg_win": (sum(win_vals) / len(win_vals)) if win_vals else None,
                "avg_loss": -(sum(loss_vals) / len(loss_vals)) if loss_vals else None,
                "rr_mean": rr,
            }
        )
    if durations:
        durations_sorted = sorted(durations)
        n_d = len(durations_sorted)
        stats["duration_seconds"] = {
            "min": durations_sorted[0],
            "max": durations_sorted[-1],
            "mean": sum(durations_sorted) / n_d,
            "p25": durations_sorted[int(0.25 * (n_d - 1))],
            "p50": durations_sorted[int(0.50 * (n_d - 1))],
            "p75": durations_sorted[int(0.75 * (n_d - 1))],
        }
    if pnl_by_day:
        sorted_days = sorted(pnl_by_day.items())
        stats["pnl_by_day"] = [{"day": day, "pnl_pct": val} for day, val in sorted_days]
    if mae_values:
        mae_sorted = sorted(mae_values)
        n_m = len(mae_sorted)
        stats["mae"] = {
            "min": mae_sorted[0],
            "max": mae_sorted[-1],
            "mean": sum(mae_sorted) / n_m,
            "p25": mae_sorted[int(0.25 * (n_m - 1))],
            "p50": mae_sorted[int(0.50 * (n_m - 1))],
            "p75": mae_sorted[int(0.75 * (n_m - 1))],
        }
    if mfe_values:
        mfe_sorted = sorted(mfe_values)
        n_f = len(mfe_sorted)
        stats["mfe"] = {
            "min": mfe_sorted[0],
            "max": mfe_sorted[-1],
            "mean": sum(mfe_sorted) / n_f,
            "p25": mfe_sorted[int(0.25 * (n_f - 1))],
            "p50": mfe_sorted[int(0.50 * (n_f - 1))],
            "p75": mfe_sorted[int(0.75 * (n_f - 1))],
        }
    if values:
        buckets = [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3]
        hist = [0] * (len(buckets) + 1)
        for val in values:
            idx = 0
            while idx < len(buckets) and val > buckets[idx]:
                idx += 1
            hist[idx] += 1
        stats["r_histogram"] = {"buckets": buckets, "counts": hist}
    return stats


def _promotion_config(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    raw = cfg.get("promotion") or {}
    if not isinstance(raw, Mapping):
        raise ValueError("optimization.promotion must be a mapping")
    return dict(raw)


def _normalize_method(method: str) -> str:
    method = (method or "grid").lower()
    if method not in {"grid", "random"}:
        LOGGER.warning("Unknown optimization method '%s', falling back to grid", method)
        return "grid"
    return method


def _normalize_behavior_mode(mode: str) -> str:
    mode = (mode or "metrics").lower()
    if mode not in {"metrics", "trades_hist", "equity_signature"}:
        LOGGER.warning("Unknown behavior_mode '%s', falling back to metrics", mode)
        return "metrics"
    return mode


def _normalize_cluster_mode(mode: str) -> str:
    mode = (mode or "kmeans").lower()
    if mode not in {"kmeans", "dbscan"}:
        LOGGER.warning("Unknown behavior_cluster.mode '%s', falling back to kmeans", mode)
        return "kmeans"
    return mode


def _objective_summary(objective: Any) -> str:
    if isinstance(objective, Mapping):
        try:
            return json.dumps(objective, sort_keys=True, default=str)
        except Exception:
            return str(objective)
    return str(objective)


def _debug_on_fail_cfg(cfg: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    raw = cfg.get("debug_on_fail") or {}
    if not isinstance(raw, Mapping):
        return None
    if raw.get("enabled", False) is not True:
        return None
    return dict(raw)


def _write_debug_failure(
    out_dir: Path,
    trial_id: int,
    params: Mapping[str, Any],
    *,
    error: str,
    payload: Optional[Mapping[str, Any]] = None,
    context: Optional[Mapping[str, Any]] = None,
    debug_cfg: Optional[Mapping[str, Any]] = None,
) -> None:
    debug_cfg = debug_cfg or {}
    debug_dir = out_dir / "debug_failures"
    debug_dir.mkdir(parents=True, exist_ok=True)
    compact_mode = str(debug_cfg.get("mode", "stats")).lower()
    equity_limit = debug_cfg.get("equity_max_points")
    trade_limit = debug_cfg.get("trade_sample_size")
    debug_payload = payload or {}
    if payload and compact_mode in {"compact", "summary", "light", "stats"}:
        debug_payload = _compact_payload(
            payload, equity_limit=equity_limit, trade_limit=trade_limit, mode=compact_mode
        )
    body = {
        "trial_id": trial_id,
        "params": dict(params),
        "error": error,
        "context": dict(context or {}),
        "payload": debug_payload,
    }
    path = debug_dir / f"trial_{trial_id}.json"
    path.write_text(json.dumps(body, indent=2, default=str))


def _merge_soft_constraints(objective: Any, promotion_cfg: Mapping[str, Any]) -> Any:
    soft = promotion_cfg.get("soft_constraints")
    if not isinstance(soft, Mapping) or not soft:
        return objective
    if isinstance(objective, Mapping):
        merged = dict(objective)
        penalties = merged.get("penalties")
        if not isinstance(penalties, Mapping):
            penalties = {}
        penalties = dict(penalties)
        for key, val in soft.items():
            if key not in penalties and isinstance(val, Mapping):
                penalties[key] = dict(val)
        merged["penalties"] = penalties
        return merged
    return {"weights": {str(objective): 1.0}, "penalties": dict(soft)}


def _log_optimization_plan(
    kind: str,
    *,
    cfg: Mapping[str, Any],
    promotion_cfg: Mapping[str, Any],
    screening_cfg: Mapping[str, Any],
    refine_cfg: Mapping[str, Any],
    objective: Any,
    method: str,
    total_trials: int,
    keys: List[str],
    out_dir: Path,
    base_meta: Mapping[str, Any],
    behavior_mode: str,
    behavior_bins: List[Any],
    behavior_points: int,
    cluster_cfg: Mapping[str, Any],
    artifacts_cfg: Mapping[str, Any],
    full_cfg: Mapping[str, Any],
) -> None:
    screening_enabled = bool(screening_cfg) and screening_cfg.get("enabled", True) is not False
    windows = screening_cfg.get("windows") or []
    hard_constraints = promotion_cfg.get("hard_constraints") if isinstance(promotion_cfg.get("hard_constraints"), Mapping) else {}
    constraints = {
        key: (hard_constraints.get(key) if hard_constraints else promotion_cfg.get(key))
        for key in (
            "min_trades",
            "max_drawdown_pct",
            "min_winrate_pct",
            "min_return_pct",
            "min_sharpe",
            "min_sortino",
        )
        if (hard_constraints.get(key) if hard_constraints else promotion_cfg.get(key)) is not None
    }
    artifacts_mode = str(artifacts_cfg.get("mode", "full")).lower()
    full_artifacts = full_cfg.get("artifacts") if isinstance(full_cfg.get("artifacts"), Mapping) else {}
    full_mode = str(full_artifacts.get("mode", "full")).lower() if full_cfg else "full"
    full_levels = _promotion_levels(full_artifacts) if full_cfg else []
    cluster_mode = _normalize_cluster_mode(cluster_cfg.get("mode", "kmeans"))
    LOGGER.info(
        "Optimization plan (%s): out_dir=%s method=%s trials=%d params=%d objective=%s",
        kind,
        out_dir,
        method,
        total_trials,
        len(keys),
        _objective_summary(objective),
    )
    LOGGER.info(
        "Optimization metadata: dataset_id=%s timeframe=%s start=%s end=%s code_version=%s",
        base_meta.get("dataset_id"),
        base_meta.get("timeframe"),
        base_meta.get("start"),
        base_meta.get("end"),
        base_meta.get("code_version"),
    )
    LOGGER.info(
        "Optimization screening: enabled=%s windows=%d aggregate=%s max_bars=%s max_trades=%s max_seconds=%s pruning=%s min_window_objective=%s min_windows_passed=%s max_windows_failed=%s",
        screening_enabled,
        len(windows),
        screening_cfg.get("aggregate", "mean"),
        screening_cfg.get("max_bars"),
        screening_cfg.get("max_trades"),
        screening_cfg.get("max_seconds"),
        screening_cfg.get("pruning"),
        screening_cfg.get("min_window_objective"),
        screening_cfg.get("min_windows_passed"),
        screening_cfg.get("max_windows_failed"),
    )
    LOGGER.info(
        "Optimization promotion: top_k=%s hard_constraints=%s soft_constraints=%s dedupe_distance=%s behavior_distance=%s log_dedupe=%s levels=%s",
        promotion_cfg.get("top_k", 0),
        constraints or None,
        promotion_cfg.get("soft_constraints"),
        promotion_cfg.get("dedupe_distance"),
        promotion_cfg.get("behavior_distance"),
        promotion_cfg.get("log_dedupe"),
        _promotion_levels(promotion_cfg),
    )
    LOGGER.info(
        "Optimization behavior: mode=%s metrics=%s bins=%d points=%d cluster_enabled=%s cluster_mode=%s",
        behavior_mode,
        promotion_cfg.get("behavior_metrics") or [],
        len(behavior_bins),
        behavior_points,
        bool(cluster_cfg.get("enabled")),
        cluster_mode,
    )
    LOGGER.info(
        "Optimization artifacts: mode=%s trade_sample_size=%s equity_max_points=%s full_pass=%s full_mode=%s full_levels=%s",
        artifacts_mode,
        artifacts_cfg.get("trade_sample_size"),
        artifacts_cfg.get("equity_max_points"),
        bool(full_cfg.get("enabled")),
        full_mode,
        full_levels,
    )
    LOGGER.info("Optimization cache_features: %s", cfg.get("cache_features"))
    LOGGER.info("Optimization debug_on_fail: %s", cfg.get("debug_on_fail"))
    LOGGER.info(
        "Optimization refine: enabled=%s top_k=%s shrink_pct=%s freeze_keys=%s freeze_prefixes=%s",
        bool(refine_cfg.get("enabled")),
        refine_cfg.get("top_k", 5),
        refine_cfg.get("shrink_pct", 0.5),
        refine_cfg.get("freeze_keys") or [],
        refine_cfg.get("freeze_prefixes") or [],
    )
    LOGGER.info(
        "Optimization flow: screening=%s -> promotion -> clustering=%s -> refine=%s -> full_pass=%s",
        screening_enabled,
        bool(cluster_cfg.get("enabled")),
        bool(refine_cfg.get("enabled")),
        bool(full_cfg.get("enabled")),
    )


def _promotion_levels(cfg: Mapping[str, Any]) -> List[Dict[str, Any]]:
    levels = cfg.get("levels") or []
    if not isinstance(levels, list):
        return []
    cleaned: List[Dict[str, Any]] = []
    for entry in levels:
        if not isinstance(entry, Mapping):
            continue
        level = dict(entry)
        if "max_rank" not in level:
            continue
        cleaned.append(level)
    return cleaned


def _level_for_rank(levels: List[Dict[str, Any]], rank: int) -> Optional[Dict[str, Any]]:
    if not levels:
        return None
    try:
        rank_val = int(rank)
    except Exception:
        rank_val = rank
    for entry in sorted(levels, key=lambda item: int(item.get("max_rank", 0) or 0)):
        try:
            max_rank = int(entry.get("max_rank", 0) or 0)
        except Exception:
            continue
        if max_rank > 0 and rank_val <= max_rank:
            return entry
    return None


def _artifacts_cfg_for_level(
    base_cfg: Mapping[str, Any],
    level_cfg: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    merged = dict(base_cfg or {})
    if not level_cfg:
        return merged
    for key in ("mode", "trade_sample_size", "equity_max_points"):
        if key in level_cfg and level_cfg.get(key) is not None:
            merged[key] = level_cfg.get(key)
    return merged


def _meets_constraints(
    run: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> bool:
    if not cfg:
        return True
    hard = cfg.get("hard_constraints") if isinstance(cfg.get("hard_constraints"), Mapping) else {}
    min_trades = hard.get("min_trades") if hard else cfg.get("min_trades")
    if min_trades is not None:
        win = run.get("winCount") or 0
        loss = run.get("lossCount") or 0
        try:
            trades = int(win) + int(loss)
        except Exception:
            trades = 0
        if trades < int(min_trades):
            return False
    max_dd = hard.get("max_drawdown_pct") if hard else cfg.get("max_drawdown_pct")
    if max_dd is not None:
        try:
            dd = float(run.get("maxDrawdownPct"))
        except Exception:
            return False
        if dd > float(max_dd):
            return False
    min_winrate = hard.get("min_winrate_pct") if hard else cfg.get("min_winrate_pct")
    if min_winrate is not None:
        try:
            winrate = float(run.get("winratePct"))
        except Exception:
            return False
        if winrate < float(min_winrate):
            return False
    min_return = hard.get("min_return_pct") if hard else cfg.get("min_return_pct")
    if min_return is not None:
        try:
            ret = float(run.get("returnPct"))
        except Exception:
            return False
        if ret < float(min_return):
            return False
    min_sharpe = hard.get("min_sharpe") if hard else cfg.get("min_sharpe")
    if min_sharpe is not None:
        try:
            sharpe = float(run.get("sharpe"))
        except Exception:
            return False
        if sharpe < float(min_sharpe):
            return False
    min_sortino = hard.get("min_sortino") if hard else cfg.get("min_sortino")
    if min_sortino is not None:
        try:
            sortino = float(run.get("sortino"))
        except Exception:
            return False
        if sortino < float(min_sortino):
            return False
    return True


def _normalized_distance(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    bounds: Mapping[str, Tuple[float, float]],
) -> Optional[float]:
    if not left or not right:
        return None
    total = 0.0
    count = 0
    for key, (vmin, vmax) in bounds.items():
        lval = left.get(key)
        rval = right.get(key)
        if not isinstance(lval, (int, float)) or not isinstance(rval, (int, float)):
            continue
        span = vmax - vmin
        if span <= 0:
            continue
        lnorm = (float(lval) - vmin) / span
        rnorm = (float(rval) - vmin) / span
        total += abs(lnorm - rnorm)
        count += 1
    if count == 0:
        return None
    return total / count


def _dominant_param_distance(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    bounds: Mapping[str, Tuple[float, float]],
) -> Tuple[Optional[str], Optional[float]]:
    if not left or not right:
        return None, None
    best_key = None
    best_dist = None
    for key, (vmin, vmax) in bounds.items():
        lval = left.get(key)
        rval = right.get(key)
        if not isinstance(lval, (int, float)) or not isinstance(rval, (int, float)):
            continue
        span = vmax - vmin
        if span <= 0:
            continue
        lnorm = (float(lval) - vmin) / span
        rnorm = (float(rval) - vmin) / span
        dist = abs(lnorm - rnorm)
        if best_dist is None or dist > best_dist:
            best_dist = dist
            best_key = key
    return best_key, best_dist


def _dominant_metric_distance(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    metrics: List[str],
    bounds: Mapping[str, Tuple[float, float]],
) -> Tuple[Optional[str], Optional[float]]:
    best_key = None
    best_dist = None
    for key in metrics:
        if key not in bounds:
            continue
        lval = left.get(key)
        rval = right.get(key)
        if not isinstance(lval, (int, float)) or not isinstance(rval, (int, float)):
            continue
        vmin, vmax = bounds[key]
        span = vmax - vmin
        if span <= 0:
            continue
        lnorm = (float(lval) - vmin) / span
        rnorm = (float(rval) - vmin) / span
        dist = abs(lnorm - rnorm)
        if best_dist is None or dist > best_dist:
            best_dist = dist
            best_key = key
    return best_key, best_dist


def _behavior_distance(
    left_run: Mapping[str, Any],
    right_run: Mapping[str, Any],
    metrics: List[str],
    bounds: Mapping[str, Tuple[float, float]],
) -> Optional[float]:
    if not left_run or not right_run or not metrics:
        return None
    total = 0.0
    count = 0
    for metric in metrics:
        key = str(metric)
        lval = left_run.get(key)
        rval = right_run.get(key)
        if not isinstance(lval, (int, float)) or not isinstance(rval, (int, float)):
            continue
        vmin, vmax = bounds.get(key, (None, None))
        if vmin is None or vmax is None:
            continue
        span = vmax - vmin
        if span <= 0:
            continue
        lnorm = (float(lval) - vmin) / span
        rnorm = (float(rval) - vmin) / span
        total += abs(lnorm - rnorm)
        count += 1
    if count == 0:
        return None
    return total / count


def _behavior_vector(
    run: Mapping[str, Any],
    metrics: List[str],
    bounds: Mapping[str, Tuple[float, float]],
) -> Optional[List[float]]:
    if not run or not metrics:
        return None
    vec: List[float] = []
    for metric in metrics:
        key = str(metric)
        val = run.get(key)
        if not isinstance(val, (int, float)):
            return None
        vmin, vmax = bounds.get(key, (None, None))
        if vmin is None or vmax is None:
            return None
        span = vmax - vmin
        if span <= 0:
            return None
        vec.append((float(val) - vmin) / span)
    return vec


def _trade_hist_signature(trades: List[Mapping[str, Any]], bins: List[float]) -> Optional[List[float]]:
    if not trades or not bins:
        return None
    values: List[float] = []
    for trade in trades:
        for key in ("grossPnlPct", "pnl_pct", "gross_pnl_pct", "r_multiple"):
            if key in trade:
                try:
                    values.append(float(trade[key]))
                    break
                except Exception:
                    continue
    if not values:
        return None
    counts = [0] * (len(bins) + 1)
    for val in values:
        idx = 0
        while idx < len(bins) and val > bins[idx]:
            idx += 1
        counts[idx] += 1
    total = sum(counts)
    if total == 0:
        return None
    return [c / total for c in counts]


def _equity_signature(trades: List[Mapping[str, Any]], points: int) -> Optional[List[float]]:
    if not trades or points <= 0:
        return None
    returns: List[float] = []
    for trade in trades:
        for key in ("grossPnlPct", "pnl_pct", "gross_pnl_pct"):
            if key in trade:
                try:
                    returns.append(float(trade[key]))
                    break
                except Exception:
                    continue
    if not returns:
        return None
    equity = []
    running = 0.0
    for r in returns:
        running += r
        equity.append(running)
    if not equity:
        return None
    step = max(1, len(equity) // points)
    sample = equity[::step][:points]
    if not sample:
        return None
    scale = max(abs(min(sample)), abs(max(sample)), 1e-9)
    return [val / scale for val in sample]


def _hist_distance(left: List[float], right: List[float]) -> float:
    if not left or not right:
        return float("inf")
    length = min(len(left), len(right))
    if length == 0:
        return float("inf")
    return sum(abs(left[i] - right[i]) for i in range(length)) / length


def _dbscan_cluster(vectors: List[List[float]], eps: float, min_samples: int) -> List[int]:
    if not vectors:
        return []
    n = len(vectors)
    eps = float(eps)
    min_samples = max(1, int(min_samples))
    labels = [-1] * n
    visited = [False] * n
    cluster_id = 0

    def neighbors(idx: int) -> List[int]:
        vec = vectors[idx]
        return [j for j in range(n) if _hist_distance(vec, vectors[j]) <= eps]

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        neigh = neighbors(i)
        if len(neigh) < min_samples:
            labels[i] = -1
            continue
        labels[i] = cluster_id
        queue = [j for j in neigh if j != i]
        while queue:
            j = queue.pop()
            if not visited[j]:
                visited[j] = True
                neigh_j = neighbors(j)
                if len(neigh_j) >= min_samples:
                    for k in neigh_j:
                        if k not in queue:
                            queue.append(k)
            if labels[j] == -1:
                labels[j] = cluster_id
        cluster_id += 1
    # Assign noise as singleton clusters.
    next_cluster = cluster_id
    for i in range(n):
        if labels[i] == -1:
            labels[i] = next_cluster
            next_cluster += 1
    return labels


def _kmeans_fit(
    vectors: List[List[float]],
    k: int,
    iterations: int = 10,
) -> Tuple[List[int], List[List[float]], float]:
    if not vectors or k <= 0:
        return [], [], 0.0
    k = min(k, len(vectors))
    centers = [list(vectors[i]) for i in range(k)]
    assignments = [0] * len(vectors)
    for _ in range(iterations):
        for i, vec in enumerate(vectors):
            best = 0
            best_dist = float("inf")
            for c_idx, center in enumerate(centers):
                dist = _hist_distance(vec, center)
                if dist < best_dist:
                    best_dist = dist
                    best = c_idx
            assignments[i] = best
        new_centers: List[List[float]] = []
        for c_idx in range(k):
            members = [vectors[i] for i, a in enumerate(assignments) if a == c_idx]
            if not members:
                new_centers.append(centers[c_idx])
                continue
            length = len(members[0])
            mean = [0.0] * length
            for m in members:
                for j in range(length):
                    mean[j] += m[j]
            new_centers.append([v / len(members) for v in mean])
        centers = new_centers
    inertia = 0.0
    for vec, idx in zip(vectors, assignments):
        inertia += _hist_distance(vec, centers[idx])
    return assignments, centers, inertia


def _silhouette_score(vectors: List[List[float]], assignments: List[int]) -> float:
    if not vectors or not assignments:
        return 0.0
    n = len(vectors)
    by_cluster: Dict[int, List[int]] = {}
    for idx, cluster_id in enumerate(assignments):
        by_cluster.setdefault(cluster_id, []).append(idx)
    scores: List[float] = []
    for i in range(n):
        vec = vectors[i]
        cluster_id = assignments[i]
        same = by_cluster.get(cluster_id, [])
        if len(same) <= 1:
            scores.append(0.0)
            continue
        a = sum(_hist_distance(vec, vectors[j]) for j in same if j != i) / (len(same) - 1)
        b = None
        for other_id, members in by_cluster.items():
            if other_id == cluster_id or not members:
                continue
            dist = sum(_hist_distance(vec, vectors[j]) for j in members) / len(members)
            if b is None or dist < b:
                b = dist
        if b is None:
            scores.append(0.0)
            continue
        denom = max(a, b)
        scores.append((b - a) / denom if denom > 0 else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def _kmeans_cluster(vectors: List[List[float]], k: int, iterations: int = 10) -> List[int]:
    assignments, _, _ = _kmeans_fit(vectors, k, iterations=iterations)
    return assignments


def _auto_kmeans(
    vectors: List[List[float]],
    *,
    min_k: int,
    max_k: int,
    iterations: int,
    mode: str,
    fallback_mode: str = "inertia",
) -> List[int]:
    if not vectors:
        return []
    min_k = max(2, int(min_k))
    max_k = max(min_k, int(max_k))
    max_k = min(max_k, len(vectors))
    best_assignments: List[int] = []
    best_score = None
    for k in range(min_k, max_k + 1):
        assignments, _, inertia = _kmeans_fit(vectors, k, iterations=iterations)
        if mode == "inertia":
            score = -inertia
        else:
            score = _silhouette_score(vectors, assignments)
        if best_score is None or score > best_score:
            best_score = score
            best_assignments = assignments
    if not best_assignments and fallback_mode != mode:
        LOGGER.info("Behavior cluster auto: fallback from %s to %s", mode, fallback_mode)
        return _auto_kmeans(
            vectors,
            min_k=min_k,
            max_k=max_k,
            iterations=iterations,
            mode=fallback_mode,
            fallback_mode=mode,
        )
    return best_assignments


def _cluster_promoted(
    promoted: List[Dict[str, Any]],
    *,
    behavior_metrics: List[str],
    behavior_bounds: Mapping[str, Tuple[float, float]],
    cluster_cfg: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    if not promoted:
        return promoted
    if not cluster_cfg.get("enabled"):
        return promoted
    mode = _normalize_cluster_mode(cluster_cfg.get("mode", "kmeans"))
    k = cluster_cfg.get("k", 3)
    auto_mode = str(cluster_cfg.get("auto_mode", "silhouette")).lower()
    fallback_mode = str(cluster_cfg.get("fallback_mode", "inertia")).lower()
    min_k = cluster_cfg.get("min_k", 2)
    max_k = cluster_cfg.get("max_k", max(2, int(len(promoted) ** 0.5)))
    iterations = int(cluster_cfg.get("iterations", 10))
    eps = float(cluster_cfg.get("eps", 0.2))
    min_samples = int(cluster_cfg.get("min_samples", 2))
    auto = isinstance(k, str) and k.lower() == "auto"
    if not auto:
        try:
            k = int(k)
        except Exception:
            k = 3
    max_per_cluster = int(cluster_cfg.get("max_per_cluster", 2))
    vectors: List[List[float]] = []
    idx_map: List[int] = []
    for idx, entry in enumerate(promoted):
        sig = entry.get("behavior_sig")
        if isinstance(sig, list):
            vectors.append([float(v) for v in sig])
            idx_map.append(idx)
            continue
        vec = _behavior_vector(entry.get("run", {}), behavior_metrics, behavior_bounds)
        if vec is not None:
            vectors.append(vec)
            idx_map.append(idx)
    if not vectors:
        return promoted
    if mode == "dbscan":
        LOGGER.info("Behavior cluster mode=dbscan eps=%.3f min_samples=%d", eps, min_samples)
        assignments = _dbscan_cluster(vectors, eps=eps, min_samples=min_samples)
    elif auto:
        LOGGER.info(
            "Behavior cluster mode=kmeans auto_k min_k=%d max_k=%d auto_mode=%s fallback=%s iterations=%d",
            min_k,
            max_k,
            auto_mode,
            fallback_mode,
            iterations,
        )
        assignments = _auto_kmeans(
            vectors,
            min_k=min_k,
            max_k=max_k,
            iterations=iterations,
            mode=auto_mode,
            fallback_mode=fallback_mode,
        )
    else:
        LOGGER.info("Behavior cluster mode=kmeans k=%s iterations=%d", k, iterations)
        assignments = _kmeans_cluster(vectors, k=k, iterations=iterations)
    by_cluster: Dict[int, List[Dict[str, Any]]] = {}
    for assign, idx in zip(assignments, idx_map):
        by_cluster.setdefault(assign, []).append(promoted[idx])
    selected: List[Dict[str, Any]] = []
    for items in by_cluster.values():
        items.sort(key=lambda item: item.get("objective", float("-inf")), reverse=True)
        selected.extend(items[:max_per_cluster])
    if not selected:
        return promoted
    return selected


def _update_topk(
    topk: List[Dict[str, Any]],
    candidate: Dict[str, Any],
    *,
    top_k: int,
    dedupe_distance: Optional[float],
    bounds: Mapping[str, Tuple[float, float]],
    behavior_distance: Optional[float],
    behavior_metrics: List[str],
    behavior_bounds: Mapping[str, Tuple[float, float]],
    behavior_mode: str,
    behavior_bins: List[float],
    log_dedupe: bool = False,
) -> None:
    if top_k <= 0:
        return
    if dedupe_distance:
        closest_idx = None
        closest_dist = None
        for idx, existing in enumerate(topk):
            dist = _normalized_distance(existing.get("params", {}), candidate.get("params", {}), bounds)
            if dist is None:
                continue
            if closest_dist is None or dist < closest_dist:
                closest_dist = dist
                closest_idx = idx
        if closest_dist is not None and closest_dist < float(dedupe_distance):
            if candidate.get("objective", float("-inf")) > topk[closest_idx].get("objective", float("-inf")):
                if log_dedupe:
                    key, dist = _dominant_param_distance(
                        topk[closest_idx].get("params", {}),
                        candidate.get("params", {}),
                        bounds,
                    )
                    LOGGER.info(
                        "Dedupe params: replace trial=%s with trial=%s distance=%.4f dominant=%s:%.4f",
                        topk[closest_idx].get("trial_id"),
                        candidate.get("trial_id"),
                        float(closest_dist),
                        key,
                        float(dist) if dist is not None else 0.0,
                    )
                topk[closest_idx] = candidate
            elif log_dedupe:
                key, dist = _dominant_param_distance(
                    topk[closest_idx].get("params", {}),
                    candidate.get("params", {}),
                    bounds,
                )
                LOGGER.info(
                    "Dedupe params: reject trial=%s close to trial=%s distance=%.4f dominant=%s:%.4f",
                    candidate.get("trial_id"),
                    topk[closest_idx].get("trial_id"),
                    float(closest_dist),
                    key,
                    float(dist) if dist is not None else 0.0,
                )
            return
    if behavior_distance and behavior_mode == "metrics":
        closest_idx = None
        closest_dist = None
        for idx, existing in enumerate(topk):
            dist = _behavior_distance(
                existing.get("run", {}),
                candidate.get("run", {}),
                behavior_metrics,
                behavior_bounds,
            )
            if dist is None:
                continue
            if closest_dist is None or dist < closest_dist:
                closest_dist = dist
                closest_idx = idx
        if closest_dist is not None and closest_dist < float(behavior_distance):
            if candidate.get("objective", float("-inf")) > topk[closest_idx].get("objective", float("-inf")):
                if log_dedupe:
                    key, dist = _dominant_metric_distance(
                        topk[closest_idx].get("run", {}),
                        candidate.get("run", {}),
                        behavior_metrics,
                        behavior_bounds,
                    )
                    LOGGER.info(
                        "Dedupe behavior: replace trial=%s with trial=%s distance=%.4f dominant=%s:%.4f",
                        topk[closest_idx].get("trial_id"),
                        candidate.get("trial_id"),
                        float(closest_dist),
                        key,
                        float(dist) if dist is not None else 0.0,
                    )
                topk[closest_idx] = candidate
            elif log_dedupe:
                key, dist = _dominant_metric_distance(
                    topk[closest_idx].get("run", {}),
                    candidate.get("run", {}),
                    behavior_metrics,
                    behavior_bounds,
                )
                LOGGER.info(
                    "Dedupe behavior: reject trial=%s close to trial=%s distance=%.4f dominant=%s:%.4f",
                    candidate.get("trial_id"),
                    topk[closest_idx].get("trial_id"),
                    float(closest_dist),
                    key,
                    float(dist) if dist is not None else 0.0,
                )
            return
    if behavior_distance and behavior_mode in {"trades_hist", "equity_signature"}:
        cand_sig = candidate.get("behavior_sig")
        if cand_sig:
            closest_idx = None
            closest_dist = None
            for idx, existing in enumerate(topk):
                exist_sig = existing.get("behavior_sig")
                if not exist_sig:
                    continue
                dist = _hist_distance(exist_sig, cand_sig)
                if closest_dist is None or dist < closest_dist:
                    closest_dist = dist
                    closest_idx = idx
            if closest_dist is not None and closest_dist < float(behavior_distance):
                if candidate.get("objective", float("-inf")) > topk[closest_idx].get("objective", float("-inf")):
                    if log_dedupe:
                        LOGGER.info(
                            "Dedupe behavior: replace trial=%s with trial=%s distance=%.4f mode=%s",
                            topk[closest_idx].get("trial_id"),
                            candidate.get("trial_id"),
                            float(closest_dist),
                            behavior_mode,
                        )
                    topk[closest_idx] = candidate
                elif log_dedupe:
                    LOGGER.info(
                        "Dedupe behavior: reject trial=%s close to trial=%s distance=%.4f mode=%s",
                        candidate.get("trial_id"),
                        topk[closest_idx].get("trial_id"),
                        float(closest_dist),
                        behavior_mode,
                    )
                return
    topk.append(candidate)
    topk.sort(key=lambda item: item.get("objective", float("-inf")), reverse=True)
    if len(topk) > top_k:
        topk.pop()


def _write_promoted(
    out_dir: Path,
    promoted: List[Dict[str, Any]],
    *,
    artifacts_cfg: Optional[Mapping[str, Any]] = None,
    promotion_cfg: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    if not promoted:
        return []
    promoted_dir = out_dir / "promoted"
    promoted_dir.mkdir(parents=True, exist_ok=True)
    stored: List[Dict[str, Any]] = []
    artifacts_cfg = artifacts_cfg or {}
    levels = _promotion_levels(promotion_cfg or {})
    logger = logging.getLogger(__name__)
    if not levels:
        logger.info("Promotion: no levels configured, using artifacts mode '%s'", artifacts_cfg.get("mode", "full"))
    for idx, entry in enumerate(promoted, start=1):
        level_cfg = _level_for_rank(levels, idx)
        if levels and level_cfg is None:
            logger.info("Promotion: no level for rank %d, using base artifacts config", idx)
        level_name = level_cfg.get("mode") if level_cfg else None
        level_artifacts_cfg = _artifacts_cfg_for_level(artifacts_cfg, level_cfg)
        compact_mode = str(level_artifacts_cfg.get("mode", "full")).lower()
        equity_limit = level_artifacts_cfg.get("equity_max_points")
        trade_limit = level_artifacts_cfg.get("trade_sample_size")
        trial_id = entry.get("trial_id", "unknown")
        payload = entry.get("payload") or {}
        if compact_mode in {"compact", "summary", "light"}:
            payload = _compact_payload(payload, equity_limit=equity_limit, trade_limit=trade_limit, mode=compact_mode)
        if compact_mode in {"stats"}:
            payload = _compact_payload(payload, equity_limit=equity_limit, trade_limit=trade_limit, mode=compact_mode)
        path = promoted_dir / f"trial_{trial_id}.json"
        path.write_text(json.dumps(payload, indent=2, default=str))
        stored.append(
            {
                "trial_id": trial_id,
                "objective": entry.get("objective"),
                "params": entry.get("params"),
                "path": str(path),
                "level": level_name or compact_mode,
            }
        )
    return stored


def _disable_screening(spec: Mapping[str, Any]) -> Dict[str, Any]:
    updated = deepcopy(spec)
    opt = updated.get("optimization")
    if not isinstance(opt, Mapping):
        updated["optimization"] = {}
        opt = updated["optimization"]
    screen = opt.get("screening")
    if not isinstance(screen, Mapping):
        opt["screening"] = {}
        screen = opt["screening"]
    screen["enabled"] = False
    for key in ("max_bars", "max_trades", "max_seconds", "windows", "aggregate", "window_start", "window_end"):
        if key in screen:
            screen.pop(key, None)
    return updated


def _run_full_pass(
    base_spec: Mapping[str, Any],
    promoted: List[Dict[str, Any]],
    *,
    out_dir: Path,
    runner_fn,
    artifacts_cfg: Optional[Mapping[str, Any]] = None,
    full_cfg: Optional[Mapping[str, Any]] = None,
    promotion_cfg: Optional[Mapping[str, Any]] = None,
    objective: Any = None,
) -> List[Dict[str, Any]]:
    if not promoted:
        return []
    full_dir = out_dir / "full_pass"
    full_dir.mkdir(parents=True, exist_ok=True)
    artifacts_cfg = artifacts_cfg or {}
    full_cfg = full_cfg or {}
    folds = full_cfg.get("folds") or []
    if not isinstance(folds, list):
        folds = []
    aggregate = full_cfg.get("aggregate", "mean")
    levels = _promotion_levels(artifacts_cfg) or _promotion_levels(promotion_cfg or {})
    logger = logging.getLogger(__name__)
    if not levels:
        logger.info("Full pass: no levels configured, using artifacts mode '%s'", artifacts_cfg.get("mode", "full"))
    stored: List[Dict[str, Any]] = []
    for idx, entry in enumerate(promoted, start=1):
        level_cfg = _level_for_rank(levels, idx) if levels else None
        if levels and level_cfg is None:
            logger.info("Full pass: no level for rank %d, using base artifacts config", idx)
        level_artifacts_cfg = _artifacts_cfg_for_level(artifacts_cfg, level_cfg)
        compact_mode = str(level_artifacts_cfg.get("mode", "full")).lower()
        equity_limit = level_artifacts_cfg.get("equity_max_points")
        trade_limit = level_artifacts_cfg.get("trade_sample_size")
        trial_id = entry.get("trial_id", "unknown")
        params = entry.get("params", {})
        if folds:
            fold_payloads: List[Dict[str, Any]] = []
            fold_scores: List[float] = []
            for fold_idx, fold in enumerate(folds):
                if not isinstance(fold, Mapping):
                    continue
                spec = _disable_screening(base_spec)
                for key, val in (params or {}).items():
                    _set_path(spec, key, val)
                spec.setdefault("optimization", {}).setdefault("screening", {})
                spec["optimization"]["screening"]["enabled"] = True
                spec["optimization"]["screening"]["window_start"] = fold.get("start")
                spec["optimization"]["screening"]["window_end"] = fold.get("end")
                result = runner_fn(spec)
                payload = result.get("payload") or {}
                run = payload.get("run") or {}
                score = _objective_value(run, objective) if objective is not None else float("nan")
                fold_scores.append(score)
                if compact_mode in {"compact", "summary", "light", "stats"}:
                    payload = _compact_payload(
                        payload, equity_limit=equity_limit, trade_limit=trade_limit, mode=compact_mode
                    )
                fold_payloads.append(
                    {
                        "fold": fold_idx,
                        "window": {"start": fold.get("start"), "end": fold.get("end")},
                        "objective": score,
                        "payload": payload,
                    }
                )
            agg_score = _aggregate_scores(fold_scores, aggregate) if fold_scores else float("-inf")
            payload = {
                "folds": fold_payloads,
                "aggregate_objective": agg_score,
                "aggregate": aggregate,
            }
        else:
            spec = _disable_screening(base_spec)
            for key, val in (params or {}).items():
                _set_path(spec, key, val)
            result = runner_fn(spec)
            payload = result.get("payload") or {}
            if compact_mode in {"compact", "summary", "light", "stats"}:
                payload = _compact_payload(
                    payload, equity_limit=equity_limit, trade_limit=trade_limit, mode=compact_mode
                )
        path = full_dir / f"trial_{trial_id}.json"
        path.write_text(json.dumps(payload, indent=2, default=str))
        stored.append(
            {
                "trial_id": trial_id,
                "params": params,
                "path": str(path),
                "level": (level_cfg.get("mode") if level_cfg else compact_mode),
            }
        )
    return stored


def _log_storage_impact(total_trials: int, promoted_count: int, full_pass_count: int) -> None:
    if total_trials <= 0:
        return
    heavy_ratio = promoted_count / total_trials if total_trials else 0.0
    LOGGER.info(
        "Optimization storage impact: trials=%d heavy_promoted=%d (%.1f%%) full_pass=%d",
        total_trials,
        promoted_count,
        heavy_ratio * 100.0,
        full_pass_count,
    )


def _read_summary(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _collect_runs(base_dir: Path) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    for child in base_dir.iterdir():
        if not child.is_dir():
            continue
        summary_path = child / "summary.json"
        summary = _read_summary(summary_path)
        if not summary:
            continue
        best = summary.get("best") or {}
        objective = best.get("objective")
        metadata = summary.get("metadata") or {}
        runs.append(
            {
                "dir": child,
                "summary": summary,
                "objective": objective,
                "metadata": metadata,
                "mtime": summary_path.stat().st_mtime if summary_path.exists() else child.stat().st_mtime,
            }
        )
    return runs


def _prune_heavy_artifacts(run_dir: Path) -> None:
    for root, dirs, _ in os.walk(run_dir):
        for dname in list(dirs):
            if dname in {"promoted", "full_pass"}:
                target = Path(root) / dname
                shutil.rmtree(target, ignore_errors=True)


def _apply_retention(out_dir: Path, cfg: Mapping[str, Any]) -> None:
    storage_cfg = cfg.get("storage") or {}
    if not isinstance(storage_cfg, Mapping):
        return
    retention = storage_cfg.get("retention") or {}
    if not isinstance(retention, Mapping):
        return
    if retention.get("enabled", True) is False:
        return
    keep_last = retention.get("keep_last_runs")
    keep_best = retention.get("keep_best_runs")
    mode = str(retention.get("mode", "heavy_only")).lower()
    dry_run = bool(retention.get("dry_run", False))
    base_dir = out_dir.parent
    runs = _collect_runs(base_dir)
    if not runs:
        return
    keep_dirs: set[Path] = {out_dir.resolve()}
    if keep_last:
        try:
            keep_last_int = int(keep_last)
        except Exception:
            keep_last_int = 0
        if keep_last_int > 0:
            recent = sorted(runs, key=lambda r: r.get("mtime", 0), reverse=True)
            keep_dirs.update({r["dir"].resolve() for r in recent[:keep_last_int]})
    if keep_best:
        try:
            keep_best_int = int(keep_best)
        except Exception:
            keep_best_int = 0
        if keep_best_int > 0:
            grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
            for run in runs:
                meta = run.get("metadata") or {}
                key = (str(meta.get("strategy_id") or "unknown"), str(meta.get("dataset_id") or "unknown"))
                grouped.setdefault(key, []).append(run)
            for items in grouped.values():
                ranked = sorted(
                    items,
                    key=lambda r: r.get("objective", float("-inf")) if r.get("objective") is not None else float("-inf"),
                    reverse=True,
                )
                keep_dirs.update({r["dir"].resolve() for r in ranked[:keep_best_int]})
    for run in runs:
        run_dir = run["dir"].resolve()
        if run_dir in keep_dirs:
            continue
        if mode == "full":
            LOGGER.info("Retention: deleting run %s", run_dir)
            if not dry_run:
                shutil.rmtree(run_dir, ignore_errors=True)
        else:
            LOGGER.info("Retention: pruning heavy artifacts in %s", run_dir)
            if not dry_run:
                _prune_heavy_artifacts(run_dir)


def run_backtest_optimization(spec: Mapping[str, Any], *, out_dir: Optional[str | Path] = None) -> Dict[str, Any]:
    cfg = _optimization_config(spec)
    promotion_cfg = _promotion_config(cfg)
    screening_cfg = cfg.get("screening") or {}
    if screening_cfg and not isinstance(screening_cfg, Mapping):
        raise ValueError("optimization.screening must be a mapping")
    refine_cfg = cfg.get("refine") or {}
    if refine_cfg and not isinstance(refine_cfg, Mapping):
        raise ValueError("optimization.refine must be a mapping")
    space = cfg.get("search_space") or {}
    if not isinstance(space, Mapping):
        raise ValueError("optimization.search_space must be a mapping")
    keys, values = _expand_search_space(space)
    bounds: Dict[str, Tuple[float, float]] = {}
    for key, vals in zip(keys, values):
        numeric = _numeric_bounds(vals)
        if numeric is not None:
            bounds[key] = numeric
    method = _normalize_method(str(cfg.get("method", "grid")))
    max_trials = cfg.get("max_trials")
    seed = cfg.get("seed")
    objective = _merge_soft_constraints(cfg.get("objective", "sharpe"), promotion_cfg)
    total_trials = _count_trials(values, method, max_trials)
    LOGGER.info("Optimization trials planned: %d", total_trials)
    top_k = int(promotion_cfg.get("top_k", 0) or 0)
    dedupe_distance = promotion_cfg.get("dedupe_distance")
    behavior_distance = promotion_cfg.get("behavior_distance")
    behavior_mode = _normalize_behavior_mode(str(promotion_cfg.get("behavior_mode", "metrics")))
    behavior_metrics = promotion_cfg.get("behavior_metrics") or []
    if not isinstance(behavior_metrics, list):
        behavior_metrics = []
    behavior_bounds = promotion_cfg.get("behavior_bounds") or {}
    if not isinstance(behavior_bounds, Mapping):
        behavior_bounds = {}
    behavior_bounds = {k: tuple(v) for k, v in behavior_bounds.items() if isinstance(v, (list, tuple)) and len(v) == 2}
    behavior_bins = promotion_cfg.get("behavior_bins") or [-5, -2, -1, -0.5, 0, 0.5, 1, 2, 5]
    if not isinstance(behavior_bins, list):
        behavior_bins = [-5, -2, -1, -0.5, 0, 0.5, 1, 2, 5]
    behavior_points = promotion_cfg.get("behavior_points", 20)
    try:
        behavior_points = int(behavior_points)
    except Exception:
        behavior_points = 20

    out_dir = Path(out_dir or cfg.get("out_dir") or "runs/optimize_backtest")
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts_cfg = cfg.get("artifacts") if isinstance(cfg.get("artifacts"), Mapping) else {}
    cluster_cfg = promotion_cfg.get("behavior_cluster") or {}
    if not isinstance(cluster_cfg, Mapping):
        cluster_cfg = {}
    full_cfg = cfg.get("full_pass") if isinstance(cfg.get("full_pass"), Mapping) else {}
    debug_cfg = _debug_on_fail_cfg(cfg)

    trials: List[Dict[str, Any]] = []
    best: Dict[str, Any] | None = None
    promoted: List[Dict[str, Any]] = []
    trial_id = 0

    base_meta = _trial_metadata(spec, cfg)
    _log_optimization_plan(
        "backtest",
        cfg=cfg,
        promotion_cfg=promotion_cfg,
        screening_cfg=screening_cfg,
        refine_cfg=refine_cfg,
        objective=objective,
        method=method,
        total_trials=total_trials,
        keys=keys,
        out_dir=out_dir,
        base_meta=base_meta,
      behavior_mode=behavior_mode,
      behavior_bins=behavior_bins,
      log_dedupe=bool(promotion_cfg.get("log_dedupe", False)),
        behavior_points=behavior_points,
        cluster_cfg=cluster_cfg,
        artifacts_cfg=artifacts_cfg,
        full_cfg=full_cfg,
    )
    for trial_spec in _trial_specs(spec, keys, values, method, max_trials, seed):
        trial_id += 1
        trial_params = {k: _get_path_value(trial_spec, k) for k in keys}
        window_scores: List[float] = []
        window_runs: List[Mapping[str, Any]] = []
        windows = screening_cfg.get("windows") or []
        if windows:
            for window in windows:
                if not isinstance(window, Mapping):
                    continue
                trial_window = deepcopy(trial_spec)
                trial_window.setdefault("optimization", {}).setdefault("screening", {})
                trial_window["optimization"]["screening"]["window_start"] = window.get("start")
                trial_window["optimization"]["screening"]["window_end"] = window.get("end")
                try:
                    result = backtest_runner.run_backtest_from_spec(trial_window)
                except Exception as exc:
                    if debug_cfg:
                        _write_debug_failure(
                            out_dir,
                            trial_id,
                            trial_params,
                            error=f"window_failure: {exc}",
                            context={"window": dict(window)},
                            debug_cfg=debug_cfg,
                        )
                    window_runs.append({"error": str(exc)})
                    window_scores.append(float("-inf"))
                    continue
                payload = result.get("payload") or {}
                run = payload.get("run") or {}
                window_runs.append(run)
                window_scores.append(_objective_value(run, objective))
            passed = _screening_pass(window_scores, screening_cfg)
            score = _aggregate_scores(window_scores, screening_cfg.get("aggregate", "mean"))
            if not passed:
                score = float("-inf")
            payload = {"runs": window_runs}
            run = {"window_scores": window_scores, "screening_failed": (not passed)}
        else:
            try:
                result = backtest_runner.run_backtest_from_spec(trial_spec)
            except Exception as exc:
                if debug_cfg:
                    _write_debug_failure(
                        out_dir,
                        trial_id,
                        trial_params,
                        error=f"trial_failure: {exc}",
                        debug_cfg=debug_cfg,
                    )
                trials.append(
                    {
                        "trial_id": trial_id,
                        "params": trial_params,
                        "objective": float("-inf"),
                        "window_scores": None,
                        "metadata": base_meta,
                    }
                )
                continue
            payload = result.get("payload") or {}
            run = payload.get("run") or {}
            score = _objective_value(run, objective)
        if not math.isfinite(score):
            if debug_cfg:
                _write_debug_failure(
                    out_dir,
                    trial_id,
                    trial_params,
                    error="non_finite_objective",
                    payload=payload,
                    debug_cfg=debug_cfg,
                )
            score = float("-inf")
        trials.append(
            {
                "trial_id": trial_id,
                "params": trial_params,
                "objective": score,
                "window_scores": window_scores or None,
                "metadata": base_meta,
            }
        )
        if best is None or score > best["objective"]:
            best = {"params": trial_params, "objective": score, "run": run}
        behavior_sig = None
        if behavior_mode == "trades_hist":
            trades = payload.get("trades")
            if isinstance(trades, list):
                behavior_sig = _trade_hist_signature(trades, behavior_bins)
        if behavior_mode == "equity_signature":
            trades = payload.get("trades")
            if isinstance(trades, list):
                behavior_sig = _equity_signature(trades, behavior_points)
        if _meets_constraints(run, promotion_cfg):
            _update_topk(
                promoted,
                {
                    "trial_id": trial_id,
                    "params": trial_params,
                    "objective": score,
                    "payload": payload,
                    "run": run,
                    "behavior_sig": behavior_sig,
                },
                top_k=top_k,
                dedupe_distance=dedupe_distance,
                bounds=bounds,
                behavior_distance=behavior_distance,
                behavior_metrics=behavior_metrics,
                behavior_bounds=behavior_bounds,
      behavior_mode=behavior_mode,
      behavior_bins=behavior_bins,
      log_dedupe=bool(promotion_cfg.get("log_dedupe", False)),
            )

    artifacts.write_trials(out_dir / "trials.json", trials)
    promoted = _cluster_promoted(
        promoted,
        behavior_metrics=behavior_metrics,
        behavior_bounds=behavior_bounds,
        cluster_cfg=cluster_cfg,
    )
    promoted_records = _write_promoted(
        out_dir,
        promoted,
        artifacts_cfg=artifacts_cfg,
        promotion_cfg=promotion_cfg,
    )
    refine_records: List[Dict[str, Any]] = []
    if refine_cfg.get("enabled"):
        refine_dir = out_dir / "refine"
        refine_dir.mkdir(parents=True, exist_ok=True)
        top_k = int(refine_cfg.get("top_k", 5))
        shrink_pct = float(refine_cfg.get("shrink_pct", 0.5))
        sorted_trials = sorted(trials, key=lambda t: t.get("objective", float("-inf")), reverse=True)
        top_trials = sorted_trials[:top_k]
        freeze_keys = refine_cfg.get("freeze_keys") or []
        if not isinstance(freeze_keys, list):
            freeze_keys = []
        freeze_prefixes = refine_cfg.get("freeze_prefixes") or []
        if not isinstance(freeze_prefixes, list):
            freeze_prefixes = []
        refined_space = _refine_search_space(
            cfg.get("search_space") or {},
            top_trials,
            shrink_pct=shrink_pct,
            top_k=top_k,
            freeze_keys=freeze_keys,
            freeze_prefixes=freeze_prefixes,
        )
        refine_spec = deepcopy(spec)
        refine_spec.setdefault("optimization", {})["search_space"] = refined_space
        refine_spec["optimization"].setdefault("refine", {})["enabled"] = False
        refine_result = run_backtest_optimization(refine_spec, out_dir=refine_dir)
        refine_records.append(
            {
                "dir": str(refine_dir),
                "top_k": top_k,
                "shrink_pct": shrink_pct,
                "trials_path": refine_result.get("trials_path"),
                "summary": refine_result.get("summary"),
            }
        )
    full_records: List[Dict[str, Any]] = []
    if full_cfg.get("enabled"):
        full_records = _run_full_pass(
            spec,
            promoted_records,
            out_dir=out_dir,
            runner_fn=backtest_runner.run_backtest_from_spec,
            artifacts_cfg=full_cfg.get("artifacts") if isinstance(full_cfg.get("artifacts"), Mapping) else {},
            full_cfg=full_cfg,
            promotion_cfg=promotion_cfg,
            objective=objective,
        )
    _log_storage_impact(total_trials, len(promoted_records), len(full_records))
    sensitivity_cfg = cfg.get("sensitivity") or {}
    sensitivity = None
    if isinstance(sensitivity_cfg, Mapping) and sensitivity_cfg.get("enabled"):
        top_k = sensitivity_cfg.get("top_k", 20)
        try:
            top_k = int(top_k)
        except Exception:
            top_k = 20
        sensitivity = _sensitivity_from_trials(trials, top_k=top_k)
    summary = {
        "objective": objective,
        "best": best,
        "total_trials": total_trials,
        "promoted": promoted_records,
        "refine": refine_records,
        "full_pass": full_records,
        "metadata": base_meta,
        "sensitivity": sensitivity,
    }
    artifacts.write_summary(out_dir / "summary.json", summary)
    _apply_retention(out_dir, cfg)
    return {
        "trials_path": str(out_dir / "trials.json"),
        "summary": str(out_dir / "summary.json"),
        "best": best,
        "total_trials": total_trials,
        "promoted": promoted_records,
        "refine": refine_records,
        "full_pass": full_records,
        "metadata": base_meta,
    }


def run_strategy_optimization(spec: Mapping[str, Any], *, out_dir: Optional[str | Path] = None) -> Dict[str, Any]:
    cfg = _optimization_config(spec)
    promotion_cfg = _promotion_config(cfg)
    screening_cfg = cfg.get("screening") or {}
    if screening_cfg and not isinstance(screening_cfg, Mapping):
        raise ValueError("optimization.screening must be a mapping")
    refine_cfg = cfg.get("refine") or {}
    if refine_cfg and not isinstance(refine_cfg, Mapping):
        raise ValueError("optimization.refine must be a mapping")
    space = cfg.get("search_space") or {}
    if not isinstance(space, Mapping):
        raise ValueError("optimization.search_space must be a mapping")
    keys, values = _expand_search_space(space)
    bounds: Dict[str, Tuple[float, float]] = {}
    for key, vals in zip(keys, values):
        numeric = _numeric_bounds(vals)
        if numeric is not None:
            bounds[key] = numeric
    method = _normalize_method(str(cfg.get("method", "grid")))
    max_trials = cfg.get("max_trials")
    seed = cfg.get("seed")
    objective = _merge_soft_constraints(cfg.get("objective", "sharpe"), promotion_cfg)
    total_trials = _count_trials(values, method, max_trials)
    LOGGER.info("Optimization trials planned: %d", total_trials)
    top_k = int(promotion_cfg.get("top_k", 0) or 0)
    dedupe_distance = promotion_cfg.get("dedupe_distance")
    behavior_distance = promotion_cfg.get("behavior_distance")
    behavior_mode = _normalize_behavior_mode(str(promotion_cfg.get("behavior_mode", "metrics")))
    behavior_metrics = promotion_cfg.get("behavior_metrics") or []
    if not isinstance(behavior_metrics, list):
        behavior_metrics = []
    behavior_bounds = promotion_cfg.get("behavior_bounds") or {}
    if not isinstance(behavior_bounds, Mapping):
        behavior_bounds = {}
    behavior_bounds = {k: tuple(v) for k, v in behavior_bounds.items() if isinstance(v, (list, tuple)) and len(v) == 2}
    behavior_bins = promotion_cfg.get("behavior_bins") or [-5, -2, -1, -0.5, 0, 0.5, 1, 2, 5]
    if not isinstance(behavior_bins, list):
        behavior_bins = [-5, -2, -1, -0.5, 0, 0.5, 1, 2, 5]
    behavior_points = promotion_cfg.get("behavior_points", 20)
    try:
        behavior_points = int(behavior_points)
    except Exception:
        behavior_points = 20

    out_dir = Path(out_dir or cfg.get("out_dir") or "runs/optimize_strategy")
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts_cfg = cfg.get("artifacts") if isinstance(cfg.get("artifacts"), Mapping) else {}
    cluster_cfg = promotion_cfg.get("behavior_cluster") or {}
    if not isinstance(cluster_cfg, Mapping):
        cluster_cfg = {}
    full_cfg = cfg.get("full_pass") if isinstance(cfg.get("full_pass"), Mapping) else {}
    debug_cfg = _debug_on_fail_cfg(cfg)

    trials: List[Dict[str, Any]] = []
    best: Dict[str, Any] | None = None
    promoted: List[Dict[str, Any]] = []
    trial_id = 0

    base_meta = _trial_metadata(spec, cfg)
    _log_optimization_plan(
        "strategy",
        cfg=cfg,
        promotion_cfg=promotion_cfg,
        screening_cfg=screening_cfg,
        refine_cfg=refine_cfg,
        objective=objective,
        method=method,
        total_trials=total_trials,
        keys=keys,
        out_dir=out_dir,
        base_meta=base_meta,
        behavior_mode=behavior_mode,
        behavior_bins=behavior_bins,
        behavior_points=behavior_points,
        cluster_cfg=cluster_cfg,
        artifacts_cfg=artifacts_cfg,
        full_cfg=full_cfg,
    )
    for trial_spec in _trial_specs(spec, keys, values, method, max_trials, seed):
        trial_id += 1
        trial_params = {k: _get_path_value(trial_spec, k) for k in keys}
        window_scores: List[float] = []
        window_runs: List[Mapping[str, Any]] = []
        windows = screening_cfg.get("windows") or []
        if windows:
            for window in windows:
                if not isinstance(window, Mapping):
                    continue
                trial_window = deepcopy(trial_spec)
                trial_window.setdefault("optimization", {}).setdefault("screening", {})
                trial_window["optimization"]["screening"]["window_start"] = window.get("start")
                trial_window["optimization"]["screening"]["window_end"] = window.get("end")
                try:
                    result = strategy_runner.run_backtest_with_payload(trial_window)
                except Exception as exc:
                    if debug_cfg:
                        _write_debug_failure(
                            out_dir,
                            trial_id,
                            trial_params,
                            error=f"window_failure: {exc}",
                            context={"window": dict(window)},
                            debug_cfg=debug_cfg,
                        )
                    window_runs.append({"error": str(exc)})
                    window_scores.append(float("-inf"))
                    continue
                payload = result.get("payload") or {}
                run = payload.get("run") or {}
                window_runs.append(run)
                window_scores.append(_objective_value(run, objective))
            passed = _screening_pass(window_scores, screening_cfg)
            score = _aggregate_scores(window_scores, screening_cfg.get("aggregate", "mean"))
            if not passed:
                score = float("-inf")
            payload = {"runs": window_runs}
            run = {"window_scores": window_scores, "screening_failed": (not passed)}
        else:
            try:
                result = strategy_runner.run_backtest_with_payload(trial_spec)
            except Exception as exc:
                if debug_cfg:
                    _write_debug_failure(
                        out_dir,
                        trial_id,
                        trial_params,
                        error=f"trial_failure: {exc}",
                        debug_cfg=debug_cfg,
                    )
                trials.append(
                    {
                        "trial_id": trial_id,
                        "params": trial_params,
                        "objective": float("-inf"),
                        "window_scores": None,
                        "metadata": base_meta,
                    }
                )
                continue
            payload = result.get("payload") or {}
            run = payload.get("run") or {}
            score = _objective_value(run, objective)
        if not math.isfinite(score):
            if debug_cfg:
                _write_debug_failure(
                    out_dir,
                    trial_id,
                    trial_params,
                    error="non_finite_objective",
                    payload=payload,
                    debug_cfg=debug_cfg,
                )
            score = float("-inf")
        trials.append(
            {
                "trial_id": trial_id,
                "params": trial_params,
                "objective": score,
                "window_scores": window_scores or None,
                "metadata": base_meta,
            }
        )
        if best is None or score > best["objective"]:
            best = {"params": trial_params, "objective": score, "run": run}
        behavior_sig = None
        if behavior_mode == "trades_hist":
            trades = payload.get("trades")
            if isinstance(trades, list):
                behavior_sig = _trade_hist_signature(trades, behavior_bins)
        if behavior_mode == "equity_signature":
            trades = payload.get("trades")
            if isinstance(trades, list):
                behavior_sig = _equity_signature(trades, behavior_points)
        if _meets_constraints(run, promotion_cfg):
            _update_topk(
                promoted,
                {
                    "trial_id": trial_id,
                    "params": trial_params,
                    "objective": score,
                    "payload": payload,
                    "run": run,
                    "behavior_sig": behavior_sig,
                },
                top_k=top_k,
                dedupe_distance=dedupe_distance,
                bounds=bounds,
                behavior_distance=behavior_distance,
                behavior_metrics=behavior_metrics,
                behavior_bounds=behavior_bounds,
                behavior_mode=behavior_mode,
                behavior_bins=behavior_bins,
            )

    artifacts.write_trials(out_dir / "trials.json", trials)
    promoted = _cluster_promoted(
        promoted,
        behavior_metrics=behavior_metrics,
        behavior_bounds=behavior_bounds,
        cluster_cfg=cluster_cfg,
    )
    promoted_records = _write_promoted(
        out_dir,
        promoted,
        artifacts_cfg=artifacts_cfg,
        promotion_cfg=promotion_cfg,
    )
    refine_records: List[Dict[str, Any]] = []
    if refine_cfg.get("enabled"):
        refine_dir = out_dir / "refine"
        refine_dir.mkdir(parents=True, exist_ok=True)
        top_k = int(refine_cfg.get("top_k", 5))
        shrink_pct = float(refine_cfg.get("shrink_pct", 0.5))
        sorted_trials = sorted(trials, key=lambda t: t.get("objective", float("-inf")), reverse=True)
        top_trials = sorted_trials[:top_k]
        freeze_keys = refine_cfg.get("freeze_keys") or []
        if not isinstance(freeze_keys, list):
            freeze_keys = []
        freeze_prefixes = refine_cfg.get("freeze_prefixes") or []
        if not isinstance(freeze_prefixes, list):
            freeze_prefixes = []
        refined_space = _refine_search_space(
            cfg.get("search_space") or {},
            top_trials,
            shrink_pct=shrink_pct,
            top_k=top_k,
            freeze_keys=freeze_keys,
            freeze_prefixes=freeze_prefixes,
        )
        refine_spec = deepcopy(spec)
        refine_spec.setdefault("optimization", {})["search_space"] = refined_space
        refine_spec["optimization"].setdefault("refine", {})["enabled"] = False
        refine_result = run_strategy_optimization(refine_spec, out_dir=refine_dir)
        refine_records.append(
            {
                "dir": str(refine_dir),
                "top_k": top_k,
                "shrink_pct": shrink_pct,
                "trials_path": refine_result.get("trials_path"),
                "summary": refine_result.get("summary"),
            }
        )
    full_records: List[Dict[str, Any]] = []
    if full_cfg.get("enabled"):
        full_records = _run_full_pass(
            spec,
            promoted_records,
            out_dir=out_dir,
            runner_fn=strategy_runner.run_backtest_with_payload,
            artifacts_cfg=full_cfg.get("artifacts") if isinstance(full_cfg.get("artifacts"), Mapping) else {},
            full_cfg=full_cfg,
            promotion_cfg=promotion_cfg,
            objective=objective,
        )
    _log_storage_impact(total_trials, len(promoted_records), len(full_records))
    sensitivity_cfg = cfg.get("sensitivity") or {}
    sensitivity = None
    if isinstance(sensitivity_cfg, Mapping) and sensitivity_cfg.get("enabled"):
        top_k = sensitivity_cfg.get("top_k", 20)
        try:
            top_k = int(top_k)
        except Exception:
            top_k = 20
        sensitivity = _sensitivity_from_trials(trials, top_k=top_k)
    summary = {
        "objective": objective,
        "best": best,
        "total_trials": total_trials,
        "promoted": promoted_records,
        "refine": refine_records,
        "full_pass": full_records,
        "metadata": base_meta,
        "sensitivity": sensitivity,
    }
    artifacts.write_summary(out_dir / "summary.json", summary)
    _apply_retention(out_dir, cfg)
    return {
        "trials_path": str(out_dir / "trials.json"),
        "summary": str(out_dir / "summary.json"),
        "best": best,
        "total_trials": total_trials,
        "promoted": promoted_records,
        "refine": refine_records,
        "full_pass": full_records,
        "metadata": base_meta,
    }


def _get_path_value(spec: Mapping[str, Any], path: str) -> Any:
    tokens = _parse_path(path)
    ref: Any = spec
    for token in tokens:
        ref = ref[token]
    return ref


__all__ = ["run_backtest_optimization", "run_strategy_optimization"]
