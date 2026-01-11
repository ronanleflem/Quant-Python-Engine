"""Optimization utilities for backtest and strategy specs."""
from __future__ import annotations

import json
import logging
import math
import random
from copy import deepcopy
from itertools import product
from pathlib import Path
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


def _numeric_bounds(values: List[Any]) -> Optional[Tuple[float, float]]:
    nums: List[float] = []
    for val in values:
        if isinstance(val, (int, float)):
            nums.append(float(val))
    if not nums:
        return None
    return min(nums), max(nums)


def _objective_value(run: Mapping[str, Any], objective: str) -> float:
    mapping = {
        "sharpe": "sharpe",
        "sortino": "sortino",
        "return_pct": "returnPct",
        "max_drawdown_pct": "maxDrawdownPct",
        "total_return": "totalReturn",
        "winrate_pct": "winratePct",
    }
    key = mapping.get(objective, objective)
    val = run.get(key)
    try:
        return float(val)
    except Exception:
        return float("-inf")


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


def _promotion_config(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    raw = cfg.get("promotion") or {}
    if not isinstance(raw, Mapping):
        raise ValueError("optimization.promotion must be a mapping")
    return dict(raw)


def _meets_constraints(
    run: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> bool:
    if not cfg:
        return True
    min_trades = cfg.get("min_trades")
    if min_trades is not None:
        win = run.get("winCount") or 0
        loss = run.get("lossCount") or 0
        try:
            trades = int(win) + int(loss)
        except Exception:
            trades = 0
        if trades < int(min_trades):
            return False
    max_dd = cfg.get("max_drawdown_pct")
    if max_dd is not None:
        try:
            dd = float(run.get("maxDrawdownPct"))
        except Exception:
            return False
        if dd > float(max_dd):
            return False
    min_winrate = cfg.get("min_winrate_pct")
    if min_winrate is not None:
        try:
            winrate = float(run.get("winratePct"))
        except Exception:
            return False
        if winrate < float(min_winrate):
            return False
    min_return = cfg.get("min_return_pct")
    if min_return is not None:
        try:
            ret = float(run.get("returnPct"))
        except Exception:
            return False
        if ret < float(min_return):
            return False
    min_sharpe = cfg.get("min_sharpe")
    if min_sharpe is not None:
        try:
            sharpe = float(run.get("sharpe"))
        except Exception:
            return False
        if sharpe < float(min_sharpe):
            return False
    min_sortino = cfg.get("min_sortino")
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


def _update_topk(
    topk: List[Dict[str, Any]],
    candidate: Dict[str, Any],
    *,
    top_k: int,
    dedupe_distance: Optional[float],
    bounds: Mapping[str, Tuple[float, float]],
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
                topk[closest_idx] = candidate
            return
    topk.append(candidate)
    topk.sort(key=lambda item: item.get("objective", float("-inf")), reverse=True)
    if len(topk) > top_k:
        topk.pop()


def _write_promoted(
    out_dir: Path,
    promoted: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not promoted:
        return []
    promoted_dir = out_dir / "promoted"
    promoted_dir.mkdir(parents=True, exist_ok=True)
    stored: List[Dict[str, Any]] = []
    for entry in promoted:
        trial_id = entry.get("trial_id", "unknown")
        payload = entry.get("payload") or {}
        path = promoted_dir / f"trial_{trial_id}.json"
        path.write_text(json.dumps(payload, indent=2, default=str))
        stored.append(
            {
                "trial_id": trial_id,
                "objective": entry.get("objective"),
                "params": entry.get("params"),
                "path": str(path),
            }
        )
    return stored


def run_backtest_optimization(spec: Mapping[str, Any], *, out_dir: Optional[str | Path] = None) -> Dict[str, Any]:
    cfg = _optimization_config(spec)
    promotion_cfg = _promotion_config(cfg)
    space = cfg.get("search_space") or {}
    if not isinstance(space, Mapping):
        raise ValueError("optimization.search_space must be a mapping")
    keys, values = _expand_search_space(space)
    bounds: Dict[str, Tuple[float, float]] = {}
    for key, vals in zip(keys, values):
        numeric = _numeric_bounds(vals)
        if numeric is not None:
            bounds[key] = numeric
    method = str(cfg.get("method", "grid")).lower()
    max_trials = cfg.get("max_trials")
    seed = cfg.get("seed")
    objective = str(cfg.get("objective", "sharpe"))
    total_trials = _count_trials(values, method, max_trials)
    LOGGER.info("Optimization trials planned: %d", total_trials)
    top_k = int(promotion_cfg.get("top_k", 0) or 0)
    dedupe_distance = promotion_cfg.get("dedupe_distance")

    out_dir = Path(out_dir or cfg.get("out_dir") or "runs/optimize_backtest")
    out_dir.mkdir(parents=True, exist_ok=True)

    trials: List[Dict[str, Any]] = []
    best: Dict[str, Any] | None = None
    promoted: List[Dict[str, Any]] = []
    trial_id = 0

    for trial_spec in _trial_specs(spec, keys, values, method, max_trials, seed):
        trial_id += 1
        result = backtest_runner.run_backtest_from_spec(trial_spec)
        payload = result.get("payload") or {}
        run = payload.get("run") or {}
        score = _objective_value(run, objective)
        trial_params = {k: _get_path_value(trial_spec, k) for k in keys}
        trials.append({"trial_id": trial_id, "params": trial_params, "objective": score})
        if best is None or score > best["objective"]:
            best = {"params": trial_params, "objective": score, "run": run}
        if _meets_constraints(run, promotion_cfg):
            _update_topk(
                promoted,
                {"trial_id": trial_id, "params": trial_params, "objective": score, "payload": payload},
                top_k=top_k,
                dedupe_distance=dedupe_distance,
                bounds=bounds,
            )

    artifacts.write_trials(out_dir / "trials.json", trials)
    promoted_records = _write_promoted(out_dir, promoted)
    summary = {
        "objective": objective,
        "best": best,
        "total_trials": total_trials,
        "promoted": promoted_records,
    }
    artifacts.write_summary(out_dir / "summary.json", summary)
    return {
        "trials_path": str(out_dir / "trials.json"),
        "summary": str(out_dir / "summary.json"),
        "best": best,
        "total_trials": total_trials,
        "promoted": promoted_records,
    }


def run_strategy_optimization(spec: Mapping[str, Any], *, out_dir: Optional[str | Path] = None) -> Dict[str, Any]:
    cfg = _optimization_config(spec)
    promotion_cfg = _promotion_config(cfg)
    space = cfg.get("search_space") or {}
    if not isinstance(space, Mapping):
        raise ValueError("optimization.search_space must be a mapping")
    keys, values = _expand_search_space(space)
    bounds: Dict[str, Tuple[float, float]] = {}
    for key, vals in zip(keys, values):
        numeric = _numeric_bounds(vals)
        if numeric is not None:
            bounds[key] = numeric
    method = str(cfg.get("method", "grid")).lower()
    max_trials = cfg.get("max_trials")
    seed = cfg.get("seed")
    objective = str(cfg.get("objective", "sharpe"))
    total_trials = _count_trials(values, method, max_trials)
    LOGGER.info("Optimization trials planned: %d", total_trials)
    top_k = int(promotion_cfg.get("top_k", 0) or 0)
    dedupe_distance = promotion_cfg.get("dedupe_distance")

    out_dir = Path(out_dir or cfg.get("out_dir") or "runs/optimize_strategy")
    out_dir.mkdir(parents=True, exist_ok=True)

    trials: List[Dict[str, Any]] = []
    best: Dict[str, Any] | None = None
    promoted: List[Dict[str, Any]] = []
    trial_id = 0

    for trial_spec in _trial_specs(spec, keys, values, method, max_trials, seed):
        trial_id += 1
        result = strategy_runner.run_backtest_with_payload(trial_spec)
        payload = result.get("payload") or {}
        run = payload.get("run") or {}
        score = _objective_value(run, objective)
        trial_params = {k: _get_path_value(trial_spec, k) for k in keys}
        trials.append({"trial_id": trial_id, "params": trial_params, "objective": score})
        if best is None or score > best["objective"]:
            best = {"params": trial_params, "objective": score, "run": run}
        if _meets_constraints(run, promotion_cfg):
            _update_topk(
                promoted,
                {"trial_id": trial_id, "params": trial_params, "objective": score, "payload": payload},
                top_k=top_k,
                dedupe_distance=dedupe_distance,
                bounds=bounds,
            )

    artifacts.write_trials(out_dir / "trials.json", trials)
    promoted_records = _write_promoted(out_dir, promoted)
    summary = {
        "objective": objective,
        "best": best,
        "total_trials": total_trials,
        "promoted": promoted_records,
    }
    artifacts.write_summary(out_dir / "summary.json", summary)
    return {
        "trials_path": str(out_dir / "trials.json"),
        "summary": str(out_dir / "summary.json"),
        "best": best,
        "total_trials": total_trials,
        "promoted": promoted_records,
    }


def _get_path_value(spec: Mapping[str, Any], path: str) -> Any:
    tokens = _parse_path(path)
    ref: Any = spec
    for token in tokens:
        ref = ref[token]
    return ref


__all__ = ["run_backtest_optimization", "run_strategy_optimization"]
