"""Optimization utilities for backtest and strategy specs."""
from __future__ import annotations

import json
import math
import random
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from ..io import artifacts
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


def run_backtest_optimization(spec: Mapping[str, Any], *, out_dir: Optional[str | Path] = None) -> Dict[str, Any]:
    cfg = _optimization_config(spec)
    space = cfg.get("search_space") or {}
    if not isinstance(space, Mapping):
        raise ValueError("optimization.search_space must be a mapping")
    keys, values = _expand_search_space(space)
    method = str(cfg.get("method", "grid")).lower()
    max_trials = cfg.get("max_trials")
    seed = cfg.get("seed")
    objective = str(cfg.get("objective", "sharpe"))

    out_dir = Path(out_dir or cfg.get("out_dir") or "runs/optimize_backtest")
    out_dir.mkdir(parents=True, exist_ok=True)

    trials: List[Dict[str, Any]] = []
    best: Dict[str, Any] | None = None

    for trial_spec in _trial_specs(spec, keys, values, method, max_trials, seed):
        result = backtest_runner.run_backtest_from_spec(trial_spec)
        payload = result.get("payload") or {}
        run = payload.get("run") or {}
        score = _objective_value(run, objective)
        trial_params = {k: _get_path_value(trial_spec, k) for k in keys}
        trials.append({"params": trial_params, "objective": score})
        if best is None or score > best["objective"]:
            best = {"params": trial_params, "objective": score, "run": run}

    artifacts.write_trials(out_dir / "trials.json", trials)
    summary = {"objective": objective, "best": best}
    artifacts.write_summary(out_dir / "summary.json", summary)
    return {"trials_path": str(out_dir / "trials.json"), "summary": str(out_dir / "summary.json"), "best": best}


def run_strategy_optimization(spec: Mapping[str, Any], *, out_dir: Optional[str | Path] = None) -> Dict[str, Any]:
    cfg = _optimization_config(spec)
    space = cfg.get("search_space") or {}
    if not isinstance(space, Mapping):
        raise ValueError("optimization.search_space must be a mapping")
    keys, values = _expand_search_space(space)
    method = str(cfg.get("method", "grid")).lower()
    max_trials = cfg.get("max_trials")
    seed = cfg.get("seed")
    objective = str(cfg.get("objective", "sharpe"))

    out_dir = Path(out_dir or cfg.get("out_dir") or "runs/optimize_strategy")
    out_dir.mkdir(parents=True, exist_ok=True)

    trials: List[Dict[str, Any]] = []
    best: Dict[str, Any] | None = None

    for trial_spec in _trial_specs(spec, keys, values, method, max_trials, seed):
        result = strategy_runner.run_backtest_with_payload(trial_spec)
        payload = result.get("payload") or {}
        run = payload.get("run") or {}
        score = _objective_value(run, objective)
        trial_params = {k: _get_path_value(trial_spec, k) for k in keys}
        trials.append({"params": trial_params, "objective": score})
        if best is None or score > best["objective"]:
            best = {"params": trial_params, "objective": score, "run": run}

    artifacts.write_trials(out_dir / "trials.json", trials)
    summary = {"objective": objective, "best": best}
    artifacts.write_summary(out_dir / "summary.json", summary)
    return {"trials_path": str(out_dir / "trials.json"), "summary": str(out_dir / "summary.json"), "best": best}


def _get_path_value(spec: Mapping[str, Any], path: str) -> Any:
    tokens = _parse_path(path)
    ref: Any = spec
    for token in tokens:
        ref = ref[token]
    return ref


__all__ = ["run_backtest_optimization", "run_strategy_optimization"]
