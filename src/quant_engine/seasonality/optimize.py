"""Optimization utilities for seasonality parameter tuning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

try:  # pragma: no cover - optional dependency during lightweight tests
    import optuna
except ModuleNotFoundError:  # pragma: no cover - exercised when optuna missing
    optuna = None  # type: ignore
import pandas as pd


DEFAULT_SEARCH_SPACE: Dict[str, Any] = {
    "dims": [["hour"], ["dow"], ["hour", "dow"]],
    "method": ["threshold", "topk"],
    "combine": ["and", "or"],
    "ret_horizon": {"low": 1, "high": 3},
    "min_samples_bin": {"low": 200, "high": 700},
    "threshold": {"low": 0.52, "high": 0.58},
    "topk": {"low": 1, "high": 6},
}


def _get_search_space(spec: Any) -> Dict[str, Any]:
    compute = getattr(spec, "compute", None)
    if compute is None:
        return {}
    if isinstance(compute, Mapping):
        return dict(compute.get("search_space", {}) or {})
    space = getattr(compute, "search_space", None)
    if space is None and hasattr(compute, "model_dump"):
        space = compute.model_dump().get("search_space")
    if isinstance(space, Mapping):
        return dict(space)
    return {}


def _normalise_choices(values: Any, default: Sequence[Any]) -> list[Any]:
    if isinstance(values, Mapping):
        for key in ("values", "choices", "options"):
            if key in values:
                values = values[key]
                break
        else:
            if "value" in values:
                values = [values["value"]]
    if isinstance(values, (list, tuple, set)):
        choices = list(values)
    elif values is None:
        choices = []
    else:
        choices = [values]
    if not choices:
        choices = list(default)
    return choices


def _normalise_dims(space: Dict[str, Any]) -> list[list[str]]:
    options = _normalise_choices(space.get("dims"), DEFAULT_SEARCH_SPACE["dims"])
    dims: list[list[str]] = []
    for opt in options:
        if isinstance(opt, str):
            dims.append([opt])
        elif isinstance(opt, Sequence):
            dims.append([str(v) for v in opt])
    if not dims:
        dims = [list(opt) for opt in DEFAULT_SEARCH_SPACE["dims"]]
    return dims


def _normalise_numeric(
    value: Any, default_low: float, default_high: float, cast_type
) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        for key in ("values", "choices", "options"):
            if key in value:
                vals = value[key]
                return {
                    "mode": "categorical",
                    "choices": [cast_type(v) for v in list(vals)],
                }
        low = value.get("low", value.get("min", default_low))
        high = value.get("high", value.get("max", default_high))
        step = value.get("step")
        return {
            "mode": "range",
            "low": cast_type(low),
            "high": cast_type(high),
            "step": cast_type(step) if step is not None else None,
        }
    if isinstance(value, (list, tuple, set)):
        return {"mode": "categorical", "choices": [cast_type(v) for v in value]}
    if value is None:
        return {
            "mode": "range",
            "low": cast_type(default_low),
            "high": cast_type(default_high),
            "step": None,
        }
    return {"mode": "categorical", "choices": [cast_type(value)]}


def _suggest_int(trial: optuna.trial.Trial, name: str, config: Dict[str, Any]) -> int:
    if config["mode"] == "categorical":
        choices = [int(v) for v in config["choices"]]
        if len(choices) == 1:
            return choices[0]
        return int(trial.suggest_categorical(name, choices))
    low = int(config["low"])
    high = int(config["high"])
    step = config.get("step")
    if low == high:
        return low
    if step is not None and int(step) > 0:
        return int(trial.suggest_int(name, low, high, step=int(step)))
    return int(trial.suggest_int(name, low, high))


def _suggest_float(trial: optuna.trial.Trial, name: str, config: Dict[str, Any]) -> float:
    if config["mode"] == "categorical":
        choices = [float(v) for v in config["choices"]]
        if len(choices) == 1:
            return choices[0]
        return float(trial.suggest_categorical(name, choices))
    low = float(config["low"])
    high = float(config["high"])
    step = config.get("step")
    if low == high:
        return low
    if step is not None:
        return float(trial.suggest_float(name, low, high, step=float(step)))
    return float(trial.suggest_float(name, low, high))


def _ensure_serialisable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _ensure_serialisable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_ensure_serialisable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # pragma: no cover - fallback for exotic scalars
            return str(value)
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:  # pragma: no cover - fallback for exotic arrays
            return str(value)
    return value


def suggest_params(trial, spec):
    """Suggest parameters for seasonality optimization."""

    space = _get_search_space(spec)
    dims_options = _normalise_dims(space)
    if len(dims_options) == 1:
        dims = dims_options[0]
    else:
        dims = trial.suggest_categorical("dims", dims_options)

    method_options = _normalise_choices(space.get("method"), DEFAULT_SEARCH_SPACE["method"])
    if len(method_options) == 1:
        method = method_options[0]
    else:
        method = trial.suggest_categorical("method", method_options)

    combine_options = _normalise_choices(space.get("combine"), DEFAULT_SEARCH_SPACE["combine"])
    if len(combine_options) == 1:
        combine = combine_options[0]
    else:
        combine = trial.suggest_categorical("combine", combine_options)

    ret_cfg = _normalise_numeric(space.get("ret_horizon"), 1, 3, int)
    min_samples_cfg = _normalise_numeric(space.get("min_samples_bin"), 200, 700, int)

    params = {
        "dims": dims,
        "method": method,
        "combine": combine,
        "ret_horizon": _suggest_int(trial, "ret_horizon", ret_cfg),
        "min_samples_bin": _suggest_int(trial, "min_samples_bin", min_samples_cfg),
    }

    if method == "threshold":
        threshold_cfg = _normalise_numeric(space.get("threshold"), 0.52, 0.58, float)
        params["threshold"] = _suggest_float(trial, "threshold", threshold_cfg)
    else:
        topk_cfg = _normalise_numeric(space.get("topk"), 1, 6, int)
        params["topk"] = _suggest_int(trial, "topk", topk_cfg)

    return params


def evaluate_params(params, spec):
    """Evaluate a set of seasonality parameters.

    Args:
        params: Dictionary of seasonality parameters to evaluate.
        spec: Seasonality specification defining evaluation context.

    Returns:
        A numeric score indicating the quality of the provided parameters.
    """

    # Create an isolated copy of the base specification so the caller's
    # instance is not mutated during the evaluation process.
    cfg = spec.model_copy(deep=True)

    profile_updates = {}
    if "ret_horizon" in params:
        profile_updates["ret_horizon"] = params["ret_horizon"]
    if "min_samples_bin" in params:
        profile_updates["min_samples_bin"] = params["min_samples_bin"]
    if profile_updates:
        cfg.profile = cfg.profile.model_copy(update=profile_updates)

    signal_updates = {}
    if "method" in params:
        signal_updates["method"] = params["method"]
    if "dims" in params:
        signal_updates["dims"] = params["dims"]
    if "combine" in params:
        signal_updates["combine"] = params["combine"]

    method = params.get("method")
    if method == "threshold" and "threshold" in params:
        signal_updates["threshold"] = params["threshold"]
    if method == "topk" and "topk" in params:
        signal_updates["topk"] = params["topk"]

    if signal_updates:
        cfg.signal = cfg.signal.model_copy(update=signal_updates)

    from . import runner

    result = runner.run(cfg)
    metrics = dict(result.get("best_metrics", {}) or {})

    n_trades = int(metrics.get("n_trades", metrics.get("trades", 0)) or 0)
    if n_trades < 30:
        objective = -1e9
    else:
        objective = float(metrics.get("sharpe", 0.0) or 0.0)

    return {
        "objective": objective,
        "metrics": metrics,
        "params": params,
    }


def run_optimization(spec):
    """Run the Optuna optimization process for seasonality tuning.

    Args:
        spec: Seasonality specification containing optimization settings.

    Returns:
        The best set of parameters discovered by the optimization process.
    """
    if optuna is None:  # pragma: no cover - exercised when optuna missing
        raise RuntimeError("optuna is required for seasonality optimisation")

    study = optuna.create_study(direction="maximize")

    if hasattr(spec, "model_copy"):
        base_spec = spec.model_copy(deep=True)
    else:  # pragma: no cover - defensive for alternative spec objects
        base_spec = spec

    compute_cfg = getattr(base_spec, "compute", None)
    max_trials = getattr(compute_cfg, "max_trials", None)
    try:
        n_trials = int(max_trials) if max_trials is not None else 30
    except (TypeError, ValueError):  # pragma: no cover - defensive parsing
        n_trials = 30
    if n_trials <= 0:
        n_trials = 30

    artifacts_cfg = getattr(base_spec, "artifacts", None)
    out_dir_value = getattr(artifacts_cfg, "out_dir", None) if artifacts_cfg else None
    out_dir = Path(out_dir_value) if out_dir_value else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    trials_path = out_dir / "seasonality_trials.parquet"
    summary_path = out_dir / "summary.json"

    def _objective(trial: optuna.trial.Trial) -> float:
        params = suggest_params(trial, base_spec)
        evaluation = evaluate_params(params, base_spec)
        metrics = _ensure_serialisable(evaluation.get("metrics", {}))
        params_clean = _ensure_serialisable(evaluation.get("params", params))
        trial.set_user_attr("metrics", metrics)
        trial.set_user_attr("params", params_clean)
        return float(evaluation.get("objective", 0.0))

    study.optimize(_objective, n_trials=n_trials)

    rows: list[Dict[str, Any]] = []
    for trial in study.trials:
        params = trial.user_attrs.get("params", {})
        metrics = trial.user_attrs.get("metrics", {})
        rows.append(
            {
                "trial_number": trial.number,
                "state": trial.state.name,
                "objective": trial.value,
                "params_json": json.dumps(_ensure_serialisable(params)),
                "metrics_json": json.dumps(_ensure_serialisable(metrics)),
            }
        )

    columns = ["trial_number", "state", "objective", "params_json", "metrics_json"]
    trials_df = pd.DataFrame(rows, columns=columns) if rows else pd.DataFrame(columns=columns)
    trials_df.to_parquet(trials_path, index=False)

    try:
        best_trial = study.best_trial
    except ValueError:  # pragma: no cover - occurs if optimisation yields no trials
        best_trial = None

    best_params = _ensure_serialisable(best_trial.user_attrs.get("params")) if best_trial else None
    best_metrics = _ensure_serialisable(best_trial.user_attrs.get("metrics")) if best_trial else None
    best_value = best_trial.value if best_trial is not None else None

    summary_payload = {
        "best_value": _ensure_serialisable(best_value),
        "best_params": best_params,
        "best_metrics": best_metrics,
        "n_trials": len(study.trials),
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2))

    return {
        "best_value": best_value,
        "best_params": best_params,
        "best_metrics": best_metrics,
        "paths": {"trials": str(trials_path), "summary": str(summary_path)},
    }
