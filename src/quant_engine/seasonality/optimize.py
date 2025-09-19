"""Optimization utilities for seasonality parameter tuning."""

import optuna  # noqa: F401


def suggest_params(trial, spec):
    """Suggest parameters for seasonality optimization.

    Args:
        trial: An Optuna trial object providing the suggestion interface.
        spec: Seasonality specification guiding the search space.

    Returns:
        A dictionary of suggested parameter values for the optimization.
    """

    dims = trial.suggest_categorical(
        "dims", [["hour"], ["dow"], ["hour", "dow"]]
    )

    method = trial.suggest_categorical("method", ["threshold", "topk"])
    params = {
        "dims": dims,
        "method": method,
        "combine": trial.suggest_categorical("combine", ["and", "or"]),
        "ret_horizon": trial.suggest_int("ret_horizon", 1, 3),
        "min_samples_bin": trial.suggest_int("min_samples_bin", 200, 700),
    }

    if method == "threshold":
        params["threshold"] = trial.suggest_float("threshold", 0.52, 0.58)
    else:
        params["topk"] = trial.suggest_int("topk", 1, 6)

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

    pass
