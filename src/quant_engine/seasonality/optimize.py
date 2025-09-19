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

    pass


def run_optimization(spec):
    """Run the Optuna optimization process for seasonality tuning.

    Args:
        spec: Seasonality specification containing optimization settings.

    Returns:
        The best set of parameters discovered by the optimization process.
    """

    pass
