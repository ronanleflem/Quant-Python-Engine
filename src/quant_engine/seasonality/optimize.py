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

    pass


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
