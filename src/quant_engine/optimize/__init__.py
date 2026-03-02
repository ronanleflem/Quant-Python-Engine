from __future__ import annotations


def run_backtest_optimization(*args, **kwargs):
    from .variants import run_backtest_optimization as _impl

    return _impl(*args, **kwargs)


def run_strategy_optimization(*args, **kwargs):
    from .variants import run_strategy_optimization as _impl

    return _impl(*args, **kwargs)


__all__ = ["run_backtest_optimization", "run_strategy_optimization"]
