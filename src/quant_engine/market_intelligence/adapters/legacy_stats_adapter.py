"""Adapter from legacy market stats outputs to normalized dataframes."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

_DEFAULT_COLUMNS = [
    "symbol",
    "event",
    "condition_name",
    "condition_value",
    "target",
    "n",
    "successes",
    "p_hat",
    "ci_low",
    "ci_high",
    "p_mean",
    "p_map",
    "hdi_low",
    "hdi_high",
    "lift_freq",
    "lift_bayes",
    "insufficient",
    "split",
    "p_value",
    "q_value",
    "significant",
]

_RENAME_MAP = {
    "p": "p_hat",
    "probability": "p_hat",
    "count": "n",
    "wins": "successes",
}


def _default_runner(spec: Any) -> pd.DataFrame:
    from quant_engine.stats.runner import run_stats

    return run_stats(spec)


class LegacyStatsAdapter:
    """Execute legacy stats runner and normalize output for market-intelligence consumers."""

    def __init__(self, runner: Callable[[Any], pd.DataFrame] = _default_runner) -> None:
        self._runner = runner

    def run(self, spec: Any) -> pd.DataFrame:
        """Run legacy stats and return a UTC-indexed dataframe with canonical columns."""
        out = self._runner(spec)
        df = out.copy() if isinstance(out, pd.DataFrame) else pd.DataFrame(out)

        if "ts" in df.columns:
            idx = pd.to_datetime(df["ts"], utc=True)
        elif isinstance(df.index, pd.DatetimeIndex):
            idx = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
        else:
            idx = pd.DatetimeIndex([pd.Timestamp("1970-01-01", tz="UTC")] * len(df))

        df = df.rename(columns=_RENAME_MAP)
        df = df.reindex(columns=_DEFAULT_COLUMNS, fill_value=pd.NA)
        df.index = idx
        df.index.name = "ts"

        for numeric_col in (
            "n",
            "successes",
            "p_hat",
            "ci_low",
            "ci_high",
            "p_mean",
            "p_map",
            "hdi_low",
            "hdi_high",
            "lift_freq",
            "lift_bayes",
            "p_value",
            "q_value",
        ):
            df[numeric_col] = pd.to_numeric(df[numeric_col], errors="coerce")

        for bool_col in ("insufficient", "significant"):
            bool_series = df[bool_col].astype("boolean")
            df[bool_col] = bool_series.fillna(False).astype(bool)

        return df


__all__ = ["LegacyStatsAdapter"]
