"""Utilities to persist backtest and statistics results."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def _write_rows(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    Path(path).write_text(json.dumps(rows))


def write_trials(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    _write_rows(path, rows)


def write_trades(path: str | Path, trades: List[Dict[str, Any]]) -> None:
    _write_rows(path, trades)


def write_equity(path: str | Path, equity: List[float]) -> None:
    rows = [{"equity": v} for v in equity]
    _write_rows(path, rows)


def write_summary(path: str | Path, summary: Dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(summary, indent=2))


def write_stats_summary(path: str | Path, df: pd.DataFrame) -> None:
    """Persist aggregate statistics to a Parquet file."""

    df.to_parquet(path, index=False)


def write_stats_details(path: str | Path, df: pd.DataFrame) -> None:
    """Persist detailed statistics to a Parquet file.

    Placeholder for future extensions (e.g. time to reversal).
    """

    df.to_parquet(path, index=False)

