"""Utilities to persist backtest and statistics results."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def _json_default(obj: Any) -> Any:
    if isinstance(obj, complex):
        return {"real": obj.real, "imag": obj.imag}
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return str(obj)


def _write_rows(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    Path(path).write_text(json.dumps(rows, default=_json_default))


def write_trials(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    _write_rows(path, rows)


def write_trades(path: str | Path, trades: List[Dict[str, Any]]) -> None:
    _write_rows(path, trades)


def write_equity(path: str | Path, equity: List[float]) -> None:
    rows = [{"equity": v} for v in equity]
    _write_rows(path, rows)


def write_summary(path: str | Path, summary: Dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(summary, indent=2, default=_json_default))


def write_stats_summary(path: str | Path, df: pd.DataFrame) -> None:
    """Persist aggregate statistics to a Parquet file."""

    if not df.empty:
        df = df.copy()
        for col in ("n", "successes"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")
    df.to_parquet(path, index=False)


def write_stats_details(path: str | Path, df: pd.DataFrame) -> None:
    """Persist detailed statistics to a Parquet file.

    Placeholder for future extensions (e.g. time to reversal).
    """

    df.to_parquet(path, index=False)

