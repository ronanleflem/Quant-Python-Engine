"""Utilities to persist backtest results."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any


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

