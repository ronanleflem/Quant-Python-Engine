"""Execution utilities for :mod:`quant_engine.stats`."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

try:  # pragma: no cover - optional dependency
    import polars as pl
except Exception:  # pragma: no cover
    pl = None

from ..api.schemas import StatsSpec


def run_stats(spec: StatsSpec) -> Dict[str, Any]:
    """Run a statistics specification.

    The current implementation is a synchronous placeholder that writes an
    empty ``stats_summary.parquet`` file and returns its path.
    """

    out_dir = Path(spec.artifacts.out_dir) if spec.artifacts and spec.artifacts.out_dir else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "stats_summary.parquet"
    if pl is not None:
        pl.DataFrame().write_parquet(summary_path)
    else:
        summary_path.touch()
    return {"summary_path": str(summary_path)}


__all__ = ["run_stats"]

