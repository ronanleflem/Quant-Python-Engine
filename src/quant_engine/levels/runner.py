"""End-to-end orchestration for levels build operations."""
from __future__ import annotations

from typing import Dict

from ..core.dataset import load_ohlcv
from .builders import build_levels
from .repo import ensure_table, get_engine, upsert_levels
from .schemas import LevelsBuildSpec


def run_levels_build(spec: LevelsBuildSpec) -> Dict[str, object]:
    """Load data, compute requested levels and persist them."""

    ohlcv = load_ohlcv(spec.data)
    records = build_levels(spec, ohlcv)
    table_fqn = f"{spec.output_schema}.{spec.output_table}" if spec.output_schema else spec.output_table
    engine = get_engine()
    ensure_table(engine, table_fqn)
    result = upsert_levels(engine, table_fqn, records) if spec.upsert else {"inserted": 0, "updated": 0}
    payload = {
        "table": table_fqn,
        "count": len(records),
        **result,
    }
    return payload


__all__ = ["run_levels_build"]
