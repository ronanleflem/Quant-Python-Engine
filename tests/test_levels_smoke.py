"""Smoke tests for the levels module."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from quant_engine.levels.runner import run_levels_build
from quant_engine.levels.schemas import LevelsBuildSpec


@pytest.mark.skipif(
    "QE_MARKETDATA_MYSQL_URL" not in os.environ,
    reason="Requires QE_MARKETDATA_MYSQL_URL for persistence",
)
def test_levels_build_smoke(tmp_path: Path) -> None:
    ts = pd.date_range("2024-01-01", periods=500, freq="H", tz="UTC")
    prices = pd.Series(1.05 + 0.001 * (pd.Series(range(len(ts))) % 24))
    df = pd.DataFrame(
        {
            "ts": ts,
            "symbol": "EURUSD",
            "open": prices,
            "high": prices + 0.0005,
            "low": prices - 0.0005,
            "close": prices + 0.0001,
            "volume": 1000,
        }
    )
    csv_path = tmp_path / "ohlcv.csv"
    df.to_csv(csv_path, index=False)

    spec_payload = {
        "data": {
            "dataset_path": str(csv_path),
            "symbols": ["EURUSD"],
            "timeframe": "H1",
            "start": ts.min().isoformat(),
            "end": ts.max().isoformat(),
        },
        "symbols": ["EURUSD"],
        "range_start": ts.min().isoformat(),
        "range_end": ts.max().isoformat(),
        "targets": [
            {"type": "PDH", "params": {}},
            {"type": "PDL", "params": {}},
            {"type": "GAP_D", "params": {}},
            {"type": "FVG", "params": {"min_size_ticks": 1}},
        ],
    }
    spec = LevelsBuildSpec.model_validate(spec_payload)
    result = run_levels_build(spec)
    assert result["count"] > 0
