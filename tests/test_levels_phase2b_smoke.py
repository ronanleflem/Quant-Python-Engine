"""Smoke tests for the Phase 2B level detectors."""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant_engine.levels.builders import build_levels
from quant_engine.levels.schemas import LevelsBuildSpec


def test_levels_phase2b_smoke(tmp_path) -> None:
    ts = pd.date_range("2024-01-01", periods=96, freq="1h", tz="UTC")
    base = 1.10 + 0.0003 * np.arange(len(ts))
    close_vals = base.copy()
    open_vals = close_vals - 0.0002
    high_vals = close_vals + 0.0003
    low_vals = close_vals - 0.0003

    idx = 10
    high_vals[idx - 1] = close_vals[idx - 1] + 0.0001
    high_vals[idx] = close_vals[idx] + 0.0001
    low_vals[idx + 1] = high_vals[idx] + 0.0002

    high_vals[0:4] = close_vals[0:4] + 0.0001
    low_vals[0:4] = close_vals[0:4] - 0.0001
    high_vals[8:12] = close_vals[8:12] + 0.0004
    low_vals[8:12] = high_vals[0:4].max() + 0.0005

    df = pd.DataFrame(
        {
            "ts": ts,
            "symbol": "TEST",
            "open": open_vals,
            "high": high_vals,
            "low": low_vals,
            "close": close_vals,
            "volume": 1000.0,
        }
    )

    csv_path = tmp_path / "ohlcv.csv"
    df.to_csv(csv_path, index=False)

    spec_payload = {
        "data": {
            "dataset_path": str(csv_path),
            "symbols": ["TEST"],
            "timeframe": "H1",
            "start": ts.min().isoformat(),
            "end": ts.max().isoformat(),
        },
        "symbols": ["TEST"],
        "range_start": ts.min().isoformat(),
        "range_end": ts.max().isoformat(),
        "targets": [
            {
                "type": "VWAP_DAY",
                "params": {"anchor": "day", "bands_sigma": [1.0, 2.0]},
            },
            {
                "type": "VWAP_BAND_1+",
                "params": {"anchor": "day", "bands_sigma": [1.0, 2.0]},
            },
            {
                "type": "VWAP_BAND_2+",
                "params": {"anchor": "day", "bands_sigma": [1.0, 2.0]},
            },
            {
                "type": "FVG_HTF",
                "params": {"htf": "H1", "min_size_ticks": 1, "price_increment": 0.0001},
            },
            {
                "type": "FVG_HTF",
                "params": {"htf": "H4", "min_size_ticks": 1, "price_increment": 0.0001},
            },
            {
                "type": "ADR_BAND_1",
                "params": {"adr_window": 1, "k_list": [1.0, 2.0]},
            },
            {
                "type": "ADR_BAND_2",
                "params": {"adr_window": 1, "k_list": [1.0, 2.0]},
            },
            {"type": "PIVOT_P", "params": {}},
            {"type": "PIVOT_R1", "params": {}},
            {"type": "PIVOT_S1", "params": {}},
        ],
    }
    spec = LevelsBuildSpec.model_validate(spec_payload)
    records = build_levels(spec, df)

    by_type: dict[str, list] = {}
    for record in records:
        by_type.setdefault(record.level_type, []).append(record)

    assert len(by_type.get("VWAP_DAY", [])) > 0
    assert len(by_type.get("VWAP_BAND_1+", [])) > 0
    assert len(by_type.get("ADR_BAND_1", [])) >= 1
    assert len(by_type.get("PIVOT_P", [])) >= 1

    fvg_timeframes = {rec.timeframe for rec in by_type.get("FVG_HTF", [])}
    assert {"H1", "H4"}.issubset(fvg_timeframes)
