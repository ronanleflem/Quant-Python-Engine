"""Smoke tests for structure and liquidity pool detectors."""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant_engine.levels.builders import build_levels
from quant_engine.levels.schemas import LevelsBuildSpec


def test_levels_phase2a_smoke(tmp_path) -> None:
    ts = pd.date_range("2024-01-01", periods=40, freq="h", tz="UTC")
    close_vals = np.array(
        [
            1.0000,
            1.0008,
            1.0016,
            1.0028,
            1.0020,
            1.0012,
            1.0004,
            0.9992,
            1.0005,
            1.0018,
            1.0032,
            1.0040,
            1.0046,
            1.0038,
            1.0024,
            1.0008,
            0.9988,
            0.9980,
            0.9992,
            1.0008,
            1.0026,
            1.0044,
            1.0062,
            1.0075,
            1.0088,
            1.0096,
            1.0104,
            1.0112,
            1.0118,
            1.0125,
            1.0132,
            1.0138,
            1.0145,
            1.0152,
            1.0158,
            1.0165,
            1.0172,
            1.0178,
            1.0185,
            1.0190,
        ]
    )
    open_vals = close_vals - 0.0002
    high_vals = close_vals + 0.0004
    low_vals = close_vals - 0.0004

    # Equal highs cluster around 1.0065 within tolerance.
    high_vals[22] = 1.0065
    high_vals[23] = 1.0066
    high_vals[24] = 1.0065
    # Equal lows cluster around 0.9982 within tolerance.
    low_vals[15] = 0.9982
    low_vals[16] = 0.9983
    low_vals[17] = 0.9982

    df = pd.DataFrame(
        {
            "ts": ts,
            "symbol": "TEST",
            "open": open_vals,
            "high": high_vals,
            "low": low_vals,
            "close": close_vals,
            "volume": 1000,
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
            {"type": "SWING_H", "params": {"left": 2, "right": 2}},
            {"type": "SWING_L", "params": {"left": 2, "right": 2}},
            {
                "type": "EQH",
                "params": {
                    "tolerance_ticks": 1,
                    "price_increment": 0.0001,
                    "min_count": 2,
                    "lookback_bars": 20,
                },
            },
            {
                "type": "EQL",
                "params": {
                    "tolerance_ticks": 1,
                    "price_increment": 0.0001,
                    "min_count": 2,
                    "lookback_bars": 20,
                },
            },
            {"type": "BOS_H", "params": {"left": 2, "right": 2}},
            {"type": "BOS_L", "params": {"left": 2, "right": 2}},
            {"type": "MSS", "params": {"left": 2, "right": 2}},
        ],
    }
    spec = LevelsBuildSpec.model_validate(spec_payload)
    records = build_levels(spec, df)

    by_type = {}
    for record in records:
        by_type.setdefault(record.level_type, []).append(record)

    assert len(by_type.get("SWING_H", [])) > 0
    assert len(by_type.get("SWING_L", [])) > 0
    assert len(by_type.get("EQH", [])) > 0
    assert len(by_type.get("EQL", [])) > 0
    assert len(by_type.get("BOS_H", [])) > 0
    assert len(by_type.get("BOS_L", [])) > 0
    assert len(by_type.get("MSS", [])) > 0
