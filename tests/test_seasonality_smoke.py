from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from quant_engine.api import schemas
from quant_engine.seasonality import runner

_ = pytest.importorskip("polars")


def _write_synthetic_csv(path: Path, *, rows: int = 240) -> None:
    start = datetime(2025, 1, 1, 0, 0)
    price = 100.0
    fieldnames = [
        "timestamp",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(rows):
            ts = start + timedelta(hours=idx)
            direction = 1 if ts.hour == 9 else 0
            change = 0.02 if direction else -0.01
            next_price = price * (1 + change)
            writer.writerow(
                {
                    "timestamp": ts.isoformat(),
                    "symbol": "TEST",
                    "open": f"{price:.4f}",
                    "high": f"{max(price, next_price) + 0.5:.4f}",
                    "low": f"{min(price, next_price) - 0.5:.4f}",
                    "close": f"{price:.4f}",
                    "volume": "1000",
                }
            )
            price = next_price


def test_seasonality_smoke(tmp_path: Path) -> None:
    dataset_path = tmp_path / "seasonality.csv"
    _write_synthetic_csv(dataset_path)

    out_dir = tmp_path / "artifacts"
    spec = schemas.SeasonalitySpec(
        data=schemas.SeasonalityDataSpec(
            dataset_path=str(dataset_path),
            symbols=["TEST"],
            timeframe="M1",
            start="2025-01-01",
            end="2025-01-20",
        ),
        profile=schemas.SeasonalityProfileSpec(
            by_hour=True,
            by_dow=False,
            by_month=False,
            measure="direction",
            ret_horizon=1,
            min_samples_bin=5,
        ),
        signal=schemas.SeasonalitySignalSpec(
            method="threshold",
            threshold=0.54,
            dims=["hour"],
            combine="and",
        ),
        validation=schemas.ValidationSpec(
            min_trades=0,
            train_months=0,
            test_months=1,
            folds=1,
            embargo_days=0,
        ),
        artifacts=schemas.ArtifactsSpec(out_dir=str(out_dir)),
    )

    result = runner.run(spec)

    profiles_path = out_dir / "fold_0" / "seasonality_profiles.parquet"
    assert profiles_path.exists()
    active_bins = result.get("active_bins", {})
    assert active_bins.get("hour")
