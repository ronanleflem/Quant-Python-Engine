from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from quant_engine.api import schemas
from quant_engine.seasonality import runner
from quant_engine.seasonality.compute import CONDITIONAL_METRIC_NAMES

_ = pytest.importorskip("polars")


def _write_tiny_csv(path: Path) -> None:
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
        for idx in range(8):
            ts = start + timedelta(hours=idx)
            next_price = price + 1.0
            writer.writerow(
                {
                    "timestamp": ts.isoformat(),
                    "symbol": "TINY",
                    "open": f"{price:.4f}",
                    "high": f"{next_price + 0.5:.4f}",
                    "low": f"{price - 0.5:.4f}",
                    "close": f"{next_price:.4f}",
                    "volume": "1000",
                }
            )
            price = next_price


def test_seasonality_integration_tiny_dataset(tmp_path: Path) -> None:
    dataset_path = tmp_path / "tiny_seasonality.csv"
    _write_tiny_csv(dataset_path)

    out_dir = tmp_path / "artifacts"
    spec = schemas.SeasonalitySpec(
        data=schemas.SeasonalityDataSpec(
            dataset_path=str(dataset_path),
            symbols=["TINY"],
            timeframe="H1",
            start="2025-01-01",
            end="2025-01-02",
        ),
        profile=schemas.SeasonalityProfileSpec(
            by_hour=True,
            by_dow=False,
            by_month=False,
            measure="direction",
            ret_horizon=1,
            min_samples_bin=1,
        ),
        signal=schemas.SeasonalitySignalSpec(
            method="threshold",
            threshold=0.5,
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
    summary_path = out_dir / "fold_0" / "summary.json"
    trades_path = out_dir / "fold_0" / "trades.parquet"
    equity_path = out_dir / "fold_0" / "equity.parquet"

    assert profiles_path.exists()
    assert summary_path.exists()
    assert trades_path.exists()
    assert equity_path.exists()

    pl = pytest.importorskip("polars")
    profiles_df = pl.read_parquet(profiles_path)
    for col in CONDITIONAL_METRIC_NAMES:
        assert col in profiles_df.columns

    best_metrics = result.get("best_metrics", {})
    for key in ["sharpe", "max_drawdown", "trades", "n_trades"]:
        assert key in best_metrics

    active_bins = result.get("active_bins", {})
    assert "hour" in active_bins
