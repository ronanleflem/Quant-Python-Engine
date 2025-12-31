import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from quant_engine.api import schemas
from quant_engine.core.spec import ArtifactsSpec, ValidationSpec
from quant_engine.stats import runner


def test_run_stats_writes_summary(tmp_path: Path) -> None:
    start = datetime(2020, 1, 1)
    rows = []
    for i in range(6):
        ts = start + timedelta(days=i)
        rows.append(
            {
                "timestamp": ts.isoformat(),
                "symbol": "TST",
                "open": i + 1,
                "high": i + 2,
                "low": i,
                "close": i + 1.5,
            }
        )

    dataset_path = tmp_path / "tiny.json"
    dataset_path.write_text(json.dumps(rows))

    out_dir = tmp_path / "artifacts"
    spec = schemas.StatsSpec(
        data=schemas.StatsDataSpec(
            dataset_path=str(dataset_path),
            symbols=["TST"],
            timeframe="1d",
            start="2020-01-01",
            end="2020-01-06",
        ),
        events=[schemas.StatsEventSpec(name="k_consecutive", params={"k": 2, "direction": "up"})],
        targets=[schemas.StatsTargetSpec(name="up_next_bar")],
        validation=ValidationSpec(
            min_trades=0,
            train_months=0,
            test_months=1,
            folds=1,
            embargo_days=0,
        ),
        artifacts=ArtifactsSpec(out_dir=str(out_dir)),
    )

    runner.run_stats(spec)

    summary_path = out_dir / "stats_summary.parquet"
    assert summary_path.exists()

    summary = pd.read_parquet(summary_path)
    for column in ["p_hat", "n", "successes"]:
        assert column in summary.columns
