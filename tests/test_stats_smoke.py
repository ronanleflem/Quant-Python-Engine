import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

pl = pytest.importorskip("polars")

from quant_engine.api import schemas
from quant_engine.core.spec import ArtifactsSpec, ValidationSpec
from quant_engine.stats import runner


def test_stats_smoke(tmp_path: Path) -> None:
    n = 30
    start = datetime(2020, 1, 1)
    dates = [start + timedelta(days=i) for i in range(n)]
    df = pl.DataFrame(
        {
            "timestamp": [d.strftime("%Y-%m-%dT%H:%M:%S") for d in dates],
            "symbol": ["ABC"] * n,
            "open": list(range(n)),
            "high": [x + 1 for x in range(n)],
            "low": list(range(n)),
            "close": [x + 0.5 for x in range(n)],
            "session_id": [d.strftime("%Y-%m-%d") for d in dates],
        }
    )
    dataset_path = tmp_path / "synth.json"
    dataset_path.write_text(json.dumps(df.to_dicts()))
    out_dir = tmp_path / "run"
    spec = schemas.StatsSpec(
        data=schemas.StatsDataSpec(
            dataset_path=str(dataset_path),
            symbols=["ABC"],
            timeframe="M1",
            start="2020-01-01",
            end="2020-01-30",
        ),
        events=[
            schemas.StatsEventSpec(
                name="k_consecutive", params={"k": 2, "direction": "up"}
            )
        ],
        conditions=[
            schemas.StatsConditionSpec(name="session", params={"col": "session_id"}),
            schemas.StatsConditionSpec(name="vol_tertile", params={"window": 14}),
        ],
        targets=[schemas.StatsTargetSpec(name="up_next_bar", params={})],
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
    df_read = pd.read_parquet(summary_path)
    assert df_read["p_hat"].notna().any()
