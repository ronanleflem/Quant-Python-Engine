from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import sys
import types

if "pymysql" not in sys.modules:
    pymysql_stub = types.ModuleType("pymysql")
    pymysql_cursors_stub = types.ModuleType("pymysql.cursors")
    pymysql_cursors_stub.DictCursor = object
    pymysql_stub.cursors = pymysql_cursors_stub
    sys.modules["pymysql"] = pymysql_stub
    sys.modules["pymysql.cursors"] = pymysql_cursors_stub

from quant_engine.api import schemas
from quant_engine.seasonality import runner, spec as seasonality_spec

pl = pytest.importorskip("polars")


def _build_train_df() -> "pl.DataFrame":
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    rows = []
    price = 100.0
    for idx in range(96):
        ts = start + timedelta(hours=idx)
        drift = 0.6 if idx < 48 else -0.4
        close = price + drift
        rows.append(
            {
                "timestamp": ts,
                "symbol": "BTC-USD",
                "open": price,
                "high": max(price, close) + 0.5,
                "low": min(price, close) - 0.5,
                "close": close,
                "volume": 1_000 + idx,
            }
        )
        price = close
    return pl.DataFrame(rows)


def test_seasonality_profiles_include_global_and_mi_segmented_stats() -> None:
    train_df = _build_train_df()
    model = schemas.SeasonalitySpec(
        data=schemas.SeasonalityDataSpec(
            symbols=["BTC-USD"],
            timeframe="H1",
            start="2025-01-01",
            end="2025-01-10",
        ),
        profile=schemas.SeasonalityProfileSpec(
            by_hour=True,
            by_dow=False,
            measure="direction",
            ret_horizon=1,
            min_samples_bin=1,
            segment_by_mi_labels=True,
        ),
    )
    cfg = seasonality_spec.normalise(model)

    profiles_df = runner._compute_profiles(train_df, cfg, fold_dir=None)

    assert not profiles_df.is_empty()
    assert {"mi_label_name", "mi_label_value"}.issubset(set(profiles_df.columns))

    global_rows = profiles_df.filter(pl.col("mi_label_name").is_null())
    segmented_rows = profiles_df.filter(pl.col("mi_label_name").is_not_null())

    assert global_rows.height > 0
    assert segmented_rows.height > 0
    assert set(segmented_rows.get_column("mi_label_name").unique().to_list()) == {"label_regime"}
    assert all(v is not None for v in segmented_rows.get_column("mi_label_value").to_list())
