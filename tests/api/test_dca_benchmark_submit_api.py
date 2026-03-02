from __future__ import annotations

import os
import sys
import types
from typing import Any

import pytest

if "pymysql" not in sys.modules:
    pymysql_stub = types.ModuleType("pymysql")
    pymysql_stub.connect = lambda *args, **kwargs: None
    cursors_stub = types.ModuleType("pymysql.cursors")
    cursors_stub.DictCursor = object
    pymysql_stub.cursors = cursors_stub
    sys.modules["pymysql"] = pymysql_stub
    sys.modules["pymysql.cursors"] = cursors_stub

from quant_engine.api import app as api_app
from quant_engine.api import worker as worker_module
from quant_engine.config import reset_settings_cache


_REQUIRED_BENCHMARK_FIELDS = {
    "variant",
    "planned_date",
    "effective_date",
    "fallback",
    "cashflow",
    "invested_capital",
    "capital_curve",
    "timezone",
}


@pytest.fixture
def api_db(tmp_path):
    os.environ["DB_SQLITE_PATH"] = str(tmp_path / "quant.db")
    reset_settings_cache()
    yield
    os.environ.pop("DB_SQLITE_PATH", None)
    reset_settings_cache()


def _submit_benchmark_run(variant: str, **params: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "spec_type": "dca",
        "catalog_version": "v1",
        "data": {
            "symbol": "SPY",
            "timeframe": "1D",
            "start_date": "2024-01-01",
            "end_date": "2024-04-30",
            "path": "tests/data/ohlcv_dca_benchmark_daily.csv",
        },
        "strategy": {
            "type": "dca_benchmark",
            "grid": [],
            "params": {
                "variant": variant,
                "amount": 100.0,
                "fallback": "next_business_day",
                "timezone": "UTC",
                **params,
            },
        },
    }

    enqueue = api_app.enqueue_run_request(payload)
    worker_module.process_next_job()
    result_payload = api_app.run_result_endpoint(enqueue.run_id)

    assert result_payload["status"] == "SUCCEEDED"
    return result_payload["result"]["result"]


@pytest.mark.parametrize(
    ("variant", "params"),
    [
        ("monthly_fixed", {"day_of_month": 15}),
        ("monthly_randomized", {"seed": 7}),
        ("mid_month", {"start_day": 10, "end_day": 20, "day_of_month": 15}),
        ("turn_of_month", {"offset": -1}),
        ("weekly_fixed", {"weekday": 2}),
    ],
)
def test_submit_api_dca_benchmark_variants_have_homogeneous_calendar_meta(api_db, variant, params) -> None:
    result = _submit_benchmark_run(variant, **params)

    signals = result["signals"]["SPY"]
    assert signals

    for signal in signals:
        benchmark_meta = signal["meta"]["benchmark"]
        assert _REQUIRED_BENCHMARK_FIELDS.issubset(benchmark_meta.keys())
        assert benchmark_meta["variant"] == variant
        assert benchmark_meta["planned_date"]
        assert benchmark_meta["effective_date"]
        assert benchmark_meta["timezone"] == "UTC"
        assert isinstance(benchmark_meta["cashflow"], (int, float))
        assert isinstance(benchmark_meta["invested_capital"], (int, float))
        assert isinstance(benchmark_meta["capital_curve"], (int, float))


def test_submit_api_monthly_randomized_is_deterministic_with_fixed_seed(api_db) -> None:
    first = _submit_benchmark_run("monthly_randomized", seed=42)
    second = _submit_benchmark_run("monthly_randomized", seed=42)

    first_effective_dates = [s["meta"]["benchmark"]["effective_date"] for s in first["signals"]["SPY"]]
    second_effective_dates = [s["meta"]["benchmark"]["effective_date"] for s in second["signals"]["SPY"]]

    assert first_effective_dates == second_effective_dates
