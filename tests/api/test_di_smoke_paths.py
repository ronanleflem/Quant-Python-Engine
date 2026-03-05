from __future__ import annotations

import sys
import types
from dataclasses import dataclass

if "pymysql" not in sys.modules:
    pymysql_stub = types.ModuleType("pymysql")
    pymysql_stub.connect = lambda *args, **kwargs: None
    cursors_stub = types.ModuleType("pymysql.cursors")
    cursors_stub.DictCursor = object
    pymysql_stub.cursors = cursors_stub
    sys.modules["pymysql"] = pymysql_stub
    sys.modules["pymysql.cursors"] = cursors_stub

from quant_engine.api import app as api_app


@dataclass
class _FakeParsedRequest:
    payload: dict

    def model_dump(self, mode: str = "json") -> dict:  # noqa: ARG002
        return self.payload


def test_optimize_path_smoke_uses_submit_dependency_injection(monkeypatch):
    captured: dict = {}

    def fake_run_job(job_type: str, payload: dict):
        captured["job_type"] = job_type
        captured["payload"] = payload
        return "opt-job", {"ok": True}

    monkeypatch.setattr(api_app, "_run_job", fake_run_job)

    spec = api_app.Spec(
        data=api_app.spec_module.DataSpec(
            dataset_path="tests/data/ohlcv_dca_benchmark_daily.csv",
            mysql=None,
            symbols=["BTCUSDT"],
            timeframe="1h",
            start="2025-01-01T00:00:00Z",
            end="2025-01-02T00:00:00Z",
        ),
        strategy=api_app.spec_module.StrategySpec(
            filters=api_app.spec_module.FiltersSpec(ema_fast=10, ema_slow=20),
            tpsl=api_app.spec_module.TPSLSpec(atr_k=1.0),
            validation=api_app.spec_module.ValidationSpec(),
            objective="sharpe",
            search_space={"ema_fast": [10]},
        ),
    )
    response = api_app.submit(spec)

    assert response.id == "opt-job"
    assert captured["job_type"] == api_app.JOB_TYPE_OPTIMIZATION
    assert captured["payload"]["spec"]["data"]["symbols"] == ["BTCUSDT"]


def test_backtest_path_smoke_queues_canonical_run_with_injected_dependencies(monkeypatch):
    monkeypatch.setattr(
        api_app,
        "validate_run_request_input",
        lambda payload: _FakeParsedRequest(payload),
    )
    monkeypatch.setattr(api_app, "_validate_canonical_market_stats_params", lambda payload: None)
    monkeypatch.setattr(api_app, "_normalize_request_id", lambda _: None)
    monkeypatch.setattr(api_app.ids, "generate_id", lambda: "run-di-001")
    monkeypatch.setattr(api_app, "_canonical_job_defaults", lambda: (3, 900))

    captured: dict = {}

    def fake_init_job(job_id, job_type, payload, *, status, max_attempts, timeout_seconds):
        captured.update(
            {
                "job_id": job_id,
                "job_type": job_type,
                "payload": payload,
                "status": status,
                "max_attempts": max_attempts,
                "timeout_seconds": timeout_seconds,
            }
        )

    monkeypatch.setattr(api_app, "_init_job", fake_init_job)

    response = api_app.enqueue_run_request(
        {
            "spec_type": "backtest",
            "request_id": None,
            "data": {"symbol": "BTCUSDT", "timeframe": "1h"},
        }
    )

    assert response.run_id == "run-di-001"
    assert response.status == api_app.JOB_STATUS_QUEUED
    assert captured["job_type"] == api_app.JOB_TYPE_CANONICAL_RUN


def test_mi_path_smoke_stats_run_uses_injected_job_runner(monkeypatch):
    captured: dict = {}

    def fake_run_job(job_type: str, payload: dict):
        captured["job_type"] = job_type
        captured["payload"] = payload
        return "stats-job", {"rows": 1}

    monkeypatch.setattr(api_app, "_run_job", fake_run_job)

    spec = api_app.schemas.StatsSpec(
        data=api_app.schemas.StatsDataSpec(
            symbols=["BTCUSDT"],
            timeframe="1h",
            start="2025-01-01T00:00:00Z",
            end="2025-01-02T00:00:00Z",
        ),
        events=[api_app.schemas.StatsEventSpec(name="next_n_bars", params={"n": 3, "direction": "up"})],
        conditions=[],
        targets=[api_app.schemas.StatsTargetSpec(name="up_3", params={"horizon": 3})],
    )

    response = api_app.stats_run(spec)

    assert response.id == "stats-job"
    assert response.status == "completed"
    assert captured["job_type"] == api_app.JOB_TYPE_STATS
