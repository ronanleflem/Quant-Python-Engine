from __future__ import annotations

from dataclasses import dataclass

import pytest

from quant_engine.api import schemas
from quant_engine.api.services import run_requests
from quant_engine.api.validation_errors import ApiValidationException


@dataclass
class _ParsedPayload:
    payload: dict

    def model_dump(self, mode: str = "json") -> dict:  # noqa: ARG002
        return self.payload


def test_enqueue_run_request_creates_new_job(monkeypatch):
    captured: dict = {}

    monkeypatch.setattr(run_requests.ids, "generate_id", lambda: "run-generated")

    response = run_requests.enqueue_run_request(
        {"spec_type": "backtest", "data": {"symbol": "BTCUSDT"}},
        validate_input=lambda payload: _ParsedPayload(payload),
        validate_market_stats_params=lambda payload: None,
        normalize_request_id=lambda _: None,
        get_job=lambda _job_id: None,
        init_job=lambda job_id, job_type, payload, **kwargs: captured.update(
            {"job_id": job_id, "job_type": job_type, "payload": payload, **kwargs}
        ),
        canonical_job_defaults=lambda: (3, 120),
        external_job_status=lambda status: status,
        canonical_job_type="canonical_run",
        queued_status="QUEUED",
    )

    assert isinstance(response, schemas.RunEnqueueResponse)
    assert response.run_id == "run-generated"
    assert response.status == "QUEUED"
    assert response.reused is False
    assert captured["job_id"] == "run-generated"
    assert captured["job_type"] == "canonical_run"
    assert captured["payload"]["request"]["spec_type"] == "backtest"


def test_enqueue_run_request_reuses_existing_canonical_job():
    response = run_requests.enqueue_run_request(
        {"spec_type": "backtest", "request_id": "run-42"},
        validate_input=lambda payload: _ParsedPayload(payload),
        validate_market_stats_params=lambda payload: None,
        normalize_request_id=lambda request_id: request_id,
        get_job=lambda _job_id: {"job_type": "canonical_run", "status": "RUNNING"},
        init_job=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not enqueue")),
        canonical_job_defaults=lambda: (3, None),
        external_job_status=lambda status: status,
        canonical_job_type="canonical_run",
        queued_status="QUEUED",
    )

    assert response.run_id == "run-42"
    assert response.status == "RUNNING"
    assert response.reused is True


def test_validate_market_stats_params_reports_errors():
    payload = {
        "spec_type": "market_stats",
        "stats": {
            "event": {"id": "k_consecutive", "params": {"k": 0, "direction": "side"}},
            "condition": {"id": "htf_trend", "params": {"tf_multiplier": "x", "ema_period": 0}},
            "target": {"id": "continuation_n", "params": {"n": "", "direction": None}},
        },
    }

    with pytest.raises(ApiValidationException) as exc:
        run_requests.validate_market_stats_params(payload)

    errors = exc.value.errors
    fields = {error["field"] for error in errors}
    assert "market_stats.stats.event.params.k" in fields
    assert "market_stats.stats.event.params.direction" in fields
    assert "market_stats.stats.condition.params.tf_multiplier" in fields
    assert "market_stats.stats.condition.params.ema_period" in fields
    assert "market_stats.stats.target.params.n" in fields
    assert "market_stats.stats.target.params.direction" in fields
