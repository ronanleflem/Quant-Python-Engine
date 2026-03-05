import os
import sys
import types

import pytest
from fastapi.responses import JSONResponse

if "pymysql" not in sys.modules:
    pymysql_stub = types.ModuleType("pymysql")
    pymysql_stub.connect = lambda *args, **kwargs: None
    cursors_stub = types.ModuleType("pymysql.cursors")
    cursors_stub.DictCursor = object
    pymysql_stub.cursors = cursors_stub
    sys.modules["pymysql"] = pymysql_stub
    sys.modules["pymysql.cursors"] = cursors_stub

from quant_engine.api import app as api_app
from quant_engine.config import reset_settings_cache


@pytest.fixture
def api_db(tmp_path):
    os.environ["DB_SQLITE_PATH"] = str(tmp_path / "quant.db")
    reset_settings_cache()
    yield
    os.environ.pop("DB_SQLITE_PATH", None)
    reset_settings_cache()


def test_run_result_endpoint_not_found_still_returns_json_error(api_db):
    response = api_app.run_result_endpoint("missing-run")
    assert isinstance(response, JSONResponse)
    assert response.status_code == 404
    assert response.body == b'{"code":"not_found","message":"Run not found"}'


def test_runs_cancel_endpoint_still_returns_conflict_when_already_finished(api_db):
    run_id = "finished-run"
    api_app._init_job(
        run_id,
        api_app.JOB_TYPE_CANONICAL_RUN,
        payload={"request": {"spec_type": "market_stats"}},
        status=api_app.JOB_STATUS_SUCCEEDED,
    )

    response = api_app.runs_service.cancel_run(
        run_id,
        get_job=api_app._get_job,
        external_job_status=api_app._external_job_status,
        request_job_cancel=api_app.request_job_cancel,
        json_error=api_app._json_error,
    )

    assert isinstance(response, JSONResponse)
    assert response.status_code == 409
    assert response.body == b'{"code":"already_finished","message":"Run already finished"}'


def test_run_detail_endpoint_still_returns_canonical_payload(api_db):
    run_id = "canonical-run"
    api_app._init_job(
        run_id,
        api_app.JOB_TYPE_CANONICAL_RUN,
        payload={"request": {"spec_type": "market_stats"}},
        status=api_app.JOB_STATUS_RUNNING_CANONICAL,
    )

    payload = api_app.run_detail_endpoint(run_id)

    assert payload["run_id"] == run_id
    assert payload["status"] == api_app.JOB_STATUS_RUNNING_CANONICAL
    assert payload["job_type"] == api_app.JOB_TYPE_CANONICAL_RUN
