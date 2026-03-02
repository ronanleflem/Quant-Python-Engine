import os
import sys
import types

import pytest

pymysql_module = types.ModuleType("pymysql")
pymysql_cursors = types.ModuleType("pymysql.cursors")
setattr(pymysql_cursors, "DictCursor", object)
setattr(pymysql_module, "cursors", pymysql_cursors)
sys.modules.setdefault("pymysql", pymysql_module)
sys.modules.setdefault("pymysql.cursors", pymysql_cursors)

from quant_engine.api import app as api_app
from quant_engine.config import reset_settings_cache


@pytest.fixture
def api_db(tmp_path):
    db_path = tmp_path / "api.sqlite"
    os.environ["DB_DSN"] = f"sqlite:///{db_path}"
    reset_settings_cache()
    yield db_path
    os.environ.pop("DB_DSN", None)
    reset_settings_cache()


def test_run_artifacts_endpoint_lists_files(api_db, tmp_path):
    out_dir = tmp_path / "run-artifacts"
    out_dir.mkdir(parents=True)
    (out_dir / "metrics.json").write_text("{}")
    (out_dir / "metrics.parquet").write_text("parquet")

    run_id = "run-artifacts-1"
    api_app._init_job(
        run_id,
        api_app.JOB_TYPE_CANONICAL_RUN,
        payload={
            "request": {
                "spec_type": "market_stats",
                "output": {"out_dir": str(out_dir)},
            }
        },
        status=api_app.JOB_STATUS_SUCCEEDED,
    )

    payload = api_app.run_artifacts_endpoint(run_id)

    assert payload["run_id"] == run_id
    assert payload["schema_version"] == "dca-grid-process-v1"
    assert payload["out_dir"] == str(out_dir)
    names = {item["name"] for item in payload["files"]}
    assert {"metrics.json", "metrics.parquet"}.issubset(names)
