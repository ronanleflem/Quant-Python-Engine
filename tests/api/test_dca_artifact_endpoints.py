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
    (out_dir / "run_manifest.json").write_text("{}")
    (out_dir / "checksums.txt").write_text("abc  metrics.json\n")
    (out_dir / "best_plausible_passive_ex_ante.json").write_text("{}")
    (out_dir / "best_plausible_passive_ex_ante.parquet").write_text("parquet")

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
    assert {"metrics.json", "metrics.parquet", "run_manifest.json", "checksums.txt", "best_plausible_passive_ex_ante.json", "best_plausible_passive_ex_ante.parquet"}.issubset(names)
    assert payload["contract_artifacts"]["best_plausible_passive_ex_ante"]["json"] == "best_plausible_passive_ex_ante.json"
    assert payload["contract_artifacts"]["best_plausible_passive_ex_ante"]["parquet"] == "best_plausible_passive_ex_ante.parquet"
    assert payload["audit_trail"]["run_manifest"] == "run_manifest.json"
    assert payload["audit_trail"]["checksums"] == "checksums.txt"


def test_canonical_market_stats_mapping_includes_performance_universe_rules_version(api_db):
    mapped = api_app._canonical_market_stats_to_spec(
        {
            "data": {
                "symbol": "AAPL",
                "timeframe": "1D",
                "start_date": "2025-01-01",
                "end_date": "2025-01-31",
            },
            "stats": {
                "event": {"id": "k_consecutive", "params": {"k": 2}},
                "condition": {"id": "session", "params": {}},
                "target": {"id": "up_next_bar", "params": {}},
            },
            "performance": {"universe_rules_version": "asset-universe-rules-v3"},
        }
    )

    assert mapped["performance"]["universe_rules_version"] == "asset-universe-rules-v3"
