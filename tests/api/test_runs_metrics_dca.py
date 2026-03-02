import sys
import types

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


def _canonical_dca_payload() -> dict:
    return {
        "spec_type": "dca",
        "catalog_version": "v1",
        "data": {
            "symbol": "BTCUSD",
            "timeframe": "H1",
            "start_date": "2025-01-01",
            "end_date": "2025-01-02",
        },
        "strategy": {
            "type": "dca_equity",
            "grid": [],
            "params": {
                "grid": [{"dd": -5.0, "weight": 1.0}],
                "execution_mode": "bar_close",
                "drawdown_reference": "ATH",
            },
        },
    }


def _setup_db(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DB_SQLITE_PATH", str(tmp_path / "quant.db"))
    reset_settings_cache()


def test_runs_metrics_endpoint_exposes_dca_metrics(tmp_path, monkeypatch) -> None:
    _setup_db(tmp_path, monkeypatch)

    def _fake_backtest(_spec):
        return {
            "result": {"counts": {"BTCUSD": 1}},
            "payload": {
                "run": {
                    "status": "ok",
                    "extra": {
                        "final_performance_normalized": 0.12,
                        "twr": 0.10,
                        "xirr": 0.08,
                        "xirr_status": "ok",
                        "max_drawdown_on_contributed_capital": 15.0,
                        "time_under_water": 3,
                        "dca_composite_score": {
                            "score": 0.66,
                            "edge": "medium",
                            "components": {
                                "performance": 0.70,
                                "irr": 0.60,
                                "drawdown": 0.80,
                                "robustness": 0.50,
                            },
                        },
                    },
                }
            },
        }

    monkeypatch.setattr(api_app.strategies_runner, "run_backtest_with_payload", _fake_backtest)

    response = api_app.enqueue_run_request(_canonical_dca_payload())
    worker_module.process_next_job()

    payload = api_app.run_metrics_endpoint(response.run_id)
    aggregated = payload["aggregated"]
    assert aggregated["final_performance_normalized"] == 0.12
    assert aggregated["twr"] == 0.10
    assert aggregated["xirr"] == 0.08
    assert aggregated["max_drawdown_on_contributed_capital"] == 15.0
    assert aggregated["time_under_water"] == 3.0
    assert aggregated["xirr_converged"] == 1.0
    assert aggregated["dca_score"] == 0.66
    assert aggregated["dca_edge_level"] == 2.0
    assert aggregated["dca_score_component_performance"] == 0.70
