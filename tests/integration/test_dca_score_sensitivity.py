import sys
import types
from copy import deepcopy

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
from quant_engine.backtest.metrics import dca_composite_score
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


def test_dca_score_weight_sensitivity_is_propagated_and_persisted(tmp_path, monkeypatch) -> None:
    _setup_db(tmp_path, monkeypatch)

    def _fake_backtest(spec):
        grid_weight = float(spec["strategy"]["params"]["grid"][0]["weight"])
        composite_weights = (
            {"performance": 0.70, "irr": 0.10, "drawdown": 0.10, "robustness": 0.10}
            if grid_weight > 1.0
            else {"performance": 0.10, "irr": 0.50, "drawdown": 0.30, "robustness": 0.10}
        )
        composite = dca_composite_score(
            final_performance_normalized_value=0.12,
            xirr_value=0.08,
            max_drawdown_on_contributed_capital_value=0.15,
            underperformance_duration_windows=2,
            underperformance_severity_pct_points=7.5,
            xirr_status="ok",
            weights=composite_weights,
        )
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
                        "time_under_water": 2,
                        "dca_composite_score": composite,
                    },
                }
            },
        }

    monkeypatch.setattr(api_app.strategies_runner, "run_backtest_with_payload", _fake_backtest)

    low_weight_payload = _canonical_dca_payload()
    high_weight_payload = deepcopy(low_weight_payload)
    high_weight_payload["strategy"]["params"]["grid"][0]["weight"] = 2.0

    low = api_app.enqueue_run_request(low_weight_payload)
    high = api_app.enqueue_run_request(high_weight_payload)
    worker_module.process_next_job()
    worker_module.process_next_job()

    low_result = api_app.run_result_endpoint(low.run_id)["result"]["payload"]["run"]["extra"]["dca_composite_score"]
    high_result = api_app.run_result_endpoint(high.run_id)["result"]["payload"]["run"]["extra"]["dca_composite_score"]

    assert set(low_result["weight_sensitivity"].keys()) == {"performance", "irr", "drawdown", "robustness"}
    assert set(high_result["weight_sensitivity"].keys()) == {"performance", "irr", "drawdown", "robustness"}
    assert high_result["score"] != low_result["score"]

    low_metrics = api_app.run_metrics_endpoint(low.run_id)["aggregated"]
    high_metrics = api_app.run_metrics_endpoint(high.run_id)["aggregated"]

    assert low_metrics["dca_score"] == low_result["score"]
    assert high_metrics["dca_score"] == high_result["score"]
    assert high_metrics["dca_score"] != low_metrics["dca_score"]
