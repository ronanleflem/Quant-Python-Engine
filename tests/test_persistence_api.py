import os

from quant_engine.persistence import (
    RunsRepository,
    MetricsRepository,
    TrialsRepository,
    session,
)
from quant_engine.api import app
from quant_engine.config import reset_settings_cache


def test_persistence_and_api(tmp_path):
    db_path = tmp_path / "runs.sqlite"
    os.environ["DB_DSN"] = f"sqlite:///{db_path}"
    reset_settings_cache()
    with session() as conn:
        runs_repo = RunsRepository(conn)
        metrics_repo = MetricsRepository(conn)
        trials_repo = TrialsRepository(conn)

        runs_repo.create_or_running(
            run_id="run1",
            spec_id="spec1",
            dataset_id="data1",
            objective="return",
            out_dir="/tmp/out",
        )
        metrics_repo.bulk_upsert_metrics("run1", {"metric": 1.0})
        trials_repo.bulk_insert_trials(
            "run1",
            [
                {
                    "trial_number": 1,
                    "params": {"x": 1},
                    "objective_value": 0.5,
                    "status": "COMPLETE",
                    "n_trades": 10,
                    "max_dd": 0.1,
                    "sharpe": 1.0,
                    "sortino": 1.0,
                    "cagr": 0.1,
                    "hit_rate": 0.5,
                    "avg_r": 0.2,
                }
            ],
        )
        runs_repo.finish("run1", "FINISHED")

    runs = app.list_runs()
    assert any(r["run_id"] == "run1" for r in runs)

    detail = app.get_run("run1")
    assert detail and detail["run"]["status"] == "FINISHED"

    metrics = app.get_run_metrics("run1")
    assert metrics["aggregated"]["metric"] == 1.0

    trials = app.get_run_trials("run1")
    assert trials[0]["trial_number"] == 1
