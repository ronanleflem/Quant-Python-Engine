import os

from quant_engine.persistence import (
    RunsRepository,
    MetricsRepository,
    TrialsRepository,
    SeasonalityProfilesRepository,
    SeasonalityRunsRepository,
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
        seasonality_runs_repo = SeasonalityRunsRepository(conn)
        seasonality_profiles_repo = SeasonalityProfilesRepository(conn)

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

        seasonality_runs_repo.create(
            "season-run",
            spec_id="spec-season",
            dataset_id="data-season",
            out_dir="/tmp/seasonality",
        )
        seasonality_profiles_repo.bulk_upsert(
            [
                {
                    "symbol": "BTCUSDT",
                    "timeframe": "1h",
                    "dim": "hour",
                    "bin": 9,
                    "measure": "direction",
                    "score": 0.6,
                    "n": 500,
                    "baseline": 0.52,
                    "lift": 0.08,
                    "metrics": {"p_breakout_up": 0.3, "run_len_up_mean": 2.1},
                    "start": "2021-01-01",
                    "end": "2021-06-01",
                    "spec_id": "spec-season",
                    "dataset_id": "data-season",
                }
            ]
        )
        seasonality_runs_repo.finish(
            "season-run",
            "completed",
            {"best_metrics": {"sharpe": 1.5}},
        )

    runs = app.list_runs()
    assert any(r["run_id"] == "run1" for r in runs)

    detail = app.get_run("run1")
    assert detail and detail["run"]["status"] == "FINISHED"

    metrics = app.get_run_metrics("run1")
    assert metrics["aggregated"]["metric"] == 1.0

    trials = app.get_run_trials("run1")
    assert trials[0]["trial_number"] == 1

    season_runs = app.list_seasonality_runs()
    assert any(r["run_id"] == "season-run" for r in season_runs)

    season_detail = app.get_seasonality_run("season-run")
    assert season_detail and season_detail["best_summary"]["best_metrics"]["sharpe"] == 1.5

    profiles = app.list_seasonality_profiles(symbol="BTCUSDT")
    assert profiles and profiles[0]["measure"] == "direction"
    assert profiles[0]["metrics"]["p_breakout_up"] == 0.3

    filtered = app.list_seasonality_profiles(
        symbol="BTCUSDT", metrics=["p_breakout_up", "run_len_up_mean"]
    )
    assert filtered and filtered[0]["metrics"]["run_len_up_mean"] == 2.1

    missing = app.list_seasonality_profiles(symbol="BTCUSDT", metrics=["amp_mean"])
    assert missing == []

    os.environ.pop("DB_DSN", None)
    reset_settings_cache()
