from quant_engine.api import app as api_app
from quant_engine.config import reset_settings_cache


def test_optimization_job_persists_experiment_metrics_and_trials(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DB_SQLITE_PATH", str(tmp_path / "quant.db"))
    reset_settings_cache()

    def _fake_spec_from_dict(_payload):
        return object()

    def _fake_run_optimisation(_spec):
        return {
            "trials_path": str(tmp_path / "runs" / "trials.parquet"),
            "best": {"metrics": {"sharpe": 1.23, "max_dd": -5.0}},
            "trials": [
                {"params": {"ema_fast": 9, "ema_slow": 26, "R": 2}, "metrics": {"sharpe": 1.23, "n_trades": 10}},
                {"params": {"ema_fast": 12, "ema_slow": 35, "R": 3}, "metrics": {"sharpe": 0.8, "n_trades": 8}},
            ],
        }

    monkeypatch.setattr(api_app.spec_module, "spec_from_dict", _fake_spec_from_dict)
    monkeypatch.setattr(api_app, "run_optimisation", _fake_run_optimisation)

    payload = {
        "spec": {
            "data": {"dataset_path": "specs/examples/data/eurusd_m1_sample.json"},
            "strategy": {"objective": "sharpe", "strategy_id": "opt_test_strategy"},
        }
    }
    run_id, result = api_app._run_job(api_app.JOB_TYPE_OPTIMIZATION, payload=payload)

    assert run_id
    assert "trials" in result

    with api_app.db.session() as conn:
        experiment = conn.execute(
            "SELECT run_id, status, objective, out_dir FROM experiment_runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        assert experiment is not None
        assert experiment["status"] == "COMPLETED"
        assert experiment["objective"] == "sharpe"
        assert "runs" in (experiment["out_dir"] or "")

        metrics_count = conn.execute(
            "SELECT COUNT(*) AS c FROM run_metrics WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        assert metrics_count is not None
        assert int(metrics_count["c"]) >= 1

        trials_count = conn.execute(
            "SELECT COUNT(*) AS c FROM trials WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        assert trials_count is not None
        assert int(trials_count["c"]) == 2
