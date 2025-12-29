import pytest

from quant_engine.config import reset_settings_cache
from quant_engine.persistence import db
from quant_engine.persistence.repo import MarketStatsRepository, SeasonalityProfilesRepository


def test_market_stats_bulk_upsert_updates_without_duplicates(monkeypatch):
    monkeypatch.setenv("DB_DSN", "sqlite:///:memory:")
    reset_settings_cache()
    conn = db.connect()
    try:
        db.init_db(conn)
        repo = MarketStatsRepository(conn)

        base_row = {
            "symbol": "EURUSD",
            "timeframe": "H1",
            "event": "breakout",
            "condition_name": "vol",
            "condition_value": "high",
            "target": "profit",
            "split": "train",
            "n": 10,
            "successes": 6,
            "p_hat": 0.6,
            "ci_low": 0.4,
            "ci_high": 0.8,
            "lift": 1.2,
            "start": "2020-01-01",
            "end": "2020-06-01",
            "spec_id": "spec-1",
            "dataset_id": "dataset-1",
        }
        repo.bulk_upsert([base_row])

        updated_row = {
            **base_row,
            "n": 20,
            "successes": 12,
            "p_hat": 0.62,
            "dataset_id": "dataset-2",
        }
        repo.bulk_upsert([updated_row])

        count = conn.execute("SELECT COUNT(*) FROM market_stats").fetchone()[0]
        assert count == 1

        stored = conn.execute(
            "SELECT n, successes, p_hat, dataset_id FROM market_stats"
        ).fetchone()
        assert stored[0] == 20
        assert stored[1] == 12
        assert stored[2] == pytest.approx(0.62)
        assert stored[3] == "dataset-2"
    finally:
        conn.close()


def test_seasonality_profiles_bulk_upsert_updates_without_duplicates(monkeypatch):
    monkeypatch.setenv("DB_DSN", "sqlite:///:memory:")
    reset_settings_cache()
    conn = db.connect()
    try:
        db.init_db(conn)
        repo = SeasonalityProfilesRepository(conn)

        base_row = {
            "symbol": "EURUSD",
            "timeframe": "H1",
            "dim": "weekday",
            "bin": 1,
            "measure": "return",
            "score": 0.1,
            "n": 5,
            "baseline": 0.05,
            "lift": 1.1,
            "metrics": {"mean": 0.1},
            "start": "2020-01-01",
            "end": "2020-06-01",
            "spec_id": "spec-2",
            "dataset_id": "dataset-3",
        }
        repo.bulk_upsert([base_row])

        updated_row = {
            **base_row,
            "score": 0.2,
            "n": 10,
            "metrics": {"mean": 0.2},
        }
        repo.bulk_upsert([updated_row])

        count = conn.execute("SELECT COUNT(*) FROM seasonality_profiles").fetchone()[0]
        assert count == 1

        stored = conn.execute(
            "SELECT score, n, metrics, dataset_id FROM seasonality_profiles"
        ).fetchone()
        assert stored[0] == pytest.approx(0.2)
        assert stored[1] == 10
        assert stored[3] == "dataset-3"
    finally:
        conn.close()
