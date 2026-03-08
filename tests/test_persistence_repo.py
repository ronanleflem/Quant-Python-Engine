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
            "p_mean": 0.58,
            "p_map": 0.6,
            "hdi_low": 0.35,
            "hdi_high": 0.79,
            "lift_freq": 0.12,
            "lift_bayes": 0.1,
            "lift": 1.2,
            "p_value": 0.04,
            "q_value": 0.05,
            "significant": True,
            "insufficient": False,
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
            "p_mean": 0.61,
            "lift_freq": 0.14,
            "q_value": 0.03,
            "dataset_id": "dataset-2",
        }
        repo.bulk_upsert([updated_row])

        count = conn.execute("SELECT COUNT(*) FROM market_stats").fetchone()[0]
        assert count == 1

        stored = conn.execute(
            "SELECT n, successes, p_hat, p_mean, lift_freq, q_value, dataset_id FROM market_stats"
        ).fetchone()
        assert stored[0] == 20
        assert stored[1] == 12
        assert stored[2] == pytest.approx(0.62)
        assert stored[3] == pytest.approx(0.61)
        assert stored[4] == pytest.approx(0.14)
        assert stored[5] == pytest.approx(0.03)
        assert stored[6] == "dataset-2"
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


def test_seasonality_profiles_bulk_upsert_accepts_text_bins(monkeypatch):
    monkeypatch.setenv("DB_DSN", "sqlite:///:memory:")
    reset_settings_cache()
    conn = db.connect()
    try:
        db.init_db(conn)
        repo = SeasonalityProfilesRepository(conn)

        row = {
            "symbol": "BTCUSDT",
            "timeframe": "1d",
            "dim": "session",
            "bin": "Asia",
            "measure": "avg_return",
            "score": 0.12,
            "n": 42,
            "baseline": 0.03,
            "lift": 0.09,
            "metrics": {"mean": 0.12},
            "start": "2022-01-01",
            "end": "2024-12-31",
            "spec_id": "seas_001",
            "dataset_id": "seasonality_ds",
        }
        repo.bulk_upsert([row])

        stored = conn.execute(
            "SELECT dim, bin, measure, dataset_id FROM seasonality_profiles"
        ).fetchone()
        assert stored[0] == "session"
        assert stored[1] == "Asia"
        assert stored[2] == "avg_return"
        assert stored[3] == "seasonality_ds"
    finally:
        conn.close()
