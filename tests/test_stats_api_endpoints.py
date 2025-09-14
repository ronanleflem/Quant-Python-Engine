import os

from quant_engine.persistence import session, MarketStatsRepository
from quant_engine.api import app
from quant_engine.config import reset_settings_cache


def setup_db(tmp_path):
    db_path = tmp_path / "stats.sqlite"
    os.environ["DB_DSN"] = f"sqlite:///{db_path}"
    reset_settings_cache()
    return db_path


def test_stats_heatmap(tmp_path):
    setup_db(tmp_path)
    with session() as conn:
        repo = MarketStatsRepository(conn)
        repo.bulk_upsert([
            {
                "symbol": "ABC",
                "timeframe": "1h",
                "event": "ev",
                "condition_name": "cond",
                "condition_value": "10",
                "target": "up",
                "split": "train",
                "n": 10,
                "successes": 5,
                "p_hat": 0.5,
                "ci_low": 0.4,
                "ci_high": 0.6,
                "lift": 0.1,
                "start": "2020",
                "end": "2020",
            },
            {
                "symbol": "ABC",
                "timeframe": "1h",
                "event": "ev",
                "condition_name": "cond",
                "condition_value": "5",
                "target": "up",
                "split": "test",
                "n": 8,
                "successes": 6,
                "p_hat": 0.75,
                "ci_low": 0.5,
                "ci_high": 0.9,
                "lift": 0.2,
                "start": "2020",
                "end": "2020",
            },
            {
                "symbol": "ABC",
                "timeframe": "1h",
                "event": "ev",
                "condition_name": "cond",
                "condition_value": "10",
                "target": "up",
                "split": "test",
                "n": 12,
                "successes": 6,
                "p_hat": 0.5,
                "ci_low": 0.3,
                "ci_high": 0.7,
                "lift": 0.05,
                "start": "2020",
                "end": "2020",
            },
        ])
    res = app.stats_heatmap(
        symbol="ABC",
        timeframe="1h",
        event="ev",
        target="up",
        condition_name="cond",
    )
    assert [r["bin"] for r in res] == ["5", "10"]
    assert res[0]["p_hat"] == 0.75


def test_stats_top(tmp_path):
    setup_db(tmp_path)
    with session() as conn:
        repo = MarketStatsRepository(conn)
        repo.bulk_upsert([
            {
                "symbol": "XYZ",
                "timeframe": "1h",
                "event": "ev1",
                "condition_name": "c",
                "condition_value": "A",
                "target": "up",
                "split": "test",
                "n": 10,
                "successes": 7,
                "p_hat": 0.7,
                "ci_low": 0.5,
                "ci_high": 0.9,
                "lift": 0.2,
                "start": "2020",
                "end": "2020",
            },
            {
                "symbol": "XYZ",
                "timeframe": "1h",
                "event": "ev2",
                "condition_name": "c",
                "condition_value": "B",
                "target": "up",
                "split": "test",
                "n": 10,
                "successes": 3,
                "p_hat": 0.3,
                "ci_low": 0.1,
                "ci_high": 0.5,
                "lift": -0.6,
                "start": "2020",
                "end": "2020",
            },
            {
                "symbol": "XYZ",
                "timeframe": "1h",
                "event": "ev3",
                "condition_name": "c",
                "condition_value": "C",
                "target": "up",
                "split": "test",
                "n": 20,
                "successes": 18,
                "p_hat": 0.9,
                "ci_low": 0.8,
                "ci_high": 0.95,
                "lift": 0.4,
                "start": "2020",
                "end": "2020",
            },
        ])
    res = app.stats_top(symbol="XYZ", timeframe="1h", k=2)
    assert len(res) == 2
    assert res[0]["event"] == "ev2"
    assert res[1]["event"] == "ev3"
