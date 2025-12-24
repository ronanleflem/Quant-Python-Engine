import os
from pathlib import Path

import pytest
from fastapi import HTTPException

from quant_engine.api import app
from quant_engine.config import reset_settings_cache
from quant_engine.persistence import session


DB_PATH = Path(__file__).resolve().parents[1] / "persistence.db"


@pytest.fixture
def persistence_db():
    os.environ["DB_DSN"] = f"sqlite:///{DB_PATH}"
    reset_settings_cache()
    with session() as conn:
        conn.execute("DELETE FROM experiment_runs")
        conn.execute("DELETE FROM run_metrics")
        conn.execute("DELETE FROM trials")
        conn.execute("DELETE FROM market_stats")
        conn.execute("DELETE FROM seasonality_profiles")
        conn.execute("DELETE FROM seasonality_runs")
    yield DB_PATH
    os.environ.pop("DB_DSN", None)
    reset_settings_cache()


def test_levels_list_invalid_date(monkeypatch):
    monkeypatch.setattr(app, "_resolve_levels_engine", lambda: object())

    with pytest.raises(HTTPException) as exc:
        app.levels_list(symbol="BTCUSDT", start="nope")

    assert exc.value.status_code == 400


def test_stats_summary_fields(persistence_db):
    with session() as conn:
        conn.execute(
            """
            INSERT INTO market_stats (
                symbol, timeframe, event, condition_name, condition_value, target,
                split, n, successes, p_hat, ci_low, ci_high, lift, start, end
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "BTCUSDT",
                "1h",
                "breakout",
                "rsi",
                "30",
                "up",
                "train",
                20,
                8,
                0.4,
                0.2,
                0.6,
                0.1,
                "2024-01-01",
                "2024-01-31",
            ),
        )
    summary = app.stats_summary(symbol="BTCUSDT", timeframe="1h", event="breakout")
    assert summary
    row = summary[0]
    expected_keys = {
        "condition_name",
        "condition_value",
        "target",
        "n",
        "successes",
        "p_hat",
        "ci_low",
        "ci_high",
    }
    assert expected_keys.issubset(row.keys())


def test_list_runs_pagination_and_filters(persistence_db):
    with session() as conn:
        conn.executemany(
            """
            INSERT INTO experiment_runs (
                run_id, status, objective, out_dir, started_at, finished_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "run-1",
                    "FINISHED",
                    "return",
                    "/tmp/run1",
                    "2024-01-02T00:00:00",
                    "2024-01-02T01:00:00",
                ),
                (
                    "run-2",
                    "FAILED",
                    "return",
                    "/tmp/run2",
                    "2024-01-03T00:00:00",
                    None,
                ),
                (
                    "run-3",
                    "FINISHED",
                    "return",
                    "/tmp/run3",
                    "2024-01-01T00:00:00",
                    None,
                ),
            ],
        )

    page_one = app.list_runs(status="FINISHED", page=1, page_size=1)
    page_two = app.list_runs(status="FINISHED", page=2, page_size=1)

    assert [r["run_id"] for r in page_one] == ["run-1"]
    assert [r["run_id"] for r in page_two] == ["run-3"]

    date_filtered = app.list_runs(
        date_from="2024-01-02T00:00:00",
        date_to="2024-01-03T23:59:59",
    )
    assert {r["run_id"] for r in date_filtered} == {"run-1", "run-2"}
