import os
import sqlite3

import pytest

from quant_engine.api import app as api_app
from quant_engine.config import reset_settings_cache
from quant_engine.persistence import session


@pytest.fixture
def api_db(tmp_path):
    db_path = tmp_path / "api.sqlite"
    os.environ["DB_DSN"] = f"sqlite:///{db_path}"
    reset_settings_cache()

    with session() as conn:
        for column, col_type in (("significant", "INTEGER"), ("q_value", "REAL")):
            try:
                conn.execute(f"ALTER TABLE market_stats ADD COLUMN {column} {col_type}")
            except sqlite3.OperationalError:
                pass

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
                    "/tmp/run-1",
                    "2024-01-03T00:00:00",
                    "2024-01-03T01:00:00",
                ),
                (
                    "run-2",
                    "FINISHED",
                    "return",
                    "/tmp/run-2",
                    "2024-01-02T00:00:00",
                    None,
                ),
                (
                    "run-3",
                    "FAILED",
                    "return",
                    "/tmp/run-3",
                    "2024-01-01T00:00:00",
                    None,
                ),
            ],
        )
        conn.executemany(
            """
            INSERT INTO run_metrics (run_id, fold, metric_name, metric_value)
            VALUES (?, ?, ?, ?)
            """,
            [
                ("run-1", None, "sharpe", 1.1),
                ("run-1", None, "return", 0.12),
                ("run-1", 0, "sharpe", 1.0),
                ("run-1", 1, "sharpe", 1.2),
                ("run-2", None, "sharpe", 0.5),
            ],
        )
        conn.executemany(
            """
            INSERT INTO market_stats (
                symbol, timeframe, event, condition_name, condition_value, target,
                split, n, successes, p_hat, ci_low, ci_high, lift, start, end,
                significant, q_value
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "BTCUSDT",
                    "1h",
                    "breakout",
                    "rsi",
                    "30",
                    "up",
                    "test",
                    100,
                    60,
                    0.6,
                    0.5,
                    0.7,
                    0.12,
                    "2024-01-01",
                    "2024-01-31",
                    1,
                    0.01,
                ),
                (
                    "BTCUSDT",
                    "1h",
                    "breakout",
                    "rsi",
                    "70",
                    "up",
                    "test",
                    100,
                    40,
                    0.4,
                    0.3,
                    0.5,
                    -0.08,
                    "2024-01-01",
                    "2024-01-31",
                    0,
                    0.2,
                ),
                (
                    "ETHUSDT",
                    "1h",
                    "breakout",
                    "rsi",
                    "30",
                    "up",
                    "test",
                    50,
                    35,
                    0.7,
                    0.6,
                    0.8,
                    0.25,
                    "2024-01-01",
                    "2024-01-31",
                    1,
                    0.02,
                ),
            ],
        )

    yield db_path

    os.environ.pop("DB_DSN", None)
    reset_settings_cache()


def test_runs_endpoints_pagination_filters(api_db):
    params = {"status": "FINISHED", "page": 1, "page_size": 1}
    expected = api_app.list_runs(**params)
    resp = api_app.runs_list_endpoint(**params)

    assert resp == expected

    params_page_two = {"status": "FINISHED", "page": 2, "page_size": 1}
    expected_page_two = api_app.list_runs(**params_page_two)
    resp_page_two = api_app.runs_list_endpoint(**params_page_two)

    assert resp_page_two == expected_page_two

    date_params = {
        "date_from": "2024-01-02T00:00:00",
        "date_to": "2024-01-03T23:59:59",
        "page": 1,
        "page_size": 50,
    }
    expected_dates = api_app.list_runs(**date_params)
    resp_dates = api_app.runs_list_endpoint(**date_params)

    assert resp_dates == expected_dates


def test_run_detail_and_metrics_endpoints(api_db):
    expected_run = api_app.get_run("run-1")
    resp = api_app.run_detail_endpoint("run-1")

    assert resp == expected_run

    expected_metrics = api_app.get_run_metrics("run-1")
    resp_metrics = api_app.run_metrics_endpoint("run-1")

    assert resp_metrics == expected_metrics


def test_stats_endpoint_filters_and_normalization(api_db):
    params = {
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "event": "breakout",
        "significant_only": "true",
        "method": "freq",
    }
    expected = api_app.list_stats(
        symbol="BTCUSDT",
        timeframe="1h",
        event="breakout",
        significant_only=True,
        method="freq",
    )
    resp = api_app.stats_list_endpoint(
        symbol=params["symbol"],
        timeframe=params["timeframe"],
        event=params["event"],
        condition_name=None,
        target=None,
        split=None,
        min_n=None,
        significant_only=True,
        method=params["method"],
        alpha=0.05,
        page=1,
        page_size=50,
    )

    assert resp == expected
    for row in resp:
        assert row["lift_freq"] == row["lift"]
        assert row["lift_bayes"] == row["lift_freq"]

    params_bayes = {
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "event": "breakout",
        "method": "bayes",
    }
    expected_bayes = api_app.list_stats(
        symbol="BTCUSDT",
        timeframe="1h",
        event="breakout",
        method="bayes",
    )
    resp_bayes = api_app.stats_list_endpoint(
        symbol=params_bayes["symbol"],
        timeframe=params_bayes["timeframe"],
        event=params_bayes["event"],
        condition_name=None,
        target=None,
        split=None,
        min_n=None,
        method=params_bayes["method"],
        alpha=0.05,
        page=1,
        page_size=50,
    )

    assert resp_bayes == expected_bayes
    for row in resp_bayes:
        assert row["lift_freq"] == row["lift"]
        assert row["lift_bayes"] == row["lift_freq"]
