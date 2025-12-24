import sqlite3

import pytest

from quant_engine.config import reset_settings_cache
from quant_engine.persistence import db


def test_init_db_creates_tables(monkeypatch):
    monkeypatch.setenv("DB_DSN", "sqlite:///:memory:")
    reset_settings_cache()
    conn = db.connect()
    try:
        db.init_db(conn)
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {row[0] for row in rows}
        expected_tables = {
            "experiment_runs",
            "run_metrics",
            "trials",
            "market_stats",
            "seasonality_profiles",
            "seasonality_runs",
        }
        assert expected_tables.issubset(table_names)
    finally:
        conn.close()


def test_run_id_unique_constraint(monkeypatch):
    monkeypatch.setenv("DB_DSN", "sqlite:///:memory:")
    reset_settings_cache()
    conn = db.connect()
    try:
        db.init_db(conn)
        conn.execute(
            "INSERT INTO experiment_runs (run_id, status) VALUES (?, ?)",
            ("run-1", "started"),
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO experiment_runs (run_id, status) VALUES (?, ?)",
                ("run-1", "started"),
            )
    finally:
        conn.close()


def test_non_sqlite_dsn_raises(monkeypatch):
    monkeypatch.setenv("DB_DSN", "mysql://localhost/db")
    reset_settings_cache()
    with pytest.raises(RuntimeError, match="Only sqlite DSNs are supported"):
        db.connect()
