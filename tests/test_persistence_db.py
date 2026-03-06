import sqlite3
import sys
import types

import pytest

if "pymysql" not in sys.modules:
    pymysql_stub = types.ModuleType("pymysql")
    pymysql_stub.connect = lambda *args, **kwargs: None
    cursors_stub = types.ModuleType("pymysql.cursors")
    cursors_stub.DictCursor = object
    pymysql_stub.cursors = cursors_stub
    sys.modules["pymysql"] = pymysql_stub
    sys.modules["pymysql.cursors"] = cursors_stub

from quant_engine.config import reset_settings_cache
from quant_engine.persistence import db
from quant_engine.persistence.repositories import extract_dca_run_metrics


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


def test_mysql_dsn_uses_mysql_connector(monkeypatch):
    class _FakeCursor:
        def execute(self, *_args, **_kwargs):
            return None

    class _FakeConn:
        def cursor(self):
            return _FakeCursor()

        def commit(self):
            return None

        def rollback(self):
            return None

        def close(self):
            return None

    monkeypatch.setenv("DB_DSN", "mysql://localhost/db")
    reset_settings_cache()
    monkeypatch.setattr(db.pymysql, "connect", lambda **_kwargs: _FakeConn())
    conn = db.connect()
    try:
        assert conn.dialect == "mysql"
    finally:
        conn.close()


def test_migrate_mysql_ignores_duplicate_column_when_version_missing() -> None:
    executed: list[tuple[str, object | None]] = []
    inserted_versions: list[int] = []

    class _FakeCursor:
        def execute(self, sql, params=None):
            executed.append((sql, params))
            if sql == "SELECT version FROM schema_migrations":
                return self
            if sql == "ALTER TABLE market_stats ADD COLUMN p_mean DOUBLE NULL":
                raise Exception(1060, "Duplicate column name 'p_mean'")
            if sql.startswith("INSERT INTO schema_migrations"):
                inserted_versions.append(int(params[0]))
            return self

        def fetchall(self):
            return [{"version": 1}, {"version": 2}, {"version": 3}]

    class _FakeConn:
        dialect = "mysql"

        def __init__(self):
            self.cursor_obj = _FakeCursor()
            self.committed = False

        def cursor(self):
            return self.cursor_obj

        def commit(self):
            self.committed = True

    conn = _FakeConn()

    db.migrate(conn)

    assert conn.committed is True
    assert 4 in inserted_versions


def test_extract_dca_run_metrics_flattens_ratio_and_efficiency() -> None:
    extra = {
        "capital_efficiency_index": 1.2,
        "return_over_stress_ratio": {
            "version": "return_over_stress_ratio_v1",
            "ratio": 0.8,
            "numerator": 0.12,
            "denominator": 0.15,
            "denominator_raw": 0.15,
            "epsilon": 1e-9,
        },
        "xirr_status": "ok",
    }
    metrics = extract_dca_run_metrics(extra)
    assert metrics["capital_efficiency_index"] == 1.2
    assert metrics["return_over_stress_ratio_ratio"] == 0.8
    assert metrics["return_over_stress_ratio_numerator"] == 0.12
    assert metrics["return_over_stress_ratio_denominator"] == 0.15
    assert metrics["return_over_stress_ratio_denominator_raw"] == 0.15
    assert metrics["return_over_stress_ratio_epsilon"] == 1e-9
    assert metrics["xirr_converged"] == 1.0
