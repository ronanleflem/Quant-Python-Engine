"""Light-weight persistence layer with SQLite fallback.

The original project targets SQLAlchemy with MySQL, however the execution
environment does not provide SQLAlchemy.  This module offers a minimal subset
using ``sqlite3`` so that tests can exercise the persistence logic.  The DSN is
controlled through environment variables and mimics the structure expected by
SQLAlchemy-based configurations.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from ..config import get_settings


def _effective_db_path() -> str:
    """Resolve the database path from settings.

    Only SQLite paths are supported in this lightweight implementation.  The
    DSN form ``sqlite:///path`` or ``sqlite:///:memory:`` is understood.  When no
    DSN is provided the default path from settings is used.
    """

    settings = get_settings()
    dsn = settings.db_dsn
    if not dsn:
        dsn = f"sqlite:///{settings.db_sqlite_path}"
    if not dsn.startswith("sqlite"):
        raise RuntimeError("Only sqlite DSNs are supported in this environment")
    path = dsn.split("sqlite:///")[1]
    return path


def connect() -> sqlite3.Connection:
    path = _effective_db_path()
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS experiment_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL UNIQUE,
            spec_id TEXT,
            dataset_id TEXT,
            status TEXT,
            objective TEXT,
            out_dir TEXT,
            started_at TEXT DEFAULT CURRENT_TIMESTAMP,
            finished_at TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS ix_experiment_runs_run_id
        ON experiment_runs(run_id)
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS run_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            fold INTEGER,
            metric_name TEXT NOT NULL,
            metric_value REAL NOT NULL,
            UNIQUE(run_id, fold, metric_name)
        )
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_run_metrics_run_id
        ON run_metrics(run_id)
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS trials (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            trial_number INTEGER NOT NULL,
            params_json TEXT,
            objective_value REAL,
            status TEXT,
            n_trades INTEGER,
            max_dd REAL,
            sharpe REAL,
            sortino REAL,
            cagr REAL,
            hit_rate REAL,
            avg_r REAL,
            UNIQUE(run_id, trial_number)
        )
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_trials_run_id
        ON trials(run_id)
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_trials_trial_number
        ON trials(trial_number)
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS market_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            event TEXT NOT NULL,
            condition_name TEXT,
            condition_value TEXT,
            target TEXT NOT NULL,
            split TEXT NOT NULL,
            n INTEGER NOT NULL,
            successes INTEGER NOT NULL,
            p_hat REAL NOT NULL,
            ci_low REAL,
            ci_high REAL,
            lift REAL NOT NULL,
            start TEXT NOT NULL,
            end TEXT NOT NULL,
            spec_id TEXT,
            dataset_id TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(
                symbol,
                timeframe,
                event,
                condition_name,
                condition_value,
                target,
                split,
                start,
                end,
                spec_id
            )
        )
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_market_stats_lookup
        ON market_stats(
            symbol,
            timeframe,
            event,
            condition_name,
            condition_value,
            target,
            split
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS seasonality_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timeframe TEXT,
            dim TEXT NOT NULL,
            bin INTEGER NOT NULL,
            measure TEXT NOT NULL,
            score REAL,
            n INTEGER,
            baseline REAL,
            lift REAL,
            metrics TEXT,
            start TEXT,
            end TEXT,
            spec_id TEXT,
            dataset_id TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, timeframe, dim, bin, measure, start, end, spec_id, dataset_id)
        )
        """
    )
    try:
        cur.execute("ALTER TABLE seasonality_profiles ADD COLUMN metrics TEXT")
    except sqlite3.OperationalError:
        pass
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_seasonality_profiles_lookup
        ON seasonality_profiles(symbol, timeframe, dim, measure)
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS seasonality_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL UNIQUE,
            spec_id TEXT,
            dataset_id TEXT,
            out_dir TEXT,
            status TEXT NOT NULL,
            best_summary TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cur.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS ix_seasonality_runs_run_id
        ON seasonality_runs(run_id)
        """
    )
    conn.commit()


@contextmanager
def session() -> Iterator[sqlite3.Connection]:
    conn = connect()
    try:
        init_db(conn)
        yield conn
        conn.commit()
    finally:
        conn.close()


__all__ = ["connect", "init_db", "session"]
