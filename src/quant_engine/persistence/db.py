"""Light-weight persistence layer with SQLite fallback.

The original project targets SQLAlchemy with MySQL, however the execution
environment does not provide SQLAlchemy. This module offers a minimal subset
using ``sqlite3`` so that tests can exercise the persistence logic. The DSN is
controlled through environment variables and mimics the structure expected by
SQLAlchemy-based configurations.

SQLite support is intended for tests and local development only. For production
MySQL deployments, use the SQLAlchemy + Alembic stack and apply the MySQL
migration statements captured alongside the SQLite migrations in this module.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Sequence

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
        raise RuntimeError(
            "Only sqlite DSNs are supported in this environment. "
            "Use SQLAlchemy + Alembic with MySQL in production."
        )
    path = dsn.split("sqlite:///")[1]
    return path


def connect() -> sqlite3.Connection:
    path = _effective_db_path()
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    sqlite_apply: Callable[[sqlite3.Connection], None]
    mysql_statements: Sequence[str]


def _migration_1(conn: sqlite3.Connection) -> None:
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
            start TEXT,
            end TEXT,
            spec_id TEXT,
            dataset_id TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, timeframe, dim, bin, measure, start, end, spec_id, dataset_id)
        )
        """
    )
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
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS api_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL UNIQUE,
            job_type TEXT NOT NULL,
            status TEXT NOT NULL,
            payload_json TEXT,
            result_json TEXT,
            error_message TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            started_at TEXT,
            finished_at TEXT,
            updated_at TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS ix_api_jobs_job_id
        ON api_jobs(job_id)
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_api_jobs_type_status
        ON api_jobs(job_type, status)
        """
    )


def _migration_2(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    try:
        cur.execute("ALTER TABLE seasonality_profiles ADD COLUMN metrics TEXT")
    except sqlite3.OperationalError:
        pass


MYSQL_MIGRATION_1 = (
    """
    CREATE TABLE IF NOT EXISTS experiment_runs (
        id BIGINT PRIMARY KEY AUTO_INCREMENT,
        run_id VARCHAR(255) NOT NULL,
        spec_id VARCHAR(255),
        dataset_id VARCHAR(255),
        status VARCHAR(64),
        objective TEXT,
        out_dir TEXT,
        started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        finished_at TIMESTAMP NULL,
        UNIQUE KEY ux_experiment_runs_run_id (run_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS run_metrics (
        id BIGINT PRIMARY KEY AUTO_INCREMENT,
        run_id VARCHAR(255) NOT NULL,
        fold INTEGER,
        metric_name VARCHAR(255) NOT NULL,
        metric_value DOUBLE NOT NULL,
        UNIQUE KEY ux_run_metrics (run_id, fold, metric_name),
        INDEX ix_run_metrics_run_id (run_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS trials (
        id BIGINT PRIMARY KEY AUTO_INCREMENT,
        run_id VARCHAR(255) NOT NULL,
        trial_number INTEGER NOT NULL,
        params_json TEXT,
        objective_value DOUBLE,
        status VARCHAR(64),
        n_trades INTEGER,
        max_dd DOUBLE,
        sharpe DOUBLE,
        sortino DOUBLE,
        cagr DOUBLE,
        hit_rate DOUBLE,
        avg_r DOUBLE,
        UNIQUE KEY ux_trials_run (run_id, trial_number),
        INDEX ix_trials_run_id (run_id),
        INDEX ix_trials_trial_number (trial_number)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS market_stats (
        id BIGINT PRIMARY KEY AUTO_INCREMENT,
        symbol VARCHAR(64) NOT NULL,
        timeframe VARCHAR(64) NOT NULL,
        event VARCHAR(255) NOT NULL,
        condition_name VARCHAR(255),
        condition_value VARCHAR(255),
        target VARCHAR(255) NOT NULL,
        split VARCHAR(64) NOT NULL,
        n INTEGER NOT NULL,
        successes INTEGER NOT NULL,
        p_hat DOUBLE NOT NULL,
        ci_low DOUBLE,
        ci_high DOUBLE,
        lift DOUBLE NOT NULL,
        start VARCHAR(255) NOT NULL,
        end VARCHAR(255) NOT NULL,
        spec_id VARCHAR(255),
        dataset_id VARCHAR(255),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE KEY ux_market_stats (
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
        ),
        INDEX ix_market_stats_lookup (
            symbol,
            timeframe,
            event,
            condition_name,
            condition_value,
            target,
            split
        )
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS seasonality_profiles (
        id BIGINT PRIMARY KEY AUTO_INCREMENT,
        symbol VARCHAR(64) NOT NULL,
        timeframe VARCHAR(64),
        dim VARCHAR(255) NOT NULL,
        bin INTEGER NOT NULL,
        measure VARCHAR(255) NOT NULL,
        score DOUBLE,
        n INTEGER,
        baseline DOUBLE,
        lift DOUBLE,
        start VARCHAR(255),
        end VARCHAR(255),
        spec_id VARCHAR(255),
        dataset_id VARCHAR(255),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE KEY ux_seasonality_profiles (
            symbol,
            timeframe,
            dim,
            bin,
            measure,
            start,
            end,
            spec_id,
            dataset_id
        ),
        INDEX ix_seasonality_profiles_lookup (symbol, timeframe, dim, measure)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS seasonality_runs (
        id BIGINT PRIMARY KEY AUTO_INCREMENT,
        run_id VARCHAR(255) NOT NULL,
        spec_id VARCHAR(255),
        dataset_id VARCHAR(255),
        out_dir TEXT,
        status VARCHAR(64) NOT NULL,
        best_summary TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE KEY ux_seasonality_runs_run_id (run_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS api_jobs (
        id BIGINT PRIMARY KEY AUTO_INCREMENT,
        job_id VARCHAR(255) NOT NULL,
        job_type VARCHAR(255) NOT NULL,
        status VARCHAR(64) NOT NULL,
        payload_json TEXT,
        result_json TEXT,
        error_message TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        started_at TIMESTAMP NULL,
        finished_at TIMESTAMP NULL,
        updated_at TIMESTAMP NULL,
        UNIQUE KEY ux_api_jobs_job_id (job_id),
        INDEX ix_api_jobs_type_status (job_type, status)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS schema_migrations (
        version INTEGER PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
)

MYSQL_MIGRATION_2 = (
    "ALTER TABLE seasonality_profiles ADD COLUMN metrics TEXT",
)

MIGRATIONS = [
    Migration(1, "initial_schema", _migration_1, MYSQL_MIGRATION_1),
    Migration(2, "seasonality_profiles_metrics", _migration_2, MYSQL_MIGRATION_2),
]


def _ensure_migrations_table(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            applied_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


def migrate(conn: sqlite3.Connection) -> None:
    _ensure_migrations_table(conn)
    cur = conn.cursor()
    applied = {
        row["version"]
        for row in cur.execute("SELECT version FROM schema_migrations").fetchall()
    }
    for migration in MIGRATIONS:
        if migration.version in applied:
            continue
        migration.sqlite_apply(conn)
        cur.execute(
            "INSERT INTO schema_migrations(version, name) VALUES (?, ?)",
            (migration.version, migration.name),
        )
    conn.commit()


def init_db(conn: sqlite3.Connection) -> None:
    migrate(conn)


@contextmanager
def session() -> Iterator[sqlite3.Connection]:
    conn = connect()
    try:
        init_db(conn)
        yield conn
        conn.commit()
    finally:
        conn.close()


def mysql_migration_plan() -> list[tuple[int, str, Sequence[str]]]:
    """Return MySQL-compatible DDL statements for each migration."""
    return [
        (migration.version, migration.name, migration.mysql_statements)
        for migration in MIGRATIONS
    ]


__all__ = ["connect", "init_db", "migrate", "mysql_migration_plan", "session"]
