"""Persistence layer supporting SQLite and MySQL.

SQLite remains the default for local development/tests.
When ``DB_DSN`` starts with ``mysql`` this module opens a PyMySQL connection and
applies MySQL migration statements from ``MIGRATIONS``.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Callable, Iterator, Sequence

import pymysql
from pymysql.cursors import DictCursor
from sqlalchemy.engine.url import make_url

from ..config import get_settings


def _effective_dsn() -> str:
    settings = get_settings()
    dsn = settings.db_dsn
    if not dsn:
        dsn = f"sqlite:///{settings.db_sqlite_path}"
    return dsn


def _dsn_dialect(dsn: str) -> str:
    if dsn.startswith("sqlite"):
        return "sqlite"
    if dsn.startswith("mysql"):
        return "mysql"
    raise RuntimeError(f"Unsupported DB_DSN dialect: {dsn}")


def _effective_db_path() -> str:
    dsn = _effective_dsn()
    if not dsn.startswith("sqlite"):
        raise RuntimeError("DB path requested for non-sqlite DSN")
    return dsn.split("sqlite:///")[1]


def _convert_placeholders_mysql(sql: str) -> str:
    return sql.replace("?", "%s")


def _rewrite_insert_or_replace_mysql(sql: str) -> str:
    pattern = re.compile(
        r"INSERT\s+OR\s+REPLACE\s+INTO\s+([a-zA-Z0-9_]+)\s*\((.*?)\)\s*VALUES\s*\((.*?)\)",
        re.IGNORECASE | re.DOTALL,
    )
    m = pattern.search(sql)
    if not m:
        return sql
    table = m.group(1)
    cols = [c.strip() for c in m.group(2).split(",") if c.strip()]
    values = m.group(3).strip()
    updates = ", ".join([f"{c}=VALUES({c})" for c in cols])
    return f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({values}) ON DUPLICATE KEY UPDATE {updates}"


def _rewrite_on_conflict_mysql(sql: str) -> str:
    normalized = sql
    do_nothing = re.search(
        r"ON\s+CONFLICT\s*\((.*?)\)\s*DO\s+NOTHING\s*$",
        normalized,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if do_nothing:
        base = normalized[: do_nothing.start()].rstrip()
        base = re.sub(r"INSERT\s+INTO", "INSERT IGNORE INTO", base, flags=re.IGNORECASE, count=1)
        return base

    do_update = re.search(
        r"ON\s+CONFLICT\s*\((.*?)\)\s*DO\s+UPDATE\s+SET\s*(.*)$",
        normalized,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if do_update:
        base = normalized[: do_update.start()].rstrip()
        set_clause = do_update.group(2).strip()
        set_clause = re.sub(
            r"excluded\.([a-zA-Z0-9_]+)",
            r"VALUES(\1)",
            set_clause,
            flags=re.IGNORECASE,
        )
        return f"{base} ON DUPLICATE KEY UPDATE {set_clause}"
    return normalized


def _prepare_sql_for_mysql(sql: str) -> str:
    out = sql
    if re.search(r"^\s*BEGIN\s+IMMEDIATE\s*$", out, flags=re.IGNORECASE):
        return "START TRANSACTION"
    if re.search(r"INSERT\s+OR\s+REPLACE", out, flags=re.IGNORECASE):
        out = _rewrite_insert_or_replace_mysql(out)
    if re.search(r"ON\s+CONFLICT", out, flags=re.IGNORECASE):
        out = _rewrite_on_conflict_mysql(out)
    out = _convert_placeholders_mysql(out)
    return out


_ISO_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")
_ISO_OFFSET_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?[+-]\d{2}:\d{2}$")


def _normalize_mysql_param_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    if _ISO_UTC_RE.match(value):
        return value[:-1].replace("T", " ")
    if _ISO_OFFSET_RE.match(value):
        # Drop timezone offset for TIMESTAMP columns.
        return value[:19].replace("T", " ")
    return value


def _normalize_mysql_params(params: Any) -> Any:
    if params is None:
        return None
    if isinstance(params, (list, tuple)):
        return tuple(_normalize_mysql_param_value(v) for v in params)
    if isinstance(params, dict):
        return {k: _normalize_mysql_param_value(v) for k, v in params.items()}
    return params


class _CompatCursor:
    def __init__(self, raw_cursor: Any, *, dialect: str):
        self._raw = raw_cursor
        self._dialect = dialect

    def execute(self, sql: str, params: Any = None) -> "_CompatCursor":
        stmt = _prepare_sql_for_mysql(sql) if self._dialect == "mysql" else sql
        exec_params = _normalize_mysql_params(params) if self._dialect == "mysql" else params
        if params is None:
            self._raw.execute(stmt)
            return self
        self._raw.execute(stmt, exec_params)
        return self

    def executemany(self, sql: str, seq_params: Sequence[Any]) -> "_CompatCursor":
        stmt = _prepare_sql_for_mysql(sql) if self._dialect == "mysql" else sql
        exec_params = [_normalize_mysql_params(p) for p in seq_params] if self._dialect == "mysql" else seq_params
        self._raw.executemany(stmt, exec_params)
        return self

    def fetchone(self) -> Any:
        return self._raw.fetchone()

    def fetchall(self) -> Any:
        return self._raw.fetchall()

    def __iter__(self):
        return iter(self._raw)

    @property
    def rowcount(self) -> int:
        return int(getattr(self._raw, "rowcount", 0))


class _CompatConnection:
    def __init__(self, raw_conn: Any, *, dialect: str):
        self._raw = raw_conn
        self.dialect = dialect

    def cursor(self) -> _CompatCursor:
        return _CompatCursor(self._raw.cursor(), dialect=self.dialect)

    def execute(self, sql: str, params: Any = None) -> _CompatCursor:
        cur = self.cursor()
        cur.execute(sql, params)
        return cur

    def commit(self) -> None:
        self._raw.commit()

    def rollback(self) -> None:
        self._raw.rollback()

    def close(self) -> None:
        self._raw.close()

    def __getattr__(self, item: str) -> Any:
        return getattr(self._raw, item)


def connect() -> _CompatConnection:
    dsn = _effective_dsn()
    dialect = _dsn_dialect(dsn)
    if dialect == "sqlite":
        path = _effective_db_path()
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        raw_conn = sqlite3.connect(path)
        raw_conn.row_factory = sqlite3.Row
        return _CompatConnection(raw_conn, dialect="sqlite")

    url = make_url(dsn)
    query = dict(url.query or {})
    charset = str(query.get("charset") or "utf8mb4")
    raw_conn = pymysql.connect(
        host=url.host or "127.0.0.1",
        user=url.username or "",
        password=url.password or "",
        database=url.database or "",
        port=int(url.port or 3306),
        charset=charset,
        autocommit=False,
        cursorclass=DictCursor,
    )
    return _CompatConnection(raw_conn, dialect="mysql")


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    sqlite_apply: Callable[[Any], None]
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


def _migration_3(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    for statement in (
        "ALTER TABLE api_jobs ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE api_jobs ADD COLUMN max_attempts INTEGER",
        "ALTER TABLE api_jobs ADD COLUMN timeout_seconds INTEGER",
        "ALTER TABLE api_jobs ADD COLUMN progress_json TEXT",
        "ALTER TABLE api_jobs ADD COLUMN cancel_requested INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE api_jobs ADD COLUMN canceled_at TEXT",
    ):
        try:
            cur.execute(statement)
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
            event(64),
            condition_name(64),
            condition_value(64),
            target(64),
            split,
            start(32),
            end(32),
            spec_id(64)
        ),
        INDEX ix_market_stats_lookup (
            symbol,
            timeframe,
            event(64),
            condition_name(64),
            condition_value(64),
            target(64),
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
            dim(64),
            bin,
            measure(64),
            start(32),
            end(32),
            spec_id(64),
            dataset_id(64)
        ),
        INDEX ix_seasonality_profiles_lookup (symbol, timeframe, dim(64), measure(64))
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

MYSQL_MIGRATION_3 = (
    "ALTER TABLE api_jobs ADD COLUMN attempts INT NOT NULL DEFAULT 0",
    "ALTER TABLE api_jobs ADD COLUMN max_attempts INT NULL",
    "ALTER TABLE api_jobs ADD COLUMN timeout_seconds INT NULL",
    "ALTER TABLE api_jobs ADD COLUMN progress_json TEXT",
    "ALTER TABLE api_jobs ADD COLUMN cancel_requested TINYINT NOT NULL DEFAULT 0",
    "ALTER TABLE api_jobs ADD COLUMN canceled_at TIMESTAMP NULL",
)

MIGRATIONS = [
    Migration(1, "initial_schema", _migration_1, MYSQL_MIGRATION_1),
    Migration(2, "seasonality_profiles_metrics", _migration_2, MYSQL_MIGRATION_2),
    Migration(3, "api_jobs_queue_fields", _migration_3, MYSQL_MIGRATION_3),
]


def _ensure_migrations_table(conn: Any) -> None:
    cur = conn.cursor()
    if getattr(conn, "dialect", "sqlite") == "mysql":
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
    else:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                applied_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def migrate(conn: Any) -> None:
    _ensure_migrations_table(conn)
    cur = conn.cursor()
    applied = {
        row["version"]
        for row in cur.execute("SELECT version FROM schema_migrations").fetchall()
    }
    for migration in MIGRATIONS:
        if migration.version in applied:
            continue
        if getattr(conn, "dialect", "sqlite") == "mysql":
            for statement in migration.mysql_statements:
                cur.execute(statement)
            cur.execute(
                "INSERT INTO schema_migrations(version, name) VALUES (?, ?)",
                (migration.version, migration.name),
            )
        else:
            migration.sqlite_apply(conn)
            cur.execute(
                "INSERT INTO schema_migrations(version, name) VALUES (?, ?)",
                (migration.version, migration.name),
            )
    conn.commit()


def init_db(conn: Any) -> None:
    migrate(conn)


@contextmanager
def session() -> Iterator[Any]:
    conn = connect()
    try:
        init_db(conn)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def mysql_migration_plan() -> list[tuple[int, str, Sequence[str]]]:
    """Return MySQL-compatible DDL statements for each migration."""
    return [
        (migration.version, migration.name, migration.mysql_statements)
        for migration in MIGRATIONS
    ]


__all__ = ["connect", "init_db", "migrate", "mysql_migration_plan", "session"]
