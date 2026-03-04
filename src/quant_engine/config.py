from __future__ import annotations

"""Simple settings loader with environment variables.

This is a lightweight stand-in for ``pydantic-settings`` which is not
available in the execution environment.  The ``get_settings`` function reads
environment variables and caches the resulting ``Settings`` object.  Tests may
call ``reset_settings_cache`` to force a reload when they modify environment
variables at runtime.
"""

from dataclasses import dataclass
import os
from functools import lru_cache


@dataclass
class Settings:
    db_dsn: str | None = None
    db_sqlite_path: str = ".db/quant.db"
    db_echo: bool = False
    mlflow_tracking_uri: str | None = None
    market_intelligence_enabled: bool = False


def _parse_bool(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    return default


@lru_cache()
def get_settings() -> Settings:
    """Return settings loaded from environment variables."""

    db_dsn = os.getenv("DB_DSN")
    db_sqlite_path = os.getenv("DB_SQLITE_PATH", ".db/quant.db")
    db_echo = os.getenv("DB_ECHO", "false").lower() == "true"
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    market_intelligence_enabled = _parse_bool(
        os.getenv("MARKET_INTELLIGENCE_ENABLED"),
        default=False,
    )
    return Settings(
        db_dsn=db_dsn,
        db_sqlite_path=db_sqlite_path,
        db_echo=db_echo,
        mlflow_tracking_uri=mlflow_tracking_uri,
        market_intelligence_enabled=market_intelligence_enabled,
    )


def reset_settings_cache() -> None:
    """Clear the settings cache (mainly for tests)."""

    get_settings.cache_clear()
