"""Specification models for strategies.

This module provides light-weight dataclass based models to parse the
JSON specification used throughout the project.  It intentionally avoids
external dependencies such as Pydantic so that the test environment can
run without optional packages installed.  The models capture just enough
structure for the unit tests in this kata.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import List, Dict, Any, Mapping


@dataclass
class MySQLDataConfig:
    """MySQL connection information for datasets."""

    connection_url: str | None = None
    env_var: str | None = "QE_MARKETDATA_MYSQL_URL"
    schema: str | None = None
    table: str = "ohlcv"
    symbol_col: str = "symbol"
    ts_col: str = "ts"
    open_col: str = "open"
    high_col: str = "high"
    low_col: str = "low"
    close_col: str = "close"
    volume_col: str = "volume"
    timeframe_col: str | None = "timeframe"
    extra_where: str | None = None
    chunk_minutes: int = 0
    symbol_lookup_table: str | None = None
    symbol_lookup_symbol_col: str = "symbol"
    symbol_lookup_id_col: str = "id"


@dataclass
class DataSpec:
    """Location and filtering information for the dataset."""

    dataset_path: str | None
    mysql: MySQLDataConfig | None
    symbols: List[str]
    timeframe: str | None
    start: str
    end: str


@dataclass
class FiltersSpec:
    """Parameters for pre-trade filters (e.g. EMAs)."""

    ema_fast: int
    ema_slow: int


@dataclass
class TPSLSpec:
    atr_k: float


@dataclass
class ValidationSpec:
    """Walk-forward validation configuration."""

    min_trades: int = 0
    train_months: int = 0
    test_months: int = 1
    folds: int = 1
    embargo_days: int = 0


@dataclass
class ArtifactsSpec:
    """Persistence configuration for locally written artifacts."""

    out_dir: str | None = None


@dataclass
class PersistenceSpec:
    """Configuration for optional database persistence."""

    enabled: bool = False
    spec_id: str | None = None
    dataset_id: str | None = None


@dataclass
class StrategySpec:
    filters: FiltersSpec
    tpsl: TPSLSpec
    validation: ValidationSpec
    objective: str
    search_space: Dict[str, List[int]]


@dataclass
class Spec:
    data: DataSpec
    strategy: StrategySpec


# ---------------------------------------------------------------------------


def _parse_spec(raw: Mapping[str, Any]) -> Spec:
    """Build a :class:`Spec` from an in-memory mapping."""

    data_raw = raw["data"]
    strat_raw = raw["strategy"]

    dataset_path = data_raw.get("dataset_path") or data_raw.get("path")
    mysql_raw = data_raw.get("mysql")
    mysql: MySQLDataConfig | None = None
    if mysql_raw is not None:
        mysql = MySQLDataConfig(
            connection_url=mysql_raw.get("connection_url"),
            env_var=mysql_raw.get("env_var", "QE_MARKETDATA_MYSQL_URL"),
            schema=mysql_raw.get("schema"),
            table=mysql_raw.get("table", "ohlcv"),
            symbol_col=mysql_raw.get("symbol_col", "symbol"),
            ts_col=mysql_raw.get("ts_col", "ts"),
            open_col=mysql_raw.get("open_col", "open"),
            high_col=mysql_raw.get("high_col", "high"),
            low_col=mysql_raw.get("low_col", "low"),
            close_col=mysql_raw.get("close_col", "close"),
            volume_col=mysql_raw.get("volume_col", "volume"),
            timeframe_col=mysql_raw.get("timeframe_col", "timeframe"),
            extra_where=mysql_raw.get("extra_where"),
            chunk_minutes=int(mysql_raw.get("chunk_minutes", 0)),
            symbol_lookup_table=mysql_raw.get("symbol_lookup_table"),
            symbol_lookup_symbol_col=mysql_raw.get("symbol_lookup_symbol_col", "symbol"),
            symbol_lookup_id_col=mysql_raw.get("symbol_lookup_id_col", "id"),
        )

    if dataset_path is None and mysql is None:
        raise ValueError("data must provide either dataset_path/path or mysql configuration")

    symbols = list(data_raw.get("symbols", []))
    timeframe = data_raw.get("timeframe")
    start = data_raw.get("start")
    end = data_raw.get("end")
    if start is None or end is None:
        raise ValueError("data.start and data.end are required")

    data = DataSpec(
        dataset_path=dataset_path,
        mysql=mysql,
        symbols=symbols,
        timeframe=timeframe,
        start=str(start),
        end=str(end),
    )
    filters = FiltersSpec(**strat_raw["filters"])
    tpsl = TPSLSpec(**strat_raw["tpsl"])
    val_raw = strat_raw.get("validation", {})
    validation = ValidationSpec(
        min_trades=val_raw.get("min_trades", 0),
        train_months=val_raw.get("train_months", 0),
        test_months=val_raw.get("test_months", 1),
        folds=val_raw.get("folds", 1),
        embargo_days=val_raw.get("embargo_days", 0),
    )
    strategy = StrategySpec(
        filters=filters,
        tpsl=tpsl,
        validation=validation,
        objective=strat_raw["objective"],
        search_space={k: list(v) for k, v in strat_raw["search_space"].items()},
    )
    return Spec(data=data, strategy=strategy)


def load_spec(path: str | Path) -> Spec:
    """Load a :class:`Spec` instance from a JSON file."""

    raw = json.loads(Path(path).read_text())
    return _parse_spec(raw)


def spec_from_dict(raw: Mapping[str, Any]) -> Spec:
    """Public helper to build a :class:`Spec` from a JSON-compatible dict."""

    return _parse_spec(raw)


