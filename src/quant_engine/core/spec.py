"""Specification models for strategies.

This module provides light-weight dataclass based models to parse the
JSON specification used throughout the project.  It intentionally avoids
external dependencies such as Pydantic so that the test environment can
run without optional packages installed.  The models capture just enough
structure for the unit tests in this kata.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
from typing import List, Dict, Any


@dataclass
class DataSpec:
    """Location and filtering information for the dataset."""

    path: str
    symbols: List[str]
    start: date
    end: date


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

    min_trades: int
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


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def load_spec(path: str | Path) -> Spec:
    """Load a :class:`Spec` instance from a JSON file."""
    raw = json.loads(Path(path).read_text())
    data_raw = raw["data"]
    strat_raw = raw["strategy"]

    data = DataSpec(
        path=data_raw["path"],
        symbols=list(data_raw["symbols"]),
        start=_parse_date(data_raw["start"]),
        end=_parse_date(data_raw["end"]),
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

