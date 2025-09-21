"""Utilities to normalise seasonality specifications."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..api.schemas import (
    SeasonalitySpec,
    SeasonalityProfileSpec,
    SeasonalitySignalSpec,
    SeasonalityComputeSpec,
    ExecutionSpec,
    RiskSpec,
    TPSSLSpec,
    ValidationSpec,
    ArtifactsSpec,
    PersistenceSpec,
)


@dataclass
class NormalisedSeasonalitySpec:
    dataset_path: Path | None
    symbols: list[str]
    timeframe: str
    start: datetime
    end: datetime
    profile: SeasonalityProfileSpec
    signal: SeasonalitySignalSpec
    compute: SeasonalityComputeSpec
    execution: ExecutionSpec
    risk: RiskSpec
    tp_sl: TPSSLSpec
    validation: ValidationSpec
    artifacts: ArtifactsSpec
    persistence: PersistenceSpec


def normalise(spec: SeasonalitySpec) -> NormalisedSeasonalitySpec:
    """Convert the Pydantic specification into python-native objects."""

    start = datetime.fromisoformat(spec.data.start)
    end = datetime.fromisoformat(spec.data.end)
    return NormalisedSeasonalitySpec(
        dataset_path=Path(spec.data.dataset_path) if spec.data.dataset_path else None,
        symbols=list(spec.data.symbols),
        timeframe=spec.data.timeframe,
        start=start,
        end=end,
        profile=spec.profile,
        signal=spec.signal,
        compute=spec.compute,
        execution=spec.execution,
        risk=spec.risk,
        tp_sl=spec.tp_sl,
        validation=spec.validation,
        artifacts=spec.artifacts,
        persistence=spec.persistence,
    )
