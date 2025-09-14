"""API request/response models (light-weight)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from ..core.spec import ValidationSpec, ArtifactsSpec, PersistenceSpec


@dataclass
class SubmitResponse:
    id: str


@dataclass
class StatusResponse:
    status: str


@dataclass
class ResultResponse:
    result: Dict[str, Any] | None


class StatsDataSpec(BaseModel):
    """Dataset location and window for statistics runs."""

    dataset_path: str
    symbols: List[str]
    timeframe: str
    start: str
    end: str


class StatsEventSpec(BaseModel):
    """Specification for an event to detect in the dataset."""

    name: str
    params: Dict[str, Any] = Field(default_factory=dict)


class StatsConditionSpec(BaseModel):
    """Specification for conditioning regime."""

    name: str
    params: Dict[str, Any] = Field(default_factory=dict)


class StatsTargetSpec(BaseModel):
    """Specification for a target metric."""

    name: str
    params: Dict[str, Any] = Field(default_factory=dict)


class StatsSpec(BaseModel):
    """Top-level specification for a statistics run."""

    data: StatsDataSpec
    events: List[StatsEventSpec] = Field(default_factory=list)
    conditions: List[StatsConditionSpec] = Field(default_factory=list)
    targets: List[StatsTargetSpec] = Field(default_factory=list)
    validation: ValidationSpec | None = None
    artifacts: ArtifactsSpec | None = None
    persistence: PersistenceSpec | None = None

