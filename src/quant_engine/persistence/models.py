"""Light-weight data models for persistence layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class MarketStat:
    """Represents a row in the ``market_stats`` table."""

    id: Optional[int] = None
    symbol: str | None = None
    timeframe: str | None = None
    event: str | None = None
    condition_name: str | None = None
    condition_value: Any = None
    target: str | None = None
    split: str | None = None
    n: int | None = None
    successes: int | None = None
    p_hat: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    lift: float | None = None
    start: str | None = None
    end: str | None = None
    spec_id: str | None = None
    dataset_id: str | None = None
    created_at: str | None = None


@dataclass
class SeasonalityProfile:
    """Represents a row in the ``seasonality_profiles`` table."""

    id: Optional[int] = None
    symbol: str | None = None
    timeframe: str | None = None
    dim: str | None = None
    bin: int | None = None
    measure: str | None = None
    score: float | None = None
    n: int | None = None
    baseline: float | None = None
    lift: float | None = None
    start: str | None = None
    end: str | None = None
    spec_id: str | None = None
    dataset_id: str | None = None
    created_at: str | None = None


@dataclass
class SeasonalityRun:
    """Represents a row in the ``seasonality_runs`` table."""

    id: Optional[int] = None
    run_id: str | None = None
    spec_id: str | None = None
    dataset_id: str | None = None
    out_dir: str | None = None
    status: str | None = None
    best_summary: Any = None
    created_at: str | None = None


__all__ = ["MarketStat", "SeasonalityProfile", "SeasonalityRun"]

