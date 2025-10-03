"""Pydantic models describing level detection and persistence."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from ..api.schemas import DataInputSpec


class LevelType(str, Enum):
    """Supported logical level categories."""

    PDH = "PDH"
    PDL = "PDL"
    PWH = "PWH"
    PWL = "PWL"
    PMH = "PMH"
    PML = "PML"
    GAP_D = "GAP_D"
    GAP_W = "GAP_W"
    FVG = "FVG"
    POC = "POC"
    RN = "RN"


class LevelRecord(BaseModel):
    """Single detected level ready for persistence."""

    model_config = ConfigDict(extra="forbid")

    symbol: str
    timeframe: str
    level_type: Literal[
        "PDH",
        "PDL",
        "PWH",
        "PWL",
        "PMH",
        "PML",
        "GAP_D",
        "GAP_W",
        "FVG",
        "POC",
        "RN",
    ]
    price: Optional[float] = None
    price_lo: Optional[float] = None
    price_hi: Optional[float] = None
    anchor_ts: datetime
    valid_from_ts: Optional[datetime] = None
    valid_to_ts: Optional[datetime] = None
    params_hash: Optional[str] = None
    source: str = "python-levels"


class LevelsBuildItem(BaseModel):
    """Description of a detector to apply."""

    type: str
    params: Dict[str, object] = Field(default_factory=dict)


class LevelsBuildSpec(BaseModel):
    """Top-level spec passed by CLI/API to construct and persist levels."""

    data: DataInputSpec
    symbols: List[str]
    range_start: str
    range_end: str
    targets: List[LevelsBuildItem]
    output_schema: str = "marketdata"
    output_table: str = "levels"
    upsert: bool = True


__all__ = [
    "LevelRecord",
    "LevelsBuildSpec",
    "LevelsBuildItem",
    "LevelType",
]
