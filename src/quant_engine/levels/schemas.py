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
    SESSION_HIGH = "SESSION_HIGH"
    SESSION_LOW = "SESSION_LOW"
    ORH = "ORH"
    ORL = "ORL"
    IBH = "IBH"
    IBL = "IBL"
    PDO = "PDO"
    PDC = "PDC"
    PWO = "PWO"
    PWC = "PWC"
    PMO = "PMO"
    PMC = "PMC"


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
        "SESSION_HIGH",
        "SESSION_LOW",
        "ORH",
        "ORL",
        "IBH",
        "IBL",
        "PDO",
        "PDC",
        "PWO",
        "PWC",
        "PMO",
        "PMC",
    ]
    price: Optional[float] = None
    price_lo: Optional[float] = None
    price_hi: Optional[float] = None
    anchor_ts: datetime
    valid_from_ts: Optional[datetime] = None
    valid_to_ts: Optional[datetime] = None
    params_hash: Optional[str] = None
    source: str = "python-levels"


class SessionWindows(BaseModel):
    """UTC session windows described by inclusive hour bounds."""

    model_config = ConfigDict(extra="forbid")

    asia: tuple[int, int] = (0, 7)
    europe: tuple[int, int] = (8, 12)
    overlap: tuple[int, int] = (13, 16)
    us: tuple[int, int] = (17, 21)


class ORIBSpec(BaseModel):
    """Opening range and initial balance configuration."""

    model_config = ConfigDict(extra="forbid")

    or_minutes: int = 30
    ib_minutes: int = 60


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
    session_windows: Optional[SessionWindows] = None
    orib: Optional[ORIBSpec] = None


__all__ = [
    "LevelRecord",
    "LevelsBuildSpec",
    "LevelsBuildItem",
    "LevelType",
    "SessionWindows",
    "ORIBSpec",
]
