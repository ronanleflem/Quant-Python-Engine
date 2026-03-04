"""Models and naming conventions for market-intelligence features."""
from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, field_validator

FEATURE_COLUMN_PREFIX = "feat_"
LABEL_COLUMN_PREFIX = "label_"
FEATURE_META_COLUMNS = (
    "feature_name",
    "feature_version",
    "timeframe",
    "symbol",
)

_VERSION_RE = re.compile(r"^v?\d+\.\d+\.\d+$")
_TIMEFRAME_RE = re.compile(r"^\d+(m|h|d|w)$")
_SYMBOL_RE = re.compile(r"^[A-Z0-9]+(?:[-_/][A-Z0-9]+)*$")


class FeatureMeta(BaseModel):
    """Metadata used to identify and version a computed feature."""

    model_config = ConfigDict(extra="forbid")

    feature_name: str
    feature_version: str
    timeframe: str
    symbol: str

    @field_validator("feature_version")
    @classmethod
    def _validate_feature_version(cls, value: str) -> str:
        if not _VERSION_RE.fullmatch(value):
            raise ValueError("feature_version must match semantic versioning, e.g. 1.0.0 or v1.0.0")
        return value

    @field_validator("timeframe")
    @classmethod
    def _validate_timeframe(cls, value: str) -> str:
        normalized = value.lower()
        if not _TIMEFRAME_RE.fullmatch(normalized):
            raise ValueError("timeframe must match <int><unit> with unit in m/h/d/w, e.g. 1h")
        return normalized

    @field_validator("symbol")
    @classmethod
    def _validate_symbol(cls, value: str) -> str:
        normalized = value.strip().upper()
        if not _SYMBOL_RE.fullmatch(normalized):
            raise ValueError("symbol must use uppercase alpha-numeric tokens separated by -, _, or /")
        return normalized


__all__ = [
    "FEATURE_COLUMN_PREFIX",
    "FEATURE_META_COLUMNS",
    "LABEL_COLUMN_PREFIX",
    "FeatureMeta",
]
