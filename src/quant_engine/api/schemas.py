"""API request/response models (light-weight)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class SubmitResponse:
    id: str


@dataclass
class StatusResponse:
    status: str


@dataclass
class ResultResponse:
    result: Dict[str, Any] | None

