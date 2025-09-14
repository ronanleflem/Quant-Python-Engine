"""Regime condition helpers for statistics runs."""
from __future__ import annotations

from typing import Any, Dict, List


def htf_trend(data: List[Dict[str, Any]], **params: Any) -> List[str]:
    """Placeholder for higher-time-frame trend regime."""

    return ["flat" for _ in data]


def vol_tertile(data: List[Dict[str, Any]], **params: Any) -> List[int]:
    """Placeholder for volatility tertile regime."""

    return [0 for _ in data]


def session(data: List[Dict[str, Any]], **params: Any) -> List[str]:
    """Placeholder for session labels."""

    return ["" for _ in data]

