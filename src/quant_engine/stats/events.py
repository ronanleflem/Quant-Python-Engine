"""Event definitions for statistics runs."""
from __future__ import annotations

from typing import Any, Dict, List


def k_consecutive(data: List[Dict[str, Any]], **params: Any) -> List[bool]:
    """Placeholder for detecting ``k`` consecutive events."""

    return [False for _ in data]


def shock_atr(data: List[Dict[str, Any]], **params: Any) -> List[bool]:
    """Placeholder for ATR shock events."""

    return [False for _ in data]


def breakout_hhll(data: List[Dict[str, Any]], **params: Any) -> List[bool]:
    """Placeholder for breakout of higher high / lower low events."""

    return [False for _ in data]

