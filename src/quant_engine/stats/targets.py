"""Target metrics for statistics runs."""
from __future__ import annotations

from typing import Any, Dict, List


def up_next_bar(data: List[Dict[str, Any]], **params: Any) -> List[bool]:
    """Placeholder for next bar up move."""

    return [False for _ in data]


def continuation_n(data: List[Dict[str, Any]], **params: Any) -> List[int]:
    """Placeholder for continuation over ``n`` bars."""

    return [0 for _ in data]


def time_to_reversal(data: List[Dict[str, Any]], **params: Any) -> List[int]:
    """Placeholder for time until reversal."""

    return [0 for _ in data]

