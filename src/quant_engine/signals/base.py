"""Signal base classes."""
from __future__ import annotations

from typing import List, Dict, Any


class Signal:
    """Base class for trading signals."""

    def generate(self, dataset: List[Dict[str, Any]]) -> List[int]:
        raise NotImplementedError

