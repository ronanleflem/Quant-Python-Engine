"""Statistics module skeleton."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .runner import run_stats as run_stats


__all__ = ["run_stats"]


def __getattr__(name: str):
    if name == "run_stats":
        from .runner import run_stats as _run_stats

        return _run_stats
    raise AttributeError(name)
