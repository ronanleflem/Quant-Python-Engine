"""Adapters bridging legacy analytics outputs to normalized contracts."""

from .legacy_filters_adapter import LegacyFiltersAdapter
from .legacy_stats_adapter import LegacyStatsAdapter

__all__ = ["LegacyFiltersAdapter", "LegacyStatsAdapter"]
