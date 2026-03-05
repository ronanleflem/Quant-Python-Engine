"""Adapter from legacy filter service outputs to normalized dataframes."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import warnings
from typing import Any

import pandas as pd

_FILTER_COLUMNS = ("hard_mask", "score", "score_pct", "final_mask")
_DEPRECATION_TARGET_VERSION = "v0.15.0"
_DEPRECATION_TARGET_DATE = "2026-04-30"


def _warn_deprecated() -> None:
    warnings.warn(
        (
            "quant_engine.market_intelligence.adapters.LegacyFiltersAdapter is deprecated "
            f"and will be removed in {_DEPRECATION_TARGET_VERSION} "
            f"(target date: {_DEPRECATION_TARGET_DATE}); use "
            "quant_engine.filters.trade_filter_service.score_filter_rules instead."
        ),
        DeprecationWarning,
        stacklevel=2,
    )


def _default_runner(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
    from quant_engine.filters.trade_filter_service import score_filter_rules

    return score_filter_rules(*args, **kwargs)


def _utc_indexed(ohlcv: pd.DataFrame) -> pd.DatetimeIndex:
    if isinstance(ohlcv.index, pd.DatetimeIndex):
        idx = ohlcv.index
        if idx.tz is None:
            return idx.tz_localize("UTC")
        return idx.tz_convert("UTC")

    if "ts" not in ohlcv.columns:
        raise ValueError("ohlcv must expose a DatetimeIndex or a 'ts' column")
    return pd.to_datetime(ohlcv["ts"], utc=True)


class LegacyFiltersAdapter:
    """Execute legacy filter rules and return a normalized UTC-indexed frame."""

    def __init__(
        self,
        runner: Callable[..., Mapping[str, Any]] = _default_runner,
    ) -> None:
        _warn_deprecated()
        self._runner = runner

    def run(
        self,
        ohlcv: pd.DataFrame,
        rules: Sequence[Mapping[str, Any]],
        *,
        symbol: str | None = None,
        strict: bool = True,
        min_score: float | None = None,
        min_score_pct: float | None = None,
    ) -> pd.DataFrame:
        """Run legacy filters and normalize output shape/columns for contracts."""
        result = self._runner(
            ohlcv,
            rules,
            symbol=symbol,
            strict=strict,
            min_score=min_score,
            min_score_pct=min_score_pct,
        )
        idx = _utc_indexed(ohlcv)

        normalized = pd.DataFrame(index=idx)
        normalized.index.name = "ts"
        for col in _FILTER_COLUMNS:
            values = result.get(col)
            if values is None:
                normalized[col] = pd.NA
                continue
            series = pd.Series(values, index=ohlcv.index if len(values) == len(ohlcv) else None)
            normalized[col] = series.to_numpy()

        for bool_col in ("hard_mask", "final_mask"):
            normalized[bool_col] = normalized[bool_col].fillna(False).astype(bool)
        for float_col in ("score", "score_pct"):
            normalized[float_col] = pd.to_numeric(normalized[float_col], errors="coerce")

        return normalized


__all__ = ["LegacyFiltersAdapter"]
