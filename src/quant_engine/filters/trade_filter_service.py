"""Filter rule adapter and weighted filter scoring service."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

import pandas as pd

from . import filters_registry
from .utils import FilterValidationError, _normalize_filter_spec, _validate_filter_inputs


@dataclass(frozen=True)
class FilterRule:
    type: str
    params: dict
    weight: float = 1.0
    mode: str = "hard"
    enabled: bool = True


def normalize_filter_rules(rules: Sequence[Mapping[str, Any]]) -> list[FilterRule]:
    """Normalize raw rule specs into FilterRule entries."""
    normalized: list[FilterRule] = []
    for raw in rules:
        flt_type, params = _normalize_filter_spec(raw)
        weight = raw.get("weight", 1.0)
        try:
            weight_val = float(weight)
        except Exception:
            weight_val = 1.0
        mode = str(raw.get("mode", "hard")).strip().lower()
        if mode not in {"hard", "soft"}:
            mode = "hard"
        enabled = raw.get("enabled", True)
        normalized.append(
            FilterRule(
                type=flt_type,
                params=dict(params),
                weight=weight_val,
                mode=mode,
                enabled=bool(enabled),
            )
        )
    return normalized


def _apply_rule(
    df: pd.DataFrame,
    rule: FilterRule,
    *,
    symbol: Optional[str] = None,
    strict: bool = True,
) -> pd.Series:
    errors = _validate_filter_inputs(df, rule.type, rule.params, symbol)
    if errors:
        msg = f"Filter '{rule.type}' cannot be evaluated: {', '.join(errors)}"
        if strict:
            raise FilterValidationError(msg)
        return pd.Series(False, index=df.index)

    fn = filters_registry.get(rule.type)
    if fn is None:
        msg = f"Unknown filter type '{rule.type}'"
        if strict:
            raise FilterValidationError(msg)
        return pd.Series(False, index=df.index)

    series = fn(df, **rule.params)
    if not isinstance(series, pd.Series):
        msg = f"Filter '{rule.type}' did not return a pandas Series"
        if strict:
            raise FilterValidationError(msg)
        return pd.Series(False, index=df.index)
    return series.reindex(df.index).fillna(False).astype(bool)


def score_filter_rules(
    df: pd.DataFrame,
    rules: Sequence[Mapping[str, Any]] | Sequence[FilterRule],
    *,
    symbol: Optional[str] = None,
    strict: bool = True,
    min_score: Optional[float] = None,
    min_score_pct: Optional[float] = None,
) -> dict:
    """Evaluate rules, returning hard mask + weighted score + final mask."""
    if rules and isinstance(rules[0], FilterRule):
        normalized = list(rules)  # type: ignore[list-item]
    else:
        normalized = normalize_filter_rules(rules)  # type: ignore[arg-type]

    hard_mask = pd.Series(True, index=df.index)
    score = pd.Series(0.0, index=df.index, dtype="float64")
    total_weight = 0.0
    details: list[dict[str, Any]] = []

    for rule in normalized:
        if not rule.enabled:
            continue
        series = _apply_rule(df, rule, symbol=symbol, strict=strict)
        if rule.mode == "hard":
            hard_mask &= series
        weight = max(0.0, float(rule.weight))
        if weight > 0:
            score += series.astype(int) * weight
            total_weight += weight
        details.append(
            {
                "type": rule.type,
                "mode": rule.mode,
                "weight": weight,
                "enabled": rule.enabled,
            }
        )

    if total_weight > 0:
        score_pct = score / total_weight
    else:
        score_pct = pd.Series(0.0, index=df.index)

    required = 0.0
    if min_score_pct is not None and total_weight > 0:
        required = float(min_score_pct) * total_weight
    elif min_score is not None:
        required = float(min_score)

    score_mask = score >= required if total_weight > 0 or required > 0 else pd.Series(True, index=df.index)
    final_mask = hard_mask & score_mask
    return {
        "hard_mask": hard_mask.astype(bool),
        "score": score,
        "score_pct": score_pct,
        "final_mask": final_mask.astype(bool),
        "details": details,
    }


__all__ = ["FilterRule", "normalize_filter_rules", "score_filter_rules"]
