"""Performance metrics used during optimisation."""
from __future__ import annotations

from datetime import datetime
import math
from typing import List, Dict, Any, Mapping, Sequence, Tuple


CashflowPoint = Tuple[datetime, float]

DCA_COMPOSITE_SCORE_VERSION = "dca_composite_v1"
DEFAULT_DCA_SCORE_WEIGHTS: Dict[str, float] = {
    "performance": 0.35,
    "irr": 0.25,
    "drawdown": 0.25,
    "robustness": 0.15,
}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _normalize_weights(weights: Mapping[str, Any] | None = None) -> Dict[str, float]:
    merged: Dict[str, float] = {key: float(value) for key, value in DEFAULT_DCA_SCORE_WEIGHTS.items()}
    if isinstance(weights, Mapping):
        for key in merged:
            raw_value = weights.get(key)
            if raw_value is None:
                continue
            try:
                merged[key] = max(0.0, float(raw_value))
            except Exception:
                continue
    total = sum(merged.values())
    if total <= 0:
        return {key: float(value) for key, value in DEFAULT_DCA_SCORE_WEIGHTS.items()}
    return {key: value / total for key, value in merged.items()}


def _weighted_score(weights: Mapping[str, float], components: Mapping[str, float]) -> float:
    return sum(float(weights.get(key, 0.0)) * float(components.get(key, 0.0)) for key in DEFAULT_DCA_SCORE_WEIGHTS)


def dca_composite_score(
    *,
    final_performance_normalized_value: float,
    xirr_value: float | None,
    max_drawdown_on_contributed_capital_value: float | None,
    underperformance_duration_windows: int = 0,
    underperformance_severity_pct_points: float = 0.0,
    xirr_status: str | None = None,
    weights: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Compute versioned and decomposable DCA composite score.

    Component rules (all in [0, 1]):
    - ``performance``: normalized performance in range [-20%, +80%].
    - ``irr``: annualized XIRR in range [-5%, +30%].
    - ``drawdown``: inverse penalty using max drawdown on contributed capital,
      with full penalty at 60% drawdown.
    - ``robustness``: combines rolling underperformance duration/severity and
      XIRR convergence status.

    Edge classes:
    - ``strong``: score >= 0.70
    - ``medium``: 0.40 <= score < 0.70
    - ``weak``: score < 0.40
    """

    performance_component = _clamp01((float(final_performance_normalized_value) + 0.20) / 1.0)
    irr_raw = float(xirr_value) if isinstance(xirr_value, (int, float)) else -0.05
    irr_component = _clamp01((irr_raw + 0.05) / 0.35)

    drawdown_raw = (
        float(max_drawdown_on_contributed_capital_value)
        if isinstance(max_drawdown_on_contributed_capital_value, (int, float))
        else 0.60
    )
    if drawdown_raw > 1.0:
        drawdown_raw /= 100.0
    drawdown_component = _clamp01(1.0 - (drawdown_raw / 0.60))

    duration_penalty = _clamp01(float(max(0, underperformance_duration_windows)) / 12.0)
    severity_penalty = _clamp01(float(max(0.0, underperformance_severity_pct_points)) / 150.0)
    convergence_bonus = 1.0 if str(xirr_status or "").strip().lower() == "ok" else 0.0
    robustness_component = _clamp01(1.0 - (0.5 * duration_penalty + 0.4 * severity_penalty + 0.1 * (1.0 - convergence_bonus)))

    components: Dict[str, float] = {
        "performance": performance_component,
        "irr": irr_component,
        "drawdown": drawdown_component,
        "robustness": robustness_component,
    }
    norm_weights = _normalize_weights(weights)
    score = _clamp01(_weighted_score(norm_weights, components))

    if score >= 0.70:
        edge = "strong"
    elif score >= 0.40:
        edge = "medium"
    else:
        edge = "weak"

    contributions = {key: norm_weights[key] * components[key] for key in components}
    sensitivity: Dict[str, float] = {}
    delta = 0.10
    for key in components:
        up_weights = dict(norm_weights)
        up_weights[key] = up_weights.get(key, 0.0) + delta
        up_weights = _normalize_weights(up_weights)
        score_up = _weighted_score(up_weights, components)
        sensitivity[key] = score_up - score

    return {
        "version": DCA_COMPOSITE_SCORE_VERSION,
        "weights": norm_weights,
        "components": components,
        "contributions": contributions,
        "score": score,
        "edge": edge,
        "edge_mapping": {
            "weak_max_exclusive": 0.40,
            "medium_max_exclusive": 0.70,
            "strong_min_inclusive": 0.70,
        },
        "weight_sensitivity": sensitivity,
    }


def final_performance_normalized(final_value: float, contributed_capital: float) -> float:
    """Return normalized performance relative to contributed capital.

    Formula: ``(final_value - contributed_capital) / contributed_capital``.
    Returns ``0.0`` when ``contributed_capital <= 0`` to keep the metric safe.
    """

    contributed = float(contributed_capital)
    if contributed <= 0:
        return 0.0
    return (float(final_value) - contributed) / contributed


def twr(period_returns: Sequence[float]) -> float:
    """Compute Time-Weighted Return for periodic returns in decimal form."""

    if not period_returns:
        return 0.0
    growth = 1.0
    for ret in period_returns:
        growth *= 1.0 + float(ret)
    return growth - 1.0


def xirr(cashflows: Sequence[CashflowPoint], *, max_iter: int = 100, tol: float = 1e-8) -> Tuple[float | None, str]:
    """Compute annualized XIRR and return ``(value, status)``.

    Status values:
    - ``ok``: converged
    - ``invalid_cashflows``: no sign change in cashflows
    - ``non_convergent``: Newton/bisection did not converge
    """

    if len(cashflows) < 2:
        return None, "invalid_cashflows"

    flows = sorted(cashflows, key=lambda item: item[0])
    amounts = [float(v) for _, v in flows]
    if not any(v < 0 for v in amounts) or not any(v > 0 for v in amounts):
        return None, "invalid_cashflows"

    t0 = flows[0][0]
    years = [max(0.0, (dt - t0).total_seconds()) / (365.25 * 24 * 3600) for dt, _ in flows]

    def _npv(rate: float) -> float:
        base = 1.0 + rate
        if base <= 0.0:
            return float("inf")
        return sum(cf / (base ** yr) for yr, (_, cf) in zip(years, flows))

    def _d_npv(rate: float) -> float:
        base = 1.0 + rate
        if base <= 0.0:
            return float("inf")
        return sum((-yr * cf) / (base ** (yr + 1.0)) for yr, (_, cf) in zip(years, flows))

    rate = 0.1
    for _ in range(max_iter):
        value = _npv(rate)
        if abs(value) < tol:
            return rate, "ok"
        deriv = _d_npv(rate)
        if deriv == 0 or not math.isfinite(deriv):
            break
        new_rate = rate - value / deriv
        if not math.isfinite(new_rate) or new_rate <= -0.999999:
            break
        if abs(new_rate - rate) < tol:
            return new_rate, "ok"
        rate = new_rate

    low, high = -0.9999, 10.0
    f_low = _npv(low)
    f_high = _npv(high)
    if not (math.isfinite(f_low) and math.isfinite(f_high)) or f_low * f_high > 0:
        return None, "non_convergent"

    for _ in range(max_iter):
        mid = (low + high) / 2.0
        f_mid = _npv(mid)
        if not math.isfinite(f_mid):
            return None, "non_convergent"
        if abs(f_mid) < tol or abs(high - low) < tol:
            return mid, "ok"
        if f_low * f_mid <= 0:
            high = mid
            f_high = f_mid
        else:
            low = mid
            f_low = f_mid
    return None, "non_convergent"


def max_drawdown_on_contributed_capital(equity_values: Sequence[float], contributed_capital: Sequence[float]) -> float:
    """Return max drawdown normalized by contributed capital at each step."""

    length = min(len(equity_values), len(contributed_capital))
    if length == 0:
        return 0.0
    peak = float(equity_values[0])
    max_dd = 0.0
    for idx in range(length):
        value = float(equity_values[idx])
        contrib = float(contributed_capital[idx])
        peak = max(peak, value)
        denom = contrib if contrib > 0 else 1.0
        dd = (peak - value) / denom
        if dd > max_dd:
            max_dd = dd
    return max_dd


def time_under_water(equity_values: Sequence[float]) -> int:
    """Return the longest consecutive period below the last peak."""

    if not equity_values:
        return 0
    peak = float(equity_values[0])
    current = 0
    longest = 0
    for val in equity_values:
        value = float(val)
        if value >= peak:
            peak = value
            current = 0
            continue
        current += 1
        longest = max(longest, current)
    return longest


def sharpe_ratio(returns: List[float]) -> float:
    if not returns:
        return 0.0
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / len(returns)
    std = math.sqrt(variance)
    if std == 0:
        return 0.0
    return mean / std * math.sqrt(len(returns))


def sortino_ratio(returns: List[float]) -> float:
    if not returns:
        return 0.0
    mean = sum(returns) / len(returns)
    downside = [min(0.0, r) for r in returns]
    denom = math.sqrt(sum(d ** 2 for d in downside) / len(returns))
    if denom == 0:
        return 0.0
    return mean / denom * math.sqrt(len(returns))


def max_drawdown(equity: List[float]) -> float:
    peak = float("-inf")
    max_dd = 0.0
    for val in equity:
        if val > peak:
            peak = val
        dd = peak - val
        if dd > max_dd:
            max_dd = dd
    return max_dd


def cagr(equity: List[float]) -> float:
    if not equity:
        return 0.0
    end_value = 1.0 + equity[-1]
    years = len(equity) / 252
    if years == 0:
        return 0.0
    return end_value ** (1 / years) - 1


def hit_rate(trades: List[Dict[str, Any]]) -> float:
    if not trades:
        return 0.0
    wins = sum(1 for t in trades if t["pnl"] > 0)
    return wins / len(trades)


def avg_r(trades: List[Dict[str, Any]]) -> float:
    if not trades:
        return 0.0
    return sum(t.get("r_multiple", 0.0) for t in trades) / len(trades)


def compute(trades: List[Dict[str, Any]], equity: List[float]) -> Dict[str, float]:
    returns = [t["pnl"] for t in trades]
    return {
        "sharpe": sharpe_ratio(returns),
        "sortino": sortino_ratio(returns),
        "max_drawdown": max_drawdown(equity),
        "cagr": cagr(equity),
        "hit_rate": hit_rate(trades),
        "avg_R": avg_r(trades),
        "trades": float(len(trades)),
    }
