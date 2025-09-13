"""Performance metrics used during optimisation."""
from __future__ import annotations

import math
from typing import List, Dict, Any


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
