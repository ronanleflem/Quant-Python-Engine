"""Build performance payloads for classic backtests.

Backtest metrics are normalized to be comparable across timeframes by
annualizing risk metrics and applying a configurable risk-free rate.
"""
from __future__ import annotations

from datetime import datetime
import math
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import pandas as pd

from .models import CompletedTrade, StrategyRunResult, to_backend_payload


def _maybe_dt(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value
    try:
        return pd.to_datetime(value, utc=True).to_pydatetime()
    except Exception:
        return None


def _equity_value_curve(equity: Iterable[float], initial_capital: float) -> List[float]:
    values: List[float] = []
    running = float(initial_capital)
    for val in equity:
        running = initial_capital + float(val)
        values.append(running)
    return values


def _parse_timeframe(timeframe: Optional[str]) -> Optional[Tuple[int, str]]:
    if not timeframe:
        return None
    raw = str(timeframe).strip()
    if not raw:
        return None

    match = re.match(r"^(?P<num>\d+)\s*(?P<unit>[A-Za-z]+)$", raw)
    if not match:
        match = re.match(r"^(?P<unit>[A-Za-z]+)\s*(?P<num>\d+)$", raw)
    if not match:
        return None

    value = int(match.group("num"))
    unit_raw = match.group("unit")
    unit_lower = unit_raw.lower()

    if unit_raw == "M" or unit_lower in {"mo", "mon", "month", "months"}:
        unit = "mo"
    elif unit_raw == "m" or unit_lower in {"min", "mins", "minute", "minutes"}:
        unit = "m"
    elif unit_lower in {"s", "sec", "secs", "second", "seconds"}:
        unit = "s"
    elif unit_lower in {"h", "hr", "hrs", "hour", "hours"}:
        unit = "h"
    elif unit_lower in {"d", "day", "days"}:
        unit = "d"
    elif unit_lower in {"w", "wk", "wks", "week", "weeks"}:
        unit = "w"
    elif unit_lower in {"y", "yr", "yrs", "year", "years"}:
        unit = "y"
    else:
        return None

    if value <= 0:
        return None
    return value, unit


def _periods_per_year(
    *,
    timeframe: Optional[str],
    asset_class: str,
    start_ts: Optional[datetime],
    end_ts: Optional[datetime],
    periods_count: int,
) -> Optional[float]:
    parsed = _parse_timeframe(timeframe)
    is_crypto = asset_class.upper() == "CRYPTO"

    if parsed:
        value, unit = parsed
        if is_crypto:
            days_per_year = 365.0
            hours_per_year = days_per_year * 24.0
        else:
            days_per_year = 252.0
            hours_per_year = days_per_year * 6.5
        minutes_per_year = hours_per_year * 60.0
        seconds_per_year = minutes_per_year * 60.0

        base_map = {
            "s": seconds_per_year,
            "m": minutes_per_year,
            "h": hours_per_year,
            "d": days_per_year,
            "w": 52.0,
            "mo": 12.0,
            "y": 1.0,
        }
        base = base_map.get(unit)
        if base:
            return base / float(value)

    if start_ts and end_ts and periods_count > 1:
        duration = (end_ts - start_ts).total_seconds()
        if duration > 0:
            years = duration / (365.25 * 24 * 60 * 60)
            if years > 0:
                return periods_count / years
    return None


def build_backtest_performance(
    *,
    strategy_id: str,
    run_id: str,
    asset_class: str,
    symbol: str,
    timeframe: Optional[str],
    trades: Iterable[Mapping[str, Any]],
    equity: Iterable[float],
    start_ts: Optional[Any],
    end_ts: Optional[Any],
    config: Optional[Mapping[str, Any]] = None,
) -> Tuple[StrategyRunResult, List[CompletedTrade]]:
    config = config or {}
    initial_capital = float(config.get("initial_capital", 10_000.0))
    start_dt = _maybe_dt(start_ts)
    end_dt = _maybe_dt(end_ts)

    completed: List[CompletedTrade] = []
    returns_pct: List[float] = []
    for tr in trades:
        entry_time = _maybe_dt(tr.get("ts_entry"))
        exit_time = _maybe_dt(tr.get("ts_exit"))
        entry_price = float(tr.get("price_entry") or 0.0)
        exit_price = float(tr.get("price_exit") or 0.0)
        quantity = float(tr.get("quantity") or 1.0)
        pnl = float(tr.get("pnl") or 0.0)
        pnl_pct = (pnl / entry_price * 100.0) if entry_price else 0.0
        returns_pct.append(pnl_pct)
        meta = {
            "r_multiple": tr.get("r_multiple"),
        }
        completed.append(
            CompletedTrade(
                strategy_id=strategy_id,
                run_id=run_id,
                symbol=symbol,
                asset_class=asset_class,
                side=str(tr.get("side") or "LONG").upper(),
                cycle_id=None,
                entry_time_utc=entry_time or start_dt or datetime.utcnow(),
                exit_time_utc=exit_time or end_dt or datetime.utcnow(),
                entry_price=entry_price,
                exit_price=exit_price,
                quantity=quantity,
                gross_pnl=pnl,
                gross_pnl_pct=pnl_pct,
                max_dd_pct=None,
                meta=meta,
            )
        )

    win_count = sum(1 for r in returns_pct if r > 0)
    loss_count = sum(1 for r in returns_pct if r <= 0)
    total_return = sum(returns_pct)
    nb_trades = len(returns_pct)
    average_trade = total_return / nb_trades if nb_trades else 0.0

    equity_values = _equity_value_curve(equity, initial_capital)
    final_capital = equity_values[-1] if equity_values else initial_capital
    return_pct = ((final_capital - initial_capital) / initial_capital * 100.0) if initial_capital else None
    cagr_pct = None
    if start_dt and end_dt and end_dt > start_dt and initial_capital > 0:
        years = (end_dt - start_dt).total_seconds() / (365.25 * 24 * 60 * 60)
        if years > 0:
            cagr_pct = ((final_capital / initial_capital) ** (1 / years) - 1) * 100.0

    max_drawdown = 0.0
    max_drawdown_pct = None
    if equity_values:
        peak = equity_values[0]
        max_dd_val = 0.0
        for val in equity_values:
            peak = max(peak, val)
            max_dd_val = max(max_dd_val, peak - val)
        max_drawdown = max_dd_val
        max_drawdown_pct = (max_dd_val / peak * 100.0) if peak else None

    periodic_returns: List[float] = []
    if len(equity_values) > 1:
        prev = equity_values[0]
        for val in equity_values[1:]:
            if prev != 0:
                periodic_returns.append((val - prev) / prev)
            else:
                periodic_returns.append(0.0)
            prev = val

    periods_per_year = _periods_per_year(
        timeframe=timeframe,
        asset_class=asset_class,
        start_ts=start_dt,
        end_ts=end_dt,
        periods_count=len(periodic_returns),
    )
    risk_free_rate = config.get("risk_free_rate")
    if risk_free_rate is None:
        risk_free_pct = config.get("risk_free_pct")
        risk_free_rate = (float(risk_free_pct) / 100.0) if risk_free_pct is not None else 0.0
    else:
        risk_free_rate = float(risk_free_rate)

    volatility_pct = None
    sharpe = None
    sortino = None
    mean_ret = None
    if periodic_returns:
        returns_series = pd.Series(periodic_returns)
        mean_ret = float(returns_series.mean())
        std = float(returns_series.std(ddof=0))
        if periods_per_year and std:
            rf_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1 if periods_per_year > 0 else 0.0
            volatility_pct = std * math.sqrt(periods_per_year) * 100.0
            sharpe = (mean_ret - rf_period) / std * math.sqrt(periods_per_year)

            downside = returns_series[returns_series < 0]
            if not downside.empty:
                downside_std = float(downside.std(ddof=0))
                if downside_std:
                    sortino = (mean_ret - rf_period) / downside_std * math.sqrt(periods_per_year)
        elif std:
            volatility_pct = std * 100.0
            sharpe = mean_ret / std
            downside = returns_series[returns_series < 0]
            if not downside.empty:
                downside_std = float(downside.std(ddof=0))
                if downside_std:
                    sortino = mean_ret / downside_std

    rr_moyen = None
    win_vals = [r for r in returns_pct if r > 0]
    loss_vals = [abs(r) for r in returns_pct if r < 0]
    if win_vals and loss_vals:
        rr_moyen = (sum(win_vals) / len(win_vals)) / (sum(loss_vals) / len(loss_vals))

    run = StrategyRunResult(
        strategy_id=strategy_id,
        run_id=run_id,
        asset_class=asset_class,
        universe=None,
        timeframe=timeframe,
        symbol=symbol,
        compared_symbol=None,
        start_ts_utc=start_dt or datetime.utcnow(),
        end_ts_utc=end_dt or datetime.utcnow(),
        win_count=win_count,
        loss_count=loss_count,
        total_return=total_return,
        max_drawdown=max_drawdown,
        average_trade=average_trade,
        average_sl=None,
        average_tp=None,
        rr_moyen=rr_moyen,
        total_net_return=total_return,
        net_win_count=win_count,
        net_loss_count=loss_count,
        average_net_trade=average_trade,
        initial_capital=initial_capital,
        final_capital=final_capital,
        return_pct=return_pct,
        max_drawdown_pct=max_drawdown_pct,
        volatility_pct=volatility_pct,
        sharpe=sharpe,
        sortino=sortino,
        winrate_pct=(win_count / nb_trades * 100.0) if nb_trades else None,
        extra={
            "note": "Backtest performance normalized with timeframe-aware annualization.",
            "periods_per_year": periods_per_year,
            "risk_free_rate": risk_free_rate,
            "cagr_pct": cagr_pct,
        },
    )
    return run, completed


def build_backtest_payload(
    *,
    strategy_id: str,
    run_id: str,
    asset_class: str,
    symbol: str,
    timeframe: Optional[str],
    trades: Iterable[Mapping[str, Any]],
    equity: Iterable[float],
    start_ts: Optional[Any],
    end_ts: Optional[Any],
    config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    run, completed = build_backtest_performance(
        strategy_id=strategy_id,
        run_id=run_id,
        asset_class=asset_class,
        symbol=symbol,
        timeframe=timeframe,
        trades=trades,
        equity=equity,
        start_ts=start_ts,
        end_ts=end_ts,
        config=config,
    )
    return to_backend_payload(run, completed)
