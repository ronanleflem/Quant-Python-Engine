"""Build performance payloads for classic backtests."""
from __future__ import annotations

from datetime import datetime
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
                entry_time_utc=entry_time or _maybe_dt(start_ts) or datetime.utcnow(),
                exit_time_utc=exit_time or _maybe_dt(end_ts) or datetime.utcnow(),
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

    volatility_pct = float(pd.Series(returns_pct).std(ddof=0)) if returns_pct else None
    mean_ret = float(pd.Series(returns_pct).mean()) if returns_pct else None
    sharpe = None
    sortino = None
    if volatility_pct and volatility_pct != 0 and mean_ret is not None:
        sharpe = mean_ret / volatility_pct * (len(returns_pct) ** 0.5)
    if returns_pct:
        downside = pd.Series([r for r in returns_pct if r < 0])
        if not downside.empty:
            downside_std = float(downside.std(ddof=0))
            if downside_std:
                sortino = mean_ret / downside_std * (len(returns_pct) ** 0.5) if mean_ret is not None else None

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
        start_ts_utc=_maybe_dt(start_ts) or datetime.utcnow(),
        end_ts_utc=_maybe_dt(end_ts) or datetime.utcnow(),
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
        extra={"note": "Backtest performance computed from pct returns."},
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
