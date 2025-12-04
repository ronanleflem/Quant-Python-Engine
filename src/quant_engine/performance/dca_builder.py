from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Tuple

import pandas as pd

from .models import CompletedTrade, StrategyRunResult, to_backend_payload

LOGGER = logging.getLogger(__name__)


class SignalLike(Protocol):
    strategy_id: str
    symbol: str
    asset_class: str
    side: str
    ts_open_utc: datetime
    qty: float
    meta: Mapping[str, Any]


def _maybe_dt(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value
    try:
        return pd.to_datetime(value, utc=True).to_pydatetime()
    except Exception:
        return None


def _start_end_from_signals(signals_by_symbol: Mapping[str, Iterable[SignalLike]]) -> Tuple[Optional[datetime], Optional[datetime]]:
    timestamps: List[datetime] = []
    for sigs in signals_by_symbol.values():
        for s in sigs:
            dt = _maybe_dt(getattr(s, "ts_open_utc", None))
            if dt:
                timestamps.append(dt)
    if not timestamps:
        return None, None
    return min(timestamps), max(timestamps)


def build_dca_performance_from_signals(
    strategy_id: str,
    run_id: str,
    asset_class: str,
    universe: Optional[str],
    timeframe: Optional[str],
    signals_by_symbol: Dict[str, List[SignalLike]],
    equity_curve: "pd.Series | None" = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[StrategyRunResult, List[CompletedTrade]]:
    config = config or {}

    start_ts, end_ts = _start_end_from_signals(signals_by_symbol)
    if equity_curve is not None and not equity_curve.empty:
        ec_start = _maybe_dt(equity_curve.index.min())
        ec_end = _maybe_dt(equity_curve.index.max())
        if ec_start and (start_ts is None or ec_start < start_ts):
            start_ts = ec_start
        if ec_end and (end_ts is None or ec_end > end_ts):
            end_ts = ec_end

    if start_ts is None or end_ts is None:
        now = datetime.utcnow()
        start_ts = start_ts or now
        end_ts = end_ts or now

    trades: List[CompletedTrade] = []
    for symbol, sigs in signals_by_symbol.items():
        ordered = sorted(sigs, key=lambda s: _maybe_dt(getattr(s, "ts_open_utc", None)) or datetime.utcnow())
        by_cycle: Dict[int, List[SignalLike]] = {}
        for s in ordered:
            cycle_id = getattr(s, "meta", {}).get("cycle_id") if getattr(s, "meta", None) else None
            if cycle_id is None:
                continue
            by_cycle.setdefault(int(cycle_id), []).append(s)

        for cycle_id, cycle_signals in by_cycle.items():
            buys = [s for s in cycle_signals if getattr(s, "side", "").upper() == "BUY"]
            sells_tp = [s for s in cycle_signals if getattr(s, "side", "").upper() == "SELL" and getattr(getattr(s, "meta", {}), "get", lambda *_: None)("action") == "take_profit"]
            if not sells_tp or not buys:
                LOGGER.warning("Skipping incomplete cycle %s for %s (buys=%d, sells=%d)", cycle_id, symbol, len(buys), len(sells_tp))
                continue

            entry_time = _maybe_dt(getattr(buys[0], "ts_open_utc", None)) or start_ts
            exit_time = _maybe_dt(getattr(sells_tp[-1], "ts_open_utc", None)) or end_ts
            quantity = sum(getattr(s, "qty", 0.0) or 0.0 for s in buys)

            max_dd_values: List[float] = []
            for s in cycle_signals:
                meta = getattr(s, "meta", {}) or {}
                for key in ("max_dd_reached", "drawdown_pct"):
                    if meta.get(key) is not None:
                        try:
                            max_dd_values.append(float(meta[key]))
                        except Exception:
                            continue
            max_dd_pct = max(max_dd_values) if max_dd_values else None

            first_grid_level = getattr(buys[0], "meta", {}).get("grid_level") if buys else None
            tp_meta = dict(getattr(sells_tp[-1], "meta", {}) or {})
            tp_meta.update(
                {
                    "first_grid_level": first_grid_level,
                    "cycle_id": cycle_id,
                    "max_dd_reached_cycle": max_dd_pct,
                }
            )

            trade = CompletedTrade(
                strategy_id=strategy_id,
                run_id=run_id,
                symbol=symbol,
                asset_class=asset_class,
                side="LONG",
                cycle_id=cycle_id,
                entry_time_utc=entry_time,
                exit_time_utc=exit_time,
                entry_price=0.0,  # TODO: plug real prices when available
                exit_price=0.0,  # TODO: plug real prices when available
                quantity=quantity,
                gross_pnl=0.0,  # TODO: compute PnL when prices are known
                gross_pnl_pct=0.0,
                max_dd_pct=max_dd_pct,
                meta=tp_meta,
            )
            trades.append(trade)

    win_count = sum(1 for t in trades if t.gross_pnl > 0)
    loss_count = sum(1 for t in trades if t.gross_pnl <= 0)
    total_return = sum(t.gross_pnl for t in trades)
    nb_trades = len(trades)
    average_trade = total_return / nb_trades if nb_trades else 0.0

    run = StrategyRunResult(
        strategy_id=strategy_id,
        run_id=run_id,
        asset_class=asset_class,
        universe=universe,
        timeframe=timeframe,
        symbol=None if len(signals_by_symbol) > 1 else next(iter(signals_by_symbol.keys()), None),
        compared_symbol=None,
        start_ts_utc=start_ts,
        end_ts_utc=end_ts,
        win_count=win_count,
        loss_count=loss_count,
        total_return=total_return,
        max_drawdown=0.0,
        average_trade=average_trade,
        average_sl=0.0,  # placeholder, DCA equity sans SL explicite
        average_tp=0.0,  # placeholder, TP en % dans meta non converti en pips ici
        rr_moyen=None,
        total_net_return=total_return,
        net_win_count=win_count,
        net_loss_count=loss_count,
        average_net_trade=average_trade,
        initial_capital=None,
        final_capital=None,
        return_pct=None,
        max_drawdown_pct=None,
        volatility_pct=None,
        sharpe=None,
        sortino=None,
        winrate_pct=(win_count / nb_trades * 100.0) if nb_trades else None,
        extra={"note": "DCA performance placeholder; prices/PNL to be enriched."},
    )

    return run, trades


def build_backend_payload_for_java(
    strategy_id: str,
    run_id: str,
    asset_class: str,
    universe: Optional[str],
    timeframe: Optional[str],
    signals_by_symbol: Dict[str, List[SignalLike]],
    equity_curve: "pd.Series | None" = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    run, trades = build_dca_performance_from_signals(
        strategy_id=strategy_id,
        run_id=run_id,
        asset_class=asset_class,
        universe=universe,
        timeframe=timeframe,
        signals_by_symbol=signals_by_symbol,
        equity_curve=equity_curve,
        config=config,
    )
    return to_backend_payload(run, trades)
