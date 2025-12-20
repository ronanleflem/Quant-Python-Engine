"""Construction des trades DCA et du payload backend.

À partir des signaux bas niveau (BUY/SELL taggés `cycle_id`), le module
reconstruit des `CompletedTrade` (1 cycle = 1 trade logique), calcule les
métriques agrégées (`StrategyRunResult`) et produit le payload `{run, trades}`
envoyé au backend Java. Les performances sont donc calculées dans Python pour
rester cohérentes entre backtests et exécution live.
"""

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
    ohlc_by_symbol: Optional[Dict[str, "pd.DataFrame"]] = None,
    equity_curve: "pd.Series | None" = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[StrategyRunResult, List[CompletedTrade]]:
    config = config or {}
    ohlc_by_symbol = ohlc_by_symbol or {}
    initial_capital = float(config.get("initial_capital", 10_000.0))
    capital_per_unit = float(config.get("capital_per_unit", config.get("max_capital_per_trade", 100.0)))

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

    def _price_at(symbol: str, ts: datetime) -> Optional[float]:
        df = ohlc_by_symbol.get(symbol)
        if df is None or df.empty:
            return None
        df_ts = df.copy()
        if "ts" not in df_ts.columns:
            return None
        df_ts["ts"] = pd.to_datetime(df_ts["ts"], utc=True, errors="coerce")
        df_ts = df_ts.dropna(subset=["ts"]).sort_values("ts")
        # close column fallback
        price_col = "close" if "close" in df_ts.columns else None
        if price_col is None:
            return None
        mask = df_ts["ts"] <= ts
        if mask.any():
            return float(df_ts.loc[mask].iloc[-1][price_col])
        # otherwise take first available after ts
        after = df_ts[df_ts["ts"] >= ts]
        if not after.empty:
            return float(after.iloc[0][price_col])
        return None

    trades: List[CompletedTrade] = []
    for symbol, sigs in signals_by_symbol.items():
        ordered = sorted(sigs, key=lambda s: _maybe_dt(getattr(s, "ts_open_utc", None)) or datetime.utcnow())
        by_cycle: Dict[int, List[SignalLike]] = {}
        missing_cycle = 0
        for s in ordered:
            cycle_id = getattr(s, "meta", {}).get("cycle_id") if getattr(s, "meta", None) else None
            if cycle_id is None:
                missing_cycle += 1
                continue
            by_cycle.setdefault(int(cycle_id), []).append(s)

        if missing_cycle:
            LOGGER.warning("Symbol %s: %d signals without cycle_id dropped", symbol, missing_cycle)

        for cycle_id, cycle_signals in by_cycle.items():
            buys = [s for s in cycle_signals if getattr(s, "side", "").upper() == "BUY"]
            sells_tp = [s for s in cycle_signals if getattr(s, "side", "").upper() == "SELL" and getattr(getattr(s, "meta", {}), "get", lambda *_: None)("action") == "take_profit"]
            if not sells_tp or not buys:
                LOGGER.warning(
                    "Skipping incomplete cycle %s for %s (buys=%d, sells=%d)", cycle_id, symbol, len(buys), len(sells_tp)
                )
                continue

            entry_time = _maybe_dt(getattr(buys[0], "ts_open_utc", None)) if buys else _maybe_dt(
                getattr(sells_tp[0], "ts_open_utc", None)
            )
            entry_time = entry_time or start_ts
            exit_time = _maybe_dt(getattr(sells_tp[-1], "ts_open_utc", None)) or end_ts
            # Quantité et prix moyen basés sur les BUY du cycle
            total_qty = 0.0
            total_cost = 0.0
            buy_entries: List[Dict[str, Any]] = []
            for idx, b in enumerate(buys):
                q = getattr(b, "qty", 0.0) or 0.0
                ts_b = _maybe_dt(getattr(b, "ts_open_utc", None)) or entry_time
                price_raw = _price_at(symbol, ts_b)
                price_b = price_raw or 0.0
                total_qty += q
                total_cost += q * price_b
                meta_b = getattr(b, "meta", {}) or {}
                buy_entries.append(
                    {
                        "index": idx,
                        "ts_utc": ts_b.isoformat() if ts_b else None,
                        "qty": q,
                        "price": price_raw,
                        "grid_level": meta_b.get("grid_level"),
                        "dd_pct": meta_b.get("dd_pct") if meta_b.get("dd_pct") is not None else meta_b.get("drawdown_pct"),
                        "palier_used": meta_b.get("palier_used"),
                    }
                )
            quantity = total_qty
            if quantity == 0 and buys:
                quantity = float(len(buys))  # fallback: 1 unité par BUY si qty absente
                # Si pas de prix moyen calculable, on utilisera l'entrée au prix du 1er BUY
                total_cost = quantity * (_price_at(symbol, entry_time) or 0.0)
            if quantity == 0:
                quantity = 1.0  # fallback global pour garder un PnL non nul

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
                    "synthetic_entry": not bool(buys),
                }
            )
            if len(buy_entries) > 1:
                tp_meta["intermediate_entries"] = buy_entries[1:]

            avg_entry_price = (total_cost / quantity) if quantity else (_price_at(symbol, entry_time) or 0.0)
            entry_price = avg_entry_price
            exit_price = _price_at(symbol, exit_time) or 0.0
            gross_pnl = (exit_price - entry_price) * quantity if entry_price is not None and exit_price is not None else 0.0
            gross_pnl_pct = ((exit_price - entry_price) / entry_price * 100.0) if entry_price else 0.0
            capital_used = capital_per_unit * quantity

            be_pct = tp_meta.get("be_pct")
            tp_meta["avg_entry_price"] = avg_entry_price
            tp_meta["position_qty"] = quantity
            tp_meta["pnl_pct_at_exit"] = gross_pnl_pct
            tp_meta["capital_used"] = capital_used
            if be_pct is not None:
                try:
                    tp_meta["break_even_reached"] = gross_pnl_pct >= float(be_pct)
                except Exception:
                    tp_meta["break_even_reached"] = False

            trade = CompletedTrade(
                strategy_id=strategy_id,
                run_id=run_id,
                symbol=symbol,
                asset_class=asset_class,
                side="LONG",
                cycle_id=cycle_id,
                entry_time_utc=entry_time,
                exit_time_utc=exit_time,
                entry_price=entry_price,
                exit_price=exit_price,
                quantity=quantity,
                gross_pnl=gross_pnl,
                gross_pnl_pct=gross_pnl_pct,
                max_dd_pct=max_dd_pct,
                meta=tp_meta,
            )
            trades.append(trade)
            LOGGER.info(
                "Trade built | sym=%s cycle=%s buys=%d sells=%d qty=%.4f entry=%.4f exit=%.4f pnl=%.4f pnl_pct=%.2f",
                symbol,
                cycle_id,
                len(buys),
                len(sells_tp),
                quantity,
                entry_price,
                exit_price,
                gross_pnl,
                gross_pnl_pct,
            )

    # Utilise les PnL en pourcentage pour les métriques agrégées (évite le biais multi-actifs)
    win_count = sum(1 for t in trades if t.gross_pnl_pct > 0)
    loss_count = sum(1 for t in trades if t.gross_pnl_pct <= 0)
    total_return = sum(t.gross_pnl_pct for t in trades)  # points de pourcentage cumulés
    nb_trades = len(trades)
    average_trade = total_return / nb_trades if nb_trades else 0.0
    # max drawdown sur l'équité cumulée en pourcentage (somme des pnl_pct)
    if trades:
        sorted_trades = sorted(trades, key=lambda t: t.exit_time_utc)
        equity = 0.0
        peak = 0.0
        max_dd_val = 0.0
        for tr in sorted_trades:
            equity += tr.gross_pnl_pct
            peak = max(peak, equity)
            max_dd_val = min(max_dd_val, equity - peak)
        max_drawdown_val = abs(max_dd_val)
    else:
        max_drawdown_val = 0.0
    # Courbe de capital en valeur (utilise capital_per_unit * qty comme notionnel)
    equity_value = initial_capital
    equity_peak_value = initial_capital
    max_dd_value = 0.0
    if trades:
        for tr in sorted(trades, key=lambda t: t.exit_time_utc):
            notionnel = capital_per_unit * (tr.quantity or 0.0)
            pnl_value = notionnel * (tr.gross_pnl_pct / 100.0)
            equity_value += pnl_value
            equity_peak_value = max(equity_peak_value, equity_value)
            max_dd_value = max(max_dd_value, equity_peak_value - equity_value)
    max_drawdown_pct_value = (max_dd_value / equity_peak_value * 100.0) if equity_peak_value > 0 else None
    return_pct_value = ((equity_value - initial_capital) / initial_capital * 100.0) if initial_capital else None
    tp_values: List[float] = []
    for tr in trades:
        tp_val = tr.meta.get("tp_pct") if isinstance(tr.meta, dict) else None
        if tp_val is not None:
            try:
                tp_values.append(float(tp_val))
            except Exception:
                continue
    average_tp = sum(tp_values) / len(tp_values) if tp_values else None
    returns_pct = [t.gross_pnl_pct for t in trades]
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
                sortino = mean_ret / downside_std * (len(returns_pct) ** 0.5)
    rr_moyen = None
    win_vals = [r for r in returns_pct if r > 0]
    loss_vals = [abs(r) for r in returns_pct if r < 0]
    if win_vals and loss_vals:
        rr_moyen = (sum(win_vals) / len(win_vals)) / (sum(loss_vals) / len(loss_vals))

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
        max_drawdown=max_drawdown_val,
        average_trade=average_trade,
        average_sl=None,  # pas de SL explicite dans cette strat
        average_tp=average_tp,
        rr_moyen=rr_moyen,
        total_net_return=total_return,
        net_win_count=win_count,
        net_loss_count=loss_count,
        average_net_trade=average_trade,
        initial_capital=initial_capital,
        final_capital=equity_value,
        return_pct=return_pct_value,
        max_drawdown_pct=max_drawdown_pct_value,
        volatility_pct=volatility_pct,
        sharpe=sharpe,
        sortino=sortino,
        winrate_pct=(win_count / nb_trades * 100.0) if nb_trades else None,
        extra={
            "capital_per_unit": capital_per_unit,
            "note": "DCA performance computed from pct PnL with simple capital model.",
        },
    )

    LOGGER.info(
        "Performance built | strategy=%s run=%s trades=%d wins=%d losses=%d total_return=%.4f avg_trade=%.4f",
        strategy_id,
        run_id,
        nb_trades,
        win_count,
        loss_count,
        total_return,
        average_trade,
    )

    return run, trades


def build_backend_payload_for_java(
    strategy_id: str,
    run_id: str,
    asset_class: str,
    universe: Optional[str],
    timeframe: Optional[str],
    signals_by_symbol: Dict[str, List[SignalLike]],
    ohlc_by_symbol: Optional[Dict[str, "pd.DataFrame"]] = None,
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
        ohlc_by_symbol=ohlc_by_symbol,
        equity_curve=equity_curve,
        config=config,
    )
    return to_backend_payload(run, trades)
