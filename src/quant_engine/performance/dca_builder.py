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
from ..backtest import metrics as backtest_metrics
from .stress_tests import run_monte_carlo_on_trades

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


def _extract_buy_cashflows(trade: CompletedTrade) -> List[Tuple[datetime, float]]:
    """Extract negative cashflows from trade metadata buy entries."""

    raw_entries = trade.meta.get("buy_entries") if isinstance(trade.meta, dict) else None
    if not isinstance(raw_entries, list):
        return []
    out: List[Tuple[datetime, float]] = []
    for item in raw_entries:
        if not isinstance(item, dict):
            continue
        dt = _maybe_dt(item.get("ts_utc"))
        qty = item.get("qty")
        price = item.get("price")
        if dt is None:
            continue
        try:
            amount = float(qty) * float(price)
        except Exception:
            continue
        if amount > 0:
            out.append((dt, -amount))
    return out




def _label_market_regime(return_pct: float) -> str:
    """Label market regime from window return.

    Rules:
    - return >= 20%: ``bull``
    - return <= -20%: ``bear``
    - otherwise: ``sideways``
    """
    if return_pct >= 20.0:
        return "bull"
    if return_pct <= -20.0:
        return "bear"
    return "sideways"


def _rolling_window_analytics(
    ohlc_by_symbol: Dict[str, "pd.DataFrame"],
    *,
    window_years: List[int],
    step_months: int,
) -> Dict[str, Any]:
    """Compute rolling return/xirr series, regime labels and structural underperformance."""
    if not ohlc_by_symbol:
        return {"series": [], "underperformance": {"duration_windows": 0, "severity_pct_points": 0.0}}

    returns_by_symbol: Dict[str, pd.Series] = {}
    for symbol, df in ohlc_by_symbol.items():
        if df is None or df.empty or "ts" not in df.columns or "close" not in df.columns:
            continue
        local = df[["ts", "close"]].copy()
        local["ts"] = pd.to_datetime(local["ts"], utc=True, errors="coerce")
        local = local.dropna(subset=["ts", "close"]).sort_values("ts")
        if local.empty:
            continue
        close = pd.to_numeric(local["close"], errors="coerce")
        ret = close.pct_change().fillna(0.0)
        returns_by_symbol[symbol] = pd.Series(ret.to_numpy(), index=local["ts"].dt.tz_convert(None))

    if not returns_by_symbol:
        return {"series": [], "underperformance": {"duration_windows": 0, "severity_pct_points": 0.0}}

    combined = pd.concat(returns_by_symbol.values(), axis=1).fillna(0.0)
    avg_returns = combined.mean(axis=1)
    synthetic_dataset = [{"timestamp": ts.isoformat()} for ts in avg_returns.index.to_pydatetime()]

    from ..validate.splitter import generate_dca_rolling_windows

    windows = generate_dca_rolling_windows(
        synthetic_dataset,
        window_years=window_years,
        step_months=step_months,
    )
    series: List[Dict[str, Any]] = []
    rolling_returns: List[float] = []
    for window in windows:
        start = pd.to_datetime(window["start"])
        end = pd.to_datetime(window["end"])
        sl = avg_returns[(avg_returns.index >= start) & (avg_returns.index < end)]
        if sl.empty:
            continue
        growth = float((1.0 + sl).prod() - 1.0)
        years = int(window["window_years"])
        irr = (1.0 + growth) ** (1.0 / max(years, 1)) - 1.0
        r_pct = growth * 100.0
        rolling_returns.append(r_pct)
        series.append(
            {
                "index_ts": window["index_ts"],
                "start": window["start"],
                "end": window["end"],
                "window_years": years,
                "return_pct": r_pct,
                "irr": irr,
                "regime": _label_market_regime(r_pct),
            }
        )

    longest = 0
    current = 0
    severity = 0.0
    for value in rolling_returns:
        if value < 0:
            current += 1
            longest = max(longest, current)
            severity += abs(value)
        else:
            current = 0

    return {
        "series": series,
        "regime_definition": {
            "version": "v1",
            "bull_min_return_pct": 20.0,
            "bear_max_return_pct": -20.0,
            "sideways_between": [-20.0, 20.0],
        },
        "underperformance": {
            "duration_windows": longest,
            "severity_pct_points": severity,
        },
    }
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
    trades_by_symbol: Dict[str, int] = {}
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
            sells_tp = [
                s
                for s in cycle_signals
                if getattr(s, "side", "").upper() == "SELL"
                and getattr(getattr(s, "meta", {}), "get", lambda *_: None)("action")
                in {"take_profit", "break_even", "stop_loss", "forced_exit_end"}
            ]
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
            tp_meta["buy_entries"] = buy_entries
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
            trades_by_symbol[symbol] = trades_by_symbol.get(symbol, 0) + 1
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
        if symbol not in trades_by_symbol:
            trades_by_symbol[symbol] = 0
            LOGGER.info("No completed trades for %s (signals=%d, cycles=%d)", symbol, len(ordered), len(by_cycle))

    # Utilise les PnL en pourcentage pour les métriques agrégées (évite le biais multi-actifs)
    win_count = sum(1 for t in trades if t.gross_pnl_pct > 0)
    loss_count = sum(1 for t in trades if t.gross_pnl_pct <= 0)
    total_return = sum(t.gross_pnl_pct for t in trades)  # points de pourcentage cumulés
    nb_trades = len(trades)
    average_trade = total_return / nb_trades if nb_trades else 0.0

    sorted_trades = sorted(trades, key=lambda t: t.exit_time_utc) if trades else []

    # max drawdown sur l'équité cumulée en pourcentage (somme des pnl_pct)
    if sorted_trades:
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

    # Courbe de capital en valeur (cashflows irréguliers: entrées BUY puis sortie SELL agrégée)
    cashflows: List[Tuple[datetime, float]] = []
    equity_values: List[float] = []
    contributed_values: List[float] = []
    contributed_capital = 0.0
    equity_value = float(initial_capital)

    if sorted_trades:
        for tr in sorted_trades:
            for dt, amount in _extract_buy_cashflows(tr):
                cashflows.append((dt, amount))
                contributed_capital += abs(amount)
                equity_value += amount
                equity_values.append(equity_value)
                contributed_values.append(contributed_capital)

            exit_dt = tr.exit_time_utc
            sale_amount = (tr.exit_price or 0.0) * (tr.quantity or 0.0)
            if sale_amount:
                cashflows.append((exit_dt, sale_amount))
                equity_value += sale_amount
                equity_values.append(equity_value)
                contributed_values.append(contributed_capital)

    max_drawdown_pct_value = (
        backtest_metrics.max_drawdown_on_contributed_capital(equity_values, contributed_values) * 100.0
        if equity_values
        else None
    )
    return_pct_value = ((equity_value - initial_capital) / initial_capital * 100.0) if initial_capital else None

    final_perf_norm = backtest_metrics.final_performance_normalized(equity_value, contributed_capital)
    twr_value = backtest_metrics.twr([t.gross_pnl_pct / 100.0 for t in sorted_trades])
    xirr_value, xirr_status = backtest_metrics.xirr(cashflows) if cashflows else (None, "invalid_cashflows")
    time_under_water_periods = backtest_metrics.time_under_water(equity_values)
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

    rolling_cfg = config.get("rolling_windows", {}) if isinstance(config.get("rolling_windows"), Mapping) else {}
    rolling_years_raw = rolling_cfg.get("years", [3, 5, 10])
    if not isinstance(rolling_years_raw, list):
        rolling_years_raw = [3, 5, 10]
    rolling_years = [int(y) for y in rolling_years_raw if int(y) > 0]
    rolling_step_months = int(rolling_cfg.get("step_months", 1) or 1)
    rolling_analytics = _rolling_window_analytics(
        ohlc_by_symbol,
        window_years=rolling_years or [3, 5, 10],
        step_months=rolling_step_months,
    )
    underperformance = rolling_analytics.get("underperformance", {}) if isinstance(rolling_analytics, dict) else {}
    score_cfg = config.get("composite_score", {}) if isinstance(config.get("composite_score"), Mapping) else {}
    score_weights = score_cfg.get("weights") if isinstance(score_cfg.get("weights"), Mapping) else None
    dca_score = backtest_metrics.dca_composite_score(
        final_performance_normalized_value=final_perf_norm,
        xirr_value=xirr_value,
        max_drawdown_on_contributed_capital_value=(
            (max_drawdown_pct_value / 100.0) if isinstance(max_drawdown_pct_value, (int, float)) else None
        ),
        underperformance_duration_windows=int(underperformance.get("duration_windows", 0) or 0),
        underperformance_severity_pct_points=float(underperformance.get("severity_pct_points", 0.0) or 0.0),
        xirr_status=xirr_status,
        weights=score_weights,
    )

    universe_rules_version = str(config.get("universe_rules_version", "asset-universe-rules-v1"))

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
            "note": "DCA performance computed from pct PnL with cashflow-aware metrics.",
            "metrics_version": "dca-grid-process-v1",
            "universe_rules_version": universe_rules_version,
            "final_performance_normalized": final_perf_norm,
            "twr": twr_value,
            "xirr": xirr_value,
            "xirr_status": xirr_status,
            "max_drawdown_on_contributed_capital": max_drawdown_pct_value,
            "time_under_water": time_under_water_periods,
            "contributed_capital": contributed_capital,
            "rolling_windows": rolling_analytics,
            "dca_composite_score": dca_score,
            "dca_edge": dca_score.get("edge"),
            "dca_score": dca_score.get("score"),
        },
    )

    _attach_stress_tests(run, trades, config=config, initial_capital=initial_capital)

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


def _stress_tests_enabled(config: Mapping[str, Any]) -> bool:
    stress_config = config.get("stress_tests", {}) if isinstance(config.get("stress_tests"), Mapping) else {}
    if "enabled" in stress_config:
        return bool(stress_config.get("enabled"))
    return bool(config.get("stress_tests_enabled", False))


def _build_monte_carlo_level1(
    trades: List[CompletedTrade],
    *,
    initial_capital: float,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    stress_config = config.get("stress_tests", {}) if isinstance(config.get("stress_tests"), Mapping) else {}
    monte_carlo_config = stress_config.get("monte_carlo", {}) if isinstance(stress_config.get("monte_carlo"), Mapping) else {}
    parameters = {"initial_capital": initial_capital, **monte_carlo_config}
    metadata = {
        "strategy_id": trades[0].strategy_id if trades else None,
        "run_id": trades[0].run_id if trades else None,
        "asset_class": trades[0].asset_class if trades else None,
    }
    result = run_monte_carlo_on_trades(trades, metadata=metadata, parameters=parameters)
    metrics = result.get("metrics", {}) if isinstance(result, Mapping) else {}
    level1_keys = {
        "final_capital",
        "return_pct",
        "max_drawdown",
        "max_drawdown_pct",
        "volatility_pct",
        "sharpe",
        "sortino",
        "winrate_pct",
        "total_return",
        "average_trade",
        "win_count",
        "loss_count",
    }
    level1_metrics = {key: metrics.get(key) for key in level1_keys if key in metrics}
    payload: Dict[str, Any] = {
        "metrics": level1_metrics,
        "parameters": result.get("parameters", {}) if isinstance(result, Mapping) else {},
    }
    warnings = result.get("warnings") if isinstance(result, Mapping) else None
    if warnings:
        payload["warnings"] = warnings
    return payload


def _attach_stress_tests(
    run: StrategyRunResult,
    trades: List[CompletedTrade],
    *,
    config: Mapping[str, Any],
    initial_capital: float,
) -> None:
    if not _stress_tests_enabled(config):
        return
    if not trades:
        return
    stress_payload = {
        "monte_carlo_level1": _build_monte_carlo_level1(
            trades,
            initial_capital=initial_capital,
            config=config,
        )
    }
    if run.extra is None:
        run.extra = {}
    run.extra["stress_tests"] = stress_payload


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
