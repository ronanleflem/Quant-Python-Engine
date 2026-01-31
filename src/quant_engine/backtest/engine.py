"""Vectorised bar-based backtest engine."""
from __future__ import annotations

from typing import List, Dict, Any, Tuple, Mapping
import time
import logging
import random

from ..tpsl.rules import StopInitializer, TakeProfit, DynamicStopLoss
from . import metrics


def _parse_tpsl_jitter(jitter_cfg: Mapping[str, Any] | None) -> Dict[str, Any] | None:
    if not isinstance(jitter_cfg, Mapping):
        return None
    if jitter_cfg.get("enabled", True) is False:
        return None
    dist = str(jitter_cfg.get("dist", "uniform")).strip().lower()
    if dist not in {"uniform", "normal"}:
        raise ValueError(f"Unsupported tpsl jitter dist: {dist}")
    tp_bps = jitter_cfg.get("tp_bps", jitter_cfg.get("tp", 0.0))
    sl_bps = jitter_cfg.get("sl_bps", jitter_cfg.get("sl", 0.0))
    try:
        tp_bps_val = abs(float(tp_bps))
    except Exception:
        tp_bps_val = 0.0
    try:
        sl_bps_val = abs(float(sl_bps))
    except Exception:
        sl_bps_val = 0.0
    if tp_bps_val <= 0 and sl_bps_val <= 0:
        return None
    seed = jitter_cfg.get("seed")
    rng = random.Random(seed) if seed is not None else random.Random()
    return {
        "dist": dist,
        "tp_bps": tp_bps_val,
        "sl_bps": sl_bps_val,
        "rng": rng,
    }


def _draw_tpsl_jitter(cfg: Mapping[str, Any], *, hit_tp: bool, hit_sl: bool) -> float:
    dist = cfg["dist"]
    rng: random.Random = cfg["rng"]
    bps = cfg["sl_bps"] if hit_sl else cfg["tp_bps"]
    if not bps:
        return 0.0
    if dist == "uniform":
        return float(rng.uniform(-bps, bps))
    return float(rng.gauss(0.0, bps))


def run(
    dataset: List[Dict[str, Any]],
    signals: List[int],
    atr_values: List[float],
    atr_mult: float,
    r_mult: float,
    slippage_bps: float = 0.0,
    fee_bps: float = 0.0,
    max_trades: int | None = None,
    max_seconds: float | None = None,
    pruning: Mapping[str, Any] | None = None,
    dynamic_sl: Mapping[str, Any] | None = None,
    tpsl_jitter: Mapping[str, Any] | None = None,
) -> Tuple[List[Dict[str, Any]], List[float], Dict[str, float]]:
    """Execute a vectorised backtest.

    The strategy is evaluated on pre-computed ``signals`` (1 for long, 0 flat).
    Only one position is allowed at any time.  Orders are filled on the next
    bar's open and exits occur on the following bar after a stop loss or take
    profit condition is triggered.
    """

    cost_rate = (slippage_bps + fee_bps) / 10000.0
    jitter_cfg = _parse_tpsl_jitter(tpsl_jitter)
    pruning_cfg = pruning if isinstance(pruning, Mapping) else {}
    pruning_enabled = bool(pruning_cfg) and pruning_cfg.get("enabled", True) is not False
    max_dd_pct = pruning_cfg.get("max_drawdown_pct") if pruning_enabled else None
    min_signals_cfg = pruning_cfg.get("min_signals_after_bars") if pruning_enabled else None
    bars_threshold = None
    min_signals = None
    if isinstance(min_signals_cfg, Mapping):
        bars_threshold = min_signals_cfg.get("bars")
        min_signals = min_signals_cfg.get("min_signals")
        try:
            bars_threshold = int(bars_threshold)
        except Exception:
            bars_threshold = None
        try:
            min_signals = int(min_signals)
        except Exception:
            min_signals = None

    logger = logging.getLogger(__name__)
    trades: List[Dict[str, Any]] = []
    equity: List[float] = []
    cash = 0.0
    position = 0
    entry_price = 0.0
    entry_ts = ""
    stop_price = 0.0
    tp_price = 0.0
    sl_distance = 0.0
    start_ts = time.monotonic()
    bars_seen = 0
    signals_seen = 0
    peak = 0.0

    n = len(dataset)
    for i in range(n - 1):
        if max_seconds is not None and max_seconds > 0 and (time.monotonic() - start_ts) >= max_seconds:
            break
        row = dataset[i]
        nxt = dataset[i + 1]
        signal = signals[i]
        bars_seen += 1
        if signal == 1:
            signals_seen += 1

        if position == 0 and signal == 1:
            entry_price = nxt["open"] * (1 + cost_rate)
            entry_ts = nxt["timestamp"]
            stop_data = StopInitializer.fixed_atr(
                atr_values, atr_mult, 1, i + 1, entry_price
            )
            if stop_data is None:
                logger.debug(
                    "Skipping entry at idx=%d: missing/invalid ATR stop inputs.",
                    i + 1,
                )
                continue
            stop_price, sl_distance = stop_data
            tp_price = TakeProfit.r_multiple(entry_price, stop_price, r_mult, 1)
            position = 1
        elif position == 1:
            if dynamic_sl and dynamic_sl.get("enabled", True) is not False:
                mode = str(dynamic_sl.get("mode", "trail_atr")).lower()
                if mode == "trail_atr":
                    dyn_mult = dynamic_sl.get("atr_mult", atr_mult)
                    updated = DynamicStopLoss.trail_atr(
                        atr_values,
                        dyn_mult,
                        1,
                        i,
                        row["close"],
                        stop_price,
                    )
                    if updated is not None:
                        stop_price = updated
                else:
                    raise ValueError(f"Unsupported dynamic_sl mode: {mode}")
            hit_tp = row["high"] >= tp_price
            hit_sl = row["low"] <= stop_price
            exit_signal = signal == 0
            if hit_tp or hit_sl or exit_signal:
                exit_price = nxt["open"] * (1 - cost_rate)
                if jitter_cfg is not None and (hit_tp or hit_sl):
                    jitter_bps = _draw_tpsl_jitter(jitter_cfg, hit_tp=hit_tp, hit_sl=hit_sl)
                    if jitter_bps:
                        exit_price *= 1 + (jitter_bps / 10000.0)
                exit_ts = nxt["timestamp"]
                pnl = exit_price - entry_price
                r_val = pnl / sl_distance if sl_distance else 0.0
                trades.append(
                    {
                        "ts_entry": entry_ts,
                        "price_entry": entry_price,
                        "ts_exit": exit_ts,
                        "price_exit": exit_price,
                        "side": "long",
                        "r_multiple": r_val,
                        "pnl": pnl,
                    }
                )
                cash += pnl
                position = 0
                if max_trades is not None and max_trades > 0 and len(trades) >= max_trades:
                    break
        equity.append(cash)
        if pruning_enabled:
            if max_dd_pct is not None:
                try:
                    max_dd_value = float(max_dd_pct)
                except Exception:
                    max_dd_value = None
                if max_dd_value is not None and max_dd_value > 0:
                    if cash > peak:
                        peak = cash
                    denom = abs(peak) if abs(peak) > 1e-9 else 1.0
                    dd_pct = (peak - cash) / denom * 100.0
                    if dd_pct >= max_dd_value:
                        logger.info(
                            "Pruning backtest: drawdown %.2f%% >= %.2f%% after %d bars",
                            dd_pct,
                            max_dd_value,
                            bars_seen,
                        )
                        break
            if (
                bars_threshold is not None
                and min_signals is not None
                and bars_threshold > 0
                and bars_seen >= bars_threshold
                and signals_seen < min_signals
            ):
                logger.info(
                    "Pruning backtest: signals=%d after %d bars (min=%d)",
                    signals_seen,
                    bars_seen,
                    min_signals,
                )
                break

    # Handle trailing equity and open position at the end
    if position == 1:
        last = dataset[-1]
        exit_price = last["close"] * (1 - cost_rate)
        pnl = exit_price - entry_price
        r_val = pnl / sl_distance if sl_distance else 0.0
        trades.append(
            {
                "ts_entry": entry_ts,
                "price_entry": entry_price,
                "ts_exit": last["timestamp"],
                "price_exit": exit_price,
                "side": "long",
                "r_multiple": r_val,
                "pnl": pnl,
            }
        )
        cash += pnl
    equity.append(cash)

    summary = metrics.compute(trades, equity)
    return trades, equity, summary
