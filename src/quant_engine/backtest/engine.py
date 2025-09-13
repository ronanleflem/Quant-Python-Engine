"""Vectorised bar-based backtest engine."""
from __future__ import annotations

from typing import List, Dict, Any, Tuple

from ..tpsl.rules import StopInitializer, TakeProfit
from . import metrics


def run(
    dataset: List[Dict[str, Any]],
    signals: List[int],
    atr_values: List[float],
    atr_mult: float,
    r_mult: float,
    slippage_bps: float = 0.0,
    fee_bps: float = 0.0,
) -> Tuple[List[Dict[str, Any]], List[float], Dict[str, float]]:
    """Execute a vectorised backtest.

    The strategy is evaluated on pre-computed ``signals`` (1 for long, 0 flat).
    Only one position is allowed at any time.  Orders are filled on the next
    bar's open and exits occur on the following bar after a stop loss or take
    profit condition is triggered.
    """

    cost_rate = (slippage_bps + fee_bps) / 10000.0
    trades: List[Dict[str, Any]] = []
    equity: List[float] = []
    cash = 0.0
    position = 0
    entry_price = 0.0
    entry_ts = ""
    stop_price = 0.0
    tp_price = 0.0
    sl_distance = 0.0

    n = len(dataset)
    for i in range(n - 1):
        row = dataset[i]
        nxt = dataset[i + 1]
        signal = signals[i]

        if position == 0 and signal == 1:
            entry_price = nxt["open"] * (1 + cost_rate)
            entry_ts = nxt["timestamp"]
            stop_price, sl_distance = StopInitializer.fixed_atr(
                atr_values, atr_mult, 1, i + 1, entry_price
            )
            tp_price = TakeProfit.r_multiple(entry_price, stop_price, r_mult, 1)
            position = 1
        elif position == 1:
            hit_tp = row["high"] >= tp_price
            hit_sl = row["low"] <= stop_price
            exit_signal = signal == 0
            if hit_tp or hit_sl or exit_signal:
                exit_price = nxt["open"] * (1 - cost_rate)
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
        equity.append(cash)

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
