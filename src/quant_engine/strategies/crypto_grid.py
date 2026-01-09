"""Macro/micro grid strategy tailored for crypto assets."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from .base import Strategy, StrategySignal


@dataclass
class _CryptoState:
    """Track drawdown cycle state for each crypto instrument."""

    cycle_id: int = 0
    cycle_active: bool = False
    consumed_levels: List[bool] = field(default_factory=list)
    max_dd: float = 0.0
    cycle_low: Optional[float] = None
    cycle_high_ref: Optional[float] = None
    prev_dd: Optional[float] = None
    last_processed_ts: Optional[pd.Timestamp] = None
    tp_emitted: bool = False

    def reset(self, level_count: int) -> None:
        self.cycle_active = False
        self.consumed_levels = [False] * level_count
        self.max_dd = 0.0
        self.cycle_low = None
        self.cycle_high_ref = None
        self.prev_dd = None
        self.tp_emitted = False


class CryptoGridStrategy(Strategy):
    """Combine macro and micro signals to rotate within a crypto universe."""

    def __init__(self, strategy_id: str, params: Dict[str, Any]) -> None:
        self.strategy_id = strategy_id
        self.params = params or {}
        grid = list(self.params.get("grid", []))
        if not grid:
            raise ValueError("CryptoGridStrategy requires a non-empty grid configuration")
        self.grid: List[Dict[str, Any]] = sorted(grid, key=lambda item: float(item["dd"]))
        self.asset_class = self.params.get("asset_class", "CRYPTO").upper()
        self.tp_sl_config: Dict[str, Any] = self.params.get("tp_sl", {})

    @staticmethod
    def compute_drawdown(close: pd.Series) -> pd.Series:
        rolling_max = close.cummax()
        dd = (close / rolling_max - 1.0) * 100.0
        return dd.fillna(0.0)

    def backtest(self, ohlc: pd.DataFrame, context: Dict[str, Any]) -> List[StrategySignal]:
        df = self._normalize_ohlc(ohlc)
        state = _CryptoState()
        state.reset(len(self.grid))
        return self._process(df, context, state, only_last_ts=None)

    def evaluate_live_bar(
        self, ohlc: pd.DataFrame, context: Dict[str, Any]
    ) -> List[StrategySignal]:
        df = self._normalize_ohlc(ohlc)
        strategy_ctx = context.setdefault("state", {})
        state = self._state_from_dict(strategy_ctx)
        last_ts = df.index.max() if not df.empty else None
        signals = self._process(df, context, state, only_last_ts=last_ts)
        self._update_context_dict(strategy_ctx, state)
        return signals

    def _process(
        self,
        df: pd.DataFrame,
        context: Dict[str, Any],
        state: _CryptoState,
        only_last_ts: Optional[pd.Timestamp],
    ) -> List[StrategySignal]:
        if df.empty:
            return []
        close = df["close"].astype(float)
        dd_series = self.compute_drawdown(close)
        rolling_max = close.cummax()
        symbol = context.get("symbol", context.get("symbol_id", ""))
        asset_class = context.get("asset_class", self.asset_class)
        macro_context = context.get("macro")
        results: List[StrategySignal] = []
        last_processed = state.last_processed_ts
        for ts, price, dd in zip(dd_series.index, close, dd_series):
            if last_processed is not None and ts <= last_processed:
                continue
            allow_entries = self._allow_entries(df, ts)
            self._ensure_state_initialized(state)
            if allow_entries and not state.cycle_active:
                self._maybe_start_cycle(state, float(dd), float(rolling_max.loc[ts]), float(price))
            if state.cycle_active:
                self._update_cycle_stats(state, float(dd), float(price))
                buys = (
                    self._check_buy_levels(state, float(dd), ts, symbol, asset_class, macro_context)
                    if allow_entries
                    else []
                )
                sells = self._check_take_profit(
                    state, float(price), ts, symbol, asset_class, macro_context
                )
                for sig in (*buys, *sells):
                    if only_last_ts is None or sig.ts_open_utc == only_last_ts:
                        results.append(sig)
            self._maybe_reset_on_recovery(state, float(price))
            state.prev_dd = float(dd)
            state.last_processed_ts = ts
        return results

    @staticmethod
    def _allow_entries(df: pd.DataFrame, ts: pd.Timestamp) -> bool:
        if "_filter_ok" not in df.columns:
            return True
        try:
            return bool(df.at[ts, "_filter_ok"])
        except Exception:
            return False

    def _ensure_state_initialized(self, state: _CryptoState) -> None:
        if not state.consumed_levels:
            state.reset(len(self.grid))

    def _maybe_start_cycle(
        self,
        state: _CryptoState,
        dd: float,
        ref_high: float,
        price: float,
    ) -> None:
        if state.cycle_active:
            return
        eligible = [idx for idx, level in enumerate(self.grid) if dd <= float(level["dd"])]
        if not eligible:
            return
        state.cycle_active = True
        state.cycle_id += 1
        state.consumed_levels = [False] * len(self.grid)
        state.cycle_high_ref = ref_high
        state.cycle_low = price
        state.max_dd = dd
        state.prev_dd = state.prev_dd if state.prev_dd is not None else 0.0

    def _update_cycle_stats(self, state: _CryptoState, dd: float, price: float) -> None:
        state.max_dd = min(state.max_dd, dd)
        state.cycle_low = price if state.cycle_low is None else min(state.cycle_low, price)

    def _check_buy_levels(
        self,
        state: _CryptoState,
        dd: float,
        ts: pd.Timestamp,
        symbol: str,
        asset_class: str,
        macro_context: Any,
    ) -> List[StrategySignal]:
        signals: List[StrategySignal] = []
        for idx, level in enumerate(self.grid):
            if state.consumed_levels[idx]:
                continue
            threshold = float(level["dd"])
            prev_dd = state.prev_dd if state.prev_dd is not None else 0.0
            if dd <= threshold and prev_dd > threshold:
                state.consumed_levels[idx] = True
                tp_rule = self._resolve_tp_rule(state.max_dd)
                meta = {
                    "dd_pct": dd,
                    "drawdown_pct": dd,
                    "grid_level": threshold,
                    "grid_config": self.grid,
                    "cycle_id": state.cycle_id,
                    "max_dd_reached": state.max_dd,
                    "action": level.get("action", "increase"),
                    "type": level.get("action", "increase"),
                    "intensity": level.get("intensity"),
                    "weight": float(level.get("weight", 0.0)),
                    "tp_target": tp_rule.get("tp_pct") if tp_rule else None,
                    "tp_mode": self.tp_sl_config.get("mode"),
                    "tp_pct": tp_rule.get("tp_pct") if tp_rule else None,
                    "be_pct": tp_rule.get("be_pct") if tp_rule else None,
                    "macro_context": macro_context,
                    "macro_regime": macro_context,
                }
                signals.append(
                    StrategySignal(
                        strategy_id=self.strategy_id,
                        symbol=symbol,
                        asset_class=asset_class,
                        side="BUY",
                        ts_open_utc=ts,
                        qty=0.0,
                        meta=meta,
                    )
                )
        return signals

    def _check_take_profit(
        self,
        state: _CryptoState,
        price: float,
        ts: pd.Timestamp,
        symbol: str,
        asset_class: str,
        macro_context: Any,
    ) -> List[StrategySignal]:
        signals: List[StrategySignal] = []
        tp_rule = self._resolve_tp_rule(state.max_dd)
        if not tp_rule or state.cycle_low is None:
            return signals
        tp_pct = tp_rule.get("tp_pct")
        if tp_pct is None:
            return signals
        rebound = (price / state.cycle_low - 1.0) * 100.0
        if rebound >= float(tp_pct) and not state.tp_emitted:
            state.tp_emitted = True
            meta = {
                "action": "rebalance",
                "type": "rotate",
                "tp_mode": self.tp_sl_config.get("mode"),
                "tp_pct": tp_rule.get("tp_pct"),
                "be_pct": tp_rule.get("be_pct"),
                "max_dd_reached": state.max_dd,
                "drawdown_pct": state.max_dd,
                "cycle_id": state.cycle_id,
                "grid_config": self.grid,
                "grid_level": None,
                "rebound_pct": rebound,
                "macro_context": macro_context,
                "macro_regime": macro_context,
            }
            signals.append(
                StrategySignal(
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    asset_class=asset_class,
                    side="SELL",
                    ts_open_utc=ts,
                    qty=0.0,
                    meta=meta,
                )
            )
            self._reset_cycle(state)
        return signals

    def _maybe_reset_on_recovery(self, state: _CryptoState, price: float) -> None:
        if not state.cycle_active:
            return
        ref = state.cycle_high_ref
        if ref is None:
            return
        if price >= ref:
            self._reset_cycle(state)

    def _reset_cycle(self, state: _CryptoState) -> None:
        prev_cycle = state.cycle_id
        state.reset(len(self.grid))
        state.cycle_id = prev_cycle

    def _resolve_tp_rule(self, max_dd: float) -> Optional[Dict[str, Any]]:
        cfg = self.tp_sl_config or {}
        if not cfg or not cfg.get("enabled", False):
            return None
        if cfg.get("mode") != "per_grid_max_dd":
            return None
        rules = cfg.get("rules", [])
        selected: Optional[Dict[str, Any]] = None
        max_dd_mag = abs(max_dd)
        for rule in sorted(rules, key=lambda r: abs(float(r.get("max_dd_reached", 0.0)))):
            threshold = abs(float(rule.get("max_dd_reached", 0.0)))
            if max_dd_mag >= threshold:
                selected = rule
        return selected

    def _normalize_ohlc(self, ohlc: pd.DataFrame) -> pd.DataFrame:
        if ohlc.empty:
            return ohlc
        df = ohlc.copy()
        if "ts" in df.columns:
            df["ts"] = pd.to_datetime(df["ts"], utc=True)
            df = df.set_index("ts")
        else:
            df.index = pd.to_datetime(df.index, utc=True)
        df = df.sort_index()
        required = {"open", "high", "low", "close"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"OHLC dataframe missing columns: {missing}")
        return df

    def _state_from_dict(self, data: Dict[str, Any]) -> _CryptoState:
        state = _CryptoState()
        state.cycle_id = int(data.get("cycle_id", 0))
        state.cycle_active = bool(data.get("cycle_active", False))
        state.consumed_levels = list(data.get("consumed_levels", []))
        state.max_dd = float(data.get("max_dd", 0.0))
        cycle_low = data.get("cycle_low")
        state.cycle_low = float(cycle_low) if cycle_low is not None else None
        ref = data.get("cycle_high_ref")
        state.cycle_high_ref = float(ref) if ref is not None else None
        prev_dd = data.get("prev_dd")
        state.prev_dd = float(prev_dd) if prev_dd is not None else None
        last_ts = data.get("last_processed_ts")
        state.last_processed_ts = (
            pd.Timestamp(last_ts).tz_convert("UTC") if last_ts is not None else None
        )
        state.tp_emitted = bool(data.get("tp_emitted", False))
        self._ensure_state_initialized(state)
        return state

    def _update_context_dict(self, target: Dict[str, Any], state: _CryptoState) -> None:
        target.update(
            {
                "cycle_id": state.cycle_id,
                "cycle_active": state.cycle_active,
                "consumed_levels": list(state.consumed_levels),
                "max_dd": state.max_dd,
                "cycle_low": state.cycle_low,
                "cycle_high_ref": state.cycle_high_ref,
                "prev_dd": state.prev_dd,
                "last_processed_ts": state.last_processed_ts.isoformat()
                if state.last_processed_ts is not None
                else None,
                "tp_emitted": state.tp_emitted,
            }
        )


__all__ = ["CryptoGridStrategy"]
