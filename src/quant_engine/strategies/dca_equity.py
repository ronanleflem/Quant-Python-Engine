"""Weighted drawdown-based DCA strategy for equities."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from .base import Strategy, StrategySignal


@dataclass
class _CycleState:
    """Internal mutable state shared across backtests and live evaluation."""

    cycle_id: int = 0
    cycle_active: bool = False
    consumed_levels: List[bool] = field(default_factory=list)
    max_dd: float = 0.0
    cycle_low: Optional[float] = None
    cycle_high_ref: Optional[float] = None
    prev_dd: Optional[float] = None
    last_processed_ts: Optional[pd.Timestamp] = None
    tp_emitted: bool = False
    position_qty: float = 0.0
    position_cost: float = 0.0  # somme prix*qty pour prix moyen

    def reset(self, level_count: int) -> None:
        self.cycle_active = False
        self.consumed_levels = [False] * level_count
        self.max_dd = 0.0
        self.cycle_low = None
        self.cycle_high_ref = None
        self.prev_dd = None
        self.tp_emitted = False
        self.position_qty = 0.0
        self.position_cost = 0.0


class DcaEquityStrategy(Strategy):
    """Drawdown-weighted DCA logic driven by configurable grids."""

    def __init__(self, strategy_id: str, params: Dict[str, Any]) -> None:
        self.strategy_id = strategy_id
        self.params = params or {}
        grid = list(self.params.get("grid", []))
        if not grid:
            raise ValueError("DcaEquityStrategy requires a non-empty grid configuration")
        self.grid: List[Dict[str, Any]] = sorted(grid, key=lambda item: float(item["dd"]))
        self.asset_class = self.params.get("asset_class", "EQUITY").upper()
        self.tp_sl_config: Dict[str, Any] = self.params.get("tp_sl", {})

    @staticmethod
    def compute_drawdown(close: pd.Series) -> pd.Series:
        """Compute drawdown (in percentage) vs rolling high (deprecated)."""

        rolling_max = close.cummax()
        dd = (close / rolling_max - 1.0) * 100.0
        return dd.fillna(0.0)

    @staticmethod
    def compute_reference_high(close: pd.Series) -> pd.Series:
        """
        Rolling high sur les 3 derniers mois (~90 jours calendaires).
        - Si on démarre en début d’historique, on prend le max des bougies disponibles (min_periods=1).
        - Inclut la bougie courante (mise à jour dès qu’un nouveau plus haut apparaît).
        """

        ref = close.rolling("90D", min_periods=1).max()
        ref = ref.ffill().fillna(close.iloc[0])
        return ref

    def backtest(self, ohlc: pd.DataFrame, context: Dict[str, Any]) -> List[StrategySignal]:
        df = self._normalize_ohlc(ohlc)
        state = _CycleState()
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
        state: _CycleState,
        only_last_ts: Optional[pd.Timestamp],
    ) -> List[StrategySignal]:
        if df.empty:
            return []
        close = df["close"].astype(float)
        ref_high = self.compute_reference_high(close)
        dd_series = ((close / ref_high) - 1.0) * 100.0
        dd_series = dd_series.fillna(0.0)
        symbol = context.get("symbol", context.get("symbol_id", ""))
        asset_class = context.get("asset_class", self.asset_class)
        results: List[StrategySignal] = []
        last_processed = state.last_processed_ts
        for ts, price, dd, ref_h in zip(dd_series.index, close, dd_series, ref_high):
            if last_processed is not None and ts <= last_processed:
                continue
            state.current_price = float(price)
            self._ensure_cycle_initialized(state)
            # fige le ref_high une fois un cycle actif (ne pas recalculer pendant un trade)
            ref_high_value = state.cycle_high_ref if state.cycle_active else float(ref_h)
            self._maybe_start_cycle(state, float(dd), ref_high_value, float(price))
            if state.cycle_active:
                self._update_cycle_stats(state, float(dd), float(price))
                buys = self._check_buy_levels(state, float(dd), ts, symbol, asset_class)
                sells = self._check_take_profit(state, float(price), ts, symbol, asset_class)
                for sig in (*buys, *sells):
                    if only_last_ts is None or sig.ts_open_utc == only_last_ts:
                        results.append(sig)
            else:
                state.max_dd = 0.0
                state.cycle_low = None
                state.prev_dd = float(dd)
            self._maybe_reset_on_recovery(state, float(price))
            state.prev_dd = float(dd)
            if not state.cycle_active:
                state.cycle_high_ref = float(ref_h)
            state.last_processed_ts = ts
        return results

    def _ensure_cycle_initialized(self, state: _CycleState) -> None:
        if not state.consumed_levels:
            state.reset(len(self.grid))

    def _maybe_start_cycle(
        self,
        state: _CycleState,
        dd: float,
        ref_high: float,
        price: float,
    ) -> None:
        if state.cycle_active:
            return
        # start a cycle as soon as dd touches the shallowest level
        if dd > float(self.grid[-1]["dd"]):
            return
        state.cycle_active = True
        state.cycle_id += 1
        # none of the levels are consumed at start; they will be flagged as we cross thresholds
        state.consumed_levels = [False] * len(self.grid)
        state.cycle_high_ref = ref_high
        state.cycle_low = price
        state.max_dd = dd
        state.prev_dd = state.prev_dd if state.prev_dd is not None else 0.0
        state.position_qty = 0.0
        state.position_cost = 0.0

    def _update_cycle_stats(self, state: _CycleState, dd: float, price: float) -> None:
        state.max_dd = min(state.max_dd, dd)
        state.cycle_low = price if state.cycle_low is None else min(state.cycle_low, price)

    def _check_buy_levels(
        self,
        state: _CycleState,
        dd: float,
        ts: pd.Timestamp,
        symbol: str,
        asset_class: str,
    ) -> List[StrategySignal]:
        signals: List[StrategySignal] = []
        for idx, level in enumerate(self.grid):
            if state.consumed_levels[idx]:
                continue
            threshold = float(level["dd"])
            prev_dd = state.prev_dd if state.prev_dd is not None else 0.0
            if dd <= threshold and prev_dd > threshold:
                state.consumed_levels[idx] = True
                meta = self._build_buy_meta(state, level, dd)
                qty = float(level.get("weight", 0.0)) or 0.0
                state.position_qty += qty
                state.position_cost += qty * state.current_price
                signals.append(
                    StrategySignal(
                        strategy_id=self.strategy_id,
                        symbol=symbol,
                        asset_class=asset_class,
                        side="BUY",
                        ts_open_utc=ts,
                        qty=qty,
                        meta=meta,
                    )
                )
        return signals

    def _check_take_profit(
        self,
        state: _CycleState,
        price: float,
        ts: pd.Timestamp,
        symbol: str,
        asset_class: str,
    ) -> List[StrategySignal]:
        signals: List[StrategySignal] = []
        tp_rule = self._resolve_tp_rule(state.max_dd)
        if not tp_rule or state.cycle_low is None:
            return signals
        tp_pct = tp_rule.get("tp_pct")
        be_pct = tp_rule.get("be_pct")
        if tp_pct is None:
            return signals
        if state.position_qty <= 0:
            return signals

        avg_entry = state.position_cost / state.position_qty if state.position_qty > 0 else price
        if avg_entry == 0:
            return signals
        pnl_pct = (price / avg_entry - 1.0) * 100.0

        should_tp = pnl_pct >= float(tp_pct)
        should_be = be_pct is not None and pnl_pct >= float(be_pct)

        # Debug trace for TP/BE decisions (muted; re-enable for troubleshooting)
        # print(
        #     f"[TP_CHECK] cycle={state.cycle_id} sym={symbol} ts={ts} "
        #     f"avg_entry={avg_entry:.4f} price={price:.4f} pnl_pct={pnl_pct:.2f} "
        #     f"tp_pct={tp_pct} be_pct={be_pct} should_tp={should_tp} should_be={should_be} "
        #     f"pos_qty={state.position_qty:.4f}"
        # )

        if should_tp and not state.tp_emitted:
            state.tp_emitted = True
            meta = {
                "action": "take_profit",
                "tp_mode": self.tp_sl_config.get("mode"),
                "tp_pct": tp_rule.get("tp_pct"),
                "be_pct": tp_rule.get("be_pct"),
                "max_dd_reached": state.max_dd,
                "drawdown_pct": state.max_dd,
                "cycle_id": state.cycle_id,
                "grid_config": self.grid,
                "grid_level": None,
                "rebound_pct": None,
                "avg_entry_price": avg_entry,
                "pnl_pct_at_exit": pnl_pct,
                "break_even_reached": should_be,
            }
            signals.append(
                StrategySignal(
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    asset_class=asset_class,
                    side="SELL",
                    ts_open_utc=ts,
                    qty=state.position_qty,
                    meta=meta,
                )
            )
            self._reset_cycle(state)
        return signals

    def _maybe_reset_on_recovery(self, state: _CycleState, price: float) -> None:
        if not state.cycle_active:
            return
        ref = state.cycle_high_ref
        if ref is None:
            return
        if price >= ref and state.tp_emitted:
            self._reset_cycle(state)

    def _reset_cycle(self, state: _CycleState) -> None:
        prev_cycle = state.cycle_id
        state.reset(len(self.grid))
        state.cycle_id = prev_cycle

    def _build_buy_meta(self, state: _CycleState, level: Dict[str, Any], dd: float) -> Dict[str, Any]:
        tp_rule = self._resolve_tp_rule(state.max_dd)
        return {
            "dd_pct": dd,
            "drawdown_pct": dd,
            "grid_level": float(level.get("dd", 0.0)),
            "palier_used": float(level.get("weight", 0.0)),
            "grid_config": self.grid,
            "cycle_id": state.cycle_id,
            "max_dd_reached": state.max_dd,
            "tp_target": tp_rule.get("tp_pct") if tp_rule else None,
            "tp_mode": self.tp_sl_config.get("mode"),
            "tp_pct": tp_rule.get("tp_pct") if tp_rule else None,
            "be_pct": tp_rule.get("be_pct") if tp_rule else None,
            "tp_rule": tp_rule,
            "position_qty": state.position_qty,
            "position_cost": state.position_cost,
        }

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

    def _state_from_dict(self, data: Dict[str, Any]) -> _CycleState:
        state = _CycleState()
        state.cycle_id = int(data.get("cycle_id", 0))
        state.cycle_active = bool(data.get("cycle_active", False))
        state.consumed_levels = list(data.get("consumed_levels", []))
        state.max_dd = float(data.get("max_dd", 0.0))
        state.cycle_low = (
            float(data.get("cycle_low")) if data.get("cycle_low") is not None else None
        )
        state.cycle_high_ref = (
            float(data.get("cycle_high_ref"))
            if data.get("cycle_high_ref") is not None
            else None
        )
        prev_dd = data.get("prev_dd")
        state.prev_dd = float(prev_dd) if prev_dd is not None else None
        last_ts = data.get("last_processed_ts")
        state.last_processed_ts = (
            pd.Timestamp(last_ts).tz_convert("UTC") if last_ts is not None else None
        )
        state.tp_emitted = bool(data.get("tp_emitted", False))
        state.position_qty = float(data.get("position_qty", 0.0))
        state.position_cost = float(data.get("position_cost", 0.0))
        state.current_price = float(data.get("current_price", 0.0))
        self._ensure_cycle_initialized(state)
        return state

    def _update_context_dict(self, target: Dict[str, Any], state: _CycleState) -> None:
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
                "position_qty": state.position_qty,
                "position_cost": state.position_cost,
                "current_price": state.current_price,
            }
        )


__all__ = ["DcaEquityStrategy"]
