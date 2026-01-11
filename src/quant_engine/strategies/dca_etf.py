"""Long-term conditional DCA strategy for ETF universes."""
from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional

import pandas as pd

from .base import Strategy, StrategySignal


@dataclass
class _EtfState:
    """Internal state for ETF accumulation cycles."""

    cycle_id: int = 0
    cycle_active: bool = False
    consumed_levels: List[bool] = field(default_factory=list)
    prev_dd: Optional[float] = None
    last_processed_ts: Optional[pd.Timestamp] = None
    activation_history: List[pd.Timestamp] = field(default_factory=list)
    cycle_high_ref: Optional[float] = None

    def reset(self, level_count: int) -> None:
        self.cycle_active = False
        self.consumed_levels = [False] * level_count
        self.prev_dd = None
        self.cycle_high_ref = None


class DcaEtfStrategy(Strategy):
    """Low-turnover DCA approach focused on ETF drawdowns."""

    def __init__(self, strategy_id: str, params: Dict[str, Any]) -> None:
        self.strategy_id = strategy_id
        self.params = params or {}
        grid = list(self.params.get("grid", []))
        if not grid:
            raise ValueError("DcaEtfStrategy requires a non-empty grid configuration")
        self.grid: List[Dict[str, Any]] = sorted(grid, key=lambda item: float(item["dd"]))
        self.asset_class = self.params.get("asset_class", "ETF").upper()
        self.activation_limit: Dict[str, Any] = self.params.get("activation_limit", {})
        self.reset_on_new_high: bool = bool(self.params.get("reset_on_new_high", True))

    @staticmethod
    def compute_drawdown(close: pd.Series) -> pd.Series:
        rolling_max = close.cummax()
        dd = (close / rolling_max - 1.0) * 100.0
        return dd.fillna(0.0)

    def backtest(self, ohlc: pd.DataFrame, context: Dict[str, Any]) -> List[StrategySignal]:
        df = self._normalize_ohlc(ohlc)
        state = _EtfState()
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
        state: _EtfState,
        only_last_ts: Optional[pd.Timestamp],
    ) -> List[StrategySignal]:
        if df.empty:
            return []
        close = df["close"].astype(float)
        dd_series = self.compute_drawdown(close)
        rolling_max = close.cummax()
        symbol = context.get("symbol", context.get("symbol_id", ""))
        asset_class = context.get("asset_class", self.asset_class)
        screening = context.get("screening") or {}
        max_seconds = screening.get("max_seconds")
        start_ts = time.monotonic()
        results: List[StrategySignal] = []
        last_processed = state.last_processed_ts
        for ts, price, dd in zip(dd_series.index, close, dd_series):
            if max_seconds is not None and max_seconds > 0 and (time.monotonic() - start_ts) >= max_seconds:
                break
            if last_processed is not None and ts <= last_processed:
                continue
            allow_entries = self._allow_entries(df, ts)
            self._ensure_state_initialized(state)
            if allow_entries and not state.cycle_active:
                self._maybe_start_cycle(state, float(dd), float(rolling_max.loc[ts]))
            if state.cycle_active:
                buys = self._check_buy_levels(state, float(dd), ts, symbol, asset_class) if allow_entries else []
                for sig in buys:
                    if only_last_ts is None or sig.ts_open_utc == only_last_ts:
                        results.append(sig)
            if self.reset_on_new_high and float(price) >= float(rolling_max.loc[ts]):
                self._reset_cycle(state)
            state.prev_dd = float(dd)
            state.cycle_high_ref = float(rolling_max.loc[ts])
            state.last_processed_ts = ts
        return results

    @staticmethod
    def _allow_entries(df: pd.DataFrame, ts: pd.Timestamp) -> bool:
        if "_filter_ok" not in df.columns:
            return True
        try:
            value = df.at[ts, "_filter_ok"]
            if isinstance(value, pd.Series):
                return bool(value.fillna(False).any())
            return bool(value)
        except Exception:
            try:
                value = df.loc[ts, "_filter_ok"]
                if isinstance(value, pd.Series):
                    return bool(value.fillna(False).any())
                return bool(value)
            except Exception:
                return False

    def _ensure_state_initialized(self, state: _EtfState) -> None:
        if not state.consumed_levels:
            state.reset(len(self.grid))

    def _maybe_start_cycle(self, state: _EtfState, dd: float, ref_high: float) -> None:
        if state.cycle_active:
            return
        eligible = [idx for idx, level in enumerate(self.grid) if dd <= float(level["dd"])]
        if not eligible:
            return
        state.cycle_active = True
        state.cycle_id += 1
        state.consumed_levels = [False] * len(self.grid)
        state.cycle_high_ref = ref_high
        state.prev_dd = state.prev_dd if state.prev_dd is not None else 0.0

    def _check_buy_levels(
        self,
        state: _EtfState,
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
                if not self._allow_activation(state, ts):
                    continue
                state.consumed_levels[idx] = True
                state.activation_history.append(ts)
                meta = {
                    "dd_pct": dd,
                    "drawdown_pct": dd,
                    "grid_level": threshold,
                    "palier_used": float(level.get("weight", 0.0)),
                    "grid_config": self.grid,
                    "cycle_id": state.cycle_id,
                    "activation_count": len(state.activation_history),
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

    def _allow_activation(self, state: _EtfState, ts: pd.Timestamp) -> bool:
        if not self.activation_limit:
            return True
        max_signals = self.activation_limit.get("max_signals")
        period_days = self.activation_limit.get("period_days")
        if not max_signals or not period_days:
            return True
        cutoff = ts - pd.Timedelta(days=float(period_days))
        state.activation_history = [t for t in state.activation_history if t > cutoff]
        return len(state.activation_history) < int(max_signals)

    def _reset_cycle(self, state: _EtfState) -> None:
        prev_cycle = state.cycle_id
        state.reset(len(self.grid))
        state.cycle_id = prev_cycle

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

    def _state_from_dict(self, data: Dict[str, Any]) -> _EtfState:
        state = _EtfState()
        state.cycle_id = int(data.get("cycle_id", 0))
        state.cycle_active = bool(data.get("cycle_active", False))
        state.consumed_levels = list(data.get("consumed_levels", []))
        state.prev_dd = (
            float(data.get("prev_dd")) if data.get("prev_dd") is not None else None
        )
        last_ts = data.get("last_processed_ts")
        state.last_processed_ts = (
            pd.Timestamp(last_ts).tz_convert("UTC") if last_ts is not None else None
        )
        history_raw = [ts for ts in data.get("activation_history", []) if ts]
        state.activation_history = [pd.Timestamp(ts).tz_convert("UTC") for ts in history_raw]
        ref = data.get("cycle_high_ref")
        state.cycle_high_ref = float(ref) if ref is not None else None
        self._ensure_state_initialized(state)
        return state

    def _update_context_dict(self, target: Dict[str, Any], state: _EtfState) -> None:
        target.update(
            {
                "cycle_id": state.cycle_id,
                "cycle_active": state.cycle_active,
                "consumed_levels": list(state.consumed_levels),
                "prev_dd": state.prev_dd,
                "last_processed_ts": state.last_processed_ts.isoformat()
                if state.last_processed_ts is not None
                else None,
                "activation_history": [ts.isoformat() for ts in state.activation_history],
                "cycle_high_ref": state.cycle_high_ref,
            }
        )


__all__ = ["DcaEtfStrategy"]
