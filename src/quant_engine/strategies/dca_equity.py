"""Weighted drawdown-based DCA strategy for equities."""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
import warnings

import pandas as pd

from .base import Strategy, StrategySignal

LOGGER = logging.getLogger(__name__)

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
    be_armed: bool = False
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
        self.be_armed = False
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
        self.execution_mode = str(self.params.get("execution_mode", "bar_close")).strip().lower()
        if self.execution_mode not in {"bar_close", "intracandle"}:
            raise ValueError("execution_mode must be 'bar_close' or 'intracandle'")
        self.dd_reference_mode, self.dd_reference_window = self._parse_drawdown_reference(
            self.params.get("drawdown_reference")
        )
        self.require_crossing = bool(self.params.get("require_crossing", True))
        self.log_drawdown_summary = bool(self.params.get("log_drawdown_summary", False))

    @staticmethod
    def compute_drawdown(close: pd.Series) -> pd.Series:
        """Compute drawdown (in percentage) vs rolling high (deprecated)."""

        warnings.warn(
            "DcaEquityStrategy.compute_drawdown is deprecated and unused; "
            "compute drawdown from _compute_reference_high instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        rolling_max = close.cummax()
        dd = (close / rolling_max - 1.0) * 100.0
        return dd.fillna(0.0)

    @staticmethod
    def _parse_drawdown_reference(value: Any) -> Tuple[str, Optional[str]]:
        """
        Configure how the reference high is computed for drawdown.

        Supported forms:
        - null / missing: defaults to rolling 90D (previous behavior)
        - "ATH": all-time-high
        - "1M", "3M", "6M", "1Y": rolling time windows (approx 30/90/180/365 days)
        - {"mode": "rolling", "window": "90D"} or {"mode": "ath"}
        """

        if value is None or value == "":
            return "rolling", "90D"

        if isinstance(value, str):
            token = value.strip().upper()
            if token in {"ATH", "ALL_TIME_HIGH", "ALL-TIME-HIGH", "HISTORICAL_HIGH"}:
                return "ath", None
            mapping = {"1M": "30D", "3M": "90D", "6M": "180D", "1Y": "365D", "12M": "365D"}
            if token in mapping:
                return "rolling", mapping[token]
            return "rolling", token

        if isinstance(value, dict):
            mode = str(value.get("mode", "rolling")).strip().lower()
            if mode in {"ath", "all_time_high", "all-time-high"}:
                return "ath", None
            if mode in {"rolling", "window", "rolling_window"}:
                window = value.get("window") or value.get("lookback") or "90D"
                return "rolling", str(window)

        raise ValueError(
            "Invalid drawdown_reference; expected 'ATH'/'3M'/'6M'/'1Y' or {mode, window}"
        )

    def _compute_reference_high(self, close: pd.Series) -> pd.Series:
        """Compute drawdown reference highs from the configured mode/window."""

        mode = (self.dd_reference_mode or "rolling").lower()
        if mode == "ath":
            ref = close.expanding(min_periods=1).max()
        else:
            window = self.dd_reference_window or "90D"
            if isinstance(window, str) and self._is_time_based_window(window):
                if not isinstance(close.index, (pd.DatetimeIndex, pd.TimedeltaIndex, pd.PeriodIndex)):
                    LOGGER.warning(
                        "Rolling drawdown window %s requires a DatetimeIndex; "
                        "falling back to expanding max.",
                        window,
                    )
                    ref = close.expanding(min_periods=1).max()
                else:
                    ref = close.rolling(window, min_periods=1).max()
            else:
                ref = close.rolling(window, min_periods=1).max()
        return ref.ffill().fillna(close.iloc[0])

    @staticmethod
    def compute_reference_high(close: pd.Series) -> pd.Series:
        """
        Deprecated: use drawdown_reference + _compute_reference_high for new behavior.

        Rolling high sur les 3 derniers mois (~90 jours calendaires).
        - Si on démarre en début d’historique, on prend le max des bougies disponibles (min_periods=1).
        - Inclut la bougie courante (mise à jour dès qu’un nouveau plus haut apparaît).
        """

        warnings.warn(
            "DcaEquityStrategy.compute_reference_high is deprecated and unused; "
            "configure drawdown_reference to control the reference high instead.",
            DeprecationWarning,
            stacklevel=2,
        )

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
        only_last_ts = None if state.last_processed_ts is None else last_ts
        signals = self._process(df, context, state, only_last_ts=only_last_ts)
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
        ref_high = self._compute_reference_high(close)
        dd_series = ((close / ref_high) - 1.0) * 100.0
        dd_series = dd_series.fillna(0.0)
        symbol = context.get("symbol", context.get("symbol_id", ""))
        asset_class = context.get("asset_class", self.asset_class)
        screening = context.get("screening") or {}
        max_trades = screening.get("max_trades")
        max_seconds = screening.get("max_seconds")
        pruning_cfg = screening.get("pruning") if isinstance(screening, dict) else None
        pruning_enabled = (
            only_last_ts is None
            and isinstance(pruning_cfg, dict)
            and pruning_cfg.get("enabled", True) is not False
        )
        max_dd_pct = pruning_cfg.get("max_drawdown_pct") if pruning_enabled else None
        min_signals_cfg = pruning_cfg.get("min_signals_after_bars") if pruning_enabled else None
        bars_threshold = None
        min_signals = None
        if isinstance(min_signals_cfg, dict):
            try:
                bars_threshold = int(min_signals_cfg.get("bars"))
            except Exception:
                bars_threshold = None
            try:
                min_signals = int(min_signals_cfg.get("min_signals"))
            except Exception:
                min_signals = None
        trade_count = 0
        start_ts = time.monotonic()
        bars_seen = 0
        signals_seen = 0
        allow_mask = df["_filter_ok"].astype(bool) if "_filter_ok" in df.columns else pd.Series(True, index=df.index)
        log_level = logging.INFO if self.log_drawdown_summary else logging.DEBUG
        if LOGGER.isEnabledFor(log_level):
            try:
                dd_allowed = dd_series[allow_mask]
                min_dd_all = float(dd_series.min()) if not dd_series.empty else 0.0
                min_dd_allowed = float(dd_allowed.min()) if not dd_allowed.empty else 0.0
                crossings: List[str] = []
                for level in self.grid:
                    threshold = float(level.get("dd", 0.0))
                    crossed = (dd_series <= threshold) & (dd_series.shift(1) > threshold) & allow_mask
                    crossings.append(f"{threshold:.2f}={int(crossed.fillna(False).sum())}")
                LOGGER.log(
                    log_level,
                    "Drawdown summary for %s: min_dd=%.2f%% min_dd_allowed=%.2f%% crossings(%s)",
                    symbol,
                    min_dd_all,
                    min_dd_allowed,
                    ", ".join(crossings),
                )
            except Exception:
                LOGGER.log(log_level, "Drawdown summary for %s: unavailable", symbol)
        results: List[StrategySignal] = []
        last_processed = state.last_processed_ts
        for ts, price, dd, ref_h, high, low in zip(
            dd_series.index, close, dd_series, ref_high, df["high"].astype(float), df["low"].astype(float)
        ):
            if max_seconds is not None and max_seconds > 0 and (time.monotonic() - start_ts) >= max_seconds:
                break
            if last_processed is not None and ts <= last_processed:
                continue
            allow_entries = self._allow_entries(df, ts)
            state.current_price = float(price)
            self._ensure_cycle_initialized(state)
            # Fige le ref_high dès qu'un cycle démarre pour garder une base stable (ATH ou rolling window).
            ref_high_value = state.cycle_high_ref if state.cycle_active else float(ref_h)
            if allow_entries and not state.cycle_active:
                self._maybe_start_cycle(state, float(dd), ref_high_value, float(price))
            if state.cycle_active:
                self._update_cycle_stats(state, float(dd), float(price))
                buys = self._check_buy_levels(state, float(dd), ts, symbol, asset_class) if allow_entries else []
                sells = self._check_take_profit(
                    state,
                    close=float(price),
                    high=float(high),
                    low=float(low),
                    dd=float(dd),
                    ref_high=ref_high_value,
                    ts=ts,
                    symbol=symbol,
                    asset_class=asset_class,
                )
                signals_seen += len(buys) + len(sells)
                for sig in (*buys, *sells):
                    if only_last_ts is None or sig.ts_open_utc == only_last_ts:
                        results.append(sig)
                for sig in sells:
                    action = (sig.meta or {}).get("action") if hasattr(sig, "meta") else None
                    if action in {"take_profit", "break_even", "stop_loss"}:
                        trade_count += 1
                        if max_trades is not None and max_trades > 0 and trade_count >= max_trades:
                            return results
            else:
                state.max_dd = 0.0
                state.cycle_low = None
                state.prev_dd = float(dd)
            self._maybe_reset_on_recovery(state, float(price))
            state.prev_dd = float(dd)
            if not state.cycle_active:
                state.cycle_high_ref = float(ref_h)
            state.last_processed_ts = ts
            bars_seen += 1
            if pruning_enabled:
                if max_dd_pct is not None:
                    try:
                        max_dd_value = float(max_dd_pct)
                    except Exception:
                        max_dd_value = None
                    if max_dd_value is not None and float(dd) <= -abs(max_dd_value):
                        LOGGER.info(
                            "Pruning %s: drawdown %.2f%% <= -%.2f%% after %d bars",
                            symbol,
                            float(dd),
                            abs(max_dd_value),
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
                    LOGGER.info(
                        "Pruning %s: signals=%d after %d bars (min=%d)",
                        symbol,
                        signals_seen,
                        bars_seen,
                        min_signals,
                    )
                    break
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
            if dd <= threshold and (not self.require_crossing or prev_dd > threshold):
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
        *,
        close: float,
        high: float,
        low: float,
        dd: float,
        ref_high: float,
        ts: pd.Timestamp,
        symbol: str,
        asset_class: str,
    ) -> List[StrategySignal]:
        signals: List[StrategySignal] = []
        cfg = self.tp_sl_config or {}
        tp_rule = self._resolve_tp_rule(state.max_dd)
        tp_pct = tp_rule.get("tp_pct") if tp_rule else None
        be_pct = tp_rule.get("be_pct") if tp_rule else None
        sl_dd = cfg.get("sl_dd")
        if state.position_qty <= 0:
            return signals

        avg_entry = state.position_cost / state.position_qty if state.position_qty > 0 else close
        if avg_entry == 0:
            return signals
        if self.execution_mode == "intracandle":
            tp_price = avg_entry * (1.0 + float(tp_pct) / 100.0) if tp_pct is not None else None
            be_arm_price = avg_entry * (1.0 + float(be_pct) / 100.0) if be_pct is not None else None
            sl_price = ref_high * (1.0 + float(sl_dd) / 100.0) if sl_dd is not None else None
            should_tp = tp_price is not None and high >= tp_price
            if be_arm_price is not None and high >= be_arm_price:
                state.be_armed = True
            should_be_exit = state.be_armed and low <= avg_entry
            should_sl = sl_price is not None and low <= sl_price
        else:
            pnl_pct = (close / avg_entry - 1.0) * 100.0
            should_tp = tp_pct is not None and pnl_pct >= float(tp_pct)
            if be_pct is not None and pnl_pct >= float(be_pct):
                state.be_armed = True
            should_be_exit = state.be_armed and pnl_pct <= 0.0
            should_sl = sl_dd is not None and dd <= float(sl_dd)

        # Debug trace for TP/BE decisions (muted; re-enable for troubleshooting)
        # print(
        #     f"[TP_CHECK] cycle={state.cycle_id} sym={symbol} ts={ts} "
        #     f"avg_entry={avg_entry:.4f} price={price:.4f} pnl_pct={pnl_pct:.2f} "
        #     f"tp_pct={tp_pct} be_pct={be_pct} should_tp={should_tp} should_be={should_be} "
        #     f"pos_qty={state.position_qty:.4f}"
        # )

        action = None
        if should_sl and not state.tp_emitted:
            action = "stop_loss"
        elif should_tp and not state.tp_emitted:
            action = "take_profit"
        elif should_be_exit and not state.tp_emitted:
            action = "break_even"

        if action:
            if self.execution_mode == "intracandle":
                if action == "stop_loss":
                    exit_price = sl_price
                elif action == "take_profit":
                    exit_price = tp_price
                else:
                    exit_price = avg_entry
            else:
                exit_price = close
            if exit_price is None:
                exit_price = close
            pnl_pct = (exit_price / avg_entry - 1.0) * 100.0
            state.tp_emitted = True
            meta = {
                "action": action,
                "tp_mode": self.tp_sl_config.get("mode"),
                "tp_pct": tp_pct,
                "be_pct": be_pct,
                "sl_dd": sl_dd,
                "max_dd_reached": state.max_dd,
                "drawdown_pct": state.max_dd,
                "cycle_id": state.cycle_id,
                "grid_config": self.grid,
                "grid_level": None,
                "avg_entry_price": avg_entry,
                "pnl_pct_at_exit": pnl_pct,
                "break_even_armed": state.be_armed,
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
            df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
            if df["ts"].isna().any():
                raise ValueError("OHLC dataframe has invalid timestamps in 'ts' column")
            df = df.set_index("ts")
        elif isinstance(df.index, pd.DatetimeIndex):
            df.index = df.index.tz_convert("UTC") if df.index.tz is not None else df.index.tz_localize("UTC")
        elif isinstance(df.index, pd.PeriodIndex):
            df.index = df.index.to_timestamp().tz_localize("UTC")
        else:
            if df.index.dtype == object:
                parsed = pd.to_datetime(df.index, utc=True, errors="coerce")
                if not parsed.isna().any():
                    df.index = parsed
        df = df.sort_index()
        required = {"open", "high", "low", "close"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"OHLC dataframe missing columns: {missing}")
        return df

    @staticmethod
    def _is_time_based_window(window: str) -> bool:
        try:
            pd.tseries.frequencies.to_offset(window)
        except (ValueError, TypeError):
            return False
        return True

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
        state.be_armed = bool(data.get("be_armed", False))
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
                "be_armed": state.be_armed,
                "position_qty": state.position_qty,
                "position_cost": state.position_cost,
                "current_price": state.current_price,
            }
        )


__all__ = ["DcaEquityStrategy"]
