"""Live runner orchestrating feed, strategy evaluation and emission."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import create_engine

from ..api.schemas import LiveSpec
from ..filters import filters_registry
from ..integrations import java_client
from ..strategies import StrategySignal, create_strategy
from .emitter import (
    TradePayload,
    build_trade,
    emit_to_java,
    ensure_trades_live_table,
    write_trade,
)
from ..notify.telegram_notify import send_telegram_message
from .feed import FeedColumns, MySQLPollFeed
from .state import LiveState

LOGGER = logging.getLogger(__name__)


@dataclass
class RuleResult:
    side: str
    details: Dict[str, float]


class LiveRunner:
    """Main loop driving the live trading pipeline."""

    def __init__(self, spec: LiveSpec) -> None:
        self.spec = spec
        self.poll_interval = max(int(spec.data.poll_interval_sec), 1)
        read_url = os.environ.get(spec.data.mysql_read_env)
        if not read_url:
            raise RuntimeError(
                f"Missing MySQL read URL in environment: {spec.data.mysql_read_env}"
            )
        self.read_engine = create_engine(read_url)
        self.write_engine = None
        self.write_table = None
        self.emit_java_enabled = False
        self.emit_java_path = "/live/signal"
        destinations = spec.destinations
        if destinations and destinations.write_db:
            write_env = destinations.write_db.mysql_write_env
            write_url = os.environ.get(write_env) if write_env else None
            if write_url:
                self.write_engine = create_engine(write_url)
                schema = destinations.write_db.schema
                table = destinations.write_db.table
                self.write_table = f"{schema}.{table}" if schema else table
                ensure_trades_live_table(self.write_engine, self.write_table)
            else:
                LOGGER.warning("MySQL write URL env %s not set", write_env)
        if destinations and destinations.emit_java and destinations.emit_java.enabled:
            self.emit_java_enabled = True
            self.emit_java_path = destinations.emit_java.path or "/live/signal"
            self.emit_java_env = destinations.emit_java.url_env
        else:
            self.emit_java_env = "QE_JAVA_LIVE_URL"
        self.strategy_id = spec.strategy.strategy_id
        self.timeframe = spec.data.timeframe
        self._feeds: Dict[str, MySQLPollFeed] = {}
        self._states: Dict[str, LiveState] = {}
        self._stop = threading.Event()
        self._emitted = 0
        self._strategy_impl = None
        self._strategy_impl_type: Optional[str] = None
        self._strategy_impl_params: Dict[str, Any] = {}
        impl_spec = getattr(spec.strategy, "impl", None)
        if impl_spec is not None:
            self._strategy_impl = create_strategy(
                impl_spec.type,
                strategy_id=self.strategy_id,
                params=impl_spec.params or {},
            )
            self._strategy_impl_type = impl_spec.type
            self._strategy_impl_params = impl_spec.params or {}
        columns = FeedColumns(
            ts=spec.data.ts_col,
            open=spec.data.open_col,
            high=spec.data.high_col,
            low=spec.data.low_col,
            close=spec.data.close_col,
            volume=spec.data.volume_col,
            symbol=spec.data.symbol_col,
        )
        resolved_symbols = self._resolve_symbols(spec.data.symbols, spec.data.scans)
        for symbol in resolved_symbols:
            feed = MySQLPollFeed(
                engine=self.read_engine,
                table=spec.data.table,
                symbol=symbol,
                timeframe=self.timeframe,
                columns=columns,
                warmup_bars=spec.data.warmup_bars,
                schema=spec.data.schema,
                timeframe_col=spec.data.timeframe_col,
            )
            self._feeds[symbol] = feed
            self._states[symbol] = LiveState(
                strategy_id=self.strategy_id, symbol=symbol, timeframe=self.timeframe
            )

    def stop(self) -> None:
        self._stop.set()

    def status(self) -> Dict[str, object]:
        last_ts = {
            sym: (
                state.last_ts_seen.isoformat() if state.last_ts_seen is not None else None
            )
            for sym, state in self._states.items()
        }
        return {
            "running": not self._stop.is_set(),
            "last_ts_seen": last_ts,
            "emitted_trades": self._emitted,
        }

    def run_forever(self) -> None:
        LOGGER.info("Starting LiveRunner for strategy %s", self.strategy_id)
        try:
            while not self._stop.is_set():
                self._run_cycle()
                time.sleep(self.poll_interval)
        except KeyboardInterrupt:  # pragma: no cover - manual stop
            LOGGER.info("LiveRunner interrupted by user")
        finally:
            LOGGER.info("LiveRunner stopped")

    def _run_cycle(self) -> None:
        if self._strategy_impl is not None:
            self._run_cycle_strategy_impl()
            return
        for symbol, feed in self._feeds.items():
            state = self._states[symbol]
            if not state.warm:
                history = feed.bootstrap()
                if history.empty:
                    LOGGER.debug("Warm-up empty for %s", symbol)
                    continue
                state.append_history(history)
                state.mark_processed(history["ts"].iloc[-1])
                state.warm = True
                LOGGER.info("Warm-up loaded %s bars for %s", len(history), symbol)
                continue
            last_ts = state.last_ts_seen
            latest = feed.poll_last_bar(last_ts)
            if latest.empty:
                continue
            bar_ts = latest["ts"].iloc[-1]
            if not state.should_process(bar_ts):
                continue
            state.append_history(latest)
            decision = self._evaluate_symbol(state)
            if decision is not None:
                self._emit_signal(state, decision, latest)
            state.mark_processed(bar_ts)

    def _run_cycle_strategy_impl(self) -> None:
        positions = self._safe_get_positions()
        grouped = self._group_positions_by_symbol(positions)
        for symbol, feed in self._feeds.items():
            state = self._states[symbol]
            if not state.warm:
                history = feed.bootstrap()
                if history.empty:
                    LOGGER.debug("Warm-up empty for %s", symbol)
                    continue
                state.append_history(history)
                state.mark_processed(history["ts"].iloc[-1])
                state.warm = True
                LOGGER.info("Warm-up loaded %s bars for %s", len(history), symbol)
                continue
            last_ts = state.last_ts_seen
            latest = feed.poll_last_bar(last_ts)
            if latest.empty:
                continue
            bar_ts = latest["ts"].iloc[-1]
            if not state.should_process(bar_ts):
                continue
            state.append_history(latest)
            history = state.history.copy()
            history["ts"] = pd.to_datetime(history["ts"], utc=True)
            history_df = history.set_index("ts")
            context = self._build_strategy_context(symbol, state, grouped)
            try:
                signals = self._strategy_impl.evaluate_live_bar(history_df, context)
            except Exception as exc:  # pragma: no cover - defensive log
                LOGGER.exception("Strategy evaluation failed for %s: %s", symbol, exc)
                signals = []
            if context.get("state") is not None:
                state.strategy_ctx = context["state"]
            if signals:
                for seq, signal in enumerate(signals):
                    self._emit_strategy_signal(state, signal, latest, seq)
            state.mark_processed(bar_ts)

    def _build_strategy_context(
        self,
        symbol: str,
        state: LiveState,
        grouped_positions: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        asset_class = self._strategy_impl_params.get("asset_class")
        if asset_class is None and self._strategy_impl is not None:
            asset_class = getattr(self._strategy_impl, "asset_class", None)
        if isinstance(asset_class, str):
            asset_class_value: Any = asset_class.upper()
        else:
            asset_class_value = asset_class
        context_state = state.strategy_ctx
        if context_state is None:
            context_state = {}
            state.strategy_ctx = context_state
        return {
            "symbol": symbol,
            "asset_class": asset_class_value or symbol,
            "positions": grouped_positions.get(symbol, []),
            "portfolio_positions": grouped_positions,
            "timeframe": self.timeframe,
            "macro": self._strategy_impl_params.get("macro_context"),
            "state": context_state,
            "tp_sl": self._strategy_impl_params.get("tp_sl"),
            "grid": self._strategy_impl_params.get("grid"),
        }

    def _resolve_symbols(
        self, static_symbols: List[str], scans: List[Dict[str, Any]]
    ) -> List[str]:
        symbols = list(static_symbols)
        for scan_spec in scans or []:
            scan_type = scan_spec.get("type")
            params = scan_spec.get("params", {})
            if not scan_type:
                continue
            try:
                results = java_client.get_market_scan(scan_type, params)
            except Exception as exc:  # pragma: no cover - network failure
                LOGGER.warning("Scan %s failed: %s", scan_type, exc)
                continue
            for item in results:
                symbol = item.get("symbol")
                if symbol and symbol not in symbols:
                    symbols.append(symbol)
        return symbols

    def _safe_get_positions(self) -> List[Dict[str, Any]]:
        try:
            return java_client.get_positions()
        except Exception as exc:  # pragma: no cover - network failure
            LOGGER.warning("Failed to fetch positions from Java backend: %s", exc)
            return []

    def _group_positions_by_symbol(
        self, positions: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for position in positions:
            symbol = position.get("symbol")
            if not symbol:
                continue
            grouped.setdefault(symbol, []).append(position)
        return grouped

    def _emit_strategy_signal(
        self,
        state: LiveState,
        signal: StrategySignal,
        latest: pd.DataFrame,
        sequence: int,
    ) -> None:
        if latest.empty:
            return
        ts = pd.Timestamp(signal.ts_open_utc)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        close_col = self.spec.data.close_col
        entry_price = float(latest[close_col].iloc[-1])
        trade_side = "LONG" if signal.side.upper() == "BUY" else "SHORT"
        payload = {
            "source": "strategy_impl",
            "strategy_type": self._strategy_impl_type,
            "signal": {
                "side": signal.side,
                "asset_class": signal.asset_class,
                "qty": signal.qty,
                "meta": signal.meta,
            },
            "details": signal.meta,
        }
        trade = build_trade(
            strategy_id=self.strategy_id,
            symbol=state.symbol,
            timeframe=state.timeframe,
            ts_open=ts.isoformat(),
            side=trade_side,
            entry_price=entry_price,
            sl=None,
            tp=None,
            expected_rr=None,
            payload=payload,
        )
        trade.uniq_hash = self._compute_strategy_trade_hash(trade, signal, sequence)
        if trade.uniq_hash in state.emitted_hashes:
            LOGGER.debug("Duplicate strategy signal skipped for %s", state.symbol)
            return
        if self.write_engine is not None and self.write_table is not None:
            try:
                write_trade(self.write_engine, self.write_table, trade)
            except Exception as exc:
                LOGGER.exception("Failed to persist trade: %s", exc)
        if self.emit_java_enabled:
            payload = trade.as_dict()
            payload.pop("uniq_hash", None)
            try:
                emit_to_java(payload, url_env=self.emit_java_env, path=self.emit_java_path)
            except Exception as exc:  # pragma: no cover - network failure
                LOGGER.warning("Java emission raised: %s", exc)
        self._notify_telegram(state, trade)
        state.emitted_hashes.add(trade.uniq_hash)
        self._emitted += 1
        LOGGER.info(
            "Emitted strategy signal %s %s at %s (entry %.5f)",
            signal.side,
            state.symbol,
            ts.isoformat(),
            entry_price,
        )

    def _compute_strategy_trade_hash(
        self, trade: TradePayload, signal: StrategySignal, sequence: int
    ) -> str:
        base_hash = trade.uniq_hash or ""
        meta_repr = json.dumps(signal.meta, sort_keys=True, default=str)
        payload = f"{base_hash}|{signal.side}|{sequence}|{meta_repr}"
        return hashlib.sha256(payload.encode()).hexdigest()

    def _evaluate_symbol(self, state: LiveState) -> Optional[RuleResult]:
        df = state.history.copy()
        if df.empty:
            return None
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df = df.set_index("ts")
        filters_ok = self._apply_filters(df)
        if not filters_ok:
            return None
        rule = self._apply_rules(df)
        if rule is None:
            return None
        if not self._apply_risk_gates(df):
            return None
        return rule

    def _apply_filters(self, df: pd.DataFrame) -> bool:
        specs = self.spec.strategy.filters
        if not specs:
            return True
        results: List[bool] = []
        for flt in specs:
            fn = filters_registry.get(flt.type)
            if fn is None:
                LOGGER.warning("Unknown filter %s", flt.type)
                continue
            series = fn(df, **flt.params)
            results.append(self._latest_bool(series))
        return all(results) if results else True

    def _apply_risk_gates(self, df: pd.DataFrame) -> bool:
        gates = self.spec.strategy.risk_gates
        if not gates:
            return True
        results: List[bool] = []
        for gate in gates:
            fn = filters_registry.get(gate.type)
            if fn is None:
                LOGGER.warning("Unknown risk gate %s", gate.type)
                continue
            series = fn(df, **gate.params)
            results.append(self._latest_bool(series))
        return all(results) if results else True

    def _apply_rules(self, df: pd.DataFrame) -> Optional[RuleResult]:
        rules = self.spec.strategy.rules
        for rule in rules:
            if rule.type == "cross_over":
                res = self._rule_cross_over(df, **rule.params)
                if res is not None:
                    return res
            else:
                LOGGER.warning("Unsupported rule type %s", rule.type)
        return None

    @staticmethod
    def _latest_bool(series: pd.Series) -> bool:
        if series.empty:
            return False
        value = series.iloc[-1]
        if isinstance(value, (bool, int)):
            return bool(value)
        if pd.isna(value):
            return False
        return bool(value)

    def _rule_cross_over(
        self,
        df: pd.DataFrame,
        fast_ema: int,
        slow_ema: int,
        side: str = "LONG",
    ) -> Optional[RuleResult]:
        if "close" not in df.columns:
            return None
        close = df["close"].astype(float)
        fast = close.ewm(span=int(fast_ema), adjust=False).mean()
        slow = close.ewm(span=int(slow_ema), adjust=False).mean()
        if len(close) < max(fast_ema, slow_ema) // 2:
            return None
        side = side.upper()
        if side == "LONG":
            cond = (fast > slow) & (fast.shift(1) <= slow.shift(1))
            if cond.iloc[-1]:
                return RuleResult(side="LONG", details={"fast": fast.iloc[-1], "slow": slow.iloc[-1]})
        elif side == "SHORT":
            cond = (fast < slow) & (fast.shift(1) >= slow.shift(1))
            if cond.iloc[-1]:
                return RuleResult(side="SHORT", details={"fast": fast.iloc[-1], "slow": slow.iloc[-1]})
        else:
            LOGGER.warning("Unsupported side %s for crossover", side)
        return None

    def _compute_tp_sl(
        self,
        df: pd.DataFrame,
        side: str,
    ) -> Dict[str, Optional[float]]:
        cfg = self.spec.strategy.tp_sl_mgmt
        if cfg is None or cfg.type != "fixed_rr":
            return {"sl": None, "tp": None, "rr": None}
        params = cfg.params
        rr = params.get("rr")
        if rr is None:
            return {"sl": None, "tp": None, "rr": None}
        sl_mode = params.get("sl_mode", "atr")
        close = float(df["close"].iloc[-1])
        if sl_mode == "atr":
            atr_window = int(params.get("atr_window", 14))
            atr_mult = float(params.get("atr_mult", 1.0))
            atr = self._atr(df, window=atr_window)
            if atr is None:
                return {"sl": None, "tp": None, "rr": rr}
            if side == "LONG":
                sl = close - atr_mult * atr
                tp = close + (close - sl) * float(rr)
            else:
                sl = close + atr_mult * atr
                tp = close - (sl - close) * float(rr)
            return {"sl": sl, "tp": tp, "rr": rr}
        return {"sl": None, "tp": None, "rr": rr}

    @staticmethod
    def _atr(df: pd.DataFrame, window: int) -> Optional[float]:
        required = {"high", "low", "close"}
        if not required.issubset(df.columns):
            return None
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        prev_close = close.shift(1)
        tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1)
        atr = tr.max(axis=1).ewm(alpha=1.0 / float(window), adjust=False).mean()
        value = atr.iloc[-1]
        return None if pd.isna(value) else float(value)

    def _emit_signal(
        self,
        state: LiveState,
        rule: RuleResult,
        latest: pd.DataFrame,
    ) -> None:
        ts_raw = latest["ts"].iloc[-1]
        ts = pd.Timestamp(ts_raw)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        entry_price = float(latest["close"].iloc[-1])
        history = state.history.copy()
        history["ts"] = pd.to_datetime(history["ts"], utc=True)
        history = history.set_index("ts")
        tpsl = self._compute_tp_sl(history, rule.side)
        trade = build_trade(
            strategy_id=self.strategy_id,
            symbol=state.symbol,
            timeframe=state.timeframe,
            ts_open=ts.isoformat(),
            side=rule.side,
            entry_price=entry_price,
            sl=tpsl["sl"],
            tp=tpsl["tp"],
            expected_rr=tpsl["rr"],
            payload={"rule": "cross_over", "details": rule.details},
        )
        if trade.uniq_hash in state.emitted_hashes:
            LOGGER.debug("Duplicate trade hash %s skipped", trade.uniq_hash)
            return
        if self.write_engine is not None and self.write_table is not None:
            try:
                write_trade(self.write_engine, self.write_table, trade)
            except Exception as exc:
                LOGGER.exception("Failed to persist trade: %s", exc)
        if self.emit_java_enabled:
            payload = trade.as_dict()
            payload.pop("uniq_hash", None)
            try:
                emit_to_java(payload, url_env=self.emit_java_env, path=self.emit_java_path)
            except Exception as exc:  # pragma: no cover - network failure
                LOGGER.warning("Java emission raised: %s", exc)
        self._notify_telegram(state, trade)
        state.emitted_hashes.add(trade.uniq_hash)
        self._emitted += 1
        LOGGER.info(
            "Emitted trade %s %s at %s (entry %.5f)",
            rule.side,
            state.symbol,
            ts.isoformat(),
            entry_price,
        )

    def _notify_telegram(self, state: LiveState, trade: TradePayload) -> None:
        """Send a Telegram alert for the emitted trade without blocking the main flow."""

        try:
            msg_lines = [
                "🚨 *Signal de trading détecté*",
                f"• Stratégie : `{trade.strategy_id}`",
                f"• Symbole : `{state.symbol}`",
                f"• Timeframe : `{state.timeframe}`",
                f"• Direction : *{trade.side}*",
                f"• Prix d'entrée : `{trade.entry_price}`",
                f"• Timestamp : `{trade.ts_open}`",
            ]
            if trade.expected_rr is not None:
                msg_lines.append(f"• RR attendu : `{trade.expected_rr}`")
            if trade.signal_payload and "details" in trade.signal_payload:
                details = trade.signal_payload["details"]
                if isinstance(details, dict) and details:
                    formatted = ", ".join(
                        f"{key}={value}" for key, value in details.items()
                    )
                    msg_lines.append(f"• Détails : `{formatted}`")
            send_telegram_message("\n".join(msg_lines))
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.error("Erreur lors de la préparation/envoi de l'alerte Telegram: %s", exc)


__all__ = ["LiveRunner"]
