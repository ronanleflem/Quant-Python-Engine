"""Live runner orchestrating feed, strategy evaluation and emission."""
from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
from sqlalchemy import create_engine

from ..api.schemas import LiveSpec
from ..filters import filters_registry
from .emitter import build_trade, emit_to_java, ensure_trades_live_table, write_trade
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
        columns = FeedColumns(
            ts=spec.data.ts_col,
            open=spec.data.open_col,
            high=spec.data.high_col,
            low=spec.data.low_col,
            close=spec.data.close_col,
            volume=spec.data.volume_col,
            symbol=spec.data.symbol_col,
        )
        for symbol in spec.data.symbols:
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
        state.emitted_hashes.add(trade.uniq_hash)
        self._emitted += 1
        LOGGER.info(
            "Emitted trade %s %s at %s (entry %.5f)",
            rule.side,
            state.symbol,
            ts.isoformat(),
            entry_price,
        )


__all__ = ["LiveRunner"]
