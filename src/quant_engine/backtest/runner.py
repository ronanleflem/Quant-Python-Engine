"""Backtest runner aligned with strategy JSON specs."""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import pandas as pd
import logging

from . import engine
from ..core import dataset
from ..core.features import atr
from ..core.spec import DataSpec, parse_data_spec, uses_strategy_sources
from ..filters.utils import apply_filter_stack, FilterValidationError
from ..performance.backtest_builder import build_backtest_payload
from ..signals.ema_cross import EmaCross
from ..strategies import runner as strategies_runner
from ..strategies.runner import persist_payload_to_db

LOGGER = logging.getLogger(__name__)
_ROWS_CACHE: Dict[str, Tuple[List[Dict[str, Any]], Optional[str]]] = {}


def load_backtest_spec(path: Path | str) -> Dict[str, Any]:
    """Load a backtest specification from disk."""

    path_obj = Path(path)
    return json.loads(path_obj.read_text())


def _single_symbol(symbols: List[str], fallback: Optional[str] = None) -> str:
    unique = [str(sym) for sym in symbols if sym]
    if not unique and fallback:
        unique = [fallback]
    if not unique:
        raise ValueError("Backtest runner requires a single symbol")
    if len(set(unique)) > 1:
        raise ValueError("Backtest runner expects a single symbol dataset")
    return unique[0]


def _rows_from_dataframe(df: pd.DataFrame, symbol: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if df is None or df.empty:
        return rows
    for rec in df.to_dict("records"):
        ts_value = rec.pop("ts", None)
        if ts_value is None:
            raise RuntimeError("OHLC source must provide a 'ts' column")
        ts = pd.Timestamp(ts_value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        rec["timestamp"] = ts.isoformat()
        if not rec.get("symbol"):
            rec["symbol"] = symbol
        if "session" not in rec and "session_id" not in rec:
            rec["session"] = dataset._assign_session(ts.to_pydatetime())
        rows.append(rec)
    rows.sort(key=lambda r: r["timestamp"])
    return rows


def _fetch_ohlc_with_source(
    symbol: str,
    asset_class: str,
    data_spec_raw: Mapping[str, Any],
) -> Tuple[pd.DataFrame, str]:
    df = strategies_runner._fetch_from_delta(symbol, asset_class, data_spec_raw)
    if df is not None and not df.empty:
        return df, "delta"
    df = strategies_runner._fetch_from_mysql(symbol, data_spec_raw)
    if df is not None and not df.empty:
        return df, "mysql"
    df = strategies_runner._fetch_from_java(symbol, asset_class, data_spec_raw)
    if df is not None and not df.empty:
        return df, "java"
    raise RuntimeError(f"Unable to load OHLC for {symbol}")


def _load_rows(
    data_spec_raw: Mapping[str, Any],
    data_spec: DataSpec,
    asset_class: str,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    cache_key = json.dumps(
        {"data": data_spec_raw, "asset_class": asset_class},
        sort_keys=True,
        default=str,
    )
    cached = _ROWS_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if uses_strategy_sources(data_spec_raw):
        symbol = _single_symbol(data_spec.symbols, data_spec_raw.get("symbol"))
        df, source = _fetch_ohlc_with_source(symbol, asset_class, data_spec_raw)
        rows = _rows_from_dataframe(df, symbol)
        _ROWS_CACHE[cache_key] = (rows, source)
        return rows, source
    rows = dataset.load_dataset(data_spec)
    _ROWS_CACHE[cache_key] = (rows, None)
    return rows, None


def _build_signal(spec: Mapping[str, Any], rows: List[Dict[str, Any]]) -> List[int]:
    signal_spec = spec.get("signal", {}) or {}
    signal_type = str(signal_spec.get("type") or "").strip().lower()
    params = signal_spec.get("params", {}) or {}
    if signal_type == "ema_cross":
        fast = params.get("fast", params.get("ema_fast"))
        slow = params.get("slow", params.get("ema_slow"))
        if fast is None or slow is None:
            raise ValueError("ema_cross requires fast/slow parameters")
        return EmaCross(int(fast), int(slow)).generate(rows)
    raise ValueError(f"Unsupported signal type '{signal_type}'")


def _detect_symbol(rows: List[Dict[str, Any]]) -> str:
    symbols = sorted({str(r.get("symbol")) for r in rows if r.get("symbol")})
    if len(symbols) > 1:
        raise ValueError("Backtest runner expects a single symbol dataset")
    return symbols[0] if symbols else "UNKNOWN"


def _rows_to_frame(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    ts_col = "timestamp" if "timestamp" in df.columns else "ts"
    if ts_col not in df.columns:
        raise ValueError("Rows missing timestamp column for filter evaluation")
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
    df = df.sort_values(ts_col)
    df = df.set_index(ts_col, drop=True)
    return df


def run_backtest_from_spec(spec: Mapping[str, Any]) -> Dict[str, Any]:
    """Execute a classic backtest based on a JSON specification."""

    data_raw = spec.get("data", {}) or {}
    data_spec = parse_data_spec(data_raw, allow_strategy_sources=True)
    strategy_cfg = spec.get("strategy", {}) or {}
    asset_class = strategy_cfg.get("asset_class") or "EQUITY"
    rows, data_source = _load_rows(data_raw, data_spec, asset_class)
    if not rows:
        raise ValueError("No data rows loaded for backtest")
    optimization_cfg = spec.get("optimization", {}) or {}
    screening_cfg = optimization_cfg.get("screening") or spec.get("screening") or {}
    cache_cfg = optimization_cfg.get("cache_features") or {}
    cache_enabled = bool(cache_cfg) and cache_cfg.get("enabled", True) is not False
    max_trades = None
    max_seconds = None
    pruning_cfg = None
    if isinstance(screening_cfg, Mapping) and screening_cfg.get("enabled"):
        window_start = screening_cfg.get("window_start")
        window_end = screening_cfg.get("window_end")
        if window_start or window_end:
            df_window = _rows_to_frame(rows)
            start_ts = pd.to_datetime(window_start, utc=True) if window_start else df_window.index.min()
            end_ts = pd.to_datetime(window_end, utc=True) if window_end else df_window.index.max()
            df_window = df_window.loc[(df_window.index >= start_ts) & (df_window.index <= end_ts)]
            rows = _rows_from_dataframe(df_window.reset_index().rename(columns={"index": "ts"}), _detect_symbol(rows))
            LOGGER.info("Screening window applied: %s -> %s", start_ts, end_ts)
        max_bars = screening_cfg.get("max_bars")
        if max_bars is not None:
            try:
                max_bars_int = int(max_bars)
            except Exception:
                max_bars_int = 0
            if max_bars_int > 0 and len(rows) > max_bars_int:
                rows = rows[-max_bars_int:]
                LOGGER.info("Screening enabled: keeping last %d bars", max_bars_int)
        max_trades = screening_cfg.get("max_trades")
        max_seconds = screening_cfg.get("max_seconds")
        candidate_pruning = screening_cfg.get("pruning")
        if isinstance(candidate_pruning, Mapping):
            pruning_cfg = candidate_pruning

    symbol = _detect_symbol(rows)
    signal = _build_signal(spec, rows)

    filters_spec = spec.get("filters") or (spec.get("strategy", {}) or {}).get("filters") or []
    if filters_spec:
        df_filters = _rows_to_frame(rows)
        df_filters = df_filters.copy()
        df_filters["entry_signal"] = [bool(val) for val in signal]
        df_filters["signal"] = df_filters["entry_signal"]
        if cache_enabled:
            cache_key = json.dumps(
                {
                    "symbol": symbol,
                    "asset_class": asset_class,
                    "data": data_raw,
                    "screening": screening_cfg,
                },
                sort_keys=True,
                default=str,
            )
            df_filters.attrs["qe_cache_key"] = cache_key
            if "max_items" in cache_cfg:
                df_filters.attrs["qe_cache_max_items"] = cache_cfg.get("max_items")
        try:
            mask = apply_filter_stack(df_filters, filters_spec, symbol=symbol, logger=LOGGER, strict=True)
        except FilterValidationError as exc:
            LOGGER.error("Backtest filters failed: %s", exc)
            raise
        signal_params = (spec.get("signal", {}) or {}).get("params", {}) or {}
        require_crossing = signal_params.get("require_crossing")
        if require_crossing is None:
            require_crossing = (spec.get("strategy", {}) or {}).get("params", {}) or {}
            require_crossing = require_crossing.get("require_crossing", True)
        if require_crossing:
            signal = [int(bool(s) and bool(m)) for s, m in zip(signal, mask)]
        else:
            gated: List[int] = []
            for idx, (s, m) in enumerate(zip(signal, mask)):
                if idx > 0 and gated[idx - 1] == 1:
                    gated.append(int(bool(s)))
                else:
                    gated.append(int(bool(s) and bool(m)))
            signal = gated

    tpsl = spec.get("tpsl", {}) or {}
    atr_window = int(tpsl.get("atr_window", tpsl.get("atr_period", 14)))
    atr_values = atr.compute(rows, {"period": atr_window})
    atr_mult = float(tpsl.get("atr_k", 1.0))
    r_mult = float(tpsl.get("r_mult", 2.0))
    slippage_bps = float(tpsl.get("slippage_bps", 0.0))
    fee_bps = float(tpsl.get("fee_bps", 0.0))

    trades, equity, summary = engine.run(
        rows,
        signal,
        atr_values,
        atr_mult,
        r_mult,
        slippage_bps=slippage_bps,
        fee_bps=fee_bps,
        max_trades=max_trades,
        max_seconds=max_seconds,
        pruning=pruning_cfg,
    )

    strategy_id = strategy_cfg.get("strategy_id", "backtest")
    run_id = spec.get("run_id") or strategy_cfg.get("run_id") or uuid.uuid4().hex
    timeframe = data_spec.timeframe

    start_ts = rows[0].get("timestamp") if rows else None
    end_ts = rows[-1].get("timestamp") if rows else None

    payload = build_backtest_payload(
        strategy_id=strategy_id,
        run_id=run_id,
        asset_class=str(asset_class).upper(),
        symbol=symbol,
        timeframe=timeframe,
        trades=trades,
        equity=equity,
        start_ts=start_ts,
        end_ts=end_ts,
        config=spec.get("performance", {}) or {},
    )

    persistence_cfg = spec.get("persistence", {})
    if not isinstance(persistence_cfg, Mapping) or persistence_cfg.get("enabled", True):
        LOGGER.info("Backtest persistence enabled; DB_DSN=%s", "set" if os.getenv("DB_DSN") else "missing")
        if data_source == "java":
            data_override = {**data_raw, "broker": "IBKR"}
            spec_for_persistence = {**spec, "data": data_override}
        else:
            spec_for_persistence = spec
        persist_payload_to_db(payload, spec_for_persistence, strategy_type="backtest")
    else:
        LOGGER.info("Backtest persistence disabled via spec.persistence.enabled=false")

    output = spec.get("output")
    if output:
        path = Path(output.get("path", "backtest_result.json"))
        fmt = output.get("format", "json").lower()
        path.parent.mkdir(parents=True, exist_ok=True)
        if fmt == "json":
            path.write_text(json.dumps(payload, indent=2))
        else:
            raise ValueError(f"Unsupported output format: {fmt}")

    return {
        "strategy_id": strategy_id,
        "run_id": run_id,
        "symbol": symbol,
        "summary": summary,
        "payload": payload,
    }


__all__ = ["load_backtest_spec", "run_backtest_from_spec"]
