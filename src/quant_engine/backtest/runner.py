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
from ..core.spec import DataSpec, MySQLDataConfig
from ..performance.backtest_builder import build_backtest_payload
from ..signals.ema_cross import EmaCross
from ..strategies import runner as strategies_runner
from ..strategies.runner import persist_payload_to_db

LOGGER = logging.getLogger(__name__)


def load_backtest_spec(path: Path | str) -> Dict[str, Any]:
    """Load a backtest specification from disk."""

    path_obj = Path(path)
    return json.loads(path_obj.read_text())


def _has_delta_config(raw: Mapping[str, Any]) -> bool:
    return any(str(key).startswith("delta_") for key in raw.keys())


def _uses_strategy_sources(raw: Mapping[str, Any]) -> bool:
    return bool(raw.get("mysql_env") or _has_delta_config(raw))


def _parse_data_spec(raw: Mapping[str, Any]) -> DataSpec:
    dataset_path = raw.get("dataset_path") or raw.get("path")
    mysql_raw = raw.get("mysql")
    mysql: MySQLDataConfig | None = None
    if mysql_raw is not None:
        mysql = MySQLDataConfig(
            connection_url=mysql_raw.get("connection_url"),
            env_var=mysql_raw.get("env_var", "QE_MARKETDATA_MYSQL_URL"),
            schema=mysql_raw.get("schema"),
            table=mysql_raw.get("table", "ohlcv"),
            symbol_col=mysql_raw.get("symbol_col", "symbol"),
            ts_col=mysql_raw.get("ts_col", "ts"),
            open_col=mysql_raw.get("open_col", "open"),
            high_col=mysql_raw.get("high_col", "high"),
            low_col=mysql_raw.get("low_col", "low"),
            close_col=mysql_raw.get("close_col", "close"),
            volume_col=mysql_raw.get("volume_col", "volume"),
            timeframe_col=mysql_raw.get("timeframe_col", "timeframe"),
            extra_where=mysql_raw.get("extra_where"),
            chunk_minutes=int(mysql_raw.get("chunk_minutes", 0)),
            symbol_lookup_table=mysql_raw.get("symbol_lookup_table"),
            symbol_lookup_symbol_col=mysql_raw.get("symbol_lookup_symbol_col", "symbol"),
            symbol_lookup_id_col=mysql_raw.get("symbol_lookup_id_col", "id"),
        )

    symbols = list(raw.get("symbols", []))
    timeframe = raw.get("timeframe")
    start = raw.get("start")
    end = raw.get("end")
    if start is None or end is None:
        raise ValueError("data.start and data.end are required")
    if dataset_path is None and mysql is None and not _uses_strategy_sources(raw):
        raise ValueError("data must provide dataset_path/path, mysql, or delta/mysql_env configuration")

    return DataSpec(
        dataset_path=dataset_path,
        mysql=mysql,
        symbols=symbols,
        timeframe=timeframe,
        start=str(start),
        end=str(end),
    )


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
    if _uses_strategy_sources(data_spec_raw):
        symbol = _single_symbol(data_spec.symbols, data_spec_raw.get("symbol"))
        df, source = _fetch_ohlc_with_source(symbol, asset_class, data_spec_raw)
        return _rows_from_dataframe(df, symbol), source
    return dataset.load_dataset(data_spec), None


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


def run_backtest_from_spec(spec: Mapping[str, Any]) -> Dict[str, Any]:
    """Execute a classic backtest based on a JSON specification."""

    data_raw = spec.get("data", {}) or {}
    data_spec = _parse_data_spec(data_raw)
    strategy_cfg = spec.get("strategy", {}) or {}
    asset_class = strategy_cfg.get("asset_class") or "EQUITY"
    rows, data_source = _load_rows(data_raw, data_spec, asset_class)
    if not rows:
        raise ValueError("No data rows loaded for backtest")

    symbol = _detect_symbol(rows)
    signal = _build_signal(spec, rows)

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
