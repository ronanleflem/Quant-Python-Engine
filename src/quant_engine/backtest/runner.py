"""Backtest runner aligned with strategy JSON specs."""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from . import engine
from ..core import dataset
from ..core.features import atr
from ..core.spec import DataSpec, MySQLDataConfig
from ..performance.backtest_builder import build_backtest_payload
from ..signals.ema_cross import EmaCross
from ..strategies.runner import persist_payload_to_db


def load_backtest_spec(path: Path | str) -> Dict[str, Any]:
    """Load a backtest specification from disk."""

    path_obj = Path(path)
    return json.loads(path_obj.read_text())


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
    if dataset_path is None and mysql is None:
        raise ValueError("data must provide dataset_path/path or mysql configuration")

    return DataSpec(
        dataset_path=dataset_path,
        mysql=mysql,
        symbols=symbols,
        timeframe=timeframe,
        start=str(start),
        end=str(end),
    )


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

    data_spec = _parse_data_spec(spec.get("data", {}) or {})
    rows = dataset.load_dataset(data_spec)
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

    strategy_cfg = spec.get("strategy", {}) or {}
    strategy_id = strategy_cfg.get("strategy_id", "backtest")
    run_id = spec.get("run_id") or strategy_cfg.get("run_id") or uuid.uuid4().hex
    asset_class = strategy_cfg.get("asset_class") or "EQUITY"
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
        persist_payload_to_db(payload, spec, strategy_type="backtest")

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
