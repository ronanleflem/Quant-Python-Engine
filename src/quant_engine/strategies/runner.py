"""Utilities to execute high-level strategy specifications."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd
from sqlalchemy import create_engine, text

from . import create_strategy
from .base import StrategySignal
from ..integrations import java_client
try:
    from deltalake import DeltaTable, write_deltalake
except Exception:  # pragma: no cover - optional dependency
    DeltaTable = None
    write_deltalake = None

LOGGER = logging.getLogger(__name__)


def load_strategy_spec(path: Path | str) -> Dict[str, Any]:
    """Load a strategy specification from disk."""

    path_obj = Path(path)
    return json.loads(path_obj.read_text())


def _expand_universe(spec: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    """Combine static universe entries with optional Java scan results."""

    explicit_universe: List[Mapping[str, Any]] = list(spec.get("universe", []))
    scans: Sequence[Mapping[str, Any]] = spec.get("scans", []) or []
    if not scans:
        return explicit_universe
    collected: Dict[str, Mapping[str, Any]] = {
        str(inst.get("symbol")): inst for inst in explicit_universe if inst.get("symbol")
    }
    for scan_spec in scans:
        scan_type = scan_spec.get("type")
        params = scan_spec.get("params", {})
        if not scan_type:
            continue
        try:
            results = java_client.get_market_scan(scan_type, params)
        except Exception:
            LOGGER.warning("Scan %s failed; skipping", scan_type)
            continue
        for item in results:
            symbol = item.get("symbol")
            if not symbol:
                continue
            asset_class = item.get("assetClass") or item.get("asset_class")
            collected[symbol] = {
                "symbol": symbol,
                "asset_class": asset_class,
                **{k: v for k, v in scan_spec.items() if k not in {"type", "params"}},
            }
    return list(collected.values())


def run_backtest_from_spec(spec: Mapping[str, Any]) -> Dict[str, Any]:
    """Execute ``strategy.backtest`` for every instrument defined in ``spec``."""

    strategy_cfg = spec.get("strategy", {})
    strategy_type = strategy_cfg.get("type")
    strategy_id = strategy_cfg.get("strategy_id", "strategy")
    if not strategy_type:
        raise ValueError("Strategy specification must define a 'type'")
    strategy = create_strategy(
        strategy_type, strategy_id=strategy_id, params=strategy_cfg.get("params", {})
    )
    data_spec: Mapping[str, Any] = spec.get("data", {})
    universe: Iterable[Mapping[str, Any]] = _expand_universe(spec)
    signals_by_symbol: Dict[str, List[Dict[str, Any]]] = {}
    counts: Dict[str, int] = {}
    for instrument in universe:
        symbol = instrument.get("symbol")
        if not symbol:
            LOGGER.warning("Skipping instrument without symbol: %s", instrument)
            continue
        asset_class = instrument.get(
            "asset_class",
            getattr(strategy, "asset_class", None) or strategy_cfg.get("asset_class"),
        )
        df = _fetch_ohlc_for_symbol(symbol, asset_class, data_spec, instrument)
        context = {"symbol": symbol, "asset_class": asset_class}
        signals = strategy.backtest(df, context)
        serialized = [_serialize_signal(sig) for sig in signals]
        signals_by_symbol[symbol] = serialized
        counts[symbol] = len(serialized)
    result = {
        "strategy_id": strategy_id,
        "strategy_type": strategy_type,
        "counts": counts,
        "signals": signals_by_symbol,
    }
    _persist_results_if_requested(result, spec.get("output"))
    return result


def _fetch_ohlc_for_symbol(
    symbol: str,
    asset_class: Optional[str],
    data_spec: Mapping[str, Any],
    instrument_spec: Mapping[str, Any],
) -> pd.DataFrame:
    # Combine top-level data spec with instrument overrides (instrument wins).
    merged_spec = {**data_spec, **instrument_spec}
    source = merged_spec.get("source")
    if source == "csv":
        path = Path(merged_spec["path"])
        df = pd.read_csv(path)
    else:
        df = _fetch_from_delta(symbol, asset_class, merged_spec)
        if df is None or df.empty:
            df = _fetch_from_mysql(symbol, merged_spec)
        if df is None or df.empty:
            df = _fetch_from_java(symbol, asset_class, merged_spec)
    if df is None or df.empty:
        raise RuntimeError(f"Unable to load OHLC for {symbol}")
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").reset_index(drop=True)
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing OHLC columns for {symbol}: {missing}")
    return df


def _build_delta_storage_options() -> Dict[str, Any]:
    return {
        "AWS_ACCESS_KEY_ID": os.getenv("DELTA_S3_ACCESS_KEY", os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")),
        "AWS_SECRET_ACCESS_KEY": os.getenv(
            "DELTA_S3_SECRET_KEY", os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
        ),
        "AWS_ENDPOINT_URL": os.getenv("DELTA_S3_ENDPOINT", "http://localhost:9000"),
        "AWS_REGION": os.getenv("DELTA_S3_REGION", "us-east-1"),
        "AWS_S3_ALLOW_UNSAFE_RENAME": "true",
        "AWS_S3_ADDRESSING_STYLE": os.getenv("DELTA_S3_ADDRESSING_STYLE", "path"),
        "AWS_ALLOW_HTTP": "true",
    }


def _delta_asset_dir(asset_class: Optional[str]) -> str:
    if not asset_class:
        return "GENERIC"
    mapping = {"EQUITY": "ACTION", "ETF": "ETF", "CRYPTO": "CRYPTO"}
    return mapping.get(asset_class.upper(), asset_class.upper())


def _fetch_from_delta(symbol: str, asset_class: Optional[str], spec: Mapping[str, Any]) -> Optional[pd.DataFrame]:
    base_uri = spec.get("delta_base") or os.getenv("DELTA_BASE_URI")
    if not base_uri or DeltaTable is None:
        if base_uri and DeltaTable is None:
            LOGGER.warning("deltalake package not installed; skipping Delta Lake source")
        return None

    asset_dir = spec.get("delta_asset_dir") or _delta_asset_dir(asset_class or spec.get("asset_class"))
    quotes: List[str] = []
    if "delta_quotes" in spec:
        quotes = [q.strip() for q in str(spec["delta_quotes"]).split(",") if q.strip()]
    elif os.getenv("DELTA_QUOTES"):
        quotes = [q.strip() for q in os.getenv("DELTA_QUOTES", "").split(",") if q.strip()]
    else:
        quotes = ["EUR", "USD", "USDT", "USDC"]

    start_str = spec.get("start")
    end_str = spec.get("end")
    timeframe = spec.get("timeframe")
    start_dt = pd.to_datetime(start_str, utc=True) if start_str else None
    end_dt = pd.to_datetime(end_str, utc=True) if end_str else None

    storage_options = _build_delta_storage_options()
    table_name = spec.get("delta_table") or spec.get("delta_symbol") or symbol
    for quote in quotes:
        table_path = "/".join([base_uri.rstrip("/"), asset_dir.strip("/"), quote.strip("/"), table_name])
        LOGGER.info("Trying Delta path: %s", table_path)
        try:
            dt = DeltaTable(table_path, storage_options=storage_options)
            df = dt.to_pyarrow_table().to_pandas()
        except Exception as exc:  # pragma: no cover - remote dependency
            LOGGER.info("Delta load failed for %s: %s", table_path, exc)
            continue

        if df.empty:
            LOGGER.info("Delta path %s returned no rows", table_path)
            continue
        if "ts" not in df.columns and "time" in df.columns:
            df = df.rename(columns={"time": "ts"})
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
        df = df.dropna(subset=["ts"])
        start_filter = start_dt if start_dt is not None else df["ts"].min()
        end_filter = end_dt if end_dt is not None else df["ts"].max()
        df_filtered = df[(df["ts"] >= start_filter) & (df["ts"] <= end_filter)]

        coverage = _coverage_ratio(df_filtered, start_dt, end_dt, timeframe, asset_class or spec.get("asset_class"))
        if coverage >= 0.999:
            LOGGER.info("Delta path %s covers requested range (%.1f%%)", table_path, coverage * 100)
            return df_filtered

        LOGGER.warning(
            "Delta path %s has partial coverage: %.1f%% (rows=%s)", table_path, coverage * 100, len(df_filtered)
        )

    LOGGER.info("No complete Delta coverage for %s; fallback to other sources", symbol)
    return None


def _coverage_ratio(
    df: pd.DataFrame,
    start_dt: Optional[pd.Timestamp],
    end_dt: Optional[pd.Timestamp],
    timeframe: Optional[str],
    asset_class: Optional[str],
) -> float:
    if df.empty or start_dt is None or end_dt is None or timeframe != "1D":
        return 1.0
    ts = df["ts"].dt.normalize()
    observed = set(ts.unique())
    if asset_class and asset_class.upper() == "CRYPTO":
        expected = pd.date_range(start_dt.normalize(), end_dt.normalize(), freq="D")
    else:
        expected = pd.date_range(start_dt.normalize(), end_dt.normalize(), freq="B")
    if len(expected) == 0:
        return 0.0
    return len(observed & set(expected)) / len(expected)


def _fetch_from_mysql(symbol: str, spec: Mapping[str, Any]) -> Optional[pd.DataFrame]:
    env = spec.get("mysql_env", "QE_MARKETDATA_MYSQL_URL")
    url = os.getenv(env)
    if not url:
        return None
    engine = create_engine(url)
    schema = spec.get("schema", "marketdata")
    table = spec.get("table", "ohlcv")
    symbol_col = spec.get("symbol_col", "symbol")
    ts_col = spec.get("ts_col", "ts")
    open_col = spec.get("open_col", "open")
    high_col = spec.get("high_col", "high")
    low_col = spec.get("low_col", "low")
    close_col = spec.get("close_col", "close")
    volume_col = spec.get("volume_col", "volume")
    start = spec.get("start")
    end = spec.get("end")
    timeframe = spec.get("timeframe")
    timeframe_col = spec.get("timeframe_col")
    sql = [
        f"SELECT {ts_col} AS ts, {open_col} AS open, {high_col} AS high,",
        f"       {low_col} AS low, {close_col} AS close, {volume_col} AS volume",
        f"  FROM {schema}.{table}",
        f" WHERE {symbol_col} = :symbol",
    ]
    params: Dict[str, Any] = {"symbol": symbol}
    if timeframe and timeframe_col:
        sql.append(f"   AND {timeframe_col} = :timeframe")
        params["timeframe"] = timeframe
    if start:
        sql.append(f"   AND {ts_col} >= :start")
        params["start"] = start
    if end:
        sql.append(f"   AND {ts_col} <= :end")
        params["end"] = end
    sql.append(f" ORDER BY {ts_col}")
    query = text("\n".join(sql))
    with engine.connect() as conn:
        df = pd.read_sql_query(query, conn, params=params)
    return df


def _fetch_from_java(
    symbol: str, asset_class: Optional[str], spec: Mapping[str, Any]
) -> Optional[pd.DataFrame]:
    start = spec.get("start")
    end = spec.get("end")
    timeframe = spec.get("timeframe")
    if not start or not end or not timeframe:
        return None
    asset = asset_class or spec.get("asset_class", "EQUITY")
    rows = java_client.get_ohlc(symbol, asset, start, end, timeframe)
    if not rows:
        _request_ingestion_on_gap(symbol, asset, spec)
        return None
    df = pd.DataFrame(rows)
    if "ts" not in df.columns and "time" in df.columns:
        df = df.rename(columns={"time": "ts"})
    expected = {"ts", "open", "high", "low", "close"}
    if not expected.issubset(df.columns):
        raise ValueError("Java OHLC response missing required columns")
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    start_ts = pd.to_datetime(start, utc=True)
    end_ts = pd.to_datetime(end, utc=True)
    if df.empty or df["ts"].min() > start_ts or df["ts"].max() < end_ts:
        _request_ingestion_on_gap(symbol, asset, spec)
    return df


def _request_ingestion_on_gap(symbol: str, asset: str, spec: Mapping[str, Any]) -> None:
    source = spec.get("ingestion_source", spec.get("source", "java"))
    start = spec.get("start")
    end = spec.get("end")
    if not start or not end:
        return
    try:
        java_client.request_historical_ingestion(symbol, asset, str(source), start, end)
    except Exception:
        LOGGER.warning("Historical ingestion request failed for %s", symbol)


def _serialize_signal(signal: StrategySignal) -> Dict[str, Any]:
    return {
        "strategy_id": signal.strategy_id,
        "symbol": signal.symbol,
        "asset_class": signal.asset_class,
        "side": signal.side,
        "ts_open_utc": signal.ts_open_utc.isoformat(),
        "qty": signal.qty,
        "meta": _jsonify(signal.meta),
    }


def _jsonify(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _jsonify(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonify(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (pd.Series, pd.DataFrame)):
        return json.loads(value.to_json())
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:  # pragma: no cover - fallback
            return str(value)
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def _persist_results_if_requested(result: Dict[str, Any], output_spec: Any) -> None:
    if output_spec:
        path = Path(output_spec.get("path", "strategy_signals.json"))
        fmt = output_spec.get("format", "json").lower()
        path.parent.mkdir(parents=True, exist_ok=True)
        if fmt == "json":
            path.write_text(json.dumps(result, indent=2))
        elif fmt == "csv":
            rows = [row for records in result.get("signals", {}).values() for row in records]
            pd.DataFrame(rows).to_csv(path, index=False)
        else:
            raise ValueError(f"Unsupported output format: {fmt}")
    _persist_results_to_delta(result)


def _flatten_signals(result: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for records in result.get("signals", {}).values():
        for rec in records:
            row = dict(rec)
            row["meta"] = json.dumps(row.get("meta", {}))
            rows.append(row)
    return pd.DataFrame(rows)


def _persist_results_to_delta(result: Dict[str, Any]) -> None:
    base_uri = os.getenv("DELTA_BASE_URI")
    if not base_uri or write_deltalake is None:
        if base_uri and write_deltalake is None:
            LOGGER.warning("deltalake package not installed; skipping Delta persistence")
        return
    strategy_id = result.get("strategy_id", "strategy")
    table_path = "/".join([base_uri.rstrip("/"), "STRATEGIE", strategy_id])
    df = _flatten_signals(result)
    if df.empty:
        return
    storage_options = _build_delta_storage_options()
    try:
        write_deltalake(
            table_path,
            df,
            mode="append",
            storage_options=storage_options,
            partition_by=["strategy_id", "symbol"],
        )
        LOGGER.info("Persisted strategy signals to Delta: %s", table_path)
    except Exception as exc:  # pragma: no cover - remote dependency
        LOGGER.warning("Failed to persist strategy signals to Delta (%s): %s", table_path, exc)


__all__ = ["load_strategy_spec", "run_backtest_from_spec"]
