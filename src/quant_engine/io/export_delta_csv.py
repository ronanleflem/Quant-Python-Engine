from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from ..strategies.runner import _delta_asset_dir, _delta_brokers, _fetch_from_delta

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeltaExportResult:
    path: Path
    rows: int
    ts_start: pd.Timestamp
    ts_end: pd.Timestamp


def export_delta_csv(
    *,
    symbol: str,
    asset_class: str,
    timeframe: str,
    output_root: Path,
    folder: str,
    delta_base: Optional[str] = None,
    delta_prefix: Optional[str] = None,
    delta_exchange: Optional[str] = None,
    delta_market_type: Optional[str] = None,
    delta_quotes: Optional[str] = None,
    delta_broker: Optional[str] = None,
    delta_asset_dir: Optional[str] = None,
    delta_table: Optional[str] = None,
    delta_calendar: Optional[str] = None,
) -> DeltaExportResult:
    if not delta_base and not os.getenv("DELTA_BASE_URI"):
        delta_base = "s3://quant-delta-dev"
    spec = {
        "delta_base": delta_base,
        "delta_prefix": delta_prefix,
        "delta_exchange": delta_exchange,
        "delta_market_type": delta_market_type,
        "delta_quotes": delta_quotes,
        "delta_broker": delta_broker,
        "delta_asset_dir": delta_asset_dir,
        "delta_table": delta_table,
        "delta_calendar": delta_calendar,
        "timeframe": timeframe,
        "asset_class": asset_class,
    }
    spec = {k: v for k, v in spec.items() if v is not None}

    LOGGER.info(
        "Export Delta CSV: symbol=%s asset_class=%s timeframe=%s",
        symbol,
        asset_class,
        timeframe,
    )
    _log_delta_candidates(symbol, asset_class, spec)
    _log_missing_exchange_hint(symbol, asset_class, spec)

    df = _fetch_from_delta(symbol, asset_class, spec)
    if df is None or df.empty:
        candidates = _candidate_delta_paths(symbol, asset_class, spec)
        hint = ""
        if candidates:
            hint = " Tried paths: " + "; ".join(candidates)
        raise RuntimeError(f"No Delta data found for {symbol} ({asset_class}).{hint}")

    df = _filter_timeframe(df, timeframe)
    if df.empty:
        raise RuntimeError(f"No Delta data matches timeframe {timeframe} for {symbol}")

    ts_col = _pick_column(df, ["ts", "timestamp", "time"])
    if not ts_col:
        raise RuntimeError("Delta data missing timestamp column")
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.dropna(subset=[ts_col]).sort_values(ts_col)

    open_col = _pick_column(df, ["open"])
    high_col = _pick_column(df, ["high"])
    low_col = _pick_column(df, ["low"])
    close_col = _pick_column(df, ["close"])
    if not all([open_col, high_col, low_col, close_col]):
        raise RuntimeError("Delta data missing OHLC columns")
    volume_col = _pick_column(df, ["volume", "vol"])

    ts_start = pd.to_datetime(df[ts_col].min(), utc=True)
    ts_end = pd.to_datetime(df[ts_col].max(), utc=True)

    label = _normalize_timeframe_label(timeframe)
    filename = f"{symbol}_{ts_start.strftime('%Y%m%d')}_{ts_end.strftime('%Y%m%d')}_{label}.csv"
    output_dir = output_root / folder
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    out = pd.DataFrame(
        {
            "Timestamp": df[ts_col].dt.strftime("%Y-%m-%dT%H:%M:%S"),
            "Open": df[open_col].astype(float),
            "High": df[high_col].astype(float),
            "Low": df[low_col].astype(float),
            "Close": df[close_col].astype(float),
            "Volume": df[volume_col].astype(float) if volume_col else 0.0,
        }
    )
    out.to_csv(output_path, index=False)
    return DeltaExportResult(path=output_path, rows=len(out), ts_start=ts_start, ts_end=ts_end)


def _pick_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    lower_map = {str(col).lower(): col for col in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        key = str(cand).lower()
        if key in lower_map:
            return lower_map[key]
    return None


def _filter_timeframe(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if "timeframe" not in df.columns and "tf" not in df.columns:
        return df
    tf_col = "timeframe" if "timeframe" in df.columns else "tf"
    variants = _timeframe_variants(timeframe)
    mask = df[tf_col].astype(str).str.strip().str.lower().isin(variants)
    return df[mask].copy()


def _normalize_timeframe_label(timeframe: str) -> str:
    tf = str(timeframe).strip().lower()
    aliases = {
        "1m": "m1",
        "1min": "m1",
        "1minute": "m1",
        "m1": "m1",
        "5m": "m5",
        "5min": "m5",
        "m5": "m5",
        "15m": "m15",
        "15min": "m15",
        "m15": "m15",
        "30m": "m30",
        "30min": "m30",
        "m30": "m30",
        "60m": "1h",
        "1h": "1h",
        "h1": "1h",
        "4h": "4h",
        "h4": "4h",
        "1d": "d1",
        "d1": "d1",
        "daily": "d1",
        "1w": "w1",
        "w1": "w1",
        "weekly": "w1",
    }
    return aliases.get(tf, tf.replace(" ", ""))


def _timeframe_variants(timeframe: str) -> set[str]:
    tf = str(timeframe).strip().lower()
    variants = {tf, tf.replace(" ", "")}
    label = _normalize_timeframe_label(tf)
    variants.add(label)
    if label.startswith("m") and label[1:].isdigit():
        value = label[1:]
        variants.update({f"{value}m", f"{value}min"})
    if label.endswith("h") and label[:-1].isdigit():
        variants.add(f"{label[:-1]}h")
    if label == "d1":
        variants.update({"1d", "daily"})
    return variants


def _candidate_delta_paths(symbol: str, asset_class: str, spec: dict) -> list[str]:
    base_uri = spec.get("delta_base") or os.getenv("DELTA_BASE_URI") or "s3://quant-delta-dev"
    if not base_uri:
        return []

    asset_dir = spec.get("delta_asset_dir") or _delta_asset_dir(asset_class)
    delta_prefix = spec.get("delta_prefix") or os.getenv("DELTA_PREFIX") or "delta"
    delta_prefix = str(delta_prefix).strip().strip("/")
    exchange = spec.get("delta_exchange") or spec.get("exchange") or os.getenv("DELTA_EXCHANGE") or "GENERIC"
    exchange = str(exchange).strip().upper() or "GENERIC"
    market_type = spec.get("delta_market_type") or spec.get("market_type") or os.getenv("DELTA_MARKET_TYPE") or "SPOT"
    market_type = str(market_type).strip().upper() or "SPOT"
    if "delta_quotes" in spec:
        quotes = [q.strip() for q in str(spec["delta_quotes"]).split(",") if q.strip()]
    elif os.getenv("DELTA_QUOTES"):
        quotes = [q.strip() for q in os.getenv("DELTA_QUOTES", "").split(",") if q.strip()]
    else:
        quotes = ["EUR", "USD", "USDT", "USDC"]

    table_name = spec.get("delta_table") or spec.get("delta_symbol") or symbol
    brokers = _delta_brokers(asset_class, spec)
    use_exchange_dir = asset_class.upper() != "CRYPTO" and exchange != "GENERIC"
    paths: list[str] = []
    for broker in (brokers or [""]):
        for quote in quotes:
            parts = [base_uri.rstrip("/")]
            if delta_prefix:
                parts.append(delta_prefix)
            parts.append(asset_dir.strip("/"))
            if broker:
                parts.append(str(broker).strip().upper())
            if market_type:
                parts.append(market_type.strip("/"))
            if use_exchange_dir:
                parts.append(exchange.strip("/"))
            parts.extend([quote.strip("/"), table_name])
            paths.append("/".join(parts))
    return paths


def _log_delta_candidates(symbol: str, asset_class: str, spec: dict) -> None:
    base_uri = spec.get("delta_base") or os.getenv("DELTA_BASE_URI") or "s3://quant-delta-dev"
    if not base_uri:
        LOGGER.warning("Delta base URI missing (delta_base or DELTA_BASE_URI).")
        return
    paths = _candidate_delta_paths(symbol, asset_class, spec)
    if not paths:
        LOGGER.warning("No Delta candidate paths resolved for %s (%s).", symbol, asset_class)
        return
    for path in paths:
        LOGGER.info("Delta candidate path: %s", path)


def _log_missing_exchange_hint(symbol: str, asset_class: str, spec: dict) -> None:
    asset = str(asset_class).strip().upper()
    if asset not in {"ACTION", "EQUITY", "STOCK", "ETF"}:
        return
    exchange = spec.get("delta_exchange") or spec.get("exchange") or os.getenv("DELTA_EXCHANGE")
    if exchange:
        return
    LOGGER.warning(
        "Delta exchange missing for %s (%s). Equity/ETF paths include an exchange segment, "
        "ex: quant-delta-dev/delta/STOCK/IBKR/SPOT/NASDAQ/USD",
        symbol,
        asset_class,
    )
