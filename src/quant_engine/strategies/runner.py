"""Utilities to execute high-level strategy specifications."""
from __future__ import annotations

import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd
import requests
from sqlalchemy import create_engine, inspect, text

from . import create_strategy
from .base import StrategySignal
from ..integrations import java_client
from ..filters.utils import apply_filter_stack, FilterValidationError
from ..performance.dca_builder import build_backend_payload_for_java
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
try:
    from deltalake import DeltaTable, write_deltalake
except Exception:  # pragma: no cover - optional dependency
    DeltaTable = None
    write_deltalake = None
try:  # pragma: no cover - optional dependency
    import pandas_market_calendars as mcal  # type: ignore
except Exception:
    mcal = None

_WARNED_MCAL_MISSING = False
_WARNED_MCAL_ERROR = False
_MARKET_SCHEDULE_CACHE: Dict[tuple[str, object, object], pd.DatetimeIndex] = {}

logging.basicConfig(level=logging.INFO)
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
    ohlc_by_symbol: Dict[str, pd.DataFrame] = {}
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
        filters_spec = strategy_cfg.get("filters") or spec.get("filters") or []
        if filters_spec:
            df = df.copy()
            df_filter = df.copy()
            if "ts" in df_filter.columns:
                df_filter["ts"] = pd.to_datetime(df_filter["ts"], utc=True)
                df_filter = df_filter.set_index("ts")
            try:
                mask = apply_filter_stack(df_filter, filters_spec, symbol=symbol, logger=LOGGER, strict=True)
            except FilterValidationError as exc:
                LOGGER.error("Filters failed for %s: %s", symbol, exc)
                raise
            if "ts" in df.columns:
                ts_index = pd.to_datetime(df["ts"], utc=True)
                df["_filter_ok"] = mask.reindex(ts_index, fill_value=False).to_numpy()
            else:
                df["_filter_ok"] = mask.reindex(df.index, fill_value=False)
        ohlc_by_symbol[symbol] = df.copy()
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
    _persist_results_to_db(result, spec, ohlc_by_symbol)
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
            LOGGER.info("Delta source empty for %s, falling back to MySQL", symbol)
            df = _fetch_from_mysql(symbol, merged_spec)
            if df is not None and not df.empty:
                LOGGER.info("Loaded OHLC for %s from MySQL (%d rows)", symbol, len(df))
        else:
            LOGGER.info("Loaded OHLC for %s from Delta (%d rows)", symbol, len(df))

        if df is None or df.empty:
            LOGGER.info("MySQL source empty for %s, falling back to Java", symbol)
            df = _fetch_from_java(symbol, asset_class, merged_spec)
            if df is not None and not df.empty:
                LOGGER.info("Loaded OHLC for %s from Java (%d rows)", symbol, len(df))
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
    mapping = {"EQUITY": "STOCK", "ACTION": "STOCK", "ETF": "ETF", "CRYPTO": "CRYPTO"}
    return mapping.get(asset_class.upper(), asset_class.upper())


def _delta_brokers(asset_class: Optional[str], spec: Mapping[str, Any]) -> List[str]:
    raw = spec.get("delta_brokers") or spec.get("brokers") or spec.get("broker") or spec.get("delta_broker")
    collected: List[str] = []
    if isinstance(raw, str):
        collected = [b.strip() for b in raw.split(",") if b.strip()]
    elif isinstance(raw, Iterable) and not isinstance(raw, (str, bytes, dict)):
        collected = [str(b).strip() for b in raw if str(b).strip()]

    asset = (asset_class or spec.get("asset_class") or "").upper()
    if asset == "CRYPTO":
        preferred = ["BINANCE", "BITGET", "MEXC"]
    else:
        preferred = ["IBKR"]

    ordered: List[str] = []
    if collected:
        for item in collected:
            key = str(item).strip().upper()
            if key and key not in ordered:
                ordered.append(key)
        if asset != "CRYPTO" and "IBKR" not in ordered:
            ordered.append("IBKR")
    else:
        for item in preferred:
            key = str(item).strip().upper()
            if key and key not in ordered:
                ordered.append(key)
    return ordered


def _fetch_from_delta(symbol: str, asset_class: Optional[str], spec: Mapping[str, Any]) -> Optional[pd.DataFrame]:
    base_uri = spec.get("delta_base") or os.getenv("DELTA_BASE_URI")
    if not base_uri or DeltaTable is None:
        if base_uri and DeltaTable is None:
            LOGGER.warning("deltalake package not installed; skipping Delta Lake source")
        return None

    asset_class_value = asset_class or spec.get("asset_class")
    if not asset_class_value and not spec.get("delta_asset_dir"):
        LOGGER.warning("Delta asset_class missing for %s; skipping Delta source", symbol)
        return None

    asset_dir = spec.get("delta_asset_dir") or _delta_asset_dir(asset_class_value)
    delta_prefix = spec.get("delta_prefix") or os.getenv("DELTA_PREFIX") or "delta"
    delta_prefix = str(delta_prefix).strip().strip("/")
    exchange = spec.get("delta_exchange") or spec.get("exchange") or os.getenv("DELTA_EXCHANGE") or "GENERIC"
    exchange = str(exchange).strip().upper() or "GENERIC"
    market_type = spec.get("delta_market_type") or spec.get("market_type") or os.getenv("DELTA_MARKET_TYPE") or "SPOT"
    market_type = str(market_type).strip().upper() or "SPOT"
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
    try:
        min_coverage = float(spec.get("delta_min_coverage") or os.getenv("DELTA_MIN_COVERAGE", 0.95))
    except Exception:
        min_coverage = 0.95
    min_coverage = max(0.0, min(min_coverage, 1.0))

    storage_options = _build_delta_storage_options()
    table_name = spec.get("delta_table") or spec.get("delta_symbol") or symbol
    brokers = _delta_brokers(asset_class, spec)
    asset_upper = (asset_class_value or "").upper()
    use_exchange_dir = asset_upper != "CRYPTO" and exchange != "GENERIC"
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
            table_path = "/".join(parts)
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
            if "ts" not in df.columns:
                if "time" in df.columns:
                    df = df.rename(columns={"time": "ts"})
                elif "timestamp" in df.columns:
                    df = df.rename(columns={"timestamp": "ts"})
                else:
                    LOGGER.warning("Delta path %s missing time column", table_path)
                    continue

            if pd.api.types.is_numeric_dtype(df["ts"]):
                df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True, errors="coerce")
            else:
                df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
            df = df.dropna(subset=["ts"])
            start_filter = start_dt if start_dt is not None else df["ts"].min()
            end_filter = end_dt if end_dt is not None else df["ts"].max()
            df_filtered = df[(df["ts"] >= start_filter) & (df["ts"] <= end_filter)]

            coverage, expected_len, observed_len, obs_start, obs_end, missing_sample = _coverage_stats(
                df_filtered,
                start_dt,
                end_dt,
                timeframe,
                asset_class or spec.get("asset_class"),
                spec.get("delta_calendar") or os.getenv("DELTA_CALENDAR"),
                exchange,
            )
            if coverage >= min_coverage:
                LOGGER.info(
                    "Delta path %s covers requested range (%.1f%%>=%.1f%%)",
                    table_path,
                    coverage * 100,
                    min_coverage * 100,
                )
                if coverage < 1.0 and missing_sample:
                    LOGGER.info(
                        "Delta path %s missing dates (first %d): %s",
                        table_path,
                        len(missing_sample),
                        missing_sample[:20],
                    )
                return df_filtered

            LOGGER.warning(
                "Delta path %s has partial coverage: %.1f%% < %.1f%% (rows=%s, expected=%s, ts_min=%s, ts_max=%s)",
                table_path,
                coverage * 100,
                min_coverage * 100,
                observed_len,
                expected_len,
                obs_start,
                obs_end,
            )
            if missing_sample:
                LOGGER.info("Sample missing dates for %s: %s", table_path, missing_sample[:20])

    LOGGER.warning(
        "No Delta data for %s (asset_class=%s, asset_dir=%s); fallback to other sources",
        symbol,
        (asset_class_value or "UNKNOWN"),
        asset_dir,
    )
    return None


def _timeframe_freq(timeframe: Optional[str], asset_class: Optional[str]) -> Optional[str]:
    if not timeframe:
        return None
    tf = str(timeframe).strip().lower()
    aliases = {
        "1m": "1min",
        "m1": "1min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "60m": "60min",
        "1h": "1h",
        "h1": "1h",
        "4h": "4h",
        "1d": "1d",
        "d1": "1d",
        "1w": "1w",
        "w1": "1w",
    }
    tf_norm = aliases.get(tf)
    if tf_norm is None:
        match = re.match(r"^(\d+)\s*([smhdw])", tf)
        if not match:
            return None
        value = match.group(1)
        unit = match.group(2).lower()
        unit_map = {"s": "S", "m": "min", "h": "H", "d": "D", "w": "W"}
        if unit not in unit_map:
            return None
        tf_norm = f"{value}{unit_map[unit]}"

    if tf_norm == "1d" and asset_class and asset_class.upper() != "CRYPTO":
        return "B"
    if tf_norm and tf_norm[-1].isalpha() and tf_norm[-1].lower() in {"h", "w", "d"}:
        return tf_norm[:-1] + tf_norm[-1].upper()
    return tf_norm


def _market_calendar_candidates(exchange: Optional[str]) -> List[str]:
    if not exchange:
        return []
    ex = str(exchange).strip().upper()
    if not ex or ex in {"GENERIC", "UNKNOWN"}:
        return []

    aliases: Dict[str, List[str]] = {
        # US
        "NASDAQ": ["NASDAQ", "XNAS"],
        "XNAS": ["NASDAQ", "XNAS"],
        "NYSE": ["NYSE", "XNYS"],
        "XNYS": ["NYSE", "XNYS"],
        "BATS": ["BATS"],
        "IEX": ["IEX", "IEX", "INVESTORS_EXCHANGE", "Investors_Exchange"],
        # EU
        "LSE": ["LSE", "XLON"],
        "XLON": ["LSE", "XLON"],
        "SIX": ["SIX", "XSWX"],
        "XSWX": ["SIX", "XSWX"],
        "XETR": ["XETR", "XFRA"],
        "XFRA": ["XFRA", "XETR"],
        "XPAR": ["XPAR"],
        "EURONEXT": ["XPAR", "XAMS", "XBRU", "XLIS"],
        # APAC
        "JPX": ["JPX", "XJPX"],
        "XJPX": ["JPX", "XJPX"],
        "HKEX": ["HKEX", "XHKG"],
        "XHKG": ["HKEX", "XHKG"],
        "ASX": ["ASX", "XASX"],
        "XASX": ["ASX", "XASX"],
        # Canada
        "TSX": ["TSX", "XTSE"],
        "XTSE": ["TSX", "XTSE"],
        # India
        "NSE": ["NSE", "XNSE"],
        "XNSE": ["XNSE", "NSE"],
        "BSE": ["BSE", "XBOM"],
        "XBOM": ["XBOM", "BSE"],
    }
    candidates = aliases.get(ex, [ex])
    # Also try the raw value as-is if it differs (some calendars are lowercase like "stock").
    if exchange and str(exchange).strip() not in candidates:
        candidates.append(str(exchange).strip())
    # Dedupe preserving order.
    seen: set[str] = set()
    out: List[str] = []
    for item in candidates:
        key = str(item)
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _expected_from_market_calendar(
    calendar_name: str,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    freq: str,
) -> Optional[pd.DatetimeIndex]:
    if not calendar_name:
        return None
    if mcal is None:
        return None
    try:
        cal = mcal.get_calendar(calendar_name)
    except Exception:
        return None
    schedule = cal.schedule(start_date=start_dt.normalize().date(), end_date=end_dt.normalize().date())
    if schedule.empty:
        return pd.DatetimeIndex([], tz="UTC")
    if freq == "B":
        expected = schedule.index
        if expected.tz is None:
            return expected.tz_localize("UTC")
        return expected.tz_convert("UTC")

    expected_chunks: List[pd.DatetimeIndex] = []
    for _, row in schedule.iterrows():
        open_ts = row.get("market_open")
        close_ts = row.get("market_close")
        if open_ts is None or close_ts is None:
            continue
        if open_ts.tz is None:
            open_ts = open_ts.tz_localize("UTC")
        else:
            open_ts = open_ts.tz_convert("UTC")
        if close_ts.tz is None:
            close_ts = close_ts.tz_localize("UTC")
        else:
            close_ts = close_ts.tz_convert("UTC")

        session_start = max(open_ts, start_dt)
        session_end = min(close_ts, end_dt)
        if session_start > session_end:
            continue
        expected_chunks.append(pd.date_range(session_start, session_end, freq=freq))
    if not expected_chunks:
        return pd.DatetimeIndex([], tz="UTC")
    expected = expected_chunks[0]
    for chunk in expected_chunks[1:]:
        expected = expected.append(chunk)
    return expected


def _coverage_stats(
    df: pd.DataFrame,
    start_dt: Optional[pd.Timestamp],
    end_dt: Optional[pd.Timestamp],
    timeframe: Optional[str],
    asset_class: Optional[str],
    calendar: Optional[str] = None,
    exchange: Optional[str] = None,
) -> tuple[float, int, int, Optional[pd.Timestamp], Optional[pd.Timestamp], List[pd.Timestamp]]:
    if df.empty:
        return 0.0, 0, 0, None, None, []
    if start_dt is None or end_dt is None:
        ts = df["ts"].dropna()
        return 1.0, len(ts), len(ts), ts.min(), ts.max(), []

    freq = _timeframe_freq(timeframe, asset_class)
    if freq is None:
        ts = df["ts"].dropna()
        return 1.0, len(ts), len(ts), ts.min(), ts.max(), []

    ts = df["ts"].dropna().dt.tz_convert("UTC")
    if ts.empty:
        return 0.0, 0, 0, None, None, []

    cal = (calendar or os.getenv("DELTA_CALENDAR") or "").strip().upper()
    is_fx_calendar = cal in {"FX", "FOREX"}
    if freq == "B":
        if not cal and asset_class and asset_class.upper() in {"EQUITY", "STOCK", "ACTION", "ETF"}:
            cal = "USFED"
        ex = (exchange or "").strip()
        use_market_calendar = bool(ex or cal)
        expected = None
        if use_market_calendar:
            try:
                global _WARNED_MCAL_MISSING
                global _WARNED_MCAL_ERROR
                if mcal is None:
                    if not _WARNED_MCAL_MISSING:
                        LOGGER.warning(
                            "pandas_market_calendars not installed; falling back to business-day calendar for %s",
                            str(ex).upper(),
                        )
                        _WARNED_MCAL_MISSING = True
                    expected = None
                else:
                    expected = _expected_from_market_calendar(cal, start_dt, end_dt, "B") if cal else None
                    if expected is None and ex:
                        start_date = start_dt.normalize().date()
                        end_date = end_dt.normalize().date()
                        for candidate in _market_calendar_candidates(ex):
                            try:
                                cache_key = (candidate, start_date, end_date)
                                sessions = _MARKET_SCHEDULE_CACHE.get(cache_key)
                                if sessions is None:
                                    schedule = mcal.get_calendar(candidate).schedule(
                                        start_date=start_date,
                                        end_date=end_date,
                                    )
                                    sessions = schedule.index
                                    _MARKET_SCHEDULE_CACHE[cache_key] = sessions
                                if sessions.tz is None:
                                    sessions = sessions.tz_localize("UTC")
                                else:
                                    sessions = sessions.tz_convert("UTC")
                                expected = sessions.normalize()
                                break
                            except Exception:
                                expected = None
            except Exception as exc:
                if not _WARNED_MCAL_ERROR:
                    LOGGER.warning(
                        "pandas_market_calendars failed (%s); falling back to business-day calendar for %s",
                        exc,
                        str(ex).upper(),
                    )
                    _WARNED_MCAL_ERROR = True
                expected = None

        if expected is None:
            expected_freq = (
                CustomBusinessDay(calendar=USFederalHolidayCalendar())
                if cal in {"USFED", "US_FED", "US-FED"}
                else "B"
            )
            expected = pd.date_range(start_dt.normalize(), end_dt.normalize(), freq=expected_freq, tz="UTC")
        observed = ts.dt.floor("D")
    elif freq.upper().endswith("W"):
        expected = pd.date_range(start_dt.normalize(), end_dt.normalize(), freq=freq)
        observed = ts.dt.to_period("W").dt.start_time
    else:
        expected = _expected_from_market_calendar(cal, start_dt, end_dt, freq) if cal else None
        if expected is None:
            expected = pd.date_range(start_dt, end_dt, freq=freq)
        if is_fx_calendar:
            expected = expected[expected.weekday < 5]
        observed = ts.dt.floor(freq)

    if len(expected) == 0:
        return 0.0, 0, len(observed.unique()), ts.min(), ts.max(), []
    observed_set = set(observed.unique())
    expected_set = set(expected)
    coverage = len(observed_set & expected_set) / len(expected_set)
    missing_sorted = sorted(list(expected_set - observed_set))[:50]
    return coverage, len(expected_set), len(observed_set), ts.min(), ts.max(), missing_sorted


def _min_coverage(spec: Mapping[str, Any], default: float) -> float:
    for key in ("java_min_coverage", "min_coverage", "coverage_min"):
        if key in spec:
            try:
                return max(0.0, min(float(spec[key]), 1.0))
            except Exception:
                return default
    return default


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
    if df.empty:
        LOGGER.warning("Java OHLC returned empty for %s; no ingestion fallback enabled", symbol)
        return None

    coverage, _, _, obs_start, obs_end, missing_sample = _coverage_stats(
        df,
        start_ts,
        end_ts,
        timeframe,
        asset,
        spec.get("delta_calendar") or os.getenv("DELTA_CALENDAR"),
        spec.get("delta_exchange") or spec.get("exchange") or os.getenv("DELTA_EXCHANGE"),
    )
    LOGGER.info(
        "Java OHLC coverage for %s [%s -> %s]: %.1f%% (rows=%d, ts_min=%s, ts_max=%s)",
        symbol,
        start,
        end,
        coverage * 100,
        len(df),
        obs_start,
        obs_end,
    )
    if missing_sample:
        LOGGER.warning(
            "Java OHLC missing %d expected bars for %s (sample=%s)",
            len(missing_sample),
            symbol,
            [ts.isoformat() for ts in missing_sample[:20]],
        )

    min_coverage = _min_coverage(spec, 0.9)
    if coverage < min_coverage:
        LOGGER.warning(
            "Java OHLC coverage insufficient for %s: %.1f%% < %.1f%% (start=%s end=%s); skipping ingestion fallback",
            symbol,
            coverage * 100,
            min_coverage * 100,
            start,
            end,
        )
    return df


def _request_ingestion_on_gap(symbol: str, asset: str, spec: Mapping[str, Any]) -> None:
    # Ingestion fallback disabled: rely solely on Delta/MySQL/Java OHLC endpoints.
    return


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

__all__ = ["load_strategy_spec", "run_backtest_from_spec", "persist_payload_to_db"]


def _compact_meta(meta: Mapping[str, Any]) -> Dict[str, Any]:
    """Drop heavy fields (grid configs etc.) and cap size for DB transport."""

    if not meta:
        return {}
    pruned: Dict[str, Any] = {
        k: v
        for k, v in meta.items()
        if k
        not in {
            "grid_config",
            "tp_rule",
            "grid",
            "rules",
            "filters",
        }
    }
    try:
        serialized = json.dumps(pruned, ensure_ascii=False)
        if len(serialized) > 2000:
            pruned = {"note": "meta truncated", "keys": list(pruned.keys())}
    except Exception:
        pruned = {"note": "meta serialization failed"}
    return pruned


def _truncate_json(obj: Any, limit: int = 250) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False)
    except Exception:
        return ""
    if len(s) > limit:
        return s[: limit - 3] + "..."
    return s


def _filter_row_for_table(engine, table: str, row: Dict[str, Any]) -> Dict[str, Any]:
    try:
        inspector = inspect(engine)
        if not inspector.has_table(table):
            return row
        cols = {col["name"] for col in inspector.get_columns(table)}
    except Exception:
        return row
    filtered = {k: v for k, v in row.items() if k in cols}
    dropped = set(row.keys()) - set(filtered.keys())
    if dropped:
        LOGGER.warning("Dropping columns not in %s: %s", table, sorted(dropped))
    return filtered


def _first_token(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        return parts[0] if parts else None
    if isinstance(value, Iterable) and not isinstance(value, (bytes, dict)):
        for item in value:
            token = _first_token(item)
            if token:
                return token
    return str(value).strip() if str(value).strip() else None


def _build_trade_context(
    spec: Mapping[str, Any],
    default_asset_class: str,
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    data_spec = spec.get("data", {}) or {}
    default_asset = str(default_asset_class or data_spec.get("asset_class") or "").upper()

    default_market_type = _first_token(
        data_spec.get("delta_market_type")
        or data_spec.get("market_type")
        or os.getenv("DELTA_MARKET_TYPE")
        or "SPOT"
    ) or "SPOT"
    default_exchange = _first_token(data_spec.get("delta_exchange") or data_spec.get("exchange"))

    default_broker = _first_token(data_spec.get("broker") or data_spec.get("delta_broker"))
    if not default_broker:
        default_broker = _first_token(data_spec.get("brokers") or data_spec.get("delta_brokers"))
    if not default_broker:
        brokers = _delta_brokers(default_asset, data_spec)
        default_broker = brokers[0] if brokers else None

    default_currency = _first_token(
        data_spec.get("currency") or data_spec.get("delta_quote") or data_spec.get("delta_quotes") or os.getenv("DELTA_QUOTES")
    )
    if not default_currency:
        default_currency = "USDT" if default_asset == "CRYPTO" else "USD"

    if not default_exchange and default_asset == "CRYPTO" and default_broker:
        default_exchange = default_broker

    default_context = {
        "broker": default_broker or "UNKNOWN",
        "exchange": default_exchange or "UNKNOWN",
        "currency": default_currency or "UNKNOWN",
        "market_type": default_market_type or "UNKNOWN",
    }

    context_by_symbol: Dict[str, Dict[str, Any]] = {}
    for inst in spec.get("universe", []) or []:
        symbol = inst.get("symbol")
        if not symbol:
            continue
        asset = str(inst.get("asset_class") or default_asset).upper()
        broker = _first_token(inst.get("broker")) or default_context["broker"]
        exchange = _first_token(inst.get("exchange")) or default_context["exchange"]
        currency = _first_token(inst.get("currency") or inst.get("quote")) or default_context["currency"]
        market_type = _first_token(inst.get("market_type") or inst.get("delta_market_type")) or default_context["market_type"]
        if asset == "CRYPTO" and (not exchange or exchange == "UNKNOWN"):
            exchange = broker
        context_by_symbol[str(symbol)] = {
            "broker": broker,
            "exchange": exchange,
            "currency": currency,
            "market_type": market_type,
        }
    return context_by_symbol, default_context


def persist_payload_to_db(
    payload: Mapping[str, Any],
    spec: Mapping[str, Any],
    *,
    strategy_type: Optional[str] = None,
) -> None:
    dsn = os.getenv("DB_DSN")
    if not dsn:
        LOGGER.info("DB_DSN not set; skipping DB persistence")
        return

    run = payload.get("run", {}) if isinstance(payload, Mapping) else {}
    trades = payload.get("trades", []) if isinstance(payload, Mapping) else []
    asset_class = run.get("assetClass") or ""
    run_id = run.get("runId")
    if not run_id:
        run_id = spec.get("run_id") or uuid.uuid4().hex
    strategy_type = strategy_type or str(run.get("strategyType") or "")
    strategy_type = str(strategy_type).strip().lower()

    try:
        engine = create_engine(dsn)
        LOGGER.info("DB engine created for DSN %s", dsn)
    except Exception as exc:
        LOGGER.warning("DB_DSN invalid, skipping DB persistence: %s", exc)
        return

    try:
        perf_row = {
            "strategy_name": run.get("strategyId"),
            "run_id": run_id,
            "asset_class": run.get("assetClass"),
            "universe": run.get("universe"),
            "timeframe": run.get("timeframe"),
            "symbol": "MUTUAL_FUNDS" if strategy_type.startswith("dca") else run.get("symbol"),
            "compared_symbol": run.get("comparedSymbol"),
            "start_strategy": pd.to_datetime(run.get("startTsUtc"), utc=True),
            "end_strategy": pd.to_datetime(run.get("endTsUtc"), utc=True),
            "win_count": run.get("winCount"),
            "loss_count": run.get("lossCount"),
            "total_return": run.get("totalReturn"),
            "max_drawdown": run.get("maxDrawdown"),
            "average_trade": run.get("averageTrade"),
            "averagesl": run.get("averageSL"),
            "averagetp": run.get("averageTP"),
            "rr_moyen": run.get("rrMoyen"),
            "total_net_return": run.get("totalNetReturn"),
            "net_win_count": run.get("netWinCount"),
            "net_loss_count": run.get("netLossCount"),
            "average_net_trade": run.get("averageNetTrade"),
            "initial_capital": run.get("initialCapital"),
            "final_capital": run.get("finalCapital"),
            "return_pct": run.get("returnPct"),
            "max_drawdown_pct": run.get("maxDrawdownPct"),
            "volatility_pct": run.get("volatilityPct"),
            "sharpe": run.get("sharpe"),
            "sortino": run.get("sortino"),
            "winrate_pct": run.get("winratePct"),
            "metric": None,
            "value": 0.0,
            "extra_json": json.dumps(run.get("extra", {}), ensure_ascii=False),
        }
        perf_row = _filter_row_for_table(engine, "performance", perf_row)
        pd.DataFrame([perf_row]).to_sql("performance", engine, if_exists="append", index=False)
        LOGGER.info("Persisted performance row for run %s to DB", run_id)
    except Exception as exc:
        LOGGER.warning("Failed to persist performance for run %s: %s", run_id, exc)

    if not trades:
        LOGGER.info("No trades to persist for run %s", run_id)
        return

    context_by_symbol, default_context = _build_trade_context(spec, asset_class)
    trade_rows: List[Dict[str, Any]] = []
    for t in trades:
        trade_ctx = context_by_symbol.get(str(t.get("symbol")), default_context)
        meta_payload = dict(t.get("meta", {}) or {})
        if "be_pct" not in meta_payload:
            meta_payload["be_pct"] = meta_payload.get("break_even_pct")
        trade_rows.append(
            {
                "strategy_name": t.get("strategyId"),
                "run_id": t.get("runId") or run_id,
                "symbol": t.get("symbol"),
                "asset_class": t.get("assetClass"),
                "broker": trade_ctx.get("broker"),
                "exchange": trade_ctx.get("exchange"),
                "currency": trade_ctx.get("currency"),
                "market_type": trade_ctx.get("market_type"),
                "cycle_id": t.get("cycleId"),
                "trade_type": t.get("side") or "LONG",
                "entry_timestamp": pd.to_datetime(t.get("entryTimeUtc"), utc=True),
                "exit_timestamp": pd.to_datetime(t.get("exitTimeUtc"), utc=True),
                "entry_price": t.get("entryPrice") or 0.0,
                "exit_price": t.get("exitPrice") or 0.0,
                "quantity": t.get("quantity") or 0.0,
                "profit_or_loss": t.get("grossPnl") or 0.0,
                "pnl_pct": t.get("grossPnlPct"),
                "max_drawdown_pct": t.get("maxDdPct"),
                "meta_json": json.dumps(meta_payload, ensure_ascii=False),
                "confidence_score": 0.0,
                "stop_loss": 0.0,
                "take_profit": 0.0,
            }
        )
    try:
        filtered_rows = [_filter_row_for_table(engine, "trades_completed", row) for row in trade_rows]
        pd.DataFrame(filtered_rows).to_sql("trades_completed", engine, if_exists="append", index=False)
        LOGGER.info("Persisted %d trades for run %s to DB", len(trade_rows), run_id)
    except Exception as exc:
        LOGGER.warning("Failed to persist trades for run %s: %s", run_id, exc)


def _persist_results_to_db(result: Dict[str, Any], spec: Mapping[str, Any], ohlc_by_symbol: Optional[Dict[str, pd.DataFrame]] = None) -> None:
    strategy_cfg = spec.get("strategy", {}) or {}
    strategy_id = strategy_cfg.get("strategy_id") or "strategy"
    strategy_type = str(strategy_cfg.get("type") or "").strip().lower()
    run_id = spec.get("run_id") or strategy_cfg.get("run_id") or uuid.uuid4().hex
    asset_class = strategy_cfg.get("params", {}).get("asset_class") or strategy_cfg.get("asset_class") or ""
    data_spec = spec.get("data", {}) or {}
    timeframe = data_spec.get("timeframe")
    universe_label = spec.get("universe_label") or None

    class _DictSignal:
        def __init__(self, payload: Mapping[str, Any]) -> None:
            self.strategy_id = payload.get("strategy_id")
            self.symbol = payload.get("symbol")
            self.asset_class = payload.get("asset_class")
            self.side = payload.get("side")
            self.ts_open_utc = payload.get("ts_open_utc")
            self.qty = payload.get("qty", 0.0) or 0.0
            self.meta = _compact_meta(payload.get("meta", {}) or {})

    signals_by_symbol: Dict[str, List[_DictSignal]] = {}
    for sym, records in (result.get("signals") or {}).items():
        signals_by_symbol[sym] = [_DictSignal(r) for r in records]

    payload = build_backend_payload_for_java(
        strategy_id=strategy_id,
        run_id=run_id,
        asset_class=asset_class,
        universe=universe_label,
        timeframe=timeframe,
        signals_by_symbol=signals_by_symbol,
        ohlc_by_symbol=ohlc_by_symbol,
        config=spec.get("performance", {}) or {},
    )
    persist_payload_to_db(payload, spec, strategy_type=strategy_type)
