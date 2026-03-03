"""Utilities to execute high-level strategy specifications."""
from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd
import requests
from sqlalchemy import create_engine, inspect, text

from . import create_strategy
from .base import StrategySignal
from ..integrations import java_client
from ..filters.utils import apply_filter_stack, FilterValidationError
from ..filters.trade_filter_service import score_filter_rules
from ..performance.dca_builder import build_backend_payload_for_java
from ..datafeeds.asset_universe_adapter import resolve_asset_universe_adapter
from ..persistence.repositories import extract_dca_run_metrics
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
_OHLC_CACHE: "OrderedDict[str, tuple[pd.DataFrame, float]]" = OrderedDict()
_OHLC_CACHE_DEFAULT_MAX = 256
_OHLC_CACHE_DEFAULT_TTL_SECONDS = 900.0
_OHLC_CACHE_STATS = {"hits": 0, "misses": 0, "evictions": 0, "expirations": 0}
_OHLC_CACHE_CONFIG = {
    "enabled": True,
    "ttl_seconds": _OHLC_CACHE_DEFAULT_TTL_SECONDS,
    "max_items": _OHLC_CACHE_DEFAULT_MAX,
}

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def _perf_enabled() -> bool:
    flag = os.getenv("QE_PERF_TRACE", "")
    return str(flag).strip().lower() in {"1", "true", "yes", "on"}


def _perf_log(label: str, start: float) -> None:
    if _perf_enabled():
        elapsed = time.monotonic() - start
        print(f"[perf] {label} {elapsed:.2f}s", flush=True)


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


def _run_backtest_core(spec: Mapping[str, Any]) -> tuple[Dict[str, Any], Dict[str, pd.DataFrame]]:
    """Run the strategy backtest and return raw results plus OHLC cache."""

    t0 = time.monotonic()
    strategy_cfg = spec.get("strategy", {})
    strategy_type = strategy_cfg.get("type")
    strategy_id = strategy_cfg.get("strategy_id", "strategy")
    if not strategy_type:
        raise ValueError("Strategy specification must define a 'type'")
    strategy = create_strategy(
        strategy_type, strategy_id=strategy_id, params=strategy_cfg.get("params", {})
    )
    data_spec: Mapping[str, Any] = spec.get("data", {})
    optimization_cfg = spec.get("optimization", {}) or {}
    screening_cfg = optimization_cfg.get("screening") or spec.get("screening") or {}
    cache_cfg = optimization_cfg.get("cache_features") or {}
    cache_enabled = bool(cache_cfg) and cache_cfg.get("enabled", True) is not False
    universe: Iterable[Mapping[str, Any]] = _expand_universe(spec)
    signals_by_symbol: Dict[str, List[Dict[str, Any]]] = {}
    counts: Dict[str, int] = {}
    ohlc_by_symbol: Dict[str, pd.DataFrame] = {}
    cache_stats_start = dict(_OHLC_CACHE_STATS)
    for instrument in universe:
        symbol = instrument.get("symbol")
        if not symbol:
            LOGGER.warning("Skipping instrument without symbol: %s", instrument)
            continue
        asset_class = instrument.get(
            "asset_class",
            getattr(strategy, "asset_class", None) or strategy_cfg.get("asset_class"),
        )
        universe_adapter = resolve_asset_universe_adapter(asset_class)
        universe_rules = universe_adapter.get_rules(data_spec=data_spec, instrument=instrument)
        adapted_data_spec = universe_adapter.adapt_data_spec(data_spec, instrument)
        t_fetch = time.monotonic()
        df = _fetch_ohlc_for_symbol(symbol, asset_class, adapted_data_spec, instrument)
        _perf_log(f"strategy.fetch_ohlc {symbol} rows={len(df)}", t_fetch)
        if isinstance(screening_cfg, Mapping) and screening_cfg.get("enabled"):
            window_start = screening_cfg.get("window_start")
            window_end = screening_cfg.get("window_end")
            if window_start or window_end:
                df["ts"] = pd.to_datetime(df["ts"], utc=True)
                start_ts = pd.to_datetime(window_start, utc=True) if window_start else df["ts"].min()
                end_ts = pd.to_datetime(window_end, utc=True) if window_end else df["ts"].max()
                df = df[(df["ts"] >= start_ts) & (df["ts"] <= end_ts)].copy()
                LOGGER.info("Screening window applied for %s: %s -> %s", symbol, start_ts, end_ts)
            max_bars = screening_cfg.get("max_bars")
            if max_bars is not None:
                try:
                    max_bars_int = int(max_bars)
                except Exception:
                    max_bars_int = 0
                if max_bars_int > 0 and len(df) > max_bars_int:
                    df = df.tail(max_bars_int).copy()
                    LOGGER.info("Screening enabled: keeping last %d bars for %s", max_bars_int, symbol)
        filters_spec = strategy_cfg.get("filters") or spec.get("filters") or []
        filter_rules_spec = strategy_cfg.get("filter_rules") or spec.get("filter_rules") or []
        filter_rules_cfg = strategy_cfg.get("filter_rules_config") or spec.get("filter_rules_config") or {}
        if filters_spec or filter_rules_spec:
            t_filters = time.monotonic()
            df = df.copy()
            df_filter = df.copy()
            if "ts" in df_filter.columns:
                df_filter["ts"] = pd.to_datetime(df_filter["ts"], utc=True)
                df_filter = df_filter.set_index("ts")
            if cache_enabled:
                cache_key = json.dumps(
                    {
                        "symbol": symbol,
                        "asset_class": asset_class,
                        "data": data_spec,
                        "screening": screening_cfg,
                    },
                    sort_keys=True,
                    default=str,
                )
                df_filter.attrs["qe_cache_key"] = cache_key
                if "max_items" in cache_cfg:
                    df_filter.attrs["qe_cache_max_items"] = cache_cfg.get("max_items")
                if "ttl_seconds" in cache_cfg or "ttl" in cache_cfg:
                    df_filter.attrs["qe_cache_ttl_seconds"] = cache_cfg.get(
                        "ttl_seconds",
                        cache_cfg.get("ttl"),
                    )
            try:
                mask = None
                if filters_spec:
                    mask = apply_filter_stack(df_filter, filters_spec, symbol=symbol, logger=LOGGER, strict=True)
                if filter_rules_spec:
                    scoring = score_filter_rules(
                        df_filter,
                        filter_rules_spec,
                        symbol=symbol,
                        strict=True,
                        min_score=filter_rules_cfg.get("min_score"),
                        min_score_pct=filter_rules_cfg.get("min_score_pct"),
                    )
                    score_mask = scoring["final_mask"]
                    df["_filter_score"] = scoring["score_pct"].reindex(df_filter.index).fillna(0.0).to_numpy()
                    mask = score_mask if mask is None else (mask & score_mask)
            except FilterValidationError as exc:
                LOGGER.error("Filters failed for %s: %s", symbol, exc)
                raise
            try:
                allowed = int(mask.fillna(False).sum())
                total = int(len(mask))
                pct = (allowed / total * 100.0) if total else 0.0
                LOGGER.info("Filters summary for %s: %d/%d bars allowed (%.1f%%)", symbol, allowed, total, pct)
            except Exception:
                LOGGER.info("Filters summary for %s: unable to compute coverage", symbol)
            if "ts" in df.columns:
                ts_index = pd.to_datetime(df["ts"], utc=True)
                df["_filter_ok"] = mask.reindex(ts_index, fill_value=False).to_numpy()
            else:
                df["_filter_ok"] = mask.reindex(df.index, fill_value=False)
            _perf_log(f"strategy.apply_filters {symbol}", t_filters)
        ohlc_by_symbol[symbol] = df.copy()
        t_signals = time.monotonic()
        context = {"symbol": symbol, "asset_class": asset_class, "screening": screening_cfg}
        context = universe_adapter.adapt_context(context, universe_rules)
        signals = strategy.backtest(df, context)
        serialized = [_serialize_signal(sig) for sig in signals]
        signals_by_symbol[symbol] = serialized
        counts[symbol] = len(serialized)
        _perf_log(f"strategy.backtest_signals {symbol} signals={len(serialized)}", t_signals)
    _log_ohlc_cache_stats_delta(cache_stats_start)
    result = {
        "strategy_id": strategy_id,
        "strategy_type": strategy_type,
        "counts": counts,
        "signals": signals_by_symbol,
    }
    _perf_log("strategy.total", t0)
    return result, ohlc_by_symbol


def run_backtest_from_spec(spec: Mapping[str, Any]) -> Dict[str, Any]:
    """Execute ``strategy.backtest`` for every instrument defined in ``spec``."""

    result, ohlc_by_symbol = _run_backtest_core(spec)
    _persist_results_if_requested(result, spec.get("output"))
    _persist_results_to_db(result, spec, ohlc_by_symbol)
    return result


def run_backtest_with_payload(spec: Mapping[str, Any]) -> Dict[str, Any]:
    """Run the strategy backtest and return results plus backend payload."""

    result, ohlc_by_symbol = _run_backtest_core(spec)
    payload = _build_payload_for_result(result, spec, ohlc_by_symbol)
    return {"result": result, "payload": payload}


def _fetch_ohlc_for_symbol(
    symbol: str,
    asset_class: Optional[str],
    data_spec: Mapping[str, Any],
    instrument_spec: Mapping[str, Any],
) -> pd.DataFrame:
    # Combine top-level data spec with instrument overrides (instrument wins).
    merged_spec = {**data_spec, **instrument_spec}
    cache_key = json.dumps(
        {"symbol": symbol, "asset_class": asset_class, "data": merged_spec},
        sort_keys=True,
        default=str,
    )
    cache_enabled, ttl_seconds, max_items = _resolve_ohlc_cache_settings(merged_spec)
    if cache_enabled and max_items > 0:
        cached = _cache_ohlc_get(cache_key, ttl_seconds)
        if cached is not None:
            return cached.copy()
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
    if cache_enabled and max_items > 0:
        _cache_ohlc_set(cache_key, df.copy(), max_items, ttl_seconds)
    return df


def _resolve_ohlc_cache_settings(merged_spec: Mapping[str, Any]) -> tuple[bool, float, int]:
    cache_cfg = merged_spec.get("ohlc_cache")
    if cache_cfg is None:
        cache_cfg = merged_spec.get("cache_ohlc")
    if cache_cfg is None:
        cache_cfg = merged_spec.get("cache")
    if not isinstance(cache_cfg, Mapping):
        cache_cfg = {}
    enabled = cache_cfg.get("enabled", True) is not False
    ttl_seconds = cache_cfg.get("ttl_seconds", cache_cfg.get("ttl", _OHLC_CACHE_DEFAULT_TTL_SECONDS))
    max_items = cache_cfg.get("max_items", _OHLC_CACHE_DEFAULT_MAX)
    try:
        ttl_seconds = float(ttl_seconds)
    except Exception:
        ttl_seconds = _OHLC_CACHE_DEFAULT_TTL_SECONDS
    try:
        max_items = int(max_items)
    except Exception:
        max_items = _OHLC_CACHE_DEFAULT_MAX
    if max_items <= 0:
        enabled = False
    if ttl_seconds <= 0:
        ttl_seconds = 0.0
    _OHLC_CACHE_CONFIG.update(
        {"enabled": enabled, "ttl_seconds": ttl_seconds, "max_items": max_items}
    )
    return enabled, ttl_seconds, max_items


def _prune_ohlc_cache(ttl_seconds: float) -> int:
    if ttl_seconds <= 0 or not _OHLC_CACHE:
        return 0
    now = time.monotonic()
    expired = [key for key, (_, created) in _OHLC_CACHE.items() if now - created > ttl_seconds]
    for key in expired:
        _OHLC_CACHE.pop(key, None)
        _OHLC_CACHE_STATS["expirations"] += 1
    return len(expired)


def _cache_ohlc_get(cache_key: str, ttl_seconds: float) -> Optional[pd.DataFrame]:
    _prune_ohlc_cache(ttl_seconds)
    entry = _OHLC_CACHE.get(cache_key)
    if entry is None:
        _OHLC_CACHE_STATS["misses"] += 1
        return None
    df, created = entry
    if ttl_seconds > 0:
        age = time.monotonic() - created
        if age > ttl_seconds:
            _OHLC_CACHE.pop(cache_key, None)
            _OHLC_CACHE_STATS["expirations"] += 1
            _OHLC_CACHE_STATS["misses"] += 1
            return None
    _OHLC_CACHE.move_to_end(cache_key)
    _OHLC_CACHE_STATS["hits"] += 1
    return df


def _cache_ohlc_set(
    cache_key: str,
    df: pd.DataFrame,
    max_items: int,
    ttl_seconds: float,
) -> None:
    _prune_ohlc_cache(ttl_seconds)
    _OHLC_CACHE[cache_key] = (df, time.monotonic())
    _OHLC_CACHE.move_to_end(cache_key)
    while len(_OHLC_CACHE) > max_items:
        _OHLC_CACHE.popitem(last=False)
        _OHLC_CACHE_STATS["evictions"] += 1


def _log_ohlc_cache_stats_delta(start_stats: Mapping[str, int]) -> None:
    delta = {
        key: _OHLC_CACHE_STATS.get(key, 0) - start_stats.get(key, 0)
        for key in _OHLC_CACHE_STATS
    }
    if any(value > 0 for value in delta.values()):
        LOGGER.info(
            "OHLC cache stats: hits=%d misses=%d expirations=%d evictions=%d size=%d max=%d ttl=%.0fs",
            delta.get("hits", 0),
            delta.get("misses", 0),
            delta.get("expirations", 0),
            delta.get("evictions", 0),
            len(_OHLC_CACHE),
            _OHLC_CACHE_CONFIG.get("max_items", _OHLC_CACHE_DEFAULT_MAX),
            _OHLC_CACHE_CONFIG.get("ttl_seconds", _OHLC_CACHE_DEFAULT_TTL_SECONDS),
        )


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
        LOGGER.info(
            "Delta source skipped for %s: base_uri_present=%s deltalake_available=%s",
            symbol,
            bool(base_uri),
            DeltaTable is not None,
        )
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
    if "delta_exchange" in spec:
        exchanges = [e.strip() for e in str(spec.get("delta_exchange", "")).split(",") if e.strip()]
    elif spec.get("exchange"):
        exchanges = [str(spec.get("exchange")).strip()]
    else:
        env_exchange = os.getenv("DELTA_EXCHANGE") or "GENERIC"
        exchanges = [env_exchange]
    exchanges = [str(ex).strip().upper() or "GENERIC" for ex in exchanges] or ["GENERIC"]
    market_type = spec.get("delta_market_type") or spec.get("market_type") or os.getenv("DELTA_MARKET_TYPE") or "SPOT"
    market_type = str(market_type).strip().upper() or "SPOT"
    quotes: List[str] = []
    if "delta_quotes" in spec:
        quotes = [q.strip() for q in str(spec["delta_quotes"]).split(",") if q.strip()]
    elif spec.get("currency") or spec.get("delta_quote"):
        quotes = [q.strip() for q in str(spec.get("currency") or spec.get("delta_quote")).split(",") if q.strip()]
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
    LOGGER.info(
        "Delta lookup config for %s: base_uri=%s asset_dir=%s table_name=%s brokers=%s market_type=%s exchanges=%s quotes=%s timeframe=%s start=%s end=%s min_coverage=%.1f%%",
        symbol,
        str(base_uri).rstrip("/"),
        asset_dir,
        table_name,
        _delta_brokers(asset_class, spec),
        market_type,
        exchanges,
        quotes,
        timeframe,
        start_str,
        end_str,
        min_coverage * 100,
    )
    brokers = _delta_brokers(asset_class, spec)
    asset_upper = (asset_class_value or "").upper()
    use_exchange_dir = asset_upper != "CRYPTO"
    attempted_paths: List[str] = []
    attempt_count = 0
    for broker in (brokers or [""]):
        for quote in quotes:
            for exchange in exchanges:
                use_exchange = use_exchange_dir and exchange != "GENERIC"
                parts = [base_uri.rstrip("/")]
                if delta_prefix:
                    parts.append(delta_prefix)
                parts.append(asset_dir.strip("/"))
                if broker:
                    parts.append(str(broker).strip().upper())
                if market_type:
                    parts.append(market_type.strip("/"))
                if use_exchange:
                    parts.append(exchange.strip("/"))
                parts.extend([quote.strip("/"), table_name])
                table_path = "/".join(parts)
                attempt_count += 1
                attempted_paths.append(table_path)
                LOGGER.info("Trying Delta path (%d): %s", attempt_count, table_path)
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
        "No Delta data for %s (asset_class=%s, asset_dir=%s, attempts=%d); fallback to other sources",
        symbol,
        (asset_class_value or "UNKNOWN"),
        asset_dir,
        attempt_count,
    )
    if attempted_paths:
        LOGGER.info("Delta attempted paths for %s: %s", symbol, attempted_paths)
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
        expected = pd.date_range(start_dt.normalize(), end_dt.normalize(), freq=freq, tz="UTC")
        observed = ts.dt.to_period("W").dt.start_time.dt.tz_localize("UTC")
    else:
        expected = _expected_from_market_calendar(cal, start_dt, end_dt, freq) if cal else None
        if expected is None:
            # Align expected bins to timeframe boundaries so they match observed floor(freq).
            try:
                start_aligned = start_dt.floor(freq)
                end_aligned = end_dt.floor(freq)
            except Exception:
                start_aligned = start_dt
                end_aligned = end_dt
            expected = pd.date_range(start_aligned, end_aligned, freq=freq)
        if expected.tz is None:
            expected = expected.tz_localize("UTC")
        else:
            expected = expected.tz_convert("UTC")
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

__all__ = ["load_strategy_spec", "run_backtest_from_spec", "run_backtest_with_payload", "persist_payload_to_db"]


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

        extra = run.get("extra", {}) if isinstance(run.get("extra"), Mapping) else {}
        dca_metrics = extract_dca_run_metrics(extra) if strategy_type.startswith("dca") else {}
        if dca_metrics:
            metric_rows = [
                {"run_id": run_id, "fold": None, "metric_name": name, "metric_value": value}
                for name, value in dca_metrics.items()
            ]
            filtered_metric_rows = [_filter_row_for_table(engine, "run_metrics", row) for row in metric_rows]
            metric_df = pd.DataFrame(filtered_metric_rows)
            if not metric_df.empty:
                with engine.begin() as conn:
                    conn.execute(text("DELETE FROM run_metrics WHERE run_id = :run_id AND fold IS NULL"), {"run_id": run_id})
                metric_df.to_sql("run_metrics", engine, if_exists="append", index=False)
                LOGGER.info("Persisted %d DCA run metrics for run %s", len(filtered_metric_rows), run_id)
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
        trade_asset = str(t.get("assetClass") or trade_ctx.get("asset_class") or "").upper()
        exchange_value = None if trade_asset == "ETF" else trade_ctx.get("exchange")
        trade_rows.append(
            {
                "strategy_name": t.get("strategyId"),
                "run_id": t.get("runId") or run_id,
                "symbol": t.get("symbol"),
                "asset_class": t.get("assetClass"),
                "broker": trade_ctx.get("broker"),
                "exchange": exchange_value,
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

    _persist_stress_tests_payload(engine, payload, run_id=run_id)


def _persist_results_to_db(result: Dict[str, Any], spec: Mapping[str, Any], ohlc_by_symbol: Optional[Dict[str, pd.DataFrame]] = None) -> None:
    payload = _build_payload_for_result(result, spec, ohlc_by_symbol)
    strategy_type = str((spec.get("strategy", {}) or {}).get("type") or "").strip().lower()
    persist_payload_to_db(payload, spec, strategy_type=strategy_type)


def _persist_stress_tests_payload(engine, payload: Mapping[str, Any], *, run_id: str) -> None:
    stress_tests = payload.get("stress_tests")
    if not isinstance(stress_tests, Mapping):
        return

    run = payload.get("run", {}) if isinstance(payload, Mapping) else {}
    base_row = {
        "strategy_id": run.get("strategyId"),
        "run_id": run_id,
        "asset_class": run.get("assetClass"),
        "symbol": run.get("symbol"),
        "timeframe": run.get("timeframe"),
    }

    rows: List[Dict[str, Any]] = []
    for mode in ("monte_carlo", "scenarios"):
        result = stress_tests.get(mode)
        if not isinstance(result, Mapping):
            continue
        row = {
            **base_row,
            "mode": mode,
            "payload_json": json.dumps(result, ensure_ascii=False),
        }
        rows.append(row)

    if not rows:
        LOGGER.warning("Stress tests payload present but no rows to persist for run %s", run_id)
        return

    try:
        filtered_rows = [_filter_row_for_table(engine, "stress_test_result", row) for row in rows]
        pd.DataFrame(filtered_rows).to_sql("stress_test_result", engine, if_exists="append", index=False)
        LOGGER.info("Persisted %d stress test payloads for run %s", len(rows), run_id)
        return
    except Exception as exc:
        message = str(exc)
        if "payload_json" not in message and "Data too long" not in message and "1406" not in message:
            LOGGER.warning("Failed to persist stress tests for run %s: %s", run_id, exc)
            return
        LOGGER.warning("Stress tests payload too large; attempting compact insert for run %s", run_id)

    max_chars = _resolve_stress_test_max_chars()
    compact_rows: List[Dict[str, Any]] = []
    for row in rows:
        try:
            raw_payload = json.loads(row.get("payload_json", "{}"))
        except Exception:
            raw_payload = {}
        mode = row.get("mode")
        compact_payload = _compact_stress_test_payload(raw_payload, mode=mode, max_chars=max_chars)
        compact_rows.append({**row, "payload_json": json.dumps(compact_payload, ensure_ascii=False)})

    try:
        filtered_rows = [_filter_row_for_table(engine, "stress_test_result", row) for row in compact_rows]
        pd.DataFrame(filtered_rows).to_sql("stress_test_result", engine, if_exists="append", index=False)
        LOGGER.info("Persisted compact stress test payloads for run %s", run_id)
    except Exception as exc:
        LOGGER.warning("Failed to persist compact stress tests for run %s: %s", run_id, exc)


def _resolve_stress_test_max_chars() -> int:
    raw = os.getenv("QE_STRESS_TEST_MAX_JSON_CHARS")
    if raw:
        try:
            value = int(raw)
            if value > 0:
                return value
        except Exception:
            return 60000
    return 60000


def _compact_stress_test_payload(
    payload: Mapping[str, Any],
    *,
    mode: Optional[str],
    max_chars: int,
) -> Dict[str, Any]:
    def _json_len(obj: Any) -> int:
        try:
            return len(json.dumps(obj, ensure_ascii=False))
        except Exception:
            return max_chars + 1

    def _reduce_equity_curves(curves: Any, *, max_curves: int, stride: int) -> List[List[float]]:
        if not isinstance(curves, list):
            return []
        max_curves = max(0, int(max_curves))
        stride = max(1, int(stride))
        trimmed = curves[:max_curves] if max_curves else []
        reduced: List[List[float]] = []
        for curve in trimmed:
            if not isinstance(curve, list):
                continue
            sliced = curve[::stride]
            if curve and (not sliced or sliced[-1] != curve[-1]):
                sliced.append(curve[-1])
            reduced.append(sliced)
        return reduced

    compact: Dict[str, Any] = {
        "metrics": payload.get("metrics"),
        "parameters": payload.get("parameters"),
        "warnings": payload.get("warnings"),
    }

    distributions = payload.get("distributions") if isinstance(payload, Mapping) else None
    if mode == "monte_carlo":
        curves = None
        if isinstance(distributions, Mapping):
            curves = distributions.get("equity_curves")
        compact["distributions"] = {
            "equity_curves": _reduce_equity_curves(curves, max_curves=20, stride=10)
        }
    elif mode == "scenarios":
        # Scenarios distributions can be large; drop by default
        pass

    if _json_len(compact) > max_chars:
        compact.pop("distributions", None)

    if _json_len(compact) > max_chars:
        compact = {"note": "payload truncated", "metrics": payload.get("metrics")}

    return compact


def _build_payload_for_result(
    result: Dict[str, Any],
    spec: Mapping[str, Any],
    ohlc_by_symbol: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[str, Any]:
    strategy_cfg = spec.get("strategy", {}) or {}
    strategy_id = strategy_cfg.get("strategy_id") or "strategy"
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

    universe_adapter = resolve_asset_universe_adapter(asset_class)
    performance_cfg = dict(spec.get("performance", {}) or {})
    performance_cfg.setdefault("universe_rules_version", universe_adapter.rules_version)

    return build_backend_payload_for_java(
        strategy_id=strategy_id,
        run_id=run_id,
        asset_class=asset_class,
        universe=universe_label,
        timeframe=timeframe,
        signals_by_symbol=signals_by_symbol,
        ohlc_by_symbol=ohlc_by_symbol,
        config=performance_cfg,
    )
