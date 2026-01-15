"""Shared utilities for validating and applying filter stacks."""
from __future__ import annotations

from dataclasses import dataclass
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import logging
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from collections import OrderedDict
import json
import time

import pandas as pd

from . import filters_registry
from . import volume_profile as vol_profile
from . import structure_ict
from . import stat_prob


@dataclass(frozen=True)
class FilterSummary:
    summary: str
    requires: str


FILTER_SUMMARIES: Dict[str, FilterSummary] = {
    "adx": FilterSummary("ADX above threshold (trend strength).", "high, low, close"),
    "atr": FilterSummary("ATR/close within a min/max band.", "high, low, close"),
    "ema_slope": FilterSummary("EMA slope above threshold.", "close"),
    "volume_surge": FilterSummary("Detect volume spikes by z-score or ratio.", "volume"),
    "vwap_side": FilterSummary("Price above/below anchored VWAP.", "close; levels optional"),
    "poc_distance": FilterSummary("Price within distance of active POC.", "close; levels required"),
    "liquidity_sweep": FilterSummary("Sweeps recent highs/lows (ICT).", "high, low, close"),
    "bos": FilterSummary("Break of structure up/down.", "high, low, close; levels optional"),
    "mss": FilterSummary("Market structure shift (BOS flip).", "high, low, close; levels optional"),
    "session_time": FilterSummary("Allow trades in a session window.", "DatetimeIndex"),
    "day_of_week": FilterSummary("Allow/deny specific weekdays.", "DatetimeIndex"),
    "day_of_month": FilterSummary("Allow first/last N days of month.", "DatetimeIndex"),
    "month_of_year": FilterSummary("Allow/deny specific months.", "DatetimeIndex"),
    "intraday_time": FilterSummary("Allow trades in time-of-day window.", "DatetimeIndex"),
    "k_consecutive": FilterSummary("K consecutive candles in same direction.", "open, close"),
    "seasonality_bin": FilterSummary("Allow specific seasonal bins.", "DatetimeIndex; stats optional"),
    "hurst_regime": FilterSummary("Hurst exponent regime filter.", "close"),
    "entropy_window": FilterSummary("Directional entropy window filter.", "close (+open optional)"),
    "daily_loss_cap": FilterSummary("Lockout after daily loss cap.", "DatetimeIndex + pnl/equity/ret"),
    "daily_trades_cap": FilterSummary("Limit number of entries per day.", "DatetimeIndex + signal_col"),
    "cooldown_bars": FilterSummary("Cooldown after each signal.", "signal_col"),
    "atr_risk_gate": FilterSummary("Block when ATR/close too high.", "high, low, close"),
    "equity_dd_lockout": FilterSummary("Lockout after equity drawdown.", "equity"),
    "benford_law": FilterSummary("Benford MSE below threshold.", "close"),
    "cycles": FilterSummary("Low dominant autocorrelation (cycles).", "close"),
    "donchian_channels": FilterSummary("Donchian breakout filter.", "high, low, close"),
    "liquidity_cmf": FilterSummary("Chaikin Money Flow threshold.", "high, low, close, volume"),
    "market_manipulation": FilterSummary(
        "Block when entropy/kurtosis/volatility indicate anomalies.", "open, high, low, close"
    ),
    "htf_poi": FilterSummary(
        "Price interacts with active HTF zones/POI levels.", "close; levels required"
    ),
    "orderflow_delta": FilterSummary(
        "Orderflow delta/ratio filter (buy vs sell volume).", "buy/sell volume or delta column"
    ),
    "macro_cot_oi": FilterSummary(
        "Macro COT/OI alignment filter.", "cot_bias/oi_change columns"
    ),
    "lower_timeframe_confluence": FilterSummary(
        "Confluence score across momentum/ADX/VWAP/delta/EMA.", "open/high/low/close (+volume/buy/sell optional)"
    ),
    "psychologic_and_news": FilterSummary(
        "Block during news/psychologic blackout windows.", "DatetimeIndex (+news_col optional)"
    ),
    "ict_poi": FilterSummary(
        "ICT POI (levels/IFVG/fib) confluence filter.", "close (+levels/high/low optional)"
    ),
    "statistical_arbitrage": FilterSummary("Omega/Info ratio threshold.", "close"),
    "psychologic_ulcer": FilterSummary("Ulcer index below threshold.", "close"),
    "stationarity": FilterSummary("Low lag-1 autocorrelation.", "close"),
    "volatility": FilterSummary("Entropy (and optional ATR) threshold.", "high, low, close"),
    "ema_structure": FilterSummary("EMA stack and trend validation.", "close"),
    "rsi_entry": FilterSummary("RSI threshold filter.", "close"),
    "macd_entry": FilterSummary("MACD cross filter.", "close"),
    "volume_above_average": FilterSummary("Volume above its rolling average.", "volume"),
    "fractal_analysis": FilterSummary("Hurst + skew/kurtosis bounds.", "close"),
    "mean_reversion": FilterSummary("BB + Keltner mean-reversion trigger.", "high, low, close"),
    "contradictory_signals": FilterSummary("Block conflicting indicator signals.", "high, low, close"),
    "linear_regression_macd_cross": FilterSummary("Predict MACD cross via regression.", "close"),
    "atr_rising": FilterSummary("ATR rising vs previous bar/lookback.", "high, low, close"),
    "market_regime": FilterSummary("Detect trend/range/compression regimes.", "high, low, close"),
    "trend": FilterSummary("Trend direction via HH/HL or EMA.", "high, low, close"),
    "biais_institutional": FilterSummary("EMA/VWAP + optional macro filters.", "close (+volume optional)"),
    "stats_gate": FilterSummary("Gate using persisted market stats.", "market_stats table"),
}

_FILTER_CACHE: "OrderedDict[tuple, tuple[pd.Series, float]]" = OrderedDict()
_CACHE_DEFAULT_MAX = 2048
_CACHE_DEFAULT_TTL_SECONDS = 900.0
_FILTER_CACHE_STATS = {"hits": 0, "misses": 0, "evictions": 0, "expirations": 0}
_FILTER_CACHE_CONFIG = {
    "max_items": _CACHE_DEFAULT_MAX,
    "ttl_seconds": _CACHE_DEFAULT_TTL_SECONDS,
}


def _make_cache_key(df: pd.DataFrame, flt_type: str, params: Mapping[str, Any]) -> Optional[tuple]:
    cache_key = df.attrs.get("qe_cache_key")
    if cache_key is None:
        return None
    try:
        params_key = json.dumps(params, sort_keys=True, default=str)
    except Exception:
        return None
    return (cache_key, flt_type, params_key)


def _prune_filter_cache(ttl_seconds: float) -> int:
    if ttl_seconds <= 0 or not _FILTER_CACHE:
        return 0
    now = time.monotonic()
    expired = [k for k, (_, created) in _FILTER_CACHE.items() if now - created > ttl_seconds]
    for k in expired:
        _FILTER_CACHE.pop(k, None)
        _FILTER_CACHE_STATS["expirations"] += 1
    return len(expired)


def _cache_get(key: tuple, ttl_seconds: float) -> Optional[pd.Series]:
    _prune_filter_cache(ttl_seconds)
    entry = _FILTER_CACHE.get(key)
    if entry is None:
        _FILTER_CACHE_STATS["misses"] += 1
        return None
    series, created = entry
    if ttl_seconds > 0:
        age = time.monotonic() - created
        if age > ttl_seconds:
            _FILTER_CACHE.pop(key, None)
            _FILTER_CACHE_STATS["expirations"] += 1
            _FILTER_CACHE_STATS["misses"] += 1
            return None
    _FILTER_CACHE.move_to_end(key)
    _FILTER_CACHE_STATS["hits"] += 1
    return series


def _cache_set(key: tuple, series: pd.Series, max_items: int, ttl_seconds: float) -> None:
    _prune_filter_cache(ttl_seconds)
    _FILTER_CACHE[key] = (series, time.monotonic())
    _FILTER_CACHE.move_to_end(key)
    while len(_FILTER_CACHE) > max_items:
        _FILTER_CACHE.popitem(last=False)
        _FILTER_CACHE_STATS["evictions"] += 1


def _log_filter_cache_stats_delta(
    start_stats: Mapping[str, int],
    logger: logging.Logger,
) -> None:
    delta = {
        key: _FILTER_CACHE_STATS.get(key, 0) - start_stats.get(key, 0)
        for key in _FILTER_CACHE_STATS
    }
    if any(value > 0 for value in delta.values()):
        logger.info(
            "Filter cache stats: hits=%d misses=%d expirations=%d evictions=%d size=%d max=%d ttl=%.0fs",
            delta.get("hits", 0),
            delta.get("misses", 0),
            delta.get("expirations", 0),
            delta.get("evictions", 0),
            len(_FILTER_CACHE),
            _FILTER_CACHE_CONFIG.get("max_items", _CACHE_DEFAULT_MAX),
            _FILTER_CACHE_CONFIG.get("ttl_seconds", _CACHE_DEFAULT_TTL_SECONDS),
        )


class FilterValidationError(ValueError):
    """Raised when a filter cannot be evaluated due to missing inputs."""


def _ensure_datetime_index(df: pd.DataFrame, errors: List[str]) -> None:
    if not isinstance(df.index, pd.DatetimeIndex):
        errors.append("requires a DatetimeIndex")


def _validate_timezone_param(params: Mapping[str, Any], errors: List[str]) -> None:
    tz = params.get("tz")
    if tz is None:
        return
    if not isinstance(tz, str):
        errors.append("tz must be a string timezone identifier")
        return
    try:
        ZoneInfo(tz)
    except ZoneInfoNotFoundError:
        errors.append(f"unknown timezone '{tz}'")


def _require_columns(df: pd.DataFrame, columns: Iterable[str], errors: List[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        errors.append(f"missing columns: {', '.join(missing)}")


def _require_levels_repo(errors: List[str]) -> None:
    if vol_profile.lvl_repo is None or vol_profile.lvl_helpers is None:
        errors.append("levels repository not available")


def _require_stats_repo(errors: List[str]) -> None:
    if stat_prob.stats_repo is None:
        errors.append("stats repository not available")


def _normalize_filter_spec(raw: Mapping[str, Any]) -> Tuple[str, Dict[str, Any]]:
    flt_type = str(raw.get("type") or "").strip()
    if not flt_type:
        raise FilterValidationError("filter spec missing 'type'")
    params = raw.get("params") or {}
    if not isinstance(params, Mapping):
        raise FilterValidationError(f"filter '{flt_type}' params must be a mapping")
    return flt_type, dict(params)


def _validate_filter_inputs(
    df: pd.DataFrame,
    flt_type: str,
    params: Mapping[str, Any],
    symbol: Optional[str],
) -> List[str]:
    errors: List[str] = []

    if flt_type in {"adx", "atr", "atr_risk_gate"}:
        _require_columns(df, ["high", "low", "close"], errors)
    elif flt_type == "ema_slope":
        _require_columns(df, ["close"], errors)
    elif flt_type == "volume_surge":
        volume_col = params.get("volume_col", "volume")
        _require_columns(df, [volume_col], errors)
    elif flt_type == "vwap_side":
        price_col = params.get("price_col", "close")
        _require_columns(df, [price_col], errors)
        if params.get("from_levels", True):
            _require_levels_repo(errors)
            if not (params.get("symbol") or symbol):
                errors.append("requires symbol for levels lookup")
    elif flt_type == "poc_distance":
        _require_columns(df, ["close"], errors)
        _require_levels_repo(errors)
        if not (params.get("symbol") or symbol):
            errors.append("requires symbol for levels lookup")
    elif flt_type in {"liquidity_sweep", "bos", "mss"}:
        _require_columns(df, ["high", "low", "close"], errors)
        if params.get("use_levels", True):
            _require_levels_repo(errors)
            if not (params.get("symbol") or symbol):
                errors.append("requires symbol for levels lookup")
    elif flt_type in {
        "session_time",
        "day_of_week",
        "day_of_month",
        "month_of_year",
        "intraday_time",
    }:
        _ensure_datetime_index(df, errors)
        _validate_timezone_param(params, errors)
    elif flt_type == "k_consecutive":
        use_body = params.get("use_body", True)
        _require_columns(df, ["close"], errors)
        if use_body:
            _require_columns(df, ["open"], errors)
    elif flt_type == "seasonality_bin":
        _ensure_datetime_index(df, errors)
        if params.get("min_winrate") is not None:
            _require_stats_repo(errors)
            if not (params.get("symbol") or symbol):
                errors.append("requires symbol for seasonality profile lookup")
    elif flt_type == "hurst_regime":
        price_col = params.get("price_col", "close")
        _require_columns(df, [price_col], errors)
    elif flt_type == "entropy_window":
        use_body = params.get("use_body", False)
        _require_columns(df, ["close"], errors)
        if use_body:
            _require_columns(df, ["open"], errors)
    elif flt_type == "daily_loss_cap":
        _ensure_datetime_index(df, errors)
        mode = params.get("mode", "pnl")
        if mode == "pnl":
            pnl_col = params.get("pnl_col", "pnl")
            _require_columns(df, [pnl_col], errors)
        elif mode == "equity":
            equity_col = params.get("equity_col", "equity")
            _require_columns(df, [equity_col], errors)
        elif mode == "ret_notional":
            ret_col = params.get("ret_col", "ret")
            _require_columns(df, [ret_col], errors)
            if params.get("notional") is None:
                errors.append("requires notional when mode=ret_notional")
        else:
            errors.append(f"unsupported mode '{mode}'")
    elif flt_type == "daily_trades_cap":
        _ensure_datetime_index(df, errors)
        signal_col = params.get("signal_col")
        if not signal_col:
            errors.append("requires signal_col")
        else:
            _require_columns(df, [signal_col], errors)
    elif flt_type == "cooldown_bars":
        signal_col = params.get("signal_col")
        if not signal_col:
            errors.append("requires signal_col")
        else:
            _require_columns(df, [signal_col], errors)
    elif flt_type == "equity_dd_lockout":
        equity_col = params.get("equity_col", "equity")
        _require_columns(df, [equity_col], errors)
    elif flt_type in {"benford_law", "cycles", "statistical_arbitrage", "psychologic_ulcer", "stationarity"}:
        price_col = params.get("price_col", "close")
        _require_columns(df, [price_col], errors)
        if flt_type == "benford_law":
            stype = str(params.get("series_type", "range")).lower()
            if stype in {"range", "hl", "delta_range", "range_delta"}:
                _require_columns(df, [params.get("high_col", "high"), params.get("low_col", "low")], errors)
            if stype in {"body", "oc"}:
                _require_columns(df, [params.get("open_col", "open")], errors)
            if stype in {"volume", "vol"}:
                _require_columns(df, [params.get("volume_col", "volume")], errors)
    elif flt_type == "donchian_channels":
        _require_columns(df, ["high", "low", "close"], errors)
    elif flt_type == "liquidity_cmf":
        _require_columns(df, ["high", "low", "close", params.get("volume_col", "volume")], errors)
    elif flt_type == "market_manipulation":
        _require_columns(
            df,
            [
                params.get("high_col", "high"),
                params.get("low_col", "low"),
                params.get("close_col", "close"),
            ],
            errors,
        )
    elif flt_type == "htf_poi":
        _require_columns(df, [params.get("close_col", "close")], errors)
        mode = str(params.get("mode", "in_zone")).lower()
        if mode == "distance" and params.get("max_distance") is None:
            errors.append("requires max_distance when mode='distance'")
        allow_if_missing = bool(params.get("allow_if_missing", True))
        if not allow_if_missing:
            _require_levels_repo(errors)
            if not (params.get("symbol") or symbol):
                errors.append("requires symbol for levels lookup")
    elif flt_type == "orderflow_delta":
        allow_if_missing = bool(params.get("allow_if_missing", True))
        delta_col = params.get("delta_col")
        buy_col = params.get("buy_col", "buy_volume")
        sell_col = params.get("sell_col", "sell_volume")
        mode = str(params.get("mode", "delta")).lower()
        if not allow_if_missing:
            if delta_col:
                _require_columns(df, [delta_col], errors)
            else:
                _require_columns(df, [buy_col, sell_col], errors)
            if mode == "ratio":
                _require_columns(df, [params.get("volume_col", "volume")], errors)
    elif flt_type == "macro_cot_oi":
        allow_if_missing = bool(params.get("allow_if_missing", True))
        if not allow_if_missing:
            _require_columns(
                df,
                [
                    params.get("cot_col", "cot_bias"),
                    params.get("oi_col", "oi_change"),
                ],
                errors,
            )
    elif flt_type == "lower_timeframe_confluence":
        allow_if_missing = bool(params.get("allow_if_missing", True))
        if not allow_if_missing:
            _require_columns(df, [params.get("close_col", "close")], errors)
            _require_columns(
                df,
                [
                    params.get("high_col", "high"),
                    params.get("low_col", "low"),
                ],
                errors,
            )
            if params.get("vwap_max_dev") is not None:
                _require_columns(df, [params.get("vwap_price_col", "close")], errors)
                if params.get("vwap_volume_col", "volume") not in df.columns:
                    errors.append("missing columns: volume")
            if params.get("delta_ratio") is not None:
                _require_columns(df, [params.get("volume_col", "volume")], errors)
            if params.get("delta_col"):
                _require_columns(df, [params.get("delta_col")], errors)
            elif params.get("delta_min") is not None or params.get("delta_ratio") is not None:
                _require_columns(
                    df,
                    [
                        params.get("buy_col", "buy_volume"),
                        params.get("sell_col", "sell_volume"),
                    ],
                    errors,
                )
    elif flt_type == "psychologic_and_news":
        allow_if_missing = bool(params.get("allow_if_missing", True))
        if not isinstance(df.index, pd.DatetimeIndex):
            if not allow_if_missing:
                errors.append("requires a DatetimeIndex")
        news_col = params.get("news_col")
        if news_col and news_col not in df.columns:
            if not allow_if_missing:
                errors.append(f"missing columns: {news_col}")
    elif flt_type == "ict_poi":
        allow_if_missing = bool(params.get("allow_if_missing", True))
        _require_columns(df, [params.get("close_col", "close")], errors)
        include_ifvg = bool(params.get("include_ifvg", False))
        include_fib = bool(params.get("include_fib", False))
        include_ob = bool(params.get("include_ob", False))
        include_breaker = bool(params.get("include_breaker", False))
        include_model10 = bool(params.get("include_model10", False))
        if include_ifvg or include_fib or include_ob or include_breaker or include_model10:
            _require_columns(
                df,
                [
                    params.get("open_col", "open"),
                    params.get("high_col", "high"),
                    params.get("low_col", "low"),
                ],
                errors,
            )
        lvl_types = params.get("level_types")
        if lvl_types and not allow_if_missing:
            _require_levels_repo(errors)
            if not (params.get("symbol") or symbol):
                errors.append("requires symbol for levels lookup")
    elif flt_type == "volatility":
        _require_columns(df, ["high", "low", "close"], errors)
    elif flt_type in {"ema_structure", "rsi_entry", "macd_entry", "fractal_analysis"}:
        price_col = params.get("price_col", "close")
        _require_columns(df, [price_col], errors)
    elif flt_type == "volume_above_average":
        _require_columns(df, [params.get("volume_col", "volume")], errors)
    elif flt_type == "mean_reversion":
        _require_columns(df, ["high", "low", "close"], errors)
    elif flt_type == "contradictory_signals":
        _require_columns(df, ["high", "low", "close"], errors)
    elif flt_type == "linear_regression_macd_cross":
        price_col = params.get("price_col", "close")
        _require_columns(df, [price_col], errors)
    elif flt_type in {"atr_rising", "market_regime", "trend"}:
        _require_columns(df, ["high", "low", "close"], errors)
    elif flt_type == "biais_institutional":
        _require_columns(df, [params.get("price_col", "close")], errors)

    return errors


def apply_filter_stack(
    df: pd.DataFrame,
    filters: Sequence[Mapping[str, Any]],
    *,
    symbol: Optional[str] = None,
    strict: bool = True,
    logger: Optional[logging.Logger] = None,
) -> pd.Series:
    """Return a boolean mask after applying all filters in order."""

    log = logger or logging.getLogger(__name__)
    if not filters:
        return pd.Series(True, index=df.index)

    mask = pd.Series(True, index=df.index)
    cache_stats_start = dict(_FILTER_CACHE_STATS)
    max_cache = df.attrs.get("qe_cache_max_items", _CACHE_DEFAULT_MAX)
    ttl_seconds = df.attrs.get("qe_cache_ttl_seconds", _CACHE_DEFAULT_TTL_SECONDS)
    try:
        max_cache = int(max_cache)
    except Exception:
        max_cache = _CACHE_DEFAULT_MAX
    try:
        ttl_seconds = float(ttl_seconds)
    except Exception:
        ttl_seconds = _CACHE_DEFAULT_TTL_SECONDS
    if ttl_seconds < 0:
        ttl_seconds = _CACHE_DEFAULT_TTL_SECONDS
    _FILTER_CACHE_CONFIG.update({"max_items": max_cache, "ttl_seconds": ttl_seconds})
    _prune_filter_cache(ttl_seconds)

    for raw in filters:
        flt_type, params = _normalize_filter_spec(raw)
        params = dict(params)
        if symbol and "symbol" not in params and flt_type in {
            "vwap_side",
            "poc_distance",
            "bos",
            "mss",
            "seasonality_bin",
        }:
            params["symbol"] = symbol

        errors = _validate_filter_inputs(df, flt_type, params, symbol)
        if errors:
            msg = f"Filter '{flt_type}' cannot be evaluated: {', '.join(errors)}"
            log.error(msg)
            if strict:
                raise FilterValidationError(msg)
            continue

        fn = filters_registry.get(flt_type)
        if fn is None:
            msg = f"Unknown filter type '{flt_type}'"
            log.error(msg)
            if strict:
                raise FilterValidationError(msg)
            continue

        cache_key = _make_cache_key(df, flt_type, params)
        if cache_key is not None:
            cached = _cache_get(cache_key, ttl_seconds)
            if cached is not None:
                series = cached
            else:
                try:
                    series = fn(df, **params)
                except Exception as exc:
                    msg = f"Filter '{flt_type}' failed: {exc}"
                    log.error(msg)
                    if strict:
                        raise FilterValidationError(msg) from exc
                    continue
                if max_cache > 0:
                    _cache_set(cache_key, series, max_cache, ttl_seconds)
        else:
            try:
                series = fn(df, **params)
            except Exception as exc:
                msg = f"Filter '{flt_type}' failed: {exc}"
                log.error(msg)
                if strict:
                    raise FilterValidationError(msg) from exc
                continue

        if not isinstance(series, pd.Series):
            msg = f"Filter '{flt_type}' did not return a pandas Series"
            log.error(msg)
            if strict:
                raise FilterValidationError(msg)
            continue

        mask &= series.reindex(df.index).fillna(False).astype(bool)

    _log_filter_cache_stats_delta(cache_stats_start, log)

    return mask
