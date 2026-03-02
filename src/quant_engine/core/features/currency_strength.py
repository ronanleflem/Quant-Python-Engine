"""Currency strength feature engineering utilities for FX symbols."""
from __future__ import annotations

from itertools import combinations
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

MAJOR_CURRENCIES: tuple[str, ...] = ("USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD")


def parse_fx_symbol(symbol: str) -> tuple[Optional[str], Optional[str]]:
    """Return (base, quote) from common FX symbol formats."""

    if not symbol:
        return None, None
    raw = str(symbol).strip().upper().replace("_", "").replace("-", "")
    if "/" in raw:
        left, right = raw.split("/", 1)
        if len(left) == 3 and len(right) == 3:
            return left, right
        return None, None
    if len(raw) == 6:
        return raw[:3], raw[3:]
    return None, None


def default_major_pairs(majors: Sequence[str] = MAJOR_CURRENCIES) -> list[str]:
    """Return deterministic list of all pair combinations for the majors basket."""

    clean = [str(ccy).strip().upper() for ccy in majors if str(ccy).strip()]
    return [f"{base}{quote}" for base, quote in combinations(clean, 2)]


def _normalize_price_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["ts", "close"])
    out = df.copy()
    if "timestamp" in out.columns and "ts" not in out.columns:
        out = out.rename(columns={"timestamp": "ts"})
    if "ts" not in out.columns or "close" not in out.columns:
        return pd.DataFrame(columns=["ts", "close"])
    out["ts"] = pd.to_datetime(out["ts"], utc=True, errors="coerce")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.dropna(subset=["ts", "close"]).sort_values("ts")
    return out[["ts", "close"]]


def build_strength_table(
    prices_by_symbol: Mapping[str, pd.DataFrame],
    *,
    majors: Sequence[str] = MAJOR_CURRENCIES,
    lookback: int = 72,
) -> pd.DataFrame:
    """Build a timestamp-indexed table with normalized currency strength columns."""

    majors_set = {str(ccy).strip().upper() for ccy in majors if str(ccy).strip()}
    if not majors_set:
        return pd.DataFrame()

    contributions: Dict[str, list[pd.Series]] = {ccy: [] for ccy in sorted(majors_set)}
    for raw_symbol, frame in prices_by_symbol.items():
        base, quote = parse_fx_symbol(raw_symbol)
        if base not in majors_set or quote not in majors_set:
            continue
        normalized = _normalize_price_frame(frame)
        if normalized.empty:
            continue
        series = np.log(normalized["close"]).diff()
        series = pd.Series(series.to_numpy(), index=normalized["ts"])
        if lookback > 1:
            window = int(lookback)
            min_periods = max(2, min(window, int(max(2, lookback // 4))))
            series = series.rolling(window, min_periods=min_periods).mean()
        contributions[base].append(series)
        contributions[quote].append(-series)

    if not any(contributions.values()):
        return pd.DataFrame()

    by_ccy: Dict[str, pd.Series] = {}
    for ccy, entries in contributions.items():
        if not entries:
            continue
        merged = pd.concat(entries, axis=1)
        by_ccy[ccy] = merged.mean(axis=1, skipna=True)

    if not by_ccy:
        return pd.DataFrame()

    strength = pd.DataFrame(by_ccy).sort_index()
    row_mean = strength.mean(axis=1)
    row_std = strength.std(axis=1).replace(0.0, np.nan)
    zscore = strength.sub(row_mean, axis=0).div(row_std, axis=0)
    zscore = zscore.replace([np.inf, -np.inf], np.nan)
    zscore.columns = [f"ccy_strength_{col}" for col in zscore.columns]
    return zscore


def enrich_with_currency_strength(
    df: pd.DataFrame,
    *,
    symbol: str,
    prices_by_symbol: Mapping[str, pd.DataFrame],
    majors: Sequence[str] = MAJOR_CURRENCIES,
    lookback: int = 72,
) -> pd.DataFrame:
    """Attach currency strength columns to a symbol dataframe."""

    out = df.copy()
    if out.empty:
        out["ccy_strength_base"] = np.nan
        out["ccy_strength_quote"] = np.nan
        out["ccy_strength_spread"] = np.nan
        return out
    base, quote = parse_fx_symbol(symbol)
    if not base or not quote:
        out["ccy_strength_base"] = np.nan
        out["ccy_strength_quote"] = np.nan
        out["ccy_strength_spread"] = np.nan
        return out

    table = build_strength_table(prices_by_symbol, majors=majors, lookback=lookback)
    if table.empty:
        out["ccy_strength_base"] = np.nan
        out["ccy_strength_quote"] = np.nan
        out["ccy_strength_spread"] = np.nan
        return out

    ts_col = "ts" if "ts" in out.columns else "timestamp"
    if ts_col not in out.columns:
        out["ccy_strength_base"] = np.nan
        out["ccy_strength_quote"] = np.nan
        out["ccy_strength_spread"] = np.nan
        return out
    ts = pd.to_datetime(out[ts_col], utc=True, errors="coerce")
    base_col = f"ccy_strength_{base}"
    quote_col = f"ccy_strength_{quote}"
    out["ccy_strength_base"] = table.get(base_col, pd.Series(dtype="float64")).reindex(ts).to_numpy()
    out["ccy_strength_quote"] = table.get(quote_col, pd.Series(dtype="float64")).reindex(ts).to_numpy()
    out["ccy_strength_spread"] = out["ccy_strength_base"] - out["ccy_strength_quote"]
    return out
