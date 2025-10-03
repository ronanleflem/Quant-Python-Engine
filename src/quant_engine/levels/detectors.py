"""Vectorised level detectors used by the levels module."""
from __future__ import annotations

from datetime import timezone
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from .schemas import LevelRecord, SessionWindows


_PERIOD_CONFIG = {
    "D": {"freq": "1D", "timeframe": "D1", "high_type": "PDH", "low_type": "PDL"},
    "W": {"freq": "1W", "timeframe": "W1", "high_type": "PWH", "low_type": "PWL"},
    "M": {"freq": "1M", "timeframe": "M1", "high_type": "PMH", "low_type": "PML"},
}


def _to_utc_timestamp(value) -> Optional[pd.Timestamp]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _ensure_utc(series: pd.Series) -> None:
    tz = getattr(series.dt, "tz", None)
    assert tz is not None, "Timestamp series must be timezone aware (UTC)."
    assert str(tz) in {"UTC", "utc"}, "Timestamp series must be UTC normalised."


def _prepare_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    assert {"ts", "open", "high", "low", "close"}.issubset(df.columns), "Missing OHLC columns"
    df_sorted = df.sort_values("ts").copy()
    _ensure_utc(df_sorted["ts"])
    df_sorted["ts_copy"] = df_sorted["ts"]
    return df_sorted


def _resample_ohlc(df: pd.DataFrame, period: str) -> pd.DataFrame:
    cfg = _PERIOD_CONFIG[period]
    resampled = (
        df.set_index("ts")
        .resample(cfg["freq"], label="right", closed="right")
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            open_ts=("ts_copy", "first"),
            close_ts=("ts_copy", "last"),
        )
        .dropna(subset=["open", "high", "low", "close"])
    )
    resampled = resampled.reset_index(drop=False)
    resampled.rename(columns={"ts": "period_end"}, inplace=True)
    return resampled


def detect_session_high_low(df_symbol: pd.DataFrame, session_windows: SessionWindows) -> List[LevelRecord]:
    """Compute per-session high and low levels for a symbol."""

    if df_symbol.empty:
        return []
    df_prep = _prepare_ohlcv(df_symbol)
    if "symbol" in df_prep.columns:
        symbol = str(df_prep["symbol"].iloc[0])
    else:  # pragma: no cover - defensive fallback
        raise ValueError("Session detection requires a symbol column")

    bounds = {
        "ASIA": session_windows.asia,
        "EUROPE": session_windows.europe,
        "EU_US_OVERLAP": session_windows.overlap,
        "US": session_windows.us,
    }
    hours = df_prep["ts"].dt.hour
    sessions = pd.Series(data=pd.NA, index=df_prep.index, dtype="object")
    for name, (start, end) in bounds.items():
        if start <= end:
            mask = hours.between(start, end, inclusive="both")
        else:
            mask = (hours >= start) | (hours <= end)
        sessions = sessions.where(~mask, name)
    df_prep["session"] = sessions
    df_prep = df_prep.dropna(subset=["session"]).copy()
    if df_prep.empty:
        return []
    df_prep["session_date"] = df_prep["ts"].dt.floor("D")

    groups: List[dict] = []
    for (session_date, session_name), group in df_prep.groupby(["session_date", "session"], sort=False):
        if group.empty:
            continue
        start_ts = group["ts"].iloc[0]
        end_ts = group["ts"].iloc[-1]
        groups.append(
            {
                "session_date": session_date,
                "session": str(session_name),
                "start_ts": start_ts,
                "end_ts": end_ts,
                "high": float(group["high"].max()),
                "low": float(group["low"].min()),
            }
        )
    if not groups:
        return []
    groups.sort(key=lambda item: (pd.Timestamp(item["start_ts"]).value))

    records: List[LevelRecord] = []
    for idx, info in enumerate(groups):
        anchor_ts = pd.Timestamp(info["end_ts"]).to_pydatetime().astimezone(timezone.utc)
        next_start = groups[idx + 1]["start_ts"] if idx + 1 < len(groups) else None
        next_start_ts = _to_utc_timestamp(next_start)
        if next_start_ts is not None:
            valid_from = next_start_ts.to_pydatetime().astimezone(timezone.utc)
        else:
            valid_from = anchor_ts
        records.append(
            LevelRecord(
                symbol=symbol,
                timeframe="SESSION",
                level_type="SESSION_HIGH",
                price=info["high"],
                anchor_ts=anchor_ts,
                valid_from_ts=valid_from,
            )
        )
        records.append(
            LevelRecord(
                symbol=symbol,
                timeframe="SESSION",
                level_type="SESSION_LOW",
                price=info["low"],
                anchor_ts=anchor_ts,
                valid_from_ts=valid_from,
            )
        )
    return records


def _detect_intraday_range(
    df_symbol: pd.DataFrame,
    minutes: int,
    high_type: str,
    low_type: str,
) -> List[LevelRecord]:
    if df_symbol.empty:
        return []
    df_prep = _prepare_ohlcv(df_symbol)
    if "symbol" not in df_prep.columns:
        raise ValueError("Intraday range detection requires a symbol column")
    symbol = str(df_prep["symbol"].iloc[0])
    window = pd.Timedelta(minutes=int(minutes))
    df_prep["session_date"] = df_prep["ts"].dt.floor("D")
    records: List[LevelRecord] = []
    for session_date, group in df_prep.groupby("session_date", sort=False):
        if group.empty:
            continue
        day_start = pd.Timestamp(session_date)
        if day_start.tzinfo is None:
            day_start = day_start.tz_localize("UTC")
        else:
            day_start = day_start.tz_convert("UTC")
        window_end = day_start + window
        window_df = group[group["ts"] < window_end]
        if window_df.empty:
            continue
        high = float(window_df["high"].max())
        low = float(window_df["low"].min())
        anchor = window_end.to_pydatetime().astimezone(timezone.utc)
        records.append(
            LevelRecord(
                symbol=symbol,
                timeframe="D1",
                level_type=high_type,
                price=high,
                anchor_ts=anchor,
                valid_from_ts=anchor,
            )
        )
        records.append(
            LevelRecord(
                symbol=symbol,
                timeframe="D1",
                level_type=low_type,
                price=low,
                anchor_ts=anchor,
                valid_from_ts=anchor,
            )
        )
    return records


def detect_opening_range(df_symbol: pd.DataFrame, minutes: int = 30) -> List[LevelRecord]:
    """Detect opening range high/low levels for each UTC session."""

    return _detect_intraday_range(df_symbol, minutes=minutes, high_type="ORH", low_type="ORL")


def detect_initial_balance(df_symbol: pd.DataFrame, minutes: int = 60) -> List[LevelRecord]:
    """Detect initial balance high/low levels for each UTC session."""

    return _detect_intraday_range(df_symbol, minutes=minutes, high_type="IBH", low_type="IBL")


def detect_previous_high_low(df: pd.DataFrame, symbol: str, period: str = "D") -> List[LevelRecord]:
    """Return previous-period high/low levels for the supplied OHLCV frame.

    The function aggregates the intraday data to the requested period (daily,
    weekly or monthly), shifts the extrema by one period, and anchors levels on
    the close timestamp of the completed period (``close_ts``). The close
    timestamp corresponds to the final bar contained in the aggregation window.
    """

    period = period.upper()
    if period not in _PERIOD_CONFIG:
        raise ValueError(f"Unsupported period '{period}' for previous high/low detection")
    if df.empty:
        return []
    df_prep = _prepare_ohlcv(df)
    grouped = _resample_ohlc(df_prep, period)
    if grouped.empty:
        return []

    cfg = _PERIOD_CONFIG[period]
    prev_high = grouped["high"].shift(1)
    prev_low = grouped["low"].shift(1)

    records: List[LevelRecord] = []
    for idx, row in grouped.iterrows():
        anchor = row["close_ts"]
        if pd.isna(anchor):
            continue
        anchor_dt = pd.Timestamp(anchor).to_pydatetime().astimezone(timezone.utc)
        if not pd.isna(prev_high.iloc[idx]):
            records.append(
                LevelRecord(
                    symbol=symbol,
                    timeframe=cfg["timeframe"],
                    level_type=cfg["high_type"],
                    price=float(prev_high.iloc[idx]),
                    anchor_ts=anchor_dt,
                    valid_from_ts=anchor_dt,
                )
            )
        if not pd.isna(prev_low.iloc[idx]):
            records.append(
                LevelRecord(
                    symbol=symbol,
                    timeframe=cfg["timeframe"],
                    level_type=cfg["low_type"],
                    price=float(prev_low.iloc[idx]),
                    anchor_ts=anchor_dt,
                    valid_from_ts=anchor_dt,
                )
            )
    return records


_OPEN_CLOSE_TYPES = {
    "D": {"open_type": "PDO", "close_type": "PDC", "timeframe": "D1"},
    "W": {"open_type": "PWO", "close_type": "PWC", "timeframe": "W1"},
    "M": {"open_type": "PMO", "close_type": "PMC", "timeframe": "M1"},
}


def detect_previous_open_close(df_symbol: pd.DataFrame, period: str = "D") -> List[LevelRecord]:
    """Compute previous period open/close levels for the provided symbol."""

    if df_symbol.empty:
        return []
    period = period.upper()
    if period not in _OPEN_CLOSE_TYPES:
        raise ValueError(f"Unsupported period '{period}' for previous open/close detection")
    df_prep = _prepare_ohlcv(df_symbol)
    if "symbol" not in df_prep.columns:
        raise ValueError("Previous open/close detection requires a symbol column")
    symbol = str(df_prep["symbol"].iloc[0])
    grouped = _resample_ohlc(df_prep, period)
    if grouped.empty:
        return []

    cfg = _OPEN_CLOSE_TYPES[period]
    prev_open = grouped["open"].shift(1)
    prev_close = grouped["close"].shift(1)
    next_open_ts = grouped["open_ts"].shift(-1)

    records: List[LevelRecord] = []
    for idx, row in grouped.iterrows():
        anchor_raw = row.get("close_ts")
        if pd.isna(anchor_raw):
            continue
        anchor_ts = pd.Timestamp(anchor_raw).to_pydatetime().astimezone(timezone.utc)
        valid_from_raw = next_open_ts.iloc[idx]
        if pd.isna(valid_from_raw):
            valid_from_ts = anchor_ts
        else:
            valid_from_ts = pd.Timestamp(valid_from_raw).to_pydatetime().astimezone(timezone.utc)
        if not pd.isna(prev_open.iloc[idx]):
            records.append(
                LevelRecord(
                    symbol=symbol,
                    timeframe=cfg["timeframe"],
                    level_type=cfg["open_type"],
                    price=float(prev_open.iloc[idx]),
                    anchor_ts=anchor_ts,
                    valid_from_ts=valid_from_ts,
                )
            )
        if not pd.isna(prev_close.iloc[idx]):
            records.append(
                LevelRecord(
                    symbol=symbol,
                    timeframe=cfg["timeframe"],
                    level_type=cfg["close_type"],
                    price=float(prev_close.iloc[idx]),
                    anchor_ts=anchor_ts,
                    valid_from_ts=valid_from_ts,
                )
            )
    return records


def detect_gaps(df: pd.DataFrame, symbol: str, period: str = "D") -> List[LevelRecord]:
    """Detect simple open/close gaps across aggregated periods."""

    period = period.upper()
    if period not in {"D", "W"}:
        raise ValueError("Gaps are only supported for daily or weekly aggregation")
    if df.empty:
        return []
    df_prep = _prepare_ohlcv(df)
    grouped = _resample_ohlc(df_prep, period)
    if grouped.empty:
        return []

    prev_close = grouped["close"].shift(1)
    prev_close_ts = grouped["close_ts"].shift(1)
    open_today = grouped["open"]
    open_ts = grouped["open_ts"]

    cfg = _PERIOD_CONFIG[period]
    level_type = "GAP_D" if period == "D" else "GAP_W"
    records: List[LevelRecord] = []
    for idx, row in grouped.iterrows():
        if pd.isna(prev_close.iloc[idx]) or pd.isna(open_today.iloc[idx]):
            continue
        if float(open_today.iloc[idx]) == float(prev_close.iloc[idx]):
            continue
        anchor_raw = prev_close_ts.iloc[idx]
        open_raw = open_ts.iloc[idx]
        if pd.isna(anchor_raw) or pd.isna(open_raw):
            continue
        anchor_dt = pd.Timestamp(anchor_raw).to_pydatetime().astimezone(timezone.utc)
        open_dt = pd.Timestamp(open_raw).to_pydatetime().astimezone(timezone.utc)
        lo = float(min(open_today.iloc[idx], prev_close.iloc[idx]))
        hi = float(max(open_today.iloc[idx], prev_close.iloc[idx]))
        records.append(
            LevelRecord(
                symbol=symbol,
                timeframe=cfg["timeframe"],
                level_type=level_type,
                price=(lo + hi) / 2.0,
                price_lo=lo,
                price_hi=hi,
                anchor_ts=anchor_dt,
                valid_from_ts=open_dt,
            )
        )
    return records


def detect_fvg(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    min_size_ticks: int = 1,
    price_increment: Optional[float] = None,
) -> List[LevelRecord]:
    """Detect three-candle Fair Value Gaps (FVG) in the dataset.

    The anchor timestamp is the close of the centre candle (``t``). A bullish
    gap is produced when ``low[t+1] > high[t-1]`` while a bearish gap occurs when
    ``high[t+1] < low[t-1]``. The optional ``price_increment`` parameter enforces
    a minimum gap size in ticks.
    """

    if df.empty or len(df) < 3:
        return []
    df_prep = _prepare_ohlcv(df)
    lows = df_prep["low"].to_numpy()
    highs = df_prep["high"].to_numpy()
    timestamps = df_prep["ts"].to_numpy()

    records: List[LevelRecord] = []
    for idx in range(1, len(df_prep) - 1):
        low_next = lows[idx + 1]
        high_next = df_prep["high"].iloc[idx + 1]
        anchor = pd.Timestamp(timestamps[idx]).to_pydatetime().astimezone(timezone.utc)
        bullish = low_next > highs[idx - 1]
        bearish = high_next < lows[idx - 1]
        if not bullish and not bearish:
            continue
        if bullish:
            lo = float(highs[idx - 1])
            hi = float(low_next)
        else:
            lo = float(high_next)
            hi = float(lows[idx - 1])
        if lo == hi:
            continue
        size = abs(hi - lo)
        if price_increment is not None and size < (min_size_ticks * price_increment):
            continue
        price_mid = (lo + hi) / 2.0
        records.append(
            LevelRecord(
                symbol=symbol,
                timeframe=timeframe,
                level_type="FVG",
                price=price_mid,
                price_lo=min(lo, hi),
                price_hi=max(lo, hi),
                anchor_ts=anchor,
                valid_from_ts=anchor,
            )
        )
    return records


def _prepare_fill_dataframe(levels: pd.DataFrame) -> pd.DataFrame:
    df = levels.copy()
    if not df.empty:
        for col in ("valid_from_ts", "anchor_ts", "valid_to_ts"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    return df


def fill_gaps(df_symbol: pd.DataFrame, gaps: pd.DataFrame) -> pd.DataFrame:
    """Populate ``valid_to_ts`` for gap zones touched by closing prices."""

    if gaps.empty or df_symbol.empty:
        return gaps.copy()
    df_prep = _prepare_ohlcv(df_symbol)
    closes = df_prep[["ts", "close"]].copy()
    closes.rename(columns={"ts": "bar_ts"}, inplace=True)
    levels = _prepare_fill_dataframe(gaps)
    if "valid_to_ts" not in levels.columns:
        levels["valid_to_ts"] = pd.NaT

    for idx, row in levels.iterrows():
        if pd.notna(row.get("valid_to_ts")):
            continue
        price_lo = row.get("price_lo")
        price_hi = row.get("price_hi")
        valid_from = row.get("valid_from_ts") or row.get("anchor_ts")
        valid_from_ts = _to_utc_timestamp(valid_from)
        if price_lo is None or price_hi is None or valid_from_ts is None:
            continue
        window = closes.loc[closes["bar_ts"] >= valid_from_ts]
        if window.empty:
            continue
        lo = float(min(price_lo, price_hi))
        hi = float(max(price_lo, price_hi))
        touched = window[(window["close"] >= lo) & (window["close"] <= hi)]
        if touched.empty:
            continue
        fill_ts = touched.iloc[0]["bar_ts"]
        levels.at[idx, "valid_to_ts"] = fill_ts
    return levels


def fill_fvgs(df_symbol: pd.DataFrame, fvgs: pd.DataFrame) -> pd.DataFrame:
    """Populate ``valid_to_ts`` for Fair Value Gaps when retouched."""

    if fvgs.empty or df_symbol.empty:
        return fvgs.copy()
    df_prep = _prepare_ohlcv(df_symbol)
    closes = df_prep[["ts", "close", "high", "low"]].copy()
    closes.rename(columns={"ts": "bar_ts"}, inplace=True)
    levels = _prepare_fill_dataframe(fvgs)
    if "valid_to_ts" not in levels.columns:
        levels["valid_to_ts"] = pd.NaT

    for idx, row in levels.iterrows():
        if pd.notna(row.get("valid_to_ts")):
            continue
        price_lo = row.get("price_lo")
        price_hi = row.get("price_hi")
        if price_lo is None or price_hi is None:
            continue
        valid_from = row.get("valid_from_ts") or row.get("anchor_ts")
        valid_from_ts = _to_utc_timestamp(valid_from)
        if valid_from_ts is None:
            continue
        lo = float(min(price_lo, price_hi))
        hi = float(max(price_lo, price_hi))
        window = closes[closes["bar_ts"] >= valid_from_ts]
        if window.empty:
            continue
        touched = window[(window["close"] >= lo) & (window["close"] <= hi)]
        if touched.empty:
            continue
        fill_ts = touched.iloc[0]["bar_ts"]
        levels.at[idx, "valid_to_ts"] = fill_ts
    return levels


def detect_poc(
    df: pd.DataFrame,
    symbol: str,
    period: str = "D",
    bins: int = 100,
    method: str = "volume",
    price_col: str = "close",
) -> List[LevelRecord]:
    """Approximate Point of Control levels using histogram counts."""

    period = period.upper()
    if period not in {"D", "W"}:
        raise ValueError("POC detector currently supports daily or weekly aggregation")
    if df.empty:
        return []
    df_prep = _prepare_ohlcv(df)
    cfg = _PERIOD_CONFIG[period]
    records: List[LevelRecord] = []
    grouper = pd.Grouper(key="ts", freq=cfg["freq"], label="right", closed="right")
    for _, group in df_prep.groupby(grouper, dropna=True):
        if group.empty:
            continue
        prices = group[price_col].dropna()
        if prices.empty:
            continue
        price_min = float(prices.min())
        price_max = float(prices.max())
        if price_min == price_max:
            poc_price = price_min
            lo = price_min
            hi = price_max
        else:
            counts, edges = np.histogram(prices.to_numpy(), bins=bins)
            if not np.any(counts):
                continue
            max_idx = int(np.argmax(counts))
            lo = float(edges[max_idx])
            hi = float(edges[max_idx + 1])
            poc_price = (lo + hi) / 2.0
        anchor_ts = group["ts"].iloc[-1].to_pydatetime().astimezone(timezone.utc)
        records.append(
            LevelRecord(
                symbol=symbol,
                timeframe=cfg["timeframe"],
                level_type="POC",
                price=poc_price,
                price_lo=min(lo, hi),
                price_hi=max(lo, hi),
                anchor_ts=anchor_ts,
                valid_from_ts=anchor_ts,
            )
        )
    return records


def generate_round_numbers(
    symbol: str,
    timeframe: str,
    level_type: str,
    price_min: float,
    price_max: float,
    step: float,
) -> List[LevelRecord]:
    """Generate static round-number levels in the provided price range."""

    if step <= 0:
        raise ValueError("step must be positive for round number generation")
    if price_max <= price_min:
        raise ValueError("price_max must be greater than price_min")
    values: Iterable[float] = np.arange(price_min, price_max + step, step)
    anchor = pd.Timestamp.utcnow(tz="UTC").to_pydatetime().astimezone(timezone.utc)
    records = [
        LevelRecord(
            symbol=symbol,
            timeframe=timeframe,
            level_type=level_type,
            price=float(val),
            anchor_ts=anchor,
            valid_from_ts=anchor,
        )
        for val in values
    ]
    return records


__all__ = [
    "detect_previous_high_low",
    "detect_gaps",
    "detect_fvg",
    "detect_poc",
    "generate_round_numbers",
    "detect_session_high_low",
    "detect_opening_range",
    "detect_initial_balance",
    "detect_previous_open_close",
    "fill_gaps",
    "fill_fvgs",
]
