"""Vectorised level detectors used by the levels module."""
from __future__ import annotations

from datetime import timezone
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from .schemas import LevelRecord


_PERIOD_CONFIG = {
    "D": {"freq": "1D", "timeframe": "D1", "high_type": "PDH", "low_type": "PDL"},
    "W": {"freq": "1W", "timeframe": "W1", "high_type": "PWH", "low_type": "PWL"},
    "M": {"freq": "1M", "timeframe": "M1", "high_type": "PMH", "low_type": "PML"},
}


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
]
