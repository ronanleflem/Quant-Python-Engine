"""High level orchestration for level detectors."""
from __future__ import annotations

import hashlib
import json
from typing import Dict, Iterable, List

import pandas as pd

from . import detectors
from .schemas import LevelRecord, LevelsBuildSpec, SessionWindows


def _compute_hash(level_type: str, params: Dict[str, object]) -> str:
    payload = {"type": level_type, "params": params}
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def build_levels(spec: LevelsBuildSpec, ohlcv: pd.DataFrame) -> List[LevelRecord]:
    """Route the provided OHLCV dataset through requested detectors."""

    if ohlcv.empty:
        return []
    df = ohlcv.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)

    range_start = pd.to_datetime(spec.range_start, utc=True)
    range_end = pd.to_datetime(spec.range_end, utc=True)
    timeframe = spec.data.timeframe.upper()

    results: List[LevelRecord] = []
    default_sessions = spec.session_windows or SessionWindows()
    or_minutes_default = spec.orib.or_minutes if spec.orib else 30
    ib_minutes_default = spec.orib.ib_minutes if spec.orib else 60

    for symbol in spec.symbols:
        sym_df = df[df["symbol"] == symbol].copy()
        sym_df.attrs["timeframe"] = timeframe
        if sym_df.empty:
            continue
        for target in spec.targets:
            params = dict(target.params)
            level_type = target.type.upper()
            detector_records: Iterable[LevelRecord]
            if level_type in {"PDH", "PDL"}:
                detector_records = detectors.detect_previous_high_low(sym_df, symbol=symbol, period="D")
            elif level_type in {"PWH", "PWL"}:
                detector_records = detectors.detect_previous_high_low(sym_df, symbol=symbol, period="W")
            elif level_type in {"PMH", "PML"}:
                detector_records = detectors.detect_previous_high_low(sym_df, symbol=symbol, period="M")
            elif level_type in {"SESSION_HIGH", "SESSION_LOW"}:
                custom_windows = params.get("session_windows")
                if custom_windows:
                    session_windows = SessionWindows(**custom_windows)
                else:
                    session_windows = default_sessions
                detector_records = detectors.detect_session_high_low(
                    sym_df,
                    session_windows=session_windows,
                )
            elif level_type in {"ORH", "ORL"}:
                minutes = int(params.get("minutes", or_minutes_default))
                detector_records = detectors.detect_opening_range(sym_df, minutes=minutes)
            elif level_type in {"IBH", "IBL"}:
                minutes = int(params.get("minutes", ib_minutes_default))
                detector_records = detectors.detect_initial_balance(sym_df, minutes=minutes)
            elif level_type in {"PDO", "PDC", "PWO", "PWC", "PMO", "PMC"}:
                period_map = {
                    "PDO": "D",
                    "PDC": "D",
                    "PWO": "W",
                    "PWC": "W",
                    "PMO": "M",
                    "PMC": "M",
                }
                detector_records = detectors.detect_previous_open_close(
                    sym_df,
                    period=period_map[level_type],
                )
            elif level_type == "GAP_D":
                detector_records = detectors.detect_gaps(sym_df, symbol=symbol, period="D")
            elif level_type == "GAP_W":
                detector_records = detectors.detect_gaps(sym_df, symbol=symbol, period="W")
            elif level_type == "FVG":
                price_increment = params.get("price_increment")
                if price_increment is not None:
                    price_increment = float(price_increment)
                detector_records = detectors.detect_fvg(
                    sym_df,
                    symbol=symbol,
                    timeframe=timeframe,
                    min_size_ticks=int(params.get("min_size_ticks", 1)),
                    price_increment=price_increment,
                )
            elif level_type == "POC":
                detector_records = detectors.detect_poc(
                    sym_df,
                    symbol=symbol,
                    period=str(params.get("period", "D")).upper(),
                    bins=int(params.get("bins", 100)),
                    method=str(params.get("method", "volume")),
                    price_col=str(params.get("price_col", "close")),
                )
            elif level_type == "RN":
                detector_records = detectors.generate_round_numbers(
                    symbol=symbol,
                    timeframe=params.get("timeframe", timeframe),
                    level_type="RN",
                    price_min=float(params["price_min"]),
                    price_max=float(params["price_max"]),
                    step=float(params.get("step", 1.0)),
                )
            elif level_type in {"SWING_H", "SWING_L"}:
                detector_records = detectors.detect_fractals(
                    sym_df,
                    left=int(params.get("left", 2)),
                    right=int(params.get("right", 2)),
                )
            elif level_type in {"EQH", "EQL"}:
                price_increment = params.get("price_increment")
                if price_increment is not None:
                    price_increment = float(price_increment)
                detector_records = detectors.detect_equal_highs_lows(
                    sym_df,
                    side="high" if level_type == "EQH" else "low",
                    tolerance_ticks=int(params.get("tolerance_ticks", 1)),
                    price_increment=price_increment,
                    min_count=int(params.get("min_count", 2)),
                    lookback_bars=int(params.get("lookback_bars", 500)),
                )
            elif level_type in {"BOS_H", "BOS_L", "MSS"}:
                use_fractals_raw = params.get("use_fractals", True)
                if isinstance(use_fractals_raw, str):
                    use_fractals = use_fractals_raw.strip().lower() not in {"false", "0", "no"}
                else:
                    use_fractals = bool(use_fractals_raw)
                detector_records = detectors.detect_bos_mss(
                    sym_df,
                    use_fractals=use_fractals,
                    left=int(params.get("left", 2)),
                    right=int(params.get("right", 2)),
                )
            else:
                raise ValueError(f"Unsupported level target: {level_type}")
            params_hash = _compute_hash(level_type, params)
            for record in detector_records:
                if record.level_type != level_type and level_type != "RN":
                    continue
                record.params_hash = params_hash
                record.source = "python-levels v0"
                results.append(record)
    # range filter on anchor timestamps
    filtered: List[LevelRecord] = []
    for rec in results:
        anchor_ts = pd.Timestamp(rec.anchor_ts)
        if anchor_ts.tzinfo is None:
            anchor = anchor_ts.tz_localize("UTC")
        else:
            anchor = anchor_ts.tz_convert("UTC")
        if range_start <= anchor <= range_end:
            filtered.append(rec)
    return filtered


__all__ = ["build_levels"]
