"""Dataset loading utilities.

The real project would use `polars` to read Parquet files.  The exercise
environment does not provide these third party packages, therefore this
module falls back to JSON input while keeping a compatible API.  The
returned dataset is a list of dictionaries sorted by timestamp.  CSV
datasets are also supported to keep fixtures fully text-based.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, date, timezone, time
from pathlib import Path
from typing import Dict, List, Any

from types import SimpleNamespace

import pandas as pd

from .spec import DataSpec


def _parse_timestamp(value: str) -> datetime:
    value = value.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    if " " in value and "T" not in value:
        try:
            return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return datetime.fromisoformat(value.replace(" ", "T"))
    return datetime.fromisoformat(value)


def _coerce_timestamp(value: Any) -> datetime:
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if isinstance(value, datetime):
        ts = value
    elif isinstance(value, date):
        ts = datetime.combine(value, time.min)
    elif isinstance(value, str):
        ts = _parse_timestamp(value)
    else:
        raise ValueError(f"Unsupported timestamp type: {type(value)!r}")
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _format_timestamp(ts: datetime) -> str:
    return _coerce_timestamp(ts).isoformat()


def _coerce_date(value: str) -> date:
    return _coerce_timestamp(value).date()


def _assign_session(ts: datetime) -> str:
    hour = ts.hour
    if 0 <= hour < 7:
        return "Asia"
    if 7 <= hour < 12:
        return "Europe"
    if 12 <= hour < 16:
        return "EU_US_overlap"
    if 16 <= hour < 21:
        return "US"
    return "Other"


def _parse_row_types(row: Dict[str, str]) -> Dict[str, Any]:
    """Convert CSV row values to appropriate python types."""

    parsed: Dict[str, Any] = {}
    for key, value in row.items():
        if value is None:
            parsed[key] = None
            continue
        value = value.strip()
        if value == "":
            parsed[key] = None
            continue
        if key in {"timestamp", "ts", "symbol"}:
            parsed[key] = value
            continue
        # Attempt integer then float conversion, falling back to the raw string
        try:
            if "." not in value and "e" not in value and "E" not in value:
                parsed[key] = int(value)
                continue
        except ValueError:
            pass
        try:
            parsed[key] = float(value)
            continue
        except ValueError:
            parsed[key] = value
    return parsed


def _read_dataset_rows(path: Path) -> List[Dict[str, Any]]:
    """Return dataset rows from either JSON or CSV sources."""

    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            return [_parse_row_types(row) for row in reader]
    return json.loads(path.read_text())


def _normalize_row_timestamp(row: Dict[str, Any]) -> datetime:
    ts_value = row.get("timestamp")
    if ts_value is None:
        ts_value = row.get("ts")
    if ts_value is None:
        raise RuntimeError("Dataset rows must provide a 'timestamp' or 'ts' column")
    ts = _coerce_timestamp(ts_value)
    row["timestamp"] = _format_timestamp(ts)
    row.pop("ts", None)
    return ts


def _normalize_dataframe_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    if "ts" in df.columns:
        ts_col = "ts"
    elif "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "ts"})
        ts_col = "ts"
    else:
        raise RuntimeError("Dataset must contain a 'ts' or 'timestamp' column.")
    df[ts_col] = pd.to_datetime(df[ts_col].map(_coerce_timestamp), utc=True)
    return df


def _ensure_row_session(row: Dict[str, Any], ts: datetime) -> None:
    if "session" not in row and "session_id" not in row:
        row["session"] = _assign_session(ts)


def _ensure_session_column(df: pd.DataFrame) -> pd.DataFrame:
    if "session" not in df.columns and "session_id" not in df.columns:
        df["session"] = df["ts"].apply(lambda ts: _assign_session(_coerce_timestamp(ts)))
    return df


def _build_data_input_proxy(spec: DataSpec) -> SimpleNamespace:
    mysql = None
    if spec.mysql is not None:
        mysql = SimpleNamespace(**spec.mysql.__dict__)
    return SimpleNamespace(
        dataset_path=spec.dataset_path,
        mysql=mysql,
        symbols=list(spec.symbols),
        timeframe=spec.timeframe,
        start=spec.start,
        end=spec.end,
    )


def load_dataset(spec: DataSpec) -> List[Dict]:
    """Load OHLCV data from a JSON/CSV file or a MySQL source.

    Only rows matching ``spec.symbols`` and located within the requested date
    window are returned, sorted by timestamp.
    """
    rows: List[Dict[str, Any]]
    if spec.dataset_path:
        rows = _read_dataset_rows(Path(spec.dataset_path))
    elif spec.mysql is not None:
        proxy = _build_data_input_proxy(spec)
        df = load_ohlcv(proxy)
        rows = df.to_dict("records") if not df.empty else []
    else:
        raise RuntimeError("data.dataset_path or data.mysql must be defined")

    start_date = _coerce_date(spec.start)
    end_date = _coerce_date(spec.end)
    out: List[Dict[str, Any]] = []
    for row in rows:
        ts_dt = _normalize_row_timestamp(row)
        ts = ts_dt.date()
        symbol = row.get("symbol")
        if spec.symbols and symbol not in spec.symbols:
            continue
        if ts < start_date or ts > end_date:
            continue
        if symbol is None and spec.symbols:
            row["symbol"] = spec.symbols[0]
        _ensure_row_session(row, ts_dt)
        out.append(row)
    out.sort(key=lambda r: r["timestamp"])
    return out


def load_ohlcv(spec_data) -> pd.DataFrame:
    """Load OHLCV data from CSV or MySQL."""

    dataset_path = getattr(spec_data, "dataset_path", None)
    if dataset_path:
        path = Path(dataset_path)
        if path.suffix.lower() == ".json":
            raw = json.loads(path.read_text())
            df = pd.DataFrame(raw)
        else:
            df = pd.read_csv(str(path))
        df = _normalize_dataframe_timestamps(df)
        df = _ensure_session_column(df)
        return df.sort_values(["symbol", "ts"]).reset_index(drop=True)

    mysql_spec = getattr(spec_data, "mysql", None)
    if mysql_spec:
        from ..datafeeds.mysql_feed import load_ohlcv_mysql
        cols = {
            "ts": mysql_spec.ts_col,
            "symbol": mysql_spec.symbol_col,
            "open": mysql_spec.open_col,
            "high": mysql_spec.high_col,
            "low": mysql_spec.low_col,
            "close": mysql_spec.close_col,
            "volume": mysql_spec.volume_col,
        }
        df = load_ohlcv_mysql(
            connection_url=mysql_spec.connection_url,
            env_var=mysql_spec.env_var,
            schema=mysql_spec.schema,
            table=mysql_spec.table,
            symbols=list(spec_data.symbols),
            timeframe=spec_data.timeframe,
            start=spec_data.start,
            end=spec_data.end,
            cols=cols,
            timeframe_col=mysql_spec.timeframe_col,
            extra_where=mysql_spec.extra_where,
            chunk_minutes=mysql_spec.chunk_minutes,
            symbol_lookup_table=getattr(mysql_spec, "symbol_lookup_table", None),
            symbol_lookup_symbol_col=getattr(mysql_spec, "symbol_lookup_symbol_col", "symbol"),
            symbol_lookup_id_col=getattr(mysql_spec, "symbol_lookup_id_col", "id"),
        )
        if df.empty:
            return df
        df = _normalize_dataframe_timestamps(df)
        df = _ensure_session_column(df)
        return df.sort_values(["symbol", "ts"]).reset_index(drop=True)

    raise RuntimeError("Aucune source data fournie : dataset_path ou data.mysql requis.")
