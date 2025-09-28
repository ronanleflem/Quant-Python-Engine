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
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Any

from types import SimpleNamespace

import pandas as pd

from quant_engine.datafeeds.mysql_feed import load_ohlcv_mysql

from .spec import DataSpec


def _parse_timestamp(value: str) -> datetime:
    if value.endswith('Z'):
        value = value[:-1] + '+00:00'
    return datetime.fromisoformat(value)


def _coerce_date(value: str) -> date:
    value = value.strip()
    if 'T' in value or value.endswith('Z'):
        return _parse_timestamp(value).date()
    if ' ' in value:
        return datetime.strptime(value, '%Y-%m-%d %H:%M:%S').date()
    return date.fromisoformat(value)

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
        if key == "timestamp" or key == "symbol":
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
        rows = []
        if not df.empty:
            for rec in df.to_dict("records"):
                ts_value = rec.pop("ts", None)
                if ts_value is None:
                    raise RuntimeError("MySQL dataset must provide a 'ts' column")
                ts = pd.Timestamp(ts_value)
                if ts.tzinfo is None:
                    ts = ts.tz_localize("UTC")
                else:
                    ts = ts.tz_convert("UTC")
                rec["timestamp"] = ts.isoformat()
                rows.append(rec)
        else:
            rows = []
    else:
        raise RuntimeError("data.dataset_path or data.mysql must be defined")

    start_date = _coerce_date(spec.start)
    end_date = _coerce_date(spec.end)
    out: List[Dict[str, Any]] = []
    for row in rows:
        ts = _parse_timestamp(row["timestamp"]).date()
        symbol = row.get("symbol")
        if spec.symbols and symbol not in spec.symbols:
            continue
        if ts < start_date or ts > end_date:
            continue
        if symbol is None and spec.symbols:
            row["symbol"] = spec.symbols[0]
        if 'session' not in row and 'session_id' not in row:
            ts_dt = _parse_timestamp(row['timestamp'])
            row['session'] = _assign_session(ts_dt)
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
        if "ts" in df.columns:
            df["ts"] = pd.to_datetime(df["ts"], utc=True)
        elif "timestamp" in df.columns:
            df.rename(columns={"timestamp": "ts"}, inplace=True)
            df["ts"] = pd.to_datetime(df["ts"], utc=True)
        else:
            raise RuntimeError("CSV must contain a 'ts' or 'timestamp' column.")
        if 'session' not in df.columns and 'session_id' not in df.columns:
            df['session'] = df['ts'].apply(lambda ts: _assign_session(ts.to_pydatetime()))
        return df.sort_values(["symbol", "ts"]).reset_index(drop=True)

    mysql_spec = getattr(spec_data, "mysql", None)
    if mysql_spec:
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
        if 'session' not in df.columns and 'session_id' not in df.columns:
            df['session'] = df['ts'].apply(lambda ts: _assign_session(ts.to_pydatetime()))
        return df.sort_values(["symbol", "ts"]).reset_index(drop=True)

    raise RuntimeError("Aucune source data fournie : dataset_path ou data.mysql requis.")


    raise RuntimeError("Aucune source data fournie : dataset_path ou data.mysql requis.")

