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
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd

from quant_engine.datafeeds.mysql_feed import load_ohlcv_mysql

from .spec import DataSpec


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value)


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


def load_dataset(spec: DataSpec) -> List[Dict]:
    """Load OHLCV data from ``spec.path``.

    Only rows matching ``spec.symbols`` and the date range are returned.
    In a production setting this would read a Parquet file using ``polars``;
    here we rely on a simple JSON file for portability.
    """
    raw = _read_dataset_rows(Path(spec.path))
    out = []
    for row in raw:
        ts = _parse_timestamp(row["timestamp"]).date()
        if row["symbol"] not in spec.symbols:
            continue
        if ts < spec.start or ts > spec.end:
            continue
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
        return load_ohlcv_mysql(
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
        )

    raise RuntimeError("Aucune source data fournie : dataset_path ou data.mysql requis.")

