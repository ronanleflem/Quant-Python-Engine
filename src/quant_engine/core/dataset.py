"""Dataset loading utilities.

The real project would use `polars` to read Parquet files.  The exercise
environment does not provide these third party packages, therefore this
module falls back to JSON input while keeping a compatible API.  The
returned dataset is a list of dictionaries sorted by timestamp.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict

from .spec import DataSpec


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value)


def load_dataset(spec: DataSpec) -> List[Dict]:
    """Load OHLCV data from ``spec.path``.

    Only rows matching ``spec.symbols`` and the date range are returned.
    In a production setting this would read a Parquet file using ``polars``;
    here we rely on a simple JSON file for portability.
    """
    raw = json.loads(Path(spec.path).read_text())
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

