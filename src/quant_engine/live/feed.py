"""Live data feed adapters."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd
from sqlalchemy import Engine, text


@dataclass
class FeedColumns:
    """Mapping of OHLCV column names in the source table."""

    ts: str
    open: str
    high: str
    low: str
    close: str
    volume: str
    symbol: str


class MySQLPollFeed:
    """Poll-based OHLCV reader for MySQL backends."""

    def __init__(
        self,
        engine: Engine,
        table: str,
        symbol: str,
        timeframe: str,
        columns: FeedColumns,
        warmup_bars: int,
        schema: Optional[str] = None,
        timeframe_col: Optional[str] = None,
    ) -> None:
        self.engine = engine
        self.symbol = symbol
        self.timeframe = timeframe
        self.columns = columns
        self.warmup_bars = int(max(warmup_bars, 1))
        self.schema = schema
        self.timeframe_col = timeframe_col
        self._qualified_table = self._qualify_table(table, schema)

    @staticmethod
    def _qualify_table(table: str, schema: Optional[str]) -> str:
        if "." in table:
            return table
        if schema:
            return f"{schema}.{table}"
        return table

    def _base_select(self) -> str:
        cols = self.columns
        return (
            f"SELECT {cols.ts} AS ts, {cols.open} AS open, {cols.high} AS high, "
            f"{cols.low} AS low, {cols.close} AS close, {cols.volume} AS volume, "
            f"{cols.symbol} AS symbol "
            f"FROM {self._qualified_table} "
            f"WHERE {cols.symbol} = :symbol"
        )

    def bootstrap(self) -> pd.DataFrame:
        """Load the warm-up window of bars in ascending order."""

        limit = self.warmup_bars
        base = self._base_select()
        if self.timeframe_col:
            base += f" AND {self.timeframe_col} = :timeframe"
        base += f" ORDER BY {self.columns.ts} DESC LIMIT {limit}"
        params: Dict[str, object] = {"symbol": self.symbol}
        if self.timeframe_col:
            params["timeframe"] = self.timeframe
        with self.engine.connect() as conn:
            df = pd.read_sql(text(base), conn, params=params)
        if df.empty:
            return df
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df = df.sort_values("ts").reset_index(drop=True)
        return df

    def poll_last_bar(self, last_ts_seen: Optional[pd.Timestamp]) -> pd.DataFrame:
        """Fetch the most recent completed bar strictly after ``last_ts_seen``."""

        params: Dict[str, object] = {"symbol": self.symbol}
        base = self._base_select()
        if self.timeframe_col:
            base += f" AND {self.timeframe_col} = :timeframe"
            params["timeframe"] = self.timeframe
        if last_ts_seen is None:
            constraint = ""
        else:
            last_ts = pd.Timestamp(last_ts_seen)
            if last_ts.tzinfo is None:
                last_ts = last_ts.tz_localize("UTC")
            else:
                last_ts = last_ts.tz_convert("UTC")
            constraint = f" AND {self.columns.ts} > :last_ts"
            params["last_ts"] = last_ts.isoformat()
        query = (
            f"SELECT * FROM ("
            f"{base}{constraint} ORDER BY {self.columns.ts} DESC LIMIT 1"
            f") AS latest ORDER BY ts ASC"
        )
        with self.engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params=params)
        if df.empty:
            return df
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df = df.sort_values("ts").reset_index(drop=True)
        return df


__all__ = ["FeedColumns", "MySQLPollFeed"]
