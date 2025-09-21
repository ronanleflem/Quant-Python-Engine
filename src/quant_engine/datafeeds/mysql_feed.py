"""MySQL data feed helpers."""
from __future__ import annotations

import os
from typing import Optional, List, Dict

import pandas as pd
from sqlalchemy import create_engine, text


def _qualify_table(table: str, schema: Optional[str]) -> str:
    if "." in table:
        return table
    return f"{schema}.{table}" if schema else table


def load_ohlcv_mysql(
    connection_url: Optional[str],
    env_var: Optional[str],
    schema: Optional[str],
    table: str,
    symbols: List[str],
    timeframe: str,
    start: str,
    end: str,
    cols: Dict[str, str],
    timeframe_col: Optional[str] = "timeframe",
    extra_where: Optional[str] = None,
    chunk_minutes: int = 0,
) -> pd.DataFrame:
    """Read OHLCV data from MySQL into a pandas dataframe."""

    url = connection_url or (os.environ.get(env_var) if env_var else None)
    if not url:
        raise RuntimeError("MySQL URL manquante (connection_url/env_var).")
    engine = create_engine(url)

    table_q = _qualify_table(table, schema)
    ts_col = cols["ts"]
    sym_col = cols["symbol"]
    o_col = cols["open"]
    h_col = cols["high"]
    l_col = cols["low"]
    c_col = cols["close"]
    v_col = cols["volume"]

    symbols_in = ",".join([f":s{i}" for i, _ in enumerate(symbols)])
    tf_filter = f"AND {timeframe_col} = :tf" if timeframe_col else ""
    extra = f"AND ({extra_where})" if extra_where else ""

    base_sql = f"""
      SELECT {ts_col} AS ts, {sym_col} AS symbol,
             {o_col} AS open, {h_col} AS high, {l_col} AS low, {c_col} AS close, {v_col} AS volume
      FROM {table_q}
      WHERE {sym_col} IN ({symbols_in})
        {tf_filter}
        AND {ts_col} >= :start AND {ts_col} <= :end
        {extra}
      ORDER BY {ts_col} ASC
    """

    params = {**{f"s{i}": s for i, s in enumerate(symbols)}, "start": start, "end": end}
    if timeframe_col:
        params["tf"] = timeframe

    if chunk_minutes and chunk_minutes > 0:
        start_dt = pd.to_datetime(start, utc=True)
        end_dt = pd.to_datetime(end, utc=True)
        out = []
        cur = start_dt
        while cur <= end_dt:
            nxt = min(cur + pd.Timedelta(minutes=chunk_minutes), end_dt)
            p = params.copy()
            p["start"] = cur.isoformat()
            p["end"] = nxt.isoformat()
            out.append(pd.read_sql(text(base_sql), engine, params=p))
            cur = nxt + pd.Timedelta(minutes=1)
        df = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    else:
        df = pd.read_sql(text(base_sql), engine, params=params)

    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.sort_values(["symbol", "ts"]).reset_index(drop=True)


__all__ = ["load_ohlcv_mysql"]
