import os
from typing import List

import pytest

from quant_engine.api.schemas import DataInputSpec, MySQLDataSpec
from quant_engine.core.dataset import load_ohlcv


def _parse_symbols() -> List[str]:
    symbols_env = os.environ.get("QE_MARKETDATA_MYSQL_SYMBOLS")
    if symbols_env:
        return [s.strip() for s in symbols_env.split(",") if s.strip()]
    symbol = os.environ.get("QE_MARKETDATA_MYSQL_SYMBOL", "EURUSD")
    return [symbol]


def _parse_optional(name: str):
    value = os.environ.get(name)
    if value is None:
        return None
    if value.strip().lower() in {"", "none", "null"}:
        return None
    return value


@pytest.mark.skipif(
    not os.environ.get("QE_MARKETDATA_MYSQL_URL"),
    reason="no MySQL url",
)
def test_mysql_feed_smoke() -> None:
    url = os.environ["QE_MARKETDATA_MYSQL_URL"]
    schema = os.environ.get("QE_MARKETDATA_MYSQL_SCHEMA")
    table = os.environ.get("QE_MARKETDATA_MYSQL_TABLE")
    timeframe_col = _parse_optional("QE_MARKETDATA_MYSQL_TIMEFRAME_COL")
    extra_where = _parse_optional("QE_MARKETDATA_MYSQL_EXTRA_WHERE")
    chunk_minutes = int(os.environ.get("QE_MARKETDATA_MYSQL_CHUNK_MINUTES", "0"))

    mysql_spec = MySQLDataSpec(
        connection_url=url,
        env_var=None,
        schema=schema or None,
        table=table or "ohlcv",
        timeframe_col=timeframe_col,
        extra_where=extra_where,
        chunk_minutes=chunk_minutes,
    )

    spec = DataInputSpec(
        dataset_path=None,
        mysql=mysql_spec,
        symbols=_parse_symbols(),
        timeframe=os.environ.get("QE_MARKETDATA_MYSQL_TIMEFRAME", "M1"),
        start=os.environ.get("QE_MARKETDATA_MYSQL_START", "2024-01-01T00:00:00Z"),
        end=os.environ.get("QE_MARKETDATA_MYSQL_END", "2024-01-01T01:00:00Z"),
    )

    df = load_ohlcv(spec)
    expected = {"ts", "symbol", "open", "high", "low", "close", "volume"}
    assert expected.issubset(df.columns)
