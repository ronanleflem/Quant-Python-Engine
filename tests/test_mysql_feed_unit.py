from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
from pandas.api.types import is_datetime64tz_dtype

from quant_engine.datafeeds.mysql_feed import load_ohlcv_mysql


def test_mysql_feed_includes_timeframe_and_extra_where(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_create_engine(url: str) -> object:
        captured["url"] = url
        return object()

    def fake_read_sql(sql, engine, params=None):
        captured["sql"] = sql
        captured["params"] = params
        return pd.DataFrame(
            [
                {
                    "ts": "2024-01-01T00:00:00Z",
                    "symbol": "EURUSD",
                    "open": 1.0,
                    "high": 1.0,
                    "low": 1.0,
                    "close": 1.0,
                    "volume": 10.0,
                }
            ]
        )

    monkeypatch.setattr("quant_engine.datafeeds.mysql_feed.create_engine", fake_create_engine)
    monkeypatch.setattr(pd, "read_sql", fake_read_sql)

    df = load_ohlcv_mysql(
        connection_url="mysql://example/db",
        env_var=None,
        schema=None,
        table="ohlcv",
        symbols=["EURUSD"],
        timeframe="M1",
        start="2024-01-01T00:00:00Z",
        end="2024-01-01T00:05:00Z",
        cols={
            "ts": "ts",
            "symbol": "symbol",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        },
        timeframe_col="tf_col",
        extra_where="volume > 0",
    )

    sql_text = captured["sql"].text
    assert "AND tf_col = :tf" in sql_text
    assert "AND (volume > 0)" in sql_text
    assert df.shape[0] == 1


def test_mysql_feed_chunk_minutes_splits_and_sorts(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    engine = object()

    def fake_create_engine(url: str) -> object:
        return engine

    frames = [
        pd.DataFrame(
            [
                {
                    "ts": "2024-01-01T00:15:00Z",
                    "symbol": "B",
                    "open": 1.0,
                    "high": 1.1,
                    "low": 0.9,
                    "close": 1.0,
                    "volume": 10.0,
                }
            ]
        ),
        pd.DataFrame(
            [
                {
                    "ts": "2024-01-01T00:16:00Z",
                    "symbol": "A",
                    "open": 2.0,
                    "high": 2.1,
                    "low": 1.9,
                    "close": 2.0,
                    "volume": 20.0,
                }
            ]
        ),
    ]

    def fake_read_sql(sql, engine_arg, params=None):
        calls.append({"sql": sql, "engine": engine_arg, "params": params})
        return frames[len(calls) - 1]

    monkeypatch.setattr("quant_engine.datafeeds.mysql_feed.create_engine", fake_create_engine)
    monkeypatch.setattr(pd, "read_sql", fake_read_sql)

    df = load_ohlcv_mysql(
        connection_url="mysql://example/db",
        env_var=None,
        schema=None,
        table="ohlcv",
        symbols=["B", "A"],
        timeframe="M1",
        start="2024-01-01T00:00:00Z",
        end="2024-01-01T00:30:00Z",
        cols={
            "ts": "ts",
            "symbol": "symbol",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        },
        chunk_minutes=15,
    )

    assert len(calls) == 2
    assert df.shape[0] == 2
    assert list(df["symbol"]) == ["A", "B"]
    assert is_datetime64tz_dtype(df["ts"])
    assert str(df["ts"].dt.tz) == "UTC"
