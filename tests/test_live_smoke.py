"""Smoke tests for the live runner pipeline."""
from __future__ import annotations

import os
import pandas as pd
from sqlalchemy import create_engine, text

from quant_engine.api.schemas import LiveSpec
from quant_engine.live.runner import LiveRunner


def _make_rows(start: pd.Timestamp, count: int, close: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for i in range(count):
        ts = start + pd.Timedelta(minutes=i)
        rows.append(
            {
                "symbol": "EURUSD",
                "ts_utc": ts.isoformat(),
                "open": close,
                "high": close + 0.1,
                "low": close - 0.1,
                "close": close,
                "volume": 1000.0,
            }
        )
    return rows


def test_live_runner_emits_once(tmp_path, monkeypatch):
    db_path = tmp_path / "marketdata.db"
    url = f"sqlite:///{db_path}"
    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE ohlcv_m1 (
                    symbol TEXT,
                    ts_utc TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL
                )
                """
            )
        )
        start = pd.Timestamp("2024-01-01 00:00:00+00:00")
        warm_rows = _make_rows(start, 300, close=1.0)
        conn.execute(
            text(
                """
                INSERT INTO ohlcv_m1 (symbol, ts_utc, open, high, low, close, volume)
                VALUES (:symbol, :ts_utc, :open, :high, :low, :close, :volume)
                """
            ),
            warm_rows,
        )
    read_env = "QE_LIVE_TEST_READ_URL"
    monkeypatch.setenv(read_env, url)
    spec_dict = {
        "data": {
            "mysql_read_env": read_env,
            "schema": "",
            "table": "ohlcv_m1",
            "symbol_col": "symbol",
            "ts_col": "ts_utc",
            "open_col": "open",
            "high_col": "high",
            "low_col": "low",
            "close_col": "close",
            "volume_col": "volume",
            "symbols": ["EURUSD"],
            "timeframe": "M1",
            "warmup_bars": 300,
            "poll_interval_sec": 1,
            "timeframe_col": None,
        },
        "strategy": {
            "strategy_id": "TEST_STRAT",
            "filters": [],
            "rules": [
                {"type": "cross_over", "params": {"fast_ema": 20, "slow_ema": 50, "side": "LONG"}}
            ],
            "risk_gates": [],
            "tp_sl_mgmt": {"type": "fixed_rr", "params": {"rr": 2.0, "sl_mode": "atr", "atr_window": 14, "atr_mult": 1.0}},
        },
        "destinations": {"write_db": None, "emit_java": {"enabled": False}},
    }
    spec = LiveSpec.model_validate(spec_dict)
    runner = LiveRunner(spec)
    state = runner._states["EURUSD"]
    assert not state.warm
    runner._run_cycle()
    assert state.warm
    assert len(state.history) == 300
    last_ts = pd.Timestamp(state.history["ts"].iloc[-1])
    if last_ts.tzinfo is None:
        expected_ts = last_ts.tz_localize("UTC")
    else:
        expected_ts = last_ts.tz_convert("UTC")
    assert state.last_ts_seen == expected_ts
    next_bar = {
        "symbol": "EURUSD",
        "ts_utc": (start + pd.Timedelta(minutes=300)).isoformat(),
        "open": 10.0,
        "high": 10.2,
        "low": 9.8,
        "close": 10.0,
        "volume": 1200.0,
    }
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO ohlcv_m1 (symbol, ts_utc, open, high, low, close, volume)
                VALUES (:symbol, :ts_utc, :open, :high, :low, :close, :volume)
                """
            ),
            [next_bar],
        )
    runner._run_cycle()
    assert runner._emitted == 1
    assert len(state.history) == 301
    assert len(state.emitted_hashes) == 1
    emitted_hashes = set(state.emitted_hashes)
    runner._run_cycle()
    assert runner._emitted == 1
    assert state.emitted_hashes == emitted_hashes
