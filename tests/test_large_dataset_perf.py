from __future__ import annotations

from pathlib import Path
import os
import time
import tracemalloc
from contextlib import contextmanager

import pandas as pd
import pytest

from quant_engine.backtest import runner as backtest_runner
from quant_engine.strategies import runner as strategies_runner


LARGE_SOURCE = Path(
    "specs/examples/data/forex/EURUSD_20250101_20250601_1min.csv"
)

if not os.getenv("QE_PERF_TRACE"):
    os.environ["QE_PERF_TRACE"] = "1"


@pytest.fixture(scope="session")
def large_csv_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if not LARGE_SOURCE.exists():
        pytest.skip(f"Missing large dataset: {LARGE_SOURCE}")
    out_dir = tmp_path_factory.mktemp("large_csv")
    out_path = out_dir / "eurusd_large_converted.csv"
    if out_path.exists():
        return out_path

    with _time_block("large_csv_convert"):
        df = pd.read_csv(
            LARGE_SOURCE,
            usecols=["Timestamp", "Open", "High", "Low", "Close", "Volume"],
        )
        df = df.rename(
            columns={
                "Timestamp": "timestamp",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        df["symbol"] = "EURUSD"
        df["ts"] = df["timestamp"]
        df.to_csv(out_path, index=False)
    return out_path


@contextmanager
def _time_block(label: str):
    start = time.monotonic()
    print(f"[perf] {label} start", flush=True)
    try:
        yield
    finally:
        elapsed = time.monotonic() - start
        print(f"[perf] {label} done in {elapsed:.2f}s", flush=True)


def _maybe_check_memory(peak_bytes: int) -> None:
    max_mb_raw = os.getenv("QE_PERF_MAX_MB")
    if not max_mb_raw:
        return
    try:
        max_mb = float(max_mb_raw)
    except Exception:
        return
    peak_mb = peak_bytes / (1024 * 1024)
    assert peak_mb <= max_mb


def _maybe_check_throughput(elapsed: float, n_rows: int) -> None:
    min_rows_raw = os.getenv("QE_PERF_MIN_ROWS_PER_SEC")
    if not min_rows_raw:
        return
    try:
        min_rows = float(min_rows_raw)
    except Exception:
        return
    if elapsed <= 0:
        return
    rows_per_sec = n_rows / elapsed
    assert rows_per_sec >= min_rows


@pytest.mark.slow
def test_backtest_large_csv_trades(large_csv_path: Path) -> None:
    max_seconds_raw = os.getenv("QE_PERF_MAX_SECONDS")
    with open(large_csv_path, "r", encoding="utf-8") as handle:
        n_rows = sum(1 for _ in handle) - 1
    start = time.monotonic()
    tracemalloc.start()
    spec = {
        "strategy": {"strategy_id": "BT_LARGE_CSV", "asset_class": "FX"},
        "data": {
            "dataset_path": str(large_csv_path),
            "symbols": ["EURUSD"],
            "timeframe": "1m",
            "start": "2025-01-01",
            "end": "2025-06-01",
        },
        "signal": {
            "type": "ema_cross",
            "params": {"fast": 1, "slow": 2, "require_crossing": False},
        },
        "tpsl": {"atr_window": 14, "atr_k": 1.0, "r_mult": 2.0},
        "performance": {"initial_capital": 10000},
        "persistence": {"enabled": False},
    }
    with _time_block(f"backtest_large_csv_trades rows={n_rows}"):
        result = backtest_runner.run_backtest_from_spec(spec)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    trades = result.get("payload", {}).get("trades", [])
    assert trades
    elapsed = time.monotonic() - start
    if max_seconds_raw:
        max_seconds = float(max_seconds_raw)
        assert elapsed <= max_seconds
    _maybe_check_memory(peak)
    _maybe_check_throughput(elapsed, n_rows)


@pytest.mark.slow
def test_strategy_large_csv_signals(large_csv_path: Path, monkeypatch) -> None:
    max_seconds_raw = os.getenv("QE_PERF_MAX_SECONDS")
    with open(large_csv_path, "r", encoding="utf-8") as handle:
        n_rows = sum(1 for _ in handle) - 1
    start = time.monotonic()
    tracemalloc.start()
    monkeypatch.delenv("DB_DSN", raising=False)
    spec = {
        "strategy": {
            "strategy_id": "DCA_LARGE_CSV",
            "type": "dca_equity",
            "params": {
                "asset_class": "EQUITY",
                "drawdown_reference": "ATH",
                "grid": [
                    {"dd": -0.1, "weight": 1.0},
                    {"dd": -0.2, "weight": 1.0},
                ],
                "tp_sl": {"enabled": False},
                "require_crossing": False,
            },
        },
        "data": {
            "source": "csv",
            "path": str(large_csv_path),
            "timeframe": "1m",
            "start": "2025-01-01",
            "end": "2025-06-01",
        },
        "universe": [{"symbol": "EURUSD", "asset_class": "EQUITY"}],
        "performance": {"initial_capital": 10000, "capital_per_unit": 100},
    }
    with _time_block(f"strategy_large_csv_signals rows={n_rows}"):
        result = strategies_runner.run_backtest_with_payload(spec)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    counts = result.get("result", {}).get("counts", {})
    assert counts.get("EURUSD", 0) > 0
    elapsed = time.monotonic() - start
    if max_seconds_raw:
        max_seconds = float(max_seconds_raw)
        assert elapsed <= max_seconds
    _maybe_check_memory(peak)
    _maybe_check_throughput(elapsed, n_rows)


@pytest.mark.slow
def test_backtest_large_csv_filters(large_csv_path: Path) -> None:
    max_seconds_raw = os.getenv("QE_PERF_MAX_SECONDS")
    start = time.monotonic()
    spec = {
        "strategy": {"strategy_id": "BT_LARGE_CSV_FILTERS", "asset_class": "FX"},
        "data": {
            "dataset_path": str(large_csv_path),
            "symbols": ["EURUSD"],
            "timeframe": "1m",
            "start": "2025-01-01",
            "end": "2025-06-01",
        },
        "signal": {
            "type": "ema_cross",
            "params": {"fast": 1, "slow": 2, "require_crossing": False},
        },
        "filters": [
            {"type": "day_of_week", "params": {"blocked_days": [6]}}
        ],
        "tpsl": {"atr_window": 14, "atr_k": 1.0, "r_mult": 2.0},
        "performance": {"initial_capital": 10000},
        "persistence": {"enabled": False},
    }
    with _time_block("backtest_large_csv_filters"):
        result = backtest_runner.run_backtest_from_spec(spec)
    trades = result.get("payload", {}).get("trades", [])
    assert trades
    elapsed = time.monotonic() - start
    if max_seconds_raw:
        max_seconds = float(max_seconds_raw)
        assert elapsed <= max_seconds
