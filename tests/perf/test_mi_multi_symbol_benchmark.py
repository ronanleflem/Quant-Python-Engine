from __future__ import annotations

import json
import statistics
import tracemalloc
import time
import sys
import types

pymysql_module = types.ModuleType("pymysql")
cursors_module = types.ModuleType("pymysql.cursors")
cursors_module.DictCursor = object
pymysql_module.cursors = cursors_module
sys.modules.setdefault("pymysql", pymysql_module)
sys.modules.setdefault("pymysql.cursors", cursors_module)

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from quant_engine.backtest import runner as backtest_runner
from quant_engine.market_intelligence.service import MarketIntelligenceServiceV1


class _DummyStatsAdapter:
    def run(self, spec: dict[str, Any]) -> pd.DataFrame:
        idx = pd.DatetimeIndex([spec["ohlcv"].index[-1]])
        return pd.DataFrame({"symbol": [spec["symbol"]], "event": ["noop"], "p_hat": [0.5]}, index=idx)


class _DummyFiltersAdapter:
    def run(self, ohlcv: pd.DataFrame, rules, **_kwargs) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "hard_mask": [True] * len(ohlcv),
                "score": [1.0] * len(ohlcv),
                "score_pct": [1.0] * len(ohlcv),
                "final_mask": [True] * len(ohlcv),
            },
            index=ohlcv.index,
        )


@dataclass(frozen=True)
class _Case:
    timeframe: str
    symbols: int


def _freq_from_timeframe(timeframe: str) -> str:
    mapping = {"1m": "min", "5m": "5min", "1h": "h"}
    return mapping[timeframe]


def _make_rows(*, symbol: str, timeframe: str, bars: int, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    freq = _freq_from_timeframe(timeframe)
    idx = pd.date_range("2024-01-01", periods=bars, freq=freq, tz="UTC")

    walk = 100.0 + np.cumsum(rng.normal(0.0, 0.2, size=bars))
    spread = np.abs(rng.normal(0.35, 0.05, size=bars))
    close = walk
    open_ = np.concatenate([[walk[0]], walk[:-1]])
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.integers(900, 2200, size=bars)

    rows: list[dict[str, Any]] = []
    for i in range(bars):
        rows.append(
            {
                "timestamp": idx[i].isoformat(),
                "symbol": symbol,
                "open": float(open_[i]),
                "high": float(high[i]),
                "low": float(low[i]),
                "close": float(close[i]),
                "volume": int(volume[i]),
            }
        )
    return rows


def _build_spec(*, symbol: str, timeframe: str, mi_enabled: bool) -> dict[str, Any]:
    return {
        "data": {
            "path": f"synthetic/{symbol}_{timeframe}.csv",
            "symbol": symbol,
            "symbols": [symbol],
            "timeframe": timeframe,
            "start": "2024-01-01",
            "end": "2024-12-31",
        },
        "strategy": {"asset_class": "FX"},
        "signal": {"type": "ema_cross", "params": {"fast": 8, "slow": 21}},
        "tpsl": {"atr_window": 14, "atr_k": 1.0, "r_mult": 1.8},
        "market_intelligence": {"enabled": mi_enabled},
    }


def _run_case(case: _Case, rows_by_symbol: dict[str, list[dict[str, Any]]], repeats: int = 3) -> dict[str, float]:
    mi_service = MarketIntelligenceServiceV1(
        stats_adapter=_DummyStatsAdapter(),
        filters_adapter=_DummyFiltersAdapter(),
        timeframe=case.timeframe,
    )

    def _stub_loader(data_spec_raw, *_args, **_kwargs):
        symbol = str(data_spec_raw.get("symbol") or data_spec_raw.get("symbols", [""])[0])
        return rows_by_symbol[symbol], None

    original_loader = backtest_runner._load_rows
    backtest_runner._load_rows = _stub_loader
    try:
        lat_off, mem_off = [], []
        lat_on, mem_on = [], []
        symbols = sorted(rows_by_symbol)

        for _ in range(repeats):
            tracemalloc.start()
            t0 = time.perf_counter()
            for symbol in symbols:
                backtest_runner.run_backtest_from_spec(_build_spec(symbol=symbol, timeframe=case.timeframe, mi_enabled=False), mi_service=mi_service)
            lat_off.append((time.perf_counter() - t0) * 1000.0)
            _cur, peak = tracemalloc.get_traced_memory()
            mem_off.append(peak / (1024 * 1024))
            tracemalloc.stop()

            tracemalloc.start()
            t1 = time.perf_counter()
            for symbol in symbols:
                backtest_runner.run_backtest_from_spec(_build_spec(symbol=symbol, timeframe=case.timeframe, mi_enabled=True), mi_service=mi_service)
            lat_on.append((time.perf_counter() - t1) * 1000.0)
            _cur, peak = tracemalloc.get_traced_memory()
            mem_on.append(peak / (1024 * 1024))
            tracemalloc.stop()

        off_p50 = statistics.median(lat_off)
        on_p50 = statistics.median(lat_on)
        mem_off_p50 = statistics.median(mem_off)
        mem_on_p50 = statistics.median(mem_on)
        overhead_ms = on_p50 - off_p50
        mem_overhead_mb = mem_on_p50 - mem_off_p50
        symbols_count = float(case.symbols)

        return {
            "latency_off_ms_p50": off_p50,
            "latency_on_ms_p50": on_p50,
            "latency_overhead_ms": overhead_ms,
            "memory_off_mb_p50": mem_off_p50,
            "memory_on_mb_p50": mem_on_p50,
            "memory_overhead_mb": mem_overhead_mb,
            "overhead_per_symbol_ms": overhead_ms / symbols_count,
            "memory_overhead_per_symbol_mb": mem_overhead_mb / symbols_count,
            "run_to_run_cv_off": statistics.pstdev(lat_off) / max(statistics.mean(lat_off), 1e-9),
            "run_to_run_cv_on": statistics.pstdev(lat_on) / max(statistics.mean(lat_on), 1e-9),
        }
    finally:
        backtest_runner._load_rows = original_loader


@pytest.mark.performance
@pytest.mark.mi
def test_mi_multi_symbol_capacity_benchmark() -> None:
    cases = [
        _Case(timeframe="1m", symbols=1),
        _Case(timeframe="1m", symbols=4),
        _Case(timeframe="5m", symbols=4),
        _Case(timeframe="1h", symbols=8),
    ]
    bars = 1200
    results: dict[str, dict[str, float]] = {}

    for case in cases:
        rows_by_symbol = {
            f"SYM{idx:02d}": _make_rows(
                symbol=f"SYM{idx:02d}",
                timeframe=case.timeframe,
                bars=bars,
                seed=1000 + idx,
            )
            for idx in range(case.symbols)
        }
        key = f"{case.timeframe}_{case.symbols}symbols"
        results[key] = _run_case(case, rows_by_symbol=rows_by_symbol, repeats=3)

    out_path = Path("artifacts/perf")
    out_path.mkdir(parents=True, exist_ok=True)
    report_path = out_path / "mi_multi_symbol_benchmark.json"
    report_path.write_text(json.dumps(results, indent=2, sort_keys=True))

    assert report_path.exists()
    assert all(item["latency_on_ms_p50"] > 0 for item in results.values())
    assert all(item["latency_off_ms_p50"] > 0 for item in results.values())
