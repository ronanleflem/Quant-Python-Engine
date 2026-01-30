"""Integration tests: DCA builder + stress tests payload."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

import pandas as pd

from quant_engine.performance.dca_builder import build_dca_performance_from_signals


@dataclass
class FakeSignal:
    strategy_id: str
    symbol: str
    asset_class: str
    side: str
    ts_open_utc: datetime
    qty: float
    meta: Mapping[str, Any]


def _make_ohlc(closes: list[float], start: str = "2024-01-01") -> pd.DataFrame:
    dates = pd.date_range(start=start, periods=len(closes), freq="D", tz="UTC")
    return pd.DataFrame({"ts": dates, "close": closes})


def _make_signal(
    *,
    cycle_id: int,
    side: str,
    ts: datetime,
    qty: float = 1.0,
    action: str | None = None,
    symbol: str = "ABC",
) -> FakeSignal:
    meta: dict[str, Any] = {"cycle_id": cycle_id}
    if action is not None:
        meta["action"] = action
    return FakeSignal(
        strategy_id="dca",
        symbol=symbol,
        asset_class="EQUITY",
        side=side,
        ts_open_utc=ts,
        qty=qty,
        meta=meta,
    )


def test_dca_builder_attaches_monte_carlo_stress_tests() -> None:
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    signals = [
        _make_signal(cycle_id=1, side="BUY", ts=base, qty=1.0),
        _make_signal(cycle_id=1, side="SELL", ts=base.replace(day=2), action="take_profit"),
    ]
    ohlc = {"ABC": _make_ohlc([100.0, 110.0], start="2024-01-01")}
    config = {
        "initial_capital": 1_000.0,
        "capital_per_unit": 100.0,
        "stress_tests": {"enabled": True, "monte_carlo": {"n_sims": 10, "seed": 1}},
    }

    run, _trades = build_dca_performance_from_signals(
        strategy_id="dca",
        run_id="run-stress",
        asset_class="EQUITY",
        universe="ABC",
        timeframe="1D",
        signals_by_symbol={"ABC": signals},
        ohlc_by_symbol=ohlc,
        config=config,
    )

    stress_tests = run.extra.get("stress_tests", {})
    assert "monte_carlo_level1" in stress_tests
    level1 = stress_tests["monte_carlo_level1"]
    assert "metrics" in level1
    assert "final_capital" in level1["metrics"]
    assert level1["parameters"]["n_simulations"] == 10
