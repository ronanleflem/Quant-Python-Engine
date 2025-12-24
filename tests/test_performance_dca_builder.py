"""Tests for DCA performance builder payload and metrics."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

import pandas as pd
import pytest

from quant_engine.performance.dca_builder import (
    build_backend_payload_for_java,
    build_dca_performance_from_signals,
)


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
    extra_meta: Mapping[str, Any] | None = None,
) -> FakeSignal:
    meta: dict[str, Any] = {"cycle_id": cycle_id}
    if action is not None:
        meta["action"] = action
    if extra_meta:
        meta.update(extra_meta)
    return FakeSignal(
        strategy_id="dca",
        symbol="ABC",
        asset_class="EQUITY",
        side=side,
        ts_open_utc=ts,
        qty=qty,
        meta=meta,
    )


def test_build_dca_performance_from_signals_handles_cycles_and_metrics() -> None:
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    signals = [
        _make_signal(cycle_id=1, side="BUY", ts=base, qty=0.0),
        _make_signal(
            cycle_id=1,
            side="SELL",
            ts=base.replace(day=2),
            action="take_profit",
            extra_meta={"be_pct": 5.0},
        ),
        _make_signal(cycle_id=2, side="BUY", ts=base.replace(day=3), qty=1.0),
        _make_signal(cycle_id=3, side="BUY", ts=base.replace(day=4), qty=1.0),
        _make_signal(
            cycle_id=3,
            side="SELL",
            ts=base.replace(day=5),
            action="take_profit",
        ),
    ]
    ohlc = {"ABC": _make_ohlc([100.0, 110.0, 95.0, 90.0, 80.0])}
    run, trades = build_dca_performance_from_signals(
        strategy_id="dca",
        run_id="run-1",
        asset_class="EQUITY",
        universe="ABC",
        timeframe="1D",
        signals_by_symbol={"ABC": signals},
        ohlc_by_symbol=ohlc,
        config={"capital_per_unit": 100.0},
    )

    assert len(trades) == 2
    assert trades[0].quantity > 0
    assert trades[0].quantity == pytest.approx(1.0)
    assert trades[0].gross_pnl_pct == pytest.approx(10.0)
    assert trades[0].meta["break_even_reached"] is True
    assert trades[1].gross_pnl_pct == pytest.approx(-11.111111, rel=1e-5)
    assert run.win_count == 1
    assert run.loss_count == 1
    assert run.total_return == pytest.approx(trades[0].gross_pnl_pct + trades[1].gross_pnl_pct)


def test_build_backend_payload_for_java_structure_and_meta() -> None:
    base = datetime(2024, 2, 1, tzinfo=timezone.utc)
    signals = [
        _make_signal(cycle_id=1, side="BUY", ts=base, qty=1.0),
        _make_signal(
            cycle_id=1,
            side="SELL",
            ts=base.replace(day=2),
            action="take_profit",
            extra_meta={"be_pct": 1.0},
        ),
    ]
    payload = build_backend_payload_for_java(
        strategy_id="dca",
        run_id="run-2",
        asset_class="EQUITY",
        universe=None,
        timeframe=None,
        signals_by_symbol={"ABC": signals},
        ohlc_by_symbol={"ABC": _make_ohlc([100.0, 101.0], start="2024-02-01")},
    )

    assert set(payload.keys()) == {"run", "trades"}
    run = payload["run"]
    trade = payload["trades"][0]
    for key in (
        "strategyId",
        "runId",
        "assetClass",
        "startTsUtc",
        "endTsUtc",
        "winCount",
        "lossCount",
        "totalReturn",
    ):
        assert key in run
    for key in (
        "strategyId",
        "runId",
        "symbol",
        "assetClass",
        "cycleId",
        "entryTimeUtc",
        "exitTimeUtc",
        "quantity",
        "grossPnlPct",
        "meta",
    ):
        assert key in trade
    assert trade["meta"]["break_even_reached"] is True
