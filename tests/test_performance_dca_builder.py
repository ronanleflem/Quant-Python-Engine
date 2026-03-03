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
    symbol: str = "ABC",
    extra_meta: Mapping[str, Any] | None = None,
) -> FakeSignal:
    meta: dict[str, Any] = {"cycle_id": cycle_id}
    if action is not None:
        meta["action"] = action
    if extra_meta:
        meta.update(extra_meta)
    return FakeSignal(
        strategy_id="dca",
        symbol=symbol,
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
    assert isinstance(run.extra, dict)
    dca_score = run.extra.get("dca_composite_score")
    assert isinstance(dca_score, dict)
    assert 0.0 <= dca_score["score"] <= 1.0
    assert dca_score["edge"] in {"weak", "medium", "strong"}
    assert run.extra["capital_efficiency_index"] is not None
    assert run.extra["return_over_stress_ratio"]["version"] == "return_over_stress_ratio_v1"
    assert run.extra["data_contract_version"] == "dca-java-import-v2"


def test_build_dca_performance_multiple_buys_and_tp_metrics() -> None:
    base = datetime(2024, 3, 1, tzinfo=timezone.utc)
    signals = [
        _make_signal(cycle_id=1, side="BUY", ts=base, qty=1.0),
        _make_signal(cycle_id=1, side="BUY", ts=base.replace(day=2), qty=1.0),
        _make_signal(cycle_id=1, side="BUY", ts=base.replace(day=3), qty=1.0),
        _make_signal(cycle_id=1, side="SELL", ts=base.replace(day=4), action="take_profit"),
        _make_signal(cycle_id=2, side="BUY", ts=base.replace(day=5), qty=1.0),
        _make_signal(cycle_id=2, side="BUY", ts=base.replace(day=6), qty=1.0),
        _make_signal(cycle_id=2, side="SELL", ts=base.replace(day=7), action="take_profit"),
    ]
    ohlc = {"ABC": _make_ohlc([100.0, 90.0, 80.0, 120.0, 110.0, 105.0, 90.0], start="2024-03-01")}
    run, trades = build_dca_performance_from_signals(
        strategy_id="dca",
        run_id="run-3",
        asset_class="EQUITY",
        universe="ABC",
        timeframe="1D",
        signals_by_symbol={"ABC": signals},
        ohlc_by_symbol=ohlc,
        config={"initial_capital": 1_000.0, "capital_per_unit": 100.0},
    )

    trades_by_cycle = {trade.cycle_id: trade for trade in trades}
    assert trades_by_cycle[1].gross_pnl_pct == pytest.approx(33.333333, rel=1e-5)
    assert trades_by_cycle[2].gross_pnl_pct == pytest.approx(-16.279070, rel=1e-5)
    assert trades_by_cycle[1].meta["avg_entry_price"] == pytest.approx(90.0)
    assert trades_by_cycle[2].meta["avg_entry_price"] == pytest.approx(107.5)
    assert run.win_count == 1
    assert run.loss_count == 1
    assert run.max_drawdown == pytest.approx(16.279070, rel=1e-5)
    assert run.total_return == pytest.approx(
        trades_by_cycle[1].gross_pnl_pct + trades_by_cycle[2].gross_pnl_pct,
        rel=1e-5,
    )


def test_build_dca_performance_multi_symbol_sets_symbol_none() -> None:
    base = datetime(2024, 4, 1, tzinfo=timezone.utc)
    signals_abc = [
        _make_signal(cycle_id=1, side="BUY", ts=base, qty=1.0, symbol="ABC"),
        _make_signal(cycle_id=1, side="SELL", ts=base.replace(day=2), action="take_profit", symbol="ABC"),
    ]
    signals_xyz = [
        _make_signal(cycle_id=1, side="BUY", ts=base, qty=1.0, symbol="XYZ"),
        _make_signal(cycle_id=1, side="SELL", ts=base.replace(day=2), action="take_profit", symbol="XYZ"),
    ]
    ohlc = {
        "ABC": _make_ohlc([100.0, 110.0], start="2024-04-01"),
        "XYZ": _make_ohlc([200.0, 210.0], start="2024-04-01"),
    }
    run, trades = build_dca_performance_from_signals(
        strategy_id="dca",
        run_id="run-4",
        asset_class="EQUITY",
        universe="MULTI",
        timeframe="1D",
        signals_by_symbol={"ABC": signals_abc, "XYZ": signals_xyz},
        ohlc_by_symbol=ohlc,
        config={"capital_per_unit": 100.0},
    )

    assert run.symbol is None
    assert {trade.symbol for trade in trades} == {"ABC", "XYZ"}


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
    assert run["extra"]["capital_efficiency_index"] is not None
    assert run["extra"]["return_over_stress_ratio"]["version"] == "return_over_stress_ratio_v1"
    assert run["extra"]["data_contract_version"] == "dca-java-import-v2"
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


def test_to_backend_payload_has_non_null_required_fields() -> None:
    base = datetime(2024, 5, 1, tzinfo=timezone.utc)
    signals = [
        _make_signal(cycle_id=1, side="BUY", ts=base, qty=1.0),
        _make_signal(cycle_id=1, side="BUY", ts=base.replace(day=2), qty=1.0),
        _make_signal(cycle_id=1, side="SELL", ts=base.replace(day=3), action="take_profit"),
    ]
    ohlc = {"ABC": _make_ohlc([100.0, 95.0, 105.0], start="2024-05-01")}
    run, trades = build_dca_performance_from_signals(
        strategy_id="dca",
        run_id="run-5",
        asset_class="EQUITY",
        universe="ABC",
        timeframe="1D",
        signals_by_symbol={"ABC": signals},
        ohlc_by_symbol=ohlc,
        config={"initial_capital": 1_000.0, "capital_per_unit": 100.0},
    )
    payload = build_backend_payload_for_java(
        strategy_id="dca",
        run_id="run-5",
        asset_class="EQUITY",
        universe="ABC",
        timeframe="1D",
        signals_by_symbol={"ABC": signals},
        ohlc_by_symbol=ohlc,
        config={"initial_capital": 1_000.0, "capital_per_unit": 100.0},
    )

    run_payload = payload["run"]
    trade_payload = payload["trades"][0]

    for key in (
        "strategyId",
        "runId",
        "assetClass",
        "startTsUtc",
        "endTsUtc",
        "winCount",
        "lossCount",
        "totalReturn",
        "maxDrawdown",
        "averageTrade",
        "totalNetReturn",
        "netWinCount",
        "netLossCount",
        "averageNetTrade",
        "initialCapital",
        "finalCapital",
        "returnPct",
        "maxDrawdownPct",
        "winratePct",
    ):
        assert run_payload[key] is not None

    for key in (
        "strategyId",
        "runId",
        "symbol",
        "assetClass",
        "side",
        "cycleId",
        "entryTimeUtc",
        "exitTimeUtc",
        "entryPrice",
        "exitPrice",
        "quantity",
        "grossPnl",
        "grossPnlPct",
        "meta",
    ):
        assert trade_payload[key] is not None

    assert trade_payload["grossPnlPct"] == pytest.approx(trades[0].gross_pnl_pct)


def test_build_dca_performance_includes_rolling_regimes_and_underperformance() -> None:
    base = datetime(2020, 1, 1, tzinfo=timezone.utc)
    signals = [
        _make_signal(cycle_id=1, side="BUY", ts=base, qty=1.0),
        _make_signal(cycle_id=1, side="SELL", ts=base.replace(year=2021), action="take_profit"),
    ]

    dates = pd.date_range(start="2020-01-01", periods=365 * 4, freq="D", tz="UTC")
    # alternating positive/negative yearly blocks to force regime and underperformance detection
    closes = []
    price = 100.0
    for i, _ in enumerate(dates):
        if i < 365 * 2:
            price *= 0.9998
        else:
            price *= 1.0004
        closes.append(price)

    run, _trades = build_dca_performance_from_signals(
        strategy_id="dca",
        run_id="run-rolling",
        asset_class="EQUITY",
        universe="ABC",
        timeframe="1D",
        signals_by_symbol={"ABC": signals},
        ohlc_by_symbol={"ABC": pd.DataFrame({"ts": dates, "close": closes})},
        config={"rolling_windows": {"years": [3], "step_months": 12}},
    )

    rolling = run.extra["rolling_windows"]
    assert rolling["series"]
    assert rolling["series"][0]["window_years"] == 3
    assert rolling["series"][0]["regime"] in {"bull", "bear", "sideways"}
    assert "duration_windows" in rolling["underperformance"]
    assert "severity_pct_points" in rolling["underperformance"]
