from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

import pandas as pd
import pytest

from quant_engine.performance.backtest_builder import build_backtest_payload
from quant_engine.performance.dca_builder import build_backend_payload_for_java


@dataclass
class FakeSignal:
    strategy_id: str
    symbol: str
    asset_class: str
    side: str
    ts_open_utc: datetime
    qty: float
    meta: Mapping[str, Any]


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


def _make_ohlc(closes: list[float], start: str = "2024-01-01") -> pd.DataFrame:
    dates = pd.date_range(start=start, periods=len(closes), freq="D", tz="UTC")
    return pd.DataFrame({"ts": dates, "close": closes})


def test_backtest_payload_exposes_segmentation_by_regime_and_magnet_failure() -> None:
    payload = build_backtest_payload(
        strategy_id="seg",
        run_id="run-seg",
        asset_class="EQUITY",
        symbol="ABC",
        timeframe="1D",
        trades=[
            {
                "ts_entry": "2024-01-01T00:00:00Z",
                "ts_exit": "2024-01-02T00:00:00Z",
                "price_entry": 100.0,
                "price_exit": 105.0,
                "quantity": 1.0,
                "pnl": 5.0,
                "side": "LONG",
                "meta": {"regime": "bull", "magnet_failure": False},
            },
            {
                "ts_entry": "2024-01-03T00:00:00Z",
                "ts_exit": "2024-01-04T00:00:00Z",
                "price_entry": 100.0,
                "price_exit": 95.0,
                "quantity": 1.0,
                "pnl": -5.0,
                "side": "LONG",
                "regime": "bear",
                "magnet_failure": True,
            },
            {
                "ts_entry": "2024-01-05T00:00:00Z",
                "ts_exit": "2024-01-06T00:00:00Z",
                "price_entry": 100.0,
                "price_exit": 103.0,
                "quantity": 1.0,
                "pnl": 3.0,
                "side": "LONG",
                "meta": {"regime": "bull", "magnet_failure": True},
            },
        ],
        equity=[0.0, 5.0, 0.0, 3.0],
        start_ts="2024-01-01T00:00:00Z",
        end_ts="2024-01-06T00:00:00Z",
    )

    segmentation = payload["run"]["extra"]["segmentation"]
    bull = segmentation["regime"]["bull"]
    bear = segmentation["regime"]["bear"]
    failed = segmentation["magnet_failure"]["True"]
    stable = segmentation["magnet_failure"]["False"]

    assert bull["trades"] == 2
    assert bull["wins"] == 2
    assert bull["losses"] == 0
    assert bull["winrate_pct"] == pytest.approx(100.0)
    assert bear["trades"] == 1
    assert bear["losses"] == 1
    assert failed["trades"] == 2
    assert stable["trades"] == 1


def test_dca_payload_exposes_segmentation_by_regime_and_magnet_failure() -> None:
    base = datetime(2024, 6, 1, tzinfo=timezone.utc)
    signals = [
        _make_signal(cycle_id=1, side="BUY", ts=base, qty=1.0),
        _make_signal(
            cycle_id=1,
            side="SELL",
            ts=base.replace(day=2),
            action="take_profit",
            extra_meta={"regime": "bull", "magnet_failure": False},
        ),
        _make_signal(cycle_id=2, side="BUY", ts=base.replace(day=3), qty=1.0),
        _make_signal(
            cycle_id=2,
            side="SELL",
            ts=base.replace(day=4),
            action="take_profit",
            extra_meta={"regime": "bear", "magnet_failure": True},
        ),
    ]

    payload = build_backend_payload_for_java(
        strategy_id="dca",
        run_id="run-seg-dca",
        asset_class="EQUITY",
        universe="ABC",
        timeframe="1D",
        signals_by_symbol={"ABC": signals},
        ohlc_by_symbol={"ABC": _make_ohlc([100.0, 110.0, 120.0, 100.0], start="2024-06-01")},
        config={"capital_per_unit": 100.0},
    )

    segmentation = payload["run"]["extra"]["segmentation"]
    bull = segmentation["regime"]["bull"]
    bear = segmentation["regime"]["bear"]
    failed = segmentation["magnet_failure"]["True"]
    stable = segmentation["magnet_failure"]["False"]

    assert bull["trades"] == 1
    assert bull["wins"] == 1
    assert bear["trades"] == 1
    assert bear["losses"] == 1
    assert failed["trades"] == 1
    assert stable["trades"] == 1
