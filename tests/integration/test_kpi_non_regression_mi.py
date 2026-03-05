from __future__ import annotations

import json
import math
import sys
import types
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

if "pymysql" not in sys.modules:
    pymysql_mod = types.ModuleType("pymysql")
    cursors_mod = types.ModuleType("pymysql.cursors")

    class _DictCursor:  # pragma: no cover - import shim
        pass

    cursors_mod.DictCursor = _DictCursor
    pymysql_mod.cursors = cursors_mod
    sys.modules["pymysql"] = pymysql_mod
    sys.modules["pymysql.cursors"] = cursors_mod

from quant_engine.backtest import runner as backtest_runner
from quant_engine.performance.dca_builder import build_dca_performance_from_signals
from quant_engine.performance.stress_tests import run_scenarios_on_returns


GOLDEN_PATH = Path("tests/data/golden/kpi_non_regression_mi.json")


@dataclass
class _FakeSignal:
    strategy_id: str
    symbol: str
    asset_class: str
    side: str
    ts_open_utc: datetime
    qty: float
    meta: Mapping[str, Any]


def _load_golden() -> dict[str, Any]:
    return json.loads(GOLDEN_PATH.read_text())


def _assert_kpis_close(actual: Mapping[str, float], expected: Mapping[str, float], *, abs_tol: float, rel_tol: float) -> None:
    diffs: list[str] = []
    for key, expected_value in expected.items():
        if key not in actual:
            diffs.append(f"{key}: missing in actual")
            continue
        actual_value = float(actual[key])
        expected_float = float(expected_value)
        delta = abs(actual_value - expected_float)
        allowed = max(abs_tol, rel_tol * max(abs(actual_value), abs(expected_float)))
        if not math.isclose(actual_value, expected_float, rel_tol=rel_tol, abs_tol=abs_tol):
            diffs.append(
                f"{key}: actual={actual_value:.12g}, expected={expected_float:.12g}, "
                f"delta={delta:.6g}, allowed={allowed:.6g}"
            )

    extra_keys = sorted(set(actual.keys()) - set(expected.keys()))
    if extra_keys:
        diffs.append(f"unexpected KPI keys: {', '.join(extra_keys)}")

    assert not diffs, "\n".join(["KPI non-regression mismatch:", *diffs])


def _run_dca_reference() -> dict[str, float]:
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)

    def _signal(cycle_id: int, side: str, day: int, *, qty: float = 1.0, action: str | None = None) -> _FakeSignal:
        meta: dict[str, Any] = {"cycle_id": cycle_id}
        if action is not None:
            meta["action"] = action
        return _FakeSignal(
            strategy_id="dca",
            symbol="ABC",
            asset_class="EQUITY",
            side=side,
            ts_open_utc=base.replace(day=day),
            qty=qty,
            meta=meta,
        )

    signals = [
        _signal(1, "BUY", 1, qty=0.0),
        _signal(1, "SELL", 2, action="take_profit"),
        _signal(2, "BUY", 3),
        _signal(3, "BUY", 4),
        _signal(3, "SELL", 5, action="take_profit"),
    ]
    ohlc = {
        "ABC": pd.DataFrame(
            {
                "ts": pd.date_range(start="2024-01-01", periods=5, freq="D", tz="UTC"),
                "close": [100.0, 110.0, 95.0, 90.0, 80.0],
            }
        )
    }

    run, _trades = build_dca_performance_from_signals(
        strategy_id="dca",
        run_id="kpi-baseline",
        asset_class="EQUITY",
        universe="ABC",
        timeframe="1D",
        signals_by_symbol={"ABC": signals},
        ohlc_by_symbol=ohlc,
        config={"capital_per_unit": 100.0},
    )
    return {
        "win_count": float(run.win_count),
        "loss_count": float(run.loss_count),
        "total_return": float(run.total_return),
        "max_drawdown": float(run.max_drawdown),
        "average_trade": float(run.average_trade),
    }


def _run_stress_reference() -> dict[str, float]:
    stress = run_scenarios_on_returns(
        [10.0, -4.0, 6.0, -2.0, 3.0],
        parameters={
            "initial_capital": 1000.0,
            "regime_labels": ["bull", "regime_shift", "vol_spike", "bull", "bull"],
            "regime_shift_shock_multiplier": 2.0,
            "vol_spike_multiplier": 1.75,
            "scenarios": [
                {"name": "crash_mid", "type": "crash", "shock_pct": -0.1, "index": 1},
                {"name": "vol_event", "type": "volatility", "vol_multiplier": 2.0},
            ],
        },
    )
    crash = stress["distributions"]["scenarios"]["crash_mid"]["metrics"]
    vol = stress["distributions"]["scenarios"]["vol_event"]["metrics"]
    return {
        "crash_mid_return_pct": float(crash["return_pct"]),
        "crash_mid_max_drawdown_pct": float(crash["max_drawdown_pct"]),
        "vol_event_return_pct": float(vol["return_pct"]),
        "vol_event_max_drawdown_pct": float(vol["max_drawdown_pct"]),
    }


def test_kpi_non_regression_mi() -> None:
    golden = _load_golden()
    abs_tol = float(golden["tolerances"]["abs"])
    rel_tol = float(golden["tolerances"]["rel"])

    backtest_spec = backtest_runner.load_backtest_spec(Path(golden["references"]["backtest"]["spec"]))
    backtest_result = backtest_runner.run_backtest_from_spec(backtest_spec)
    actual_backtest = {k: float(v) for k, v in backtest_result["summary"].items()}
    _assert_kpis_close(
        actual_backtest,
        golden["references"]["backtest"]["kpis"],
        abs_tol=abs_tol,
        rel_tol=rel_tol,
    )

    actual_dca = _run_dca_reference()
    _assert_kpis_close(
        actual_dca,
        golden["references"]["dca"]["kpis"],
        abs_tol=abs_tol,
        rel_tol=rel_tol,
    )

    actual_stress = _run_stress_reference()
    _assert_kpis_close(
        actual_stress,
        golden["references"]["stress"]["kpis"],
        abs_tol=abs_tol,
        rel_tol=rel_tol,
    )
