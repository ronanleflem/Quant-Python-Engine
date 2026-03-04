from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping
import sys
import types

import pandas as pd

pymysql_module = types.ModuleType("pymysql")
cursors_module = types.ModuleType("pymysql.cursors")
cursors_module.DictCursor = object
pymysql_module.cursors = cursors_module
sys.modules.setdefault("pymysql", pymysql_module)
sys.modules.setdefault("pymysql.cursors", cursors_module)

from quant_engine.backtest import runner as backtest_runner
from quant_engine.config import reset_settings_cache

GOLDEN_DIR = Path("tests/data/golden")
SNAPSHOT_OFF = GOLDEN_DIR / "payload_mi_off.json"
SNAPSHOT_ON = GOLDEN_DIR / "payload_mi_on.json"


class _DummyMIService:
    def build_snapshot(self, symbol: str, ohlcv: pd.DataFrame) -> Dict[str, Any]:
        return {
            "symbol": symbol,
            "features": {
                "mi_mode": "on",
                "feat_regime": "trend",
                "feat_liquidity": "high",
            },
            "ohlcv_rows": len(ohlcv),
        }


def _build_rows() -> list[dict[str, Any]]:
    return [
        {
            "timestamp": "2024-01-01T00:00:00Z",
            "symbol": "EURUSD",
            "open": 1.1,
            "high": 1.2,
            "low": 1.0,
            "close": 1.15,
            "volume": 100,
        },
        {
            "timestamp": "2024-01-02T00:00:00Z",
            "symbol": "EURUSD",
            "open": 1.2,
            "high": 1.3,
            "low": 1.1,
            "close": 1.25,
            "volume": 110,
        },
    ]


BACKTEST_SPEC = {
    "run_id": "snapshot-mi-run",
    "data": {"source": "csv", "path": "unused.csv", "symbol": "EURUSD", "start": "2024-01-01", "end": "2024-01-02"},
    "strategy": {"asset_class": "FX", "strategy_id": "snapshot-mi"},
    "signal": {"type": "ema_cross", "params": {"fast": 1, "slow": 2}},
    "persistence": {"enabled": False},
}


def _run_with_mi_toggle(monkeypatch, *, enabled: bool) -> Dict[str, Any]:
    monkeypatch.setattr(backtest_runner, "_load_rows", lambda *_args, **_kwargs: (_build_rows(), None))

    observed: Dict[str, Any] = {}
    original_build_signal = backtest_runner._build_signal

    def wrapped_build_signal(spec_input, rows, features=None):
        observed["signal_features"] = features
        return original_build_signal(spec_input, rows, features)

    monkeypatch.setattr(backtest_runner, "_build_signal", wrapped_build_signal)

    monkeypatch.setenv("MARKET_INTELLIGENCE_ENABLED", "true" if enabled else "false")
    reset_settings_cache()

    result = backtest_runner.run_backtest_from_spec(BACKTEST_SPEC, mi_service=_DummyMIService())

    return {
        "payload": result["payload"],
        "signal_features": observed["signal_features"],
    }


def _assert_snapshot(path: Path, actual: Mapping[str, Any]) -> None:
    if os.getenv("QE_UPDATE_SNAPSHOTS", "0") == "1":
        path.write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n")
    expected = json.loads(path.read_text())
    assert actual == expected


def test_payload_snapshot_mi_on_off(monkeypatch) -> None:
    snapshot_off = _run_with_mi_toggle(monkeypatch, enabled=False)
    snapshot_on = _run_with_mi_toggle(monkeypatch, enabled=True)

    # Invariants: canonical payload stays byte-for-byte stable regardless of MI toggle.
    assert snapshot_off["payload"] == snapshot_on["payload"]

    # MI OFF must keep the contract but expose an empty feature map.
    assert snapshot_off["signal_features"] == {
        "symbol": "EURUSD",
        "features": {},
        "ohlcv_rows": 2,
    }

    # MI ON may add fields under `features`; this is a tolerated exception.
    assert snapshot_on["signal_features"]["symbol"] == "EURUSD"
    assert snapshot_on["signal_features"]["ohlcv_rows"] == 2
    assert snapshot_on["signal_features"]["features"] == {
        "mi_mode": "on",
        "feat_regime": "trend",
        "feat_liquidity": "high",
    }

    _assert_snapshot(SNAPSHOT_OFF, snapshot_off)
    _assert_snapshot(SNAPSHOT_ON, snapshot_on)
