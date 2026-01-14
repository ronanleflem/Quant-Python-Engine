from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_engine.backtest import runner as backtest_runner
from quant_engine.core import dataset as dataset_module
from quant_engine.strategies import runner as strategies_runner


DATA_CSV = Path("tests/data/ohlcv.csv")
SPEC_DIR = Path("specs/tests")


def _load_spec(name: str) -> dict:
    return backtest_runner.load_backtest_spec(SPEC_DIR / name)


def _load_source_frame() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV)
    df = df.rename(columns={"timestamp": "ts"})
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def _reset_cache() -> None:
    backtest_runner._ROWS_CACHE.clear()


def _assert_payload(result: dict) -> None:
    payload = result.get("payload", {})
    assert "run" in payload
    assert "trades" in payload
    assert payload["run"]["symbol"] == "EURUSD"


def _patch_fetch(monkeypatch, *, delta=None, mysql=None, java=None) -> None:
    if delta is not None:
        monkeypatch.setattr(strategies_runner, "_fetch_from_delta", delta)
    if mysql is not None:
        monkeypatch.setattr(strategies_runner, "_fetch_from_mysql", mysql)
    if java is not None:
        monkeypatch.setattr(strategies_runner, "_fetch_from_java", java)


def test_backtest_csv_dataset_path() -> None:
    _reset_cache()
    spec = _load_spec("backtest_csv_basic.json")
    result = backtest_runner.run_backtest_from_spec(spec)
    _assert_payload(result)


def test_backtest_delta_source(monkeypatch) -> None:
    _reset_cache()
    spec = _load_spec("backtest_delta_source.json")
    df = _load_source_frame()

    def fetch_delta(symbol: str, asset_class: str, spec_raw: dict) -> pd.DataFrame:
        assert symbol == "EURUSD"
        assert asset_class == "FX"
        return df.copy()

    def should_not_call(*_args, **_kwargs):
        raise AssertionError("Unexpected fallback source")

    _patch_fetch(monkeypatch, delta=fetch_delta, mysql=should_not_call, java=should_not_call)
    result = backtest_runner.run_backtest_from_spec(spec)
    _assert_payload(result)


def test_backtest_mysql_source(monkeypatch) -> None:
    _reset_cache()
    spec = _load_spec("backtest_mysql_source.json")
    df = _load_source_frame()

    def fetch_delta(*_args, **_kwargs):
        return None

    def fetch_mysql(symbol: str, spec_raw: dict) -> pd.DataFrame:
        assert symbol == "EURUSD"
        return df.copy()

    def should_not_call(*_args, **_kwargs):
        raise AssertionError("Unexpected java fallback")

    _patch_fetch(monkeypatch, delta=fetch_delta, mysql=fetch_mysql, java=should_not_call)
    result = backtest_runner.run_backtest_from_spec(spec)
    _assert_payload(result)


def test_backtest_java_source(monkeypatch) -> None:
    _reset_cache()
    spec = _load_spec("backtest_java_source.json")
    df = _load_source_frame()

    def fetch_delta(*_args, **_kwargs):
        return None

    def fetch_mysql(*_args, **_kwargs):
        return None

    def fetch_java(symbol: str, asset_class: str, spec_raw: dict) -> pd.DataFrame:
        assert symbol == "EURUSD"
        assert asset_class == "FX"
        return df.copy()

    _patch_fetch(monkeypatch, delta=fetch_delta, mysql=fetch_mysql, java=fetch_java)
    result = backtest_runner.run_backtest_from_spec(spec)
    _assert_payload(result)


def test_backtest_rows_cache_hit(monkeypatch) -> None:
    _reset_cache()
    spec = _load_spec("backtest_csv_basic.json")
    calls = {"count": 0}
    original = dataset_module.load_dataset

    def wrapped(spec_data):
        calls["count"] += 1
        return original(spec_data)

    monkeypatch.setattr(dataset_module, "load_dataset", wrapped)
    backtest_runner.run_backtest_from_spec(spec)
    backtest_runner.run_backtest_from_spec(spec)
    assert calls["count"] == 1
