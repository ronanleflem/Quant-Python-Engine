from __future__ import annotations

import pandas as pd

from quant_engine.api.schemas import DataInputSpec, MySQLDataSpec
from quant_engine.core.dataset import load_ohlcv


def test_load_ohlcv_uses_delta_first(monkeypatch) -> None:
    calls = {"delta": 0, "mysql": 0}

    def _fake_delta(symbol: str, asset_class: str | None, spec: dict) -> pd.DataFrame:
        calls["delta"] += 1
        assert symbol == "BTC"
        assert asset_class == "CRYPTO"
        assert spec.get("delta_quotes") == "USDT"
        return pd.DataFrame(
            [
                {
                    "ts": "2024-01-01T00:00:00Z",
                    "symbol": "BTC",
                    "open": 1.0,
                    "high": 2.0,
                    "low": 0.5,
                    "close": 1.5,
                    "volume": 10.0,
                }
            ]
        )

    def _fake_mysql(*args, **kwargs):  # pragma: no cover - defensive
        calls["mysql"] += 1
        return pd.DataFrame()

    from quant_engine.strategies import runner as strategies_runner
    from quant_engine.datafeeds import mysql_feed

    monkeypatch.setattr(strategies_runner, "_fetch_from_delta", _fake_delta)
    monkeypatch.setattr(mysql_feed, "load_ohlcv_mysql", _fake_mysql)

    spec = DataInputSpec(
        dataset_path=None,
        mysql=None,
        symbols=["BTC"],
        timeframe="1h",
        start="2024-01-01T00:00:00Z",
        end="2024-01-02T00:00:00Z",
        asset_class="CRYPTO",
        currency="USDT",
        delta_quotes="USDT",
    )

    df = load_ohlcv(spec)
    assert calls["delta"] == 1
    assert calls["mysql"] == 0
    assert not df.empty
    assert list(df["symbol"].unique()) == ["BTC"]


def test_load_ohlcv_falls_back_to_mysql_when_delta_empty(monkeypatch) -> None:
    calls = {"delta": 0, "mysql": 0}

    def _fake_delta(symbol: str, asset_class: str | None, spec: dict) -> pd.DataFrame:
        calls["delta"] += 1
        return pd.DataFrame()

    def _fake_mysql(**kwargs) -> pd.DataFrame:
        calls["mysql"] += 1
        return pd.DataFrame(
            [
                {
                    "ts": "2024-01-01T00:00:00Z",
                    "symbol": "BTC",
                    "open": 1.0,
                    "high": 2.0,
                    "low": 0.5,
                    "close": 1.5,
                    "volume": 10.0,
                }
            ]
        )

    from quant_engine.strategies import runner as strategies_runner
    from quant_engine.datafeeds import mysql_feed

    monkeypatch.setattr(strategies_runner, "_fetch_from_delta", _fake_delta)
    monkeypatch.setattr(mysql_feed, "load_ohlcv_mysql", _fake_mysql)

    spec = DataInputSpec(
        dataset_path=None,
        mysql=MySQLDataSpec(connection_url="mysql+pymysql://u:p@localhost:3306/db", env_var=None),
        symbols=["BTC"],
        timeframe="1h",
        start="2024-01-01T00:00:00Z",
        end="2024-01-02T00:00:00Z",
        asset_class="CRYPTO",
        currency="USDT",
        delta_quotes="USDT",
    )

    df = load_ohlcv(spec)
    assert calls["delta"] == 1
    assert calls["mysql"] == 1
    assert not df.empty
