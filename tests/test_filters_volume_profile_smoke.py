"""Smoke tests for volume & market profile filters."""
from __future__ import annotations

import pandas as pd
import numpy as np

from quant_engine.filters.volume_profile import (
    volume_surge_filter,
    vwap_side_filter,
    poc_distance_filter,
)


def _make_df(n: int = 200) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    base_price = 1.10 + np.cumsum(np.random.normal(0, 0.0001, size=n))
    df = pd.DataFrame(
        {
            "open": base_price + np.random.normal(0, 0.00005, size=n),
            "high": base_price + np.abs(np.random.normal(0, 0.0001, size=n)),
            "low": base_price - np.abs(np.random.normal(0, 0.0001, size=n)),
            "close": base_price,
            "volume": np.random.randint(100, 200, size=n).astype(float),
        },
        index=idx,
    )
    return df


def test_volume_surge_filter_smoke():
    df = _make_df()
    result = volume_surge_filter(df, window=10, mode="z", z_thresh=1.0)
    assert isinstance(result, pd.Series)
    assert result.index.equals(df.index)
    assert result.dtype == bool
    assert result.any() or (~result).any()


def test_vwap_side_filter_local_fallback():
    df = _make_df()
    result = vwap_side_filter(df, from_levels=False)
    assert isinstance(result, pd.Series)
    assert result.index.equals(df.index)
    assert result.dtype == bool


def test_poc_distance_filter_with_and_without_levels(monkeypatch):
    df = _make_df()

    # Without symbol or repo -> False everywhere
    result = poc_distance_filter(df, max_distance=0.0005, symbol=None)
    assert isinstance(result, pd.Series)
    assert not result.any()

    # Mock levels repo/helpers
    class DummyRepo:
        def select_levels(self, **kwargs):
            ts = pd.date_range(df.index.min(), periods=5, freq="30min")
            return pd.DataFrame(
                {
                    "anchor_ts": ts,
                    "price": np.linspace(df["close"].min(), df["close"].max(), len(ts)),
                }
            )

    class DummyHelpers:
        @staticmethod
        def distance_to(df_local, levels_df, level_type="POC", side="mid"):
            prices = levels_df["price"].to_numpy()
            anchored = pd.Series(prices[-1], index=df_local.index)
            return (df_local["close"] - anchored).abs()

    monkeypatch.setattr(
        "quant_engine.filters.volume_profile.lvl_repo",
        DummyRepo(),
        raising=False,
    )
    monkeypatch.setattr(
        "quant_engine.filters.volume_profile.lvl_helpers",
        DummyHelpers,
        raising=False,
    )

    result = poc_distance_filter(df, max_distance=0.01, symbol="EURUSD")
    assert result.any()
    assert result.index.equals(df.index)
    assert result.dtype == bool
