from __future__ import annotations

import pandas as pd

from quant_engine.filters.mtf_anomaly import mtf_anomaly_filter


def _make_df() -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=12, freq="1min", tz="UTC")
    data = {
        "open": [1.0] * len(idx),
        "high": [1.1] * len(idx),
        "low": [0.9] * len(idx),
        "close": [1.0] * len(idx),
        "volume": [100.0] * len(idx),
    }
    return pd.DataFrame(data, index=idx)


def test_mtf_anomaly_returns_mask() -> None:
    df = _make_df()
    mask = mtf_anomaly_filter(
        df,
        higher_timeframe="2m",
        base_window=3,
        higher_window=3,
        mad_threshold=0.01,
        require_both=True,
    )
    assert len(mask) == len(df)
    assert mask.dtype == bool


def test_mtf_anomaly_blocks_when_both_anomalous() -> None:
    df = _make_df()
    mask = mtf_anomaly_filter(
        df,
        higher_timeframe="2m",
        base_window=3,
        higher_window=3,
        mad_threshold=0.001,
        require_both=True,
    )
    assert bool(mask.iloc[-1]) is False


def test_mtf_anomaly_entropy_mode() -> None:
    df = _make_df()
    mask = mtf_anomaly_filter(
        df,
        higher_timeframe="2m",
        base_window=3,
        higher_window=3,
        metric="entropy",
        entropy_threshold=0.1,
        require_both=True,
    )
    assert len(mask) == len(df)
    assert mask.dtype == bool
