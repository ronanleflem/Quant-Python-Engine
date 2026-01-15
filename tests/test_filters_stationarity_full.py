from __future__ import annotations

import importlib.util
import numpy as np
import pandas as pd

from quant_engine.filters.stationarity import stationarity_filter


def test_stationarity_full_allows_missing_dependency() -> None:
    idx = pd.date_range("2025-01-01", periods=5, freq="min", tz="UTC")
    df = pd.DataFrame({"close": [1, 1, 1, 1, 1]}, index=idx)
    if importlib.util.find_spec("statsmodels") is None:
        mask = stationarity_filter(df, method="adf", allow_if_missing=True)
        assert mask.all()


def test_stationarity_full_adf() -> None:
    if importlib.util.find_spec("statsmodels") is None:
        return
    idx = pd.date_range("2025-01-01", periods=60, freq="min", tz="UTC")
    rng = np.random.default_rng(42)
    close = rng.normal(0, 1, size=len(idx)).cumsum() + 100
    df = pd.DataFrame({"close": close}, index=idx)
    mask = stationarity_filter(df, method="adf", window=20, adf_pvalue=0.99)
    assert mask.iloc[-1] == True
