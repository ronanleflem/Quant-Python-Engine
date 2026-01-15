from __future__ import annotations

import pandas as pd

from quant_engine.filters.htf_poi import htf_poi_filter


def test_htf_poi_allow_if_missing() -> None:
    idx = pd.date_range("2025-01-01", periods=5, freq="min", tz="UTC")
    df = pd.DataFrame({"close": [1.0, 1.01, 1.0, 1.02, 1.01]}, index=idx)
    mask = htf_poi_filter(df, symbol="EURUSD", allow_if_missing=True)
    assert isinstance(mask, pd.Series)
    assert mask.index.equals(df.index)
    assert mask.dtype == bool
    assert mask.all()
