from __future__ import annotations

import pandas as pd
import pytest

from quant_engine.market_intelligence.pipeline import compute_features


def _sample_ohlcv() -> pd.DataFrame:
    idx = pd.date_range("2026-01-01 00:00:00", periods=6, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "open": [99, 100, 101, 100, 102, 103],
            "high": [100, 101, 102, 101, 103, 104],
            "low": [98, 99, 100, 99, 101, 102],
            "close": [99.3, 100.2, 101.4, 100.5, 102.2, 103.3],
            "volume": [1100, 1200, 1300, 1250, 1400, 1450],
        },
        index=idx,
    )


def assert_feature_frame_contract(frame: pd.DataFrame, required_columns: set[str]) -> None:
    assert isinstance(frame.index, pd.DatetimeIndex)
    assert frame.index.is_monotonic_increasing
    assert str(frame.index.tz) == "UTC"
    assert required_columns.issubset(set(frame.columns))


@pytest.fixture()
def feature_frame() -> pd.DataFrame:
    ohlcv = _sample_ohlcv()
    features = compute_features(ohlcv)
    return features.sort_index()


def test_feature_frame_contract_index_and_required_columns(feature_frame: pd.DataFrame) -> None:
    assert_feature_frame_contract(
        feature_frame,
        required_columns={"feat_return_1", "feat_volatility_5", "feat_corr_close_volume_5"},
    )


@pytest.mark.parametrize(
    ("column", "allow_nan"),
    [
        ("feat_return_1", False),
        ("feat_volatility_5", False),
        ("feat_corr_close_volume_5", False),
    ],
)
def test_feature_nan_policy_by_feature(feature_frame: pd.DataFrame, column: str, allow_nan: bool) -> None:
    has_nan = bool(feature_frame[column].isna().any())
    assert has_nan is allow_nan
