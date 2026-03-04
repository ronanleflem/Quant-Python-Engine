from __future__ import annotations

import pytest
from pydantic import ValidationError

from quant_engine.market_intelligence.models import (
    FEATURE_COLUMN_PREFIX,
    FEATURE_META_COLUMNS,
    LABEL_COLUMN_PREFIX,
    FeatureMeta,
)


def test_column_and_label_conventions_are_stable() -> None:
    assert FEATURE_COLUMN_PREFIX == "feat_"
    assert LABEL_COLUMN_PREFIX == "label_"
    assert FEATURE_META_COLUMNS == ("feature_name", "feature_version", "timeframe", "symbol")


def test_feature_meta_validation_success() -> None:
    meta = FeatureMeta(
        feature_name="volatility_20",
        feature_version="v1.2.3",
        timeframe="1H",
        symbol=" btc-usd ",
    )

    assert meta.feature_name == "volatility_20"
    assert meta.feature_version == "v1.2.3"
    assert meta.timeframe == "1h"
    assert meta.symbol == "BTC-USD"


@pytest.mark.parametrize("version", ["1", "1.0", "version-1.0.0"])
def test_feature_meta_rejects_invalid_version(version: str) -> None:
    with pytest.raises(ValidationError):
        FeatureMeta(
            feature_name="rsi_14",
            feature_version=version,
            timeframe="1h",
            symbol="BTC-USD",
        )


@pytest.mark.parametrize("timeframe", ["h1", "15min", "1q"])
def test_feature_meta_rejects_invalid_timeframe(timeframe: str) -> None:
    with pytest.raises(ValidationError):
        FeatureMeta(
            feature_name="rsi_14",
            feature_version="1.0.0",
            timeframe=timeframe,
            symbol="BTC-USD",
        )


@pytest.mark.parametrize("symbol", ["btc usd", "BTC@USD", ""])
def test_feature_meta_rejects_invalid_symbol(symbol: str) -> None:
    with pytest.raises(ValidationError):
        FeatureMeta(
            feature_name="rsi_14",
            feature_version="1.0.0",
            timeframe="1h",
            symbol=symbol,
        )
