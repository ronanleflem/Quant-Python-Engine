from __future__ import annotations

import pandas as pd

from quant_engine.market_intelligence.adapters.legacy_filters_adapter import LegacyFiltersAdapter
from quant_engine.market_intelligence.adapters.legacy_stats_adapter import LegacyStatsAdapter


def test_legacy_filters_adapter_normalizes_shape_and_columns() -> None:
    index = pd.date_range("2026-01-01", periods=4, freq="h")
    ohlcv = pd.DataFrame(
        {
            "open": [1.0, 2.0, 3.0, 4.0],
            "high": [1.1, 2.1, 3.1, 4.1],
            "low": [0.9, 1.9, 2.9, 3.9],
            "close": [1.0, 2.0, 3.0, 4.0],
            "volume": [100, 110, 120, 130],
        },
        index=index,
    )

    def _fake_runner(*_args, **_kwargs):
        return {
            "hard_mask": pd.Series([True, True, False, False], index=ohlcv.index),
            "score": pd.Series([1.0, 0.8, 0.2, 0.0], index=ohlcv.index),
            "score_pct": pd.Series([1.0, 0.8, 0.2, 0.0], index=ohlcv.index),
            "final_mask": pd.Series([True, True, False, False], index=ohlcv.index),
        }

    adapter = LegacyFiltersAdapter(runner=_fake_runner)
    out = adapter.run(ohlcv, rules=[{"type": "atr", "params": {"period": 14}}])

    assert out.shape == (4, 4)
    assert list(out.columns) == ["hard_mask", "score", "score_pct", "final_mask"]
    assert isinstance(out.index, pd.DatetimeIndex)
    assert str(out.index.tz) == "UTC"


def test_legacy_stats_adapter_normalizes_shape_and_columns() -> None:
    legacy = pd.DataFrame(
        {
            "ts": ["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"],
            "symbol": ["BTC-USD", "BTC-USD"],
            "event": ["cross", "cross"],
            "target": ["up_1", "up_1"],
            "count": [20, 30],
            "wins": [11, 17],
            "p": [0.55, 0.57],
            "split": ["test", "test"],
        }
    )

    adapter = LegacyStatsAdapter(runner=lambda _spec: legacy)
    out = adapter.run(spec={"ignored": True})

    assert out.shape[0] == 2
    assert list(out.columns) == [
        "symbol",
        "event",
        "condition_name",
        "condition_value",
        "target",
        "n",
        "successes",
        "p_hat",
        "ci_low",
        "ci_high",
        "p_mean",
        "p_map",
        "hdi_low",
        "hdi_high",
        "lift_freq",
        "lift_bayes",
        "insufficient",
        "split",
        "p_value",
        "q_value",
        "significant",
    ]
    assert str(out.index.tz) == "UTC"
    assert out["n"].tolist() == [20, 30]
    assert out["successes"].tolist() == [11, 17]
    assert out["p_hat"].tolist() == [0.55, 0.57]
