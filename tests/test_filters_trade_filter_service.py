from __future__ import annotations

import pandas as pd

from quant_engine.filters.trade_filter_service import normalize_filter_rules, score_filter_rules


def _make_df() -> pd.DataFrame:
    idx = pd.date_range("2025-01-01 09:00:00", periods=4, freq="min", tz="UTC")
    return pd.DataFrame(
        {
            "open": [1.0, 1.0, 1.0, 1.0],
            "close": [1.01, 1.02, 1.03, 1.04],
            "high": [1.02, 1.03, 1.04, 1.05],
            "low": [0.99, 0.99, 0.99, 0.99],
        },
        index=idx,
    )


def test_normalize_filter_rules_defaults() -> None:
    rules = [
        {"type": "k_consecutive", "params": {"k": 2, "direction": "up"}, "weight": "2"},
        {"type": "intraday_time", "params": {"start": "09:00", "end": "09:02"}, "mode": "soft"},
        {"type": "adx", "params": {"window": 3}, "enabled": False},
    ]
    normalized = normalize_filter_rules(rules)
    assert normalized[0].weight == 2.0
    assert normalized[1].mode == "soft"
    assert normalized[2].enabled is False


def test_score_filter_rules_hard_soft() -> None:
    df = _make_df()
    rules = [
        {"type": "k_consecutive", "params": {"k": 2, "direction": "up"}, "mode": "hard"},
        {"type": "intraday_time", "params": {"start": "09:00", "end": "09:02"}, "mode": "soft"},
    ]
    result = score_filter_rules(df, rules, min_score_pct=1.0)
    final_mask = result["final_mask"]
    assert final_mask.iloc[1] == True
    assert final_mask.iloc[2] == False
