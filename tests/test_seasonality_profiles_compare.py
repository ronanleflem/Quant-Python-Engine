from __future__ import annotations

import math

import pytest

from quant_engine.seasonality import profiles


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / (n - 1)
    var_x = sum((x - mean_x) ** 2 for x in xs) / (n - 1)
    var_y = sum((y - mean_y) ** 2 for y in ys) / (n - 1)
    return cov / math.sqrt(var_x * var_y)


def test_compare_profiles_returns_correlation() -> None:
    pl = pytest.importorskip("polars")
    df_a = pl.DataFrame(
        {
            "symbol": ["EURUSD", "EURUSD", "EURUSD"],
            "dim": ["hour", "hour", "hour"],
            "bin": [0, 1, 2],
            "lift": [0.10, 0.20, 0.05],
            "baseline": [0.0, 0.0, 0.0],
            "n": [10, 15, 12],
            "insufficient": [False, False, False],
        }
    )
    df_b = pl.DataFrame(
        {
            "symbol": ["DXY", "DXY", "DXY"],
            "dim": ["hour", "hour", "hour"],
            "bin": [0, 1, 2],
            "lift": [0.08, 0.25, 0.00],
            "baseline": [0.0, 0.0, 0.0],
            "n": [8, 12, 9],
            "insufficient": [False, False, False],
        }
    )

    comparison, corr = profiles.compare_profiles(df_a, df_b, "hour")

    assert not comparison.is_empty()
    assert {"lift_EURUSD", "lift_DXY", "lift_diff"}.issubset(set(comparison.columns))
    lifts_a = comparison.get_column("lift_EURUSD").to_list()
    lifts_b = comparison.get_column("lift_DXY").to_list()
    assert corr is not None
    expected_corr = _pearson(lifts_a, lifts_b)
    assert corr == pytest.approx(expected_corr)
    assert comparison.get_column("lift_diff").to_list() == [
        pytest.approx(a - b) for a, b in zip(lifts_a, lifts_b)
    ]
