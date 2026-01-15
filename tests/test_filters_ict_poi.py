from __future__ import annotations

import pandas as pd

from quant_engine.filters.ict_poi import ict_poi_filter


def test_ict_poi_ifvg_detection() -> None:
    idx = pd.date_range("2025-01-01", periods=6, freq="min", tz="UTC")
    df = pd.DataFrame(
        {
            "high": [1.00, 1.02, 1.20, 1.15, 0.95, 1.06],
            "low": [0.90, 0.95, 1.10, 1.05, 0.85, 1.00],
            "close": [0.95, 1.00, 1.15, 1.10, 0.90, 1.05],
        },
        index=idx,
    )
    mask = ict_poi_filter(
        df,
        include_ifvg=True,
        ifvg_lookback=10,
        ifvg_fill_threshold=0.5,
        ifvg_fill_count=2,
        tolerance=0.0,
        allow_if_missing=False,
        allow_if_empty=False,
    )
    assert mask.iloc[-1] == True


def test_ict_poi_fib_detection() -> None:
    idx = pd.date_range("2025-01-01", periods=10, freq="min", tz="UTC")
    close = [1.0, 1.1, 1.2, 1.15, 1.1, 1.05, 1.08, 1.12, 1.18, 1.2]
    df = pd.DataFrame(
        {
            "high": [c + 0.02 for c in close],
            "low": [c - 0.02 for c in close],
            "close": close,
        },
        index=idx,
    )
    mask = ict_poi_filter(
        df,
        include_fib=True,
        fib_lookback=10,
        fib_levels=(0.5,),
        fib_tolerance=0.05,
        allow_if_missing=False,
        allow_if_empty=False,
    )
    assert mask.any() == True


def test_ict_poi_order_block_detection() -> None:
    idx = pd.date_range("2025-01-01", periods=8, freq="min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [1.0, 1.02, 1.01, 1.05, 1.10, 1.08, 1.06, 1.07],
            "high": [1.03, 1.03, 1.02, 1.08, 1.12, 1.10, 1.08, 1.09],
            "low": [0.99, 1.00, 1.00, 1.04, 1.09, 1.06, 1.05, 1.06],
            "close": [1.02, 1.01, 1.05, 1.10, 1.11, 1.07, 1.06, 1.08],
        },
        index=idx,
    )
    mask = ict_poi_filter(
        df,
        include_ob=True,
        ob_lookback=5,
        ob_atr_window=3,
        ob_impulse_atr_mult=0.5,
        ob_use_body=True,
        allow_if_missing=False,
        allow_if_empty=False,
    )
    assert mask.any() == True


def test_ict_poi_breaker_and_model10_masks() -> None:
    idx = pd.date_range("2025-01-01", periods=10, freq="min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [1.0, 1.01, 0.98, 1.02, 1.05, 1.02, 1.01, 1.01, 1.02, 1.06],
            "high": [1.02, 1.03, 1.00, 1.04, 1.08, 1.04, 1.03, 1.03, 1.04, 1.09],
            "low": [0.98, 0.99, 0.95, 1.00, 1.03, 1.01, 1.00, 1.00, 1.01, 1.04],
            "close": [1.01, 1.00, 0.97, 1.03, 1.07, 1.02, 1.02, 1.01, 1.03, 1.08],
        },
        index=idx,
    )
    mask_breaker = ict_poi_filter(
        df,
        include_breaker=True,
        breaker_sweep_lookback=3,
        breaker_retrace_level=0.5,
        breaker_retrace_tol=0.2,
        breaker_continue_bars=3,
        allow_if_missing=False,
        allow_if_empty=False,
    )
    mask_model10 = ict_poi_filter(
        df,
        include_model10=True,
        model10_sweep_lookback=3,
        model10_consolidation_bars=3,
        model10_max_range_pct=0.1,
        allow_if_missing=False,
        allow_if_empty=False,
    )
    assert isinstance(mask_breaker, pd.Series)
    assert isinstance(mask_model10, pd.Series)
