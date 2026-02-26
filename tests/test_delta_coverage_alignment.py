from __future__ import annotations

import pandas as pd

from quant_engine.strategies.runner import _coverage_stats


def test_coverage_alignment_daily_non_aligned_window() -> None:
    start = pd.Timestamp("2024-10-14T14:10:49.491609Z")
    end = pd.Timestamp("2024-10-20T14:10:49.491609Z")
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(
                [
                    "2024-10-14T00:00:00Z",
                    "2024-10-15T00:00:00Z",
                    "2024-10-16T00:00:00Z",
                    "2024-10-17T00:00:00Z",
                    "2024-10-18T00:00:00Z",
                    "2024-10-19T00:00:00Z",
                    "2024-10-20T00:00:00Z",
                ],
                utc=True,
            )
        }
    )

    coverage, expected_len, observed_len, _, _, missing = _coverage_stats(
        df=df,
        start_dt=start,
        end_dt=end,
        timeframe="1d",
        asset_class="CRYPTO",
    )

    assert expected_len == 7
    assert observed_len == 7
    assert coverage == 1.0
    assert missing == []
