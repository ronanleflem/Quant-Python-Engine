import json
from pathlib import Path

import pytest

from quant_engine.backtest import engine


DATASET = [
    {
        "timestamp": "2020-01-01T00:00:00Z",
        "open": 100.0,
        "high": 102.0,
        "low": 99.0,
        "close": 101.0,
    },
    {
        "timestamp": "2020-01-02T00:00:00Z",
        "open": 101.0,
        "high": 103.0,
        "low": 100.0,
        "close": 102.0,
    },
    {
        "timestamp": "2020-01-03T00:00:00Z",
        "open": 102.0,
        "high": 104.0,
        "low": 101.0,
        "close": 103.0,
    },
    {
        "timestamp": "2020-01-04T00:00:00Z",
        "open": 103.0,
        "high": 105.0,
        "low": 102.0,
        "close": 104.0,
    },
    {
        "timestamp": "2020-01-05T00:00:00Z",
        "open": 104.0,
        "high": 106.0,
        "low": 103.0,
        "close": 105.0,
    },
]

SIGNALS = [1, 1, 0, 0, 0]
ATR_VALUES = [1.0] * len(DATASET)


def test_backtest_summary_matches_golden(tmp_path):
    _, _, summary = engine.run(DATASET, SIGNALS, ATR_VALUES, 1.0, 2.0)

    output_path = tmp_path / "summary.json"
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    exported = json.loads(output_path.read_text())

    golden_path = Path("tests/data/golden/backtest_summary.json")
    expected = json.loads(golden_path.read_text())

    assert exported.keys() == expected.keys()
    for key, value in expected.items():
        assert exported[key] == pytest.approx(value)
