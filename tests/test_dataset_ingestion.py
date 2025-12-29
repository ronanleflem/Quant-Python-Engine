import csv
import json
from datetime import date, datetime

from quant_engine.core.dataset import load_dataset
from quant_engine.core.spec import DataSpec


def _sample_rows():
    return [
        {
            "timestamp": "2020-01-01T00:00:00",
            "open": 10.0,
            "high": 11.0,
            "low": 9.5,
            "close": 10.5,
            "volume": 100,
            "symbol": "AAA",
        },
        {
            "timestamp": "2020-01-02T00:00:00",
            "open": 10.5,
            "high": 11.5,
            "low": 10.0,
            "close": 11.0,
            "volume": 110,
            "symbol": "AAA",
        },
        {
            "timestamp": "2020-01-03T00:00:00",
            "open": 11.0,
            "high": 12.0,
            "low": 10.5,
            "close": 11.5,
            "volume": 120,
            "symbol": "AAA",
        },
        {
            "timestamp": "2020-01-04T00:00:00",
            "open": 11.5,
            "high": 12.5,
            "low": 11.0,
            "close": 12.0,
            "volume": 130,
            "symbol": "AAA",
        },
        {
            "timestamp": "2020-01-02T00:00:00",
            "open": 20.0,
            "high": 21.0,
            "low": 19.5,
            "close": 20.5,
            "volume": 200,
            "symbol": "BBB",
        },
        {
            "timestamp": "2020-01-03T00:00:00",
            "open": 20.5,
            "high": 21.5,
            "low": 20.0,
            "close": 21.0,
            "volume": 210,
            "symbol": "BBB",
        },
    ]


def _run_assertions(rows):
    assert rows, "Expected filtered rows to be returned"
    assert {row["symbol"] for row in rows} == {"AAA"}
    for row in rows:
        ts_date = datetime.fromisoformat(row["timestamp"]).date()
        assert date(2020, 1, 2) <= ts_date <= date(2020, 1, 3)
        assert "session" in row


def test_load_dataset_filters_json(tmp_path):
    path = tmp_path / "tiny.json"
    path.write_text(json.dumps(_sample_rows()))

    spec = DataSpec(
        dataset_path=str(path),
        mysql=None,
        symbols=["AAA"],
        timeframe="1D",
        start="2020-01-02",
        end="2020-01-03",
    )

    rows = load_dataset(spec)

    _run_assertions(rows)


def test_load_dataset_filters_csv(tmp_path):
    path = tmp_path / "tiny.csv"
    rows = _sample_rows()
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    spec = DataSpec(
        dataset_path=str(path),
        mysql=None,
        symbols=["AAA"],
        timeframe="1D",
        start="2020-01-02",
        end="2020-01-03",
    )

    rows = load_dataset(spec)

    _run_assertions(rows)
