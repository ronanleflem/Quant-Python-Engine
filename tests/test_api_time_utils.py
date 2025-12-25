from datetime import datetime

from quant_engine.api import app


def test_parse_format_round_trip_with_z_suffix() -> None:
    value = "2024-01-02T03:04:05Z"

    parsed = app._parse_iso_ts(value, "timestamp")

    assert parsed == datetime(2024, 1, 2, 3, 4, 5)
    assert app._format_ts(parsed) == "2024-01-02T03:04:05Z"


def test_parse_format_round_trip_with_utc_offset() -> None:
    value = "2024-01-02T03:04:05+00:00"

    parsed = app._parse_iso_ts(value, "timestamp")

    assert parsed == datetime(2024, 1, 2, 3, 4, 5)
    assert app._format_ts(parsed) == "2024-01-02T03:04:05Z"


def test_parse_format_round_trip_with_naive_timestamp() -> None:
    value = "2024-01-02T03:04:05"

    parsed = app._parse_iso_ts(value, "timestamp")

    assert parsed == datetime(2024, 1, 2, 3, 4, 5)
    assert app._format_ts(parsed) == "2024-01-02T03:04:05Z"
