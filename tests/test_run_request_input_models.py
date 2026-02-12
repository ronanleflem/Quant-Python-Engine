import pytest
from pydantic import ValidationError

from quant_engine.api.run_request_input import validate_run_request_input


def test_accepts_backtest_payload_with_java_aliases() -> None:
    payload = {
        "specType": "backtest",
        "catalogVersion": "v1",
        "data": {
            "symbol": "EURUSD",
            "timeframe": "M1",
            "startDate": "2025-01-01",
            "endDate": "2025-01-31",
        },
        "signal": {"type": "ema_cross", "fast": 9, "slow": 21, "requireCrossing": True},
    }

    parsed = validate_run_request_input(payload)

    assert parsed.spec_type == "backtest"
    assert parsed.catalog_version == "v1"
    assert parsed.data.start_date == "2025-01-01"


@pytest.mark.parametrize(
    "payload",
    [
        {
            "spec_type": "dca",
            "catalog_version": "v1",
            "data": {"symbol": "SPY", "timeframe": "D1", "start_date": "2020-01-01", "end_date": "2025-01-01"},
            "strategy": {"type": "dca_equity", "grid": ["monthly"], "params": {}},
        },
        {
            "spec_type": "market_stats",
            "catalog_version": "v1",
            "data": {"symbol": "EURUSD", "timeframe": "M5", "lookback": 2000},
            "stats": {
                "event": {"id": "ema_cross"},
                "condition": {"id": "session", "params": {"session": "london"}},
                "target": {"id": "next_bar_up"},
            },
        },
        {
            "spec_type": "seasonality",
            "catalog_version": "v1",
            "data": {"symbol": "EURUSD", "timeframe": "H1"},
            "seasonality": {"profile": {"id": "intraday"}, "signal": {"method": "threshold"}},
        },
        {
            "spec_type": "stress_tests",
            "catalog_version": "v1",
            "data": {"symbol": "EURUSD", "timeframe": "M1", "start_date": "2024-01-01", "end_date": "2024-06-01"},
        },
    ],
)
def test_accepts_all_spec_types(payload: dict) -> None:
    parsed = validate_run_request_input(payload)
    assert parsed.spec_type == payload["spec_type"]


def test_rejects_unknown_spec_type() -> None:
    payload = {"spec_type": "unknown", "catalog_version": "v1", "data": {}}

    with pytest.raises(ValidationError) as exc_info:
        validate_run_request_input(payload)

    assert "spec_type" in str(exc_info.value)


def test_rejects_missing_required_branch_block() -> None:
    payload = {
        "spec_type": "market_stats",
        "catalog_version": "v1",
        "data": {"symbol": "EURUSD", "timeframe": "M5"},
    }

    with pytest.raises(ValidationError) as exc_info:
        validate_run_request_input(payload)

    assert "stats" in str(exc_info.value)


def test_rejects_unknown_top_level_field() -> None:
    payload = {
        "spec_type": "dca",
        "catalog_version": "v1",
        "data": {"symbol": "SPY", "timeframe": "D1", "start_date": "2020-01-01", "end_date": "2025-01-01"},
        "strategy": {"type": "dca_equity", "grid": ["monthly"], "params": {}},
        "unexpected": True,
    }

    with pytest.raises(ValidationError) as exc_info:
        validate_run_request_input(payload)

    assert "unexpected" in str(exc_info.value)
