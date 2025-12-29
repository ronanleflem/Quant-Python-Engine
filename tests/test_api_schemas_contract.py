import pytest
from pydantic import ValidationError

from quant_engine.api.schemas import DataInputSpec, MySQLDataSpec, StatsPersistenceSpec


def test_mysql_data_spec_schema_alias_and_property() -> None:
    spec = MySQLDataSpec(schema="foo")

    assert spec.schema_ == "foo"
    assert spec.schema == "foo"


def test_stats_persistence_spec_enabled_from_legacy_flag() -> None:
    spec = StatsPersistenceSpec(store_stats_in_db=True)

    assert spec.enabled is True


@pytest.mark.parametrize(
    "payload,missing_field",
    [
        ({"timeframe": "M1", "start": "2024-01-01", "end": "2024-01-02"}, "symbols"),
        ({"symbols": ["EURUSD"], "start": "2024-01-01", "end": "2024-01-02"}, "timeframe"),
        ({"symbols": ["EURUSD"], "timeframe": "M1", "end": "2024-01-02"}, "start"),
        ({"symbols": ["EURUSD"], "timeframe": "M1", "start": "2024-01-01"}, "end"),
    ],
)
def test_data_input_spec_requires_fields(payload, missing_field) -> None:
    with pytest.raises(ValidationError) as excinfo:
        DataInputSpec(**payload)

    assert missing_field in str(excinfo.value)
