import json
from pathlib import Path

import pytest

from quant_engine.core import spec as spec_module


SPEC_PATH = Path(__file__).parent / "data" / "spec_example.json"


def _load_base_spec() -> dict:
    return json.loads(SPEC_PATH.read_text())


@pytest.mark.parametrize("missing_field", ["start", "end"])
def test_parse_spec_requires_start_end(missing_field: str) -> None:
    payload = _load_base_spec()
    payload["data"].pop(missing_field, None)

    with pytest.raises(ValueError, match="data.start and data.end are required"):
        spec_module.spec_from_dict(payload)


def test_parse_spec_requires_dataset_or_mysql() -> None:
    payload = _load_base_spec()
    payload["data"].pop("path", None)
    payload["data"].pop("dataset_path", None)

    with pytest.raises(ValueError, match="dataset_path/path or mysql"):
        spec_module.spec_from_dict(payload)


def test_parse_spec_rejects_non_iterable_search_space() -> None:
    payload = _load_base_spec()
    payload["strategy"]["search_space"]["ema_fast"] = 5

    with pytest.raises(TypeError):
        spec_module.spec_from_dict(payload)


def test_parse_spec_casts_mysql_chunk_minutes_to_int() -> None:
    payload = _load_base_spec()
    payload["data"].pop("path", None)
    payload["data"].pop("dataset_path", None)
    payload["data"]["mysql"] = {"chunk_minutes": "15"}

    parsed = spec_module.spec_from_dict(payload)

    assert parsed.data.mysql is not None
    assert parsed.data.mysql.chunk_minutes == 15
    assert isinstance(parsed.data.mysql.chunk_minutes, int)
