from quant_engine.api import app
from quant_engine.core import spec


def test_submit_status_result():
    sp = spec.load_spec("tests/data/spec_example.json")
    resp = app.submit(sp)
    status = app.status(resp.id)
    result = app.result(resp.id)
    assert status.status == "completed"
    assert result.result is not None
