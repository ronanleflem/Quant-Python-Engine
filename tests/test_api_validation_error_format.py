from fastapi.testclient import TestClient

from quant_engine.api import app as api_app


client = TestClient(api_app.fastapi_app)


def test_request_validation_errors_are_normalized() -> None:
    response = client.post("/stats/run", json={})

    assert response.status_code == 422
    payload = response.json()
    assert "errors" in payload
    assert isinstance(payload["errors"], list)
    assert payload["errors"]
    first = payload["errors"][0]
    assert set(first.keys()) == {"field", "code", "message"}
    assert first["field"]
    assert first["code"]
    assert first["message"]


def test_submit_invalid_spec_uses_same_error_shape() -> None:
    response = client.post("/submit", json={})

    assert response.status_code == 422
    payload = response.json()
    assert "errors" in payload
    assert payload["errors"][0]["field"] == "spec"
