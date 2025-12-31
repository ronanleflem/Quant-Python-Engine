from fastapi.testclient import TestClient

from quant_engine.api import app as api_app


client = TestClient(api_app.fastapi_app)


def test_stats_conditions_http_contract():
    response = client.get("/stats/conditions")

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, list)
    assert "session" in payload
    assert all(isinstance(item, str) for item in payload)


def test_filters_list_http_contract():
    response = client.get("/filters/list")

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, list)
    assert "adx" in payload
    assert all(isinstance(item, str) for item in payload)
