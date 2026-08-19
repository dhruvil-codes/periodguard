import pytest
from fastapi.testclient import TestClient
from periodguard.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_evaluate_endpoint(client):
    response = client.post("/evaluate")
    assert response.status_code == 200
    data = response.json()
    assert "correct_mode" in data
    assert "broken_mode" in data
    assert data["correct_mode"]["status"] == "PASS"
    assert data["broken_mode"]["status"] == "FAIL"


def test_report_endpoint(client):
    response = client.get("/report")
    assert response.status_code == 200
    data = response.json()
    assert data["correct_mode"]["status"] == "PASS"
    assert data["broken_mode"]["status"] == "FAIL"


def test_dashboard_html_render(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "PeriodGuard" in response.text
    assert "Date-Filtered Mode" in response.text
    assert "FUTURE_PERIOD_LEAK" in response.text
