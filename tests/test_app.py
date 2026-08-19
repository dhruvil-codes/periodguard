import io
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


def test_presets_endpoint(client):
    response = client.get("/api/presets")
    assert response.status_code == 200
    presets = response.json()
    assert len(presets) >= 3
    preset_ids = [p["id"] for p in presets]
    assert any("future_leak" in pid for pid in preset_ids)


def test_corpus_endpoints(client):
    # 1. List corpus
    resp = client.get("/api/corpus")
    assert resp.status_code == 200
    docs = resp.json()
    assert len(docs) >= 4

    # 2. Add manual document
    add_payload = {
        "id": "test_q1_fy25_doc",
        "company": "Acme Industries",
        "doc_type": "Press Release",
        "reporting_period": "Q1 FY25",
        "publication_date": "2024-08-10",
        "page": 1,
        "text": "Acme Industries Q1 FY25 EBITDA was steady.",
        "source_url": "https://example.com",
    }
    add_resp = client.post("/api/corpus/add", json=add_payload)
    assert add_resp.status_code == 200
    assert add_resp.json()["status"] == "success"

    # 3. Reset corpus
    reset_resp = client.post("/api/corpus/reset")
    assert reset_resp.status_code == 200
    assert reset_resp.json()["status"] == "success"


def test_upload_document_endpoint(client):
    file_content = b"Acme Industries Q3 FY25 revenue reached $400M with 18% EBITDA margin."
    files = {"file": ("acme_q3_report.txt", io.BytesIO(file_content), "text/plain")}
    data = {
        "company": "Acme Industries",
        "doc_type": "Quarterly Report",
        "reporting_period": "Q3 FY25",
        "publication_date": "2025-02-15",
    }
    response = client.post("/api/corpus/upload", files=files, data=data)
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert "acme_q3_report" in response.json()["doc"]["id"]


def test_evaluate_custom_endpoint(client):
    payload = {
        "company": "Acme Industries",
        "question": "What was EBITDA margin in Q4 FY25?",
        "as_of_date": "2025-05-15",
        "as_of_reporting_period": "Q4 FY25",
        "use_llm": False,
    }
    response = client.post("/api/evaluate/custom", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["correct_mode"]["status"] == "PASS"
    assert data["broken_mode"]["status"] == "FAIL"


def test_dashboard_html_render(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "PeriodGuard" in response.text
    assert "VERIFICATION ENGINE" in response.text
    assert "Manage Filings" in response.text
