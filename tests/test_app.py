import io
from fastapi.testclient import TestClient
import pytest

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
    assert "case" in data
    assert data["case"]["company"] == "Acme Industries"


def test_presets_endpoint(client):
    response = client.get("/api/presets")
    assert response.status_code == 200
    presets = response.json()
    assert len(presets) >= 4
    assert any(p["id"] == "future_leak_default" for p in presets)


def test_corpus_endpoints(client):
    response = client.get("/api/corpus")
    assert response.status_code == 200
    docs = response.json()
    assert len(docs) >= 4

    # Add doc
    payload = {
        "id": "test_add_doc",
        "company": "Acme Industries",
        "doc_type": "Filing",
        "reporting_period": "Q1 FY25",
        "publication_date": "2025-01-10",
        "page": 1,
        "text": "Acme had 100M revenue.",
    }
    add_resp = client.post("/api/corpus/add", json=payload)
    assert add_resp.status_code == 200

    # Delete doc
    del_resp = client.delete("/api/corpus/test_add_doc")
    assert del_resp.status_code == 200
    assert del_resp.json()["status"] == "success"

    # Reset
    reset_resp = client.post("/api/corpus/reset")
    assert reset_resp.status_code == 200
    assert reset_resp.json()["status"] == "success"


def test_clear_corpus_endpoint(client):
    clear_resp = client.post("/api/corpus/clear")
    assert clear_resp.status_code == 200
    assert clear_resp.json()["count"] == 0

    # Reset
    reset_resp = client.post("/api/corpus/reset")
    assert reset_resp.status_code == 200
    assert reset_resp.json()["count"] >= 4


def test_detect_meta_endpoint(client):
    file_content = b"APPLE INC. - Condensed Financial Statements Q1 FY25. Published: January 30, 2025. Total revenue was $124.3B."
    files = {"file": ("apple_q1_fy25.txt", io.BytesIO(file_content), "text/plain")}
    response = client.post("/api/corpus/detect-meta", files=files)
    assert response.status_code == 200
    meta = response.json()["metadata"]
    assert meta["company"] == "Apple Inc."
    assert "Q1" in meta["reporting_period"]
    assert meta["publication_date"] == "2025-01-30"


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
