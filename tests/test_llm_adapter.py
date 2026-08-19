from datetime import date
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from periodguard.answer import LLMAnswerAdapter
from periodguard.corpus import Corpus
from periodguard.models import RetrievalMode


@pytest.fixture
def corpus():
    path = Path(__file__).parent.parent / "data" / "corpus.json"
    return Corpus.from_json_file(path)


def test_llm_adapter_fallback_when_no_api_key(corpus):
    adapter = LLMAnswerAdapter(api_key=None)
    assert not adapter.is_available

    docs = corpus.filter_documents(company="Acme Industries", as_of_date=date(2025, 5, 15))
    answer = adapter.generate_answer("What was EBITDA margin?", docs, "Acme Industries", RetrievalMode.CORRECT)
    assert len(answer.claims) > 0
    assert answer.claims[0].citations[0].document_id == "acme_q4_fy25_results"


def test_llm_adapter_live_mock_response(corpus):
    adapter = LLMAnswerAdapter(api_key="mock-test-key")
    assert adapter.is_available

    inner_content = {
        "text": "Acme EBITDA margin increased 40 bps.",
        "claims": [
            {
                "text": "Acme EBITDA margin increased 40 bps.",
                "metric": "EBITDA margin",
                "value": 40.0,
                "unit": "bps",
                "period": "Q4 FY25",
                "citations": [
                    {
                        "document_id": "acme_q4_fy25_results",
                        "page": 12,
                        "quoted_text": "EBITDA margin increased by 40 bps sequentially.",
                    }
                ],
            }
        ],
    }

    mock_llm_payload = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(inner_content),
                }
            }
        ]
    }

    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps(mock_llm_payload).encode("utf-8")
    mock_response.__enter__.return_value = mock_response

    with patch("urllib.request.urlopen", return_value=mock_response):
        docs = corpus.filter_documents(company="Acme Industries", as_of_date=date(2025, 5, 15))
        answer = adapter.generate_answer("Query", docs, "Acme Industries", RetrievalMode.CORRECT)
        assert len(answer.claims) == 1
        assert answer.claims[0].value == 40.0
        assert answer.claims[0].citations[0].document_id == "acme_q4_fy25_results"
