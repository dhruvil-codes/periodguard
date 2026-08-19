from datetime import date
from pathlib import Path
import pytest

from periodguard.corpus import Corpus
from periodguard.models import Document, RetrievalMode
from periodguard.retrieve import DeterministicRetriever, score_document


@pytest.fixture
def corpus_path():
    return Path(__file__).parent.parent / "data" / "corpus.json"


@pytest.fixture
def corpus(corpus_path):
    return Corpus.from_json_file(corpus_path)


def test_corpus_loading(corpus):
    docs = corpus.all_documents()
    assert len(docs) == 4
    doc_ids = {d.id for d in docs}
    assert "acme_q4_fy25_results" in doc_ids
    assert "acme_q4_fy25_earnings_call" in doc_ids
    assert "acme_fy26_annual_report" in doc_ids
    assert "globex_q4_fy25_results" in doc_ids


def test_corpus_company_filtering(corpus):
    acme_docs = corpus.filter_documents(company="Acme Industries")
    assert len(acme_docs) == 3
    for d in acme_docs:
        assert d.company == "Acme Industries"

    globex_docs = corpus.filter_documents(company="Globex Corp")
    assert len(globex_docs) == 1
    assert globex_docs[0].id == "globex_q4_fy25_results"


def test_corpus_date_filtering(corpus):
    as_of_date = date(2025, 5, 15)
    filtered = corpus.filter_documents(company="Acme Industries", as_of_date=as_of_date)
    
    # fy26_annual_report published 2025-08-20 must be excluded
    filtered_ids = [d.id for d in filtered]
    assert "acme_q4_fy25_results" in filtered_ids
    assert "acme_q4_fy25_earnings_call" in filtered_ids
    assert "acme_fy26_annual_report" not in filtered_ids
    assert len(filtered) == 2


def test_retrieval_correct_mode_excludes_future_document(corpus):
    retriever = DeterministicRetriever(corpus)
    query = "As of 15 May 2025, did Acme Industries' EBITDA margin improve in Q4 FY25 versus Q3 FY25, and what reason did management give?"
    as_of = date(2025, 5, 15)

    results = retriever.retrieve(
        query=query,
        company="Acme Industries",
        as_of_date=as_of,
        mode=RetrievalMode.CORRECT,
        top_k=5,
    )
    retrieved_ids = [doc.id for doc, _ in results]
    assert "acme_q4_fy25_results" in retrieved_ids
    assert "acme_q4_fy25_earnings_call" in retrieved_ids
    assert "acme_fy26_annual_report" not in retrieved_ids


def test_retrieval_broken_mode_includes_future_document(corpus):
    retriever = DeterministicRetriever(corpus)
    query = "As of 15 May 2025, did Acme Industries' EBITDA margin improve in Q4 FY25 versus Q3 FY25, and what reason did management give?"
    as_of = date(2025, 5, 15)

    results = retriever.retrieve(
        query=query,
        company="Acme Industries",
        as_of_date=as_of,
        mode=RetrievalMode.BROKEN,
        top_k=5,
    )
    retrieved_ids = [doc.id for doc, _ in results]
    assert "acme_fy26_annual_report" in retrieved_ids
    assert len(retrieved_ids) == 3
