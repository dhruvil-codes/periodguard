from datetime import date
from pathlib import Path
import pytest

from periodguard.corpus import Corpus
from periodguard.evaluator import Evaluator, get_default_case
from periodguard.models import (
    Citation,
    Claim,
    EvaluationCase,
    FailureType,
    RetrievalMode,
    StructuredAnswer,
    ValidationStatus,
)
from periodguard.validators import (
    CitationResolutionValidator,
    CitationSupportProxyValidator,
    EntityPeriodConsistencyValidator,
    TemporalConsistencyValidator,
)


@pytest.fixture
def corpus():
    path = Path(__file__).parent.parent / "data" / "corpus.json"
    return Corpus.from_json_file(path)


@pytest.fixture
def case():
    return get_default_case()


@pytest.fixture
def evaluator(corpus):
    return Evaluator(corpus=corpus)


# T1: Valid Q4 FY25 citation passes all checks
def test_t1_valid_q4_fy25_citation(evaluator, case):
    report = evaluator.evaluate(case=case, mode=RetrievalMode.CORRECT)
    assert report.status == ValidationStatus.PASS
    assert len(report.failures) == 0
    assert all(status == ValidationStatus.PASS for status in report.checks.values())


# T2: FY26 citation used for a 15 May 2025 question fails with FUTURE_PERIOD_LEAK
def test_t2_future_period_leak_detected(corpus, case):
    validator = TemporalConsistencyValidator()
    claims = [
        Claim(
            text="EBITDA margin improved by 40 bps in Q4 FY25 sequentially.",
            citations=[
                Citation(
                    document_id="acme_fy26_annual_report",
                    page=45,
                    quoted_text="EBITDA margin improved by 40 bps in Q4 FY25 sequentially before contracting in H1 FY26 due to subsequent global tariff hikes.",
                )
            ],
        )
    ]
    retrieved_docs = corpus.filter_documents(company="Acme Industries")
    status, failures = validator.validate(claims, retrieved_docs, corpus, case)

    assert status == ValidationStatus.FAIL
    assert len(failures) == 1
    assert failures[0].type == FailureType.FUTURE_PERIOD_LEAK
    assert failures[0].document_id == "acme_fy26_annual_report"
    assert failures[0].publication_date == date(2025, 8, 20)
    assert failures[0].as_of_date == date(2025, 5, 15)


# T3: Citation references nonexistent document fails with INVALID_CITATION
def test_t3_invalid_citation_nonexistent_doc(corpus, case):
    validator = CitationResolutionValidator()
    claims = [
        Claim(
            text="EBITDA margin improved by 40 bps.",
            citations=[
                Citation(
                    document_id="nonexistent_doc_123",
                    page=1,
                    quoted_text="Some fake quote",
                )
            ],
        )
    ]
    retrieved_docs = corpus.filter_documents(company="Acme Industries")
    status, failures = validator.validate(claims, retrieved_docs, corpus, case)

    assert status == ValidationStatus.FAIL
    assert len(failures) >= 1
    assert any(f.type == FailureType.INVALID_CITATION for f in failures)


# T4: Peer-company citation used fails with ENTITY_OR_PERIOD_MISMATCH
def test_t4_peer_company_entity_mismatch(corpus, case):
    validator = EntityPeriodConsistencyValidator()
    claims = [
        Claim(
            text="EBITDA margin increased by 50 bps.",
            citations=[
                Citation(
                    document_id="globex_q4_fy25_results",
                    page=8,
                    quoted_text="Globex Corp reported Q4 FY25 results. EBITDA margin increased by 50 bps sequentially to 19.1%",
                )
            ],
        )
    ]
    retrieved_docs = corpus.all_documents()
    status, failures = validator.validate(claims, retrieved_docs, corpus, case)

    assert status == ValidationStatus.FAIL
    assert len(failures) == 1
    assert failures[0].type == FailureType.ENTITY_OR_PERIOD_MISMATCH
    assert "Globex Corp" in failures[0].message


# T5: Citation has correct topic but no numeric value fails with UNSUPPORTED_CLAIM
def test_t5_citation_missing_numeric_value(corpus, case):
    validator = CitationSupportProxyValidator()
    claims = [
        Claim(
            text="EBITDA margin improved by 95 bps sequentially.",
            metric="EBITDA margin",
            value=95.0,
            unit="bps",
            citations=[
                Citation(
                    document_id="acme_q4_fy25_results",
                    page=12,
                    quoted_text="EBITDA margin increased by 40 bps sequentially to 18.4% compared to 18.0% in Q3 FY25.",
                )
            ],
        )
    ]
    retrieved_docs = [corpus.get_document("acme_q4_fy25_results")]
    status, failures = validator.validate(claims, retrieved_docs, corpus, case)

    assert status == ValidationStatus.FAIL
    assert len(failures) == 1
    assert failures[0].type == FailureType.UNSUPPORTED_CLAIM
    assert "95" in failures[0].message


# T6: Correct value uses wrong unit fails with UNSUPPORTED_CLAIM
def test_t6_wrong_unit_unsupported_claim(corpus, case):
    validator = CitationSupportProxyValidator()
    claims = [
        Claim(
            text="EBITDA margin improved by 40 percentage points.",
            metric="EBITDA margin",
            value=40.0,
            unit="percentage points",  # Text says "bps", not "percentage points"
            citations=[
                Citation(
                    document_id="acme_q4_fy25_results",
                    page=12,
                    quoted_text="EBITDA margin increased by 40 bps sequentially to 18.4% compared to 18.0% in Q3 FY25.",
                )
            ],
        )
    ]
    retrieved_docs = [corpus.get_document("acme_q4_fy25_results")]
    status, failures = validator.validate(claims, retrieved_docs, corpus, case)

    assert status == ValidationStatus.FAIL
    assert len(failures) == 1
    assert failures[0].type == FailureType.UNSUPPORTED_CLAIM
    assert "percentage points" in failures[0].message


# T7: Correct mode excludes later document and passes
def test_t7_correct_mode_acceptance(evaluator, case):
    report = evaluator.evaluate(case=case, mode=RetrievalMode.CORRECT)
    assert report.status == ValidationStatus.PASS
    doc_ids = [d.id for d in report.retrieved_documents]
    assert "acme_fy26_annual_report" not in doc_ids


# T8: Broken mode includes later document and fails with FUTURE_PERIOD_LEAK
def test_t8_broken_mode_acceptance(evaluator, case):
    report = evaluator.evaluate(case=case, mode=RetrievalMode.BROKEN)
    assert report.status == ValidationStatus.FAIL
    failure_types = [f.type for f in report.failures]
    assert FailureType.FUTURE_PERIOD_LEAK in failure_types
    assert report.checks["temporal_consistency"] == ValidationStatus.FAIL
