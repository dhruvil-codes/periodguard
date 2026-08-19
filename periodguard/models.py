from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RetrievalMode(str, Enum):
    CORRECT = "correct_date_filtered"
    BROKEN = "broken_no_date_filter"


class FailureType(str, Enum):
    INVALID_CITATION = "INVALID_CITATION"
    FUTURE_PERIOD_LEAK = "FUTURE_PERIOD_LEAK"
    ENTITY_OR_PERIOD_MISMATCH = "ENTITY_OR_PERIOD_MISMATCH"
    UNSUPPORTED_CLAIM = "UNSUPPORTED_CLAIM"


class ValidationStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"


class Document(BaseModel):
    id: str = Field(..., description="Unique document identifier")
    company: str = Field(..., description="Company name")
    doc_type: str = Field(..., description="Type of document (e.g. 10-Q, Results, Earnings Call, Annual Report)")
    reporting_period: str = Field(..., description="Financial reporting period (e.g. Q4 FY25, FY26)")
    publication_date: date = Field(..., description="Date document was published or call occurred")
    page: int = Field(..., description="Page number of the document section")
    text: str = Field(..., description="Text content of the document or excerpt")
    source_url: str = Field(..., description="Source URL or provenance metadata")


class Citation(BaseModel):
    document_id: str = Field(..., description="ID of cited document")
    page: int = Field(..., description="Page number cited")
    quoted_text: str = Field(..., description="Exact quoted text from the document")


class Claim(BaseModel):
    text: str = Field(..., description="The factual claim text")
    metric: Optional[str] = Field(None, description="Financial metric name (e.g. EBITDA margin)")
    value: Optional[float] = Field(None, description="Numeric value extracted or asserted")
    unit: Optional[str] = Field(None, description="Unit of measurement (e.g. bps, %, USD)")
    period: Optional[str] = Field(None, description="Financial period the claim refers to")
    citations: List[Citation] = Field(default_factory=list, description="Citations supporting this claim")


class StructuredAnswer(BaseModel):
    text: str = Field(..., description="Overall synthesized answer text")
    claims: List[Claim] = Field(default_factory=list, description="Atomic claims with citations")


class EvaluationCase(BaseModel):
    id: str = Field(..., description="Unique case ID")
    company: str = Field(..., description="Target company")
    question: str = Field(..., description="The financial research question")
    as_of_date: date = Field(..., description="Cut-off date for eligible evidence")
    as_of_reporting_period: str = Field(..., description="Target reporting period")
    expected_metric: Optional[str] = Field(None, description="Expected metric name")
    expected_unit: Optional[str] = Field(None, description="Expected metric unit")
    negative_control_documents: List[str] = Field(default_factory=list, description="Documents that must be excluded")


class ValidationFailure(BaseModel):
    type: FailureType = Field(..., description="Categorized failure type")
    document_id: Optional[str] = Field(None, description="ID of offending document if applicable")
    publication_date: Optional[date] = Field(None, description="Publication date of offending document")
    as_of_date: Optional[date] = Field(None, description="As-of date requested by evaluation case")
    claim_text: Optional[str] = Field(None, description="Claim that caused the failure")
    citation: Optional[Citation] = Field(None, description="Offending citation object")
    message: str = Field(..., description="Human-readable explanation of why this check failed")


class EvaluationReport(BaseModel):
    status: ValidationStatus = Field(..., description="Overall evaluation status (PASS/FAIL)")
    mode: RetrievalMode = Field(..., description="Retrieval mode used during evaluation")
    question: str = Field(..., description="The financial question evaluated")
    as_of_date: date = Field(..., description="The cut-off as-of date")
    checks: Dict[str, ValidationStatus] = Field(..., description="Status per validator check")
    failures: List[ValidationFailure] = Field(default_factory=list, description="List of detected failures")
    claims: List[Claim] = Field(default_factory=list, description="Structured claims produced")
    retrieved_documents: List[Document] = Field(default_factory=list, description="Documents retrieved for the case")
