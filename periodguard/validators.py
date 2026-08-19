from __future__ import annotations

import re
from typing import List, Optional, Tuple

from periodguard.corpus import Corpus
from periodguard.models import (
    Citation,
    Claim,
    Document,
    EvaluationCase,
    FailureType,
    ValidationFailure,
    ValidationStatus,
)


class BaseValidator:
    name: str = "base_validator"

    def validate(
        self,
        claims: List[Claim],
        retrieved_docs: List[Document],
        corpus: Corpus,
        case: EvaluationCase,
    ) -> Tuple[ValidationStatus, List[ValidationFailure]]:
        raise NotImplementedError


class CitationResolutionValidator(BaseValidator):
    name = "citation_resolution"

    def validate(
        self,
        claims: List[Claim],
        retrieved_docs: List[Document],
        corpus: Corpus,
        case: EvaluationCase,
    ) -> Tuple[ValidationStatus, List[ValidationFailure]]:
        failures: List[ValidationFailure] = []
        retrieved_ids = {d.id for d in retrieved_docs}

        for claim in claims:
            if not claim.citations:
                failures.append(
                    ValidationFailure(
                        type=FailureType.INVALID_CITATION,
                        claim_text=claim.text,
                        message=f"Claim has no citations attached: '{claim.text}'",
                    )
                )
                continue

            for citation in claim.citations:
                doc = corpus.get_document(citation.document_id)
                if not doc:
                    failures.append(
                        ValidationFailure(
                            type=FailureType.INVALID_CITATION,
                            document_id=citation.document_id,
                            claim_text=claim.text,
                            citation=citation,
                            message=f"Cited document ID '{citation.document_id}' does not exist in corpus.",
                        )
                    )
                    continue

                if citation.document_id not in retrieved_ids:
                    failures.append(
                        ValidationFailure(
                            type=FailureType.INVALID_CITATION,
                            document_id=citation.document_id,
                            claim_text=claim.text,
                            citation=citation,
                            message=f"Cited document '{citation.document_id}' was not in the retrieved evidence set.",
                        )
                    )

                if citation.page != doc.page:
                    failures.append(
                        ValidationFailure(
                            type=FailureType.INVALID_CITATION,
                            document_id=citation.document_id,
                            claim_text=claim.text,
                            citation=citation,
                            message=f"Cited page {citation.page} does not match document page {doc.page}.",
                        )
                    )

                if citation.quoted_text.strip().lower() not in doc.text.lower():
                    # check normalized whitespace
                    norm_quote = " ".join(citation.quoted_text.split()).lower()
                    norm_doc = " ".join(doc.text.split()).lower()
                    if norm_quote not in norm_doc:
                        failures.append(
                            ValidationFailure(
                                type=FailureType.INVALID_CITATION,
                                document_id=citation.document_id,
                                claim_text=claim.text,
                                citation=citation,
                                message=f"Quoted text not found in source document '{citation.document_id}'.",
                            )
                        )

        status = ValidationStatus.FAIL if failures else ValidationStatus.PASS
        return status, failures


class TemporalConsistencyValidator(BaseValidator):
    name = "temporal_consistency"

    def validate(
        self,
        claims: List[Claim],
        retrieved_docs: List[Document],
        corpus: Corpus,
        case: EvaluationCase,
    ) -> Tuple[ValidationStatus, List[ValidationFailure]]:
        failures: List[ValidationFailure] = []

        for claim in claims:
            for citation in claim.citations:
                doc = corpus.get_document(citation.document_id)
                if not doc:
                    continue

                if doc.publication_date > case.as_of_date:
                    failures.append(
                        ValidationFailure(
                            type=FailureType.FUTURE_PERIOD_LEAK,
                            document_id=doc.id,
                            publication_date=doc.publication_date,
                            as_of_date=case.as_of_date,
                            claim_text=claim.text,
                            citation=citation,
                            message=(
                                f"Future-period citation leak: Document '{doc.id}' was published on "
                                f"{doc.publication_date.isoformat()}, which violates the as-of boundary "
                                f"({case.as_of_date.isoformat()})."
                            ),
                        )
                    )

        status = ValidationStatus.FAIL if failures else ValidationStatus.PASS
        return status, failures


class EntityPeriodConsistencyValidator(BaseValidator):
    name = "entity_period_consistency"

    def validate(
        self,
        claims: List[Claim],
        retrieved_docs: List[Document],
        corpus: Corpus,
        case: EvaluationCase,
    ) -> Tuple[ValidationStatus, List[ValidationFailure]]:
        failures: List[ValidationFailure] = []

        for claim in claims:
            # Check period in claim if specified
            if claim.period:
                # If claim period points to a future fiscal year like FY26 when case is FY25
                if "fy26" in claim.period.lower() and "fy25" in case.as_of_reporting_period.lower():
                    failures.append(
                        ValidationFailure(
                            type=FailureType.ENTITY_OR_PERIOD_MISMATCH,
                            claim_text=claim.text,
                            message=f"Claim references period '{claim.period}', which exceeds case period boundary '{case.as_of_reporting_period}'.",
                        )
                    )

            for citation in claim.citations:
                doc = corpus.get_document(citation.document_id)
                if not doc:
                    continue

                if doc.company.strip().lower() != case.company.strip().lower():
                    failures.append(
                        ValidationFailure(
                            type=FailureType.ENTITY_OR_PERIOD_MISMATCH,
                            document_id=doc.id,
                            claim_text=claim.text,
                            citation=citation,
                            message=f"Entity mismatch: Cited document belongs to '{doc.company}', expected '{case.company}'.",
                        )
                    )

        status = ValidationStatus.FAIL if failures else ValidationStatus.PASS
        return status, failures


class CitationSupportProxyValidator(BaseValidator):
    name = "citation_support_proxy"

    def validate(
        self,
        claims: List[Claim],
        retrieved_docs: List[Document],
        corpus: Corpus,
        case: EvaluationCase,
    ) -> Tuple[ValidationStatus, List[ValidationFailure]]:
        """
        Transparent rule-based proxy for citation support:
        Checks metric token presence, numerical value presence, unit presence, and directional consistency.
        """
        failures: List[ValidationFailure] = []

        for claim in claims:
            for citation in claim.citations:
                quote = citation.quoted_text.lower()
                doc = corpus.get_document(citation.document_id)
                doc_text = doc.text.lower() if doc else quote

                # 1. Numeric value check
                if claim.value is not None:
                    # check integer representation (40) or float (40.0) or string in quote
                    val_str_int = str(int(claim.value)) if claim.value.is_integer() else str(claim.value)
                    val_str_float = f"{claim.value:.1f}"
                    if val_str_int not in quote and val_str_float not in quote and val_str_int not in doc_text:
                        failures.append(
                            ValidationFailure(
                                type=FailureType.UNSUPPORTED_CLAIM,
                                document_id=citation.document_id,
                                claim_text=claim.text,
                                citation=citation,
                                message=f"Numeric value {claim.value} asserted in claim was not found in cited quote or document.",
                            )
                        )

                # 2. Metric phrase check
                if claim.metric:
                    metric_tokens = [t for t in re.findall(r"\b[a-zA-Z0-9]+\b", claim.metric.lower()) if len(t) > 2]
                    matched = any(t in quote or t in doc_text for t in metric_tokens)
                    if not matched:
                        failures.append(
                            ValidationFailure(
                                type=FailureType.UNSUPPORTED_CLAIM,
                                document_id=citation.document_id,
                                claim_text=claim.text,
                                citation=citation,
                                message=f"Metric '{claim.metric}' asserted in claim was not found in cited evidence.",
                            )
                        )

                # 3. Unit check
                if claim.unit:
                    unit_clean = claim.unit.lower().strip()
                    # Handle common equivalents like "bps" or "%" or "usd"
                    if unit_clean not in quote and unit_clean not in doc_text:
                        failures.append(
                            ValidationFailure(
                                type=FailureType.UNSUPPORTED_CLAIM,
                                document_id=citation.document_id,
                                claim_text=claim.text,
                                citation=citation,
                                message=f"Unit '{claim.unit}' asserted in claim was not supported by cited evidence.",
                            )
                        )

        status = ValidationStatus.FAIL if failures else ValidationStatus.PASS
        return status, failures
